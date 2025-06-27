# === scripts/train_persona_lora.py (Adapted for Qwen3 Persona LoRA) ===
"""
Fine-tune a LoRA adapter on the persona dataset. This script
wraps HuggingFace Transformers + PEFT and is designed to run on consumer GPUs
using 4-bit QLoRA.

Example:
    python scripts/train_persona_lora.py \\
        --base-model Qwen/Qwen2-4B-Instruct \\
        --dataset ./data/persona_lora_traindata_cleaned_v1.jsonl \\
        --output-dir ./trained_adapters/oscar_c_persona_qwen3_4b_v1 \\
        --lora-r 32 --lora-alpha 64 --epochs 3 --batch-size 2 \\
        --max-seq-length 4096 
"""

import argparse
import json
import os
from pathlib import Path
import logging # Added logging

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling, # SFTTrainer might be better for chat, but sticking to provided script structure
    Trainer,
    TrainingArguments,
    # SFTTrainer, # Alternative for chat-formatted data
    # TrainingArguments as SFTTrainingArguments,
)

logger = logging.getLogger(__name__)

# ----------------------- Main ---------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a Persona LoRA adapter for OSCAR-C (qwen3:4b)")
    # --- Model and Data ---
    parser.add_argument("--base-model", required=True, help="HF model ID for the Qwen3 4B base model (e.g., Qwen/Qwen2-4B-Instruct)")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset file (e.g., ./data/persona_lora_traindata_cleaned_v1.jsonl)")
    parser.add_argument("--output-dir", required=True, help="Directory to save the trained LoRA adapter (e.g., ./trained_adapters/oscar_c_persona_qwen3_4b_v1)")
    
    # --- LoRA Hyperparameters ---
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank (r)")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha (typically 2*r)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout rate")
    # Target modules for Qwen models (as per qwen3 lora guide.txt)
    parser.add_argument("--lora-target-modules", nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        help="Modules to apply LoRA to. Default for Qwen-like models.")

    # --- Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per device training batch size (adjust based on VRAM)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (AdamW optimizer usually needs smaller LR for LLMs)") # Adjusted LR
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Maximum sequence length for tokenization") # Increased
    parser.add_argument("--max-steps", type=int, default=-1, help="Override total training steps (for quick tests)")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log training progress every N steps")
    parser.add_argument("--save-steps", type=int, default=0, help="Save checkpoint every N steps. 0 means save per epoch.")


    # --- Quantization and Hardware ---
    parser.add_argument("--use-qlora", action="store_true", default=True, help="Use QLoRA (4-bit quantization). Default: True")
    parser.add_argument("--no-qlora", action="store_false", dest="use_qlora", help="Disable QLoRA (train in bf16/fp16 if VRAM allows).")
    
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])


    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="[%(asctime)s] [%(levelname)-7s] [%(name)-20s] %(message)s")

    logger.info(f"Starting LoRA training with parameters: {args}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load tokenizer & base model ---
    logger.info(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)

    # Configure padding token for Qwen models
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            logger.info(f"Tokenizer pad_token not set. Using eos_token ({tokenizer.eos_token}) as pad_token.")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Add a generic pad token if EOS is also missing (less common for instruct models)
            logger.warning("Tokenizer pad_token and eos_token are not set. Adding a new pad_token '<|pad|>'.")
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    
    # Ensure model is resized if new tokens were added (though pad_token usually exists or uses eos)
    # model.resize_token_embeddings(len(tokenizer)) # Do this after model load

    quantization_config = None
    if args.use_qlora:
        logger.info("Using QLoRA (4-bit quantization).")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, # or torch.float16 if bfloat16 not supported
        )
    else:
        logger.info("QLoRA disabled. Model will be loaded in default precision (or bf16/fp16 if specified elsewhere).")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config if args.use_qlora else None,
        torch_dtype=torch.bfloat16 if args.use_qlora else "auto", # bf16 for QLoRA, auto for others
        device_map="auto", # Automatically distribute model across available GPUs
        trust_remote_code=True,
    )
    
    # Important: Resize token embeddings if tokenizer had new tokens added (e.g. pad_token)
    # This should be done *before* prepare_model_for_kbit_training or get_peft_model
    if tokenizer.pad_token_id == tokenizer.eos_token_id and tokenizer.eos_token_id is None: # check if new pad added
         base_model.resize_token_embeddings(len(tokenizer))


    if args.use_qlora:
        base_model = prepare_model_for_kbit_training(base_model)

    # --- Apply PEFT LoRA ---
    logger.info(f"Applying LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    logger.info(f"Targeting modules: {args.lora_target_modules}")
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # --- Dataset Loading and Processing ---
    logger.info(f"Loading dataset from: {args.dataset}")
    try:
        dataset = load_dataset("json", data_files=args.dataset, split="train")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    logger.info(f"Dataset loaded. Number of examples: {len(dataset)}")

    # Tokenization function for Qwen chat format
    # Dataset fields: "instruction" (detailed system prompt for LoRA), "input" (UserQuery + State), "output" (Thinking + Response)
    def tokenize_fn(examples):
        # Using Qwen specific chat tokens: <|im_start|>, <|im_end|>
        # For Qwen, eos_token is often <|im_end|> or <|endoftext|>.
        # We assume tokenizer.eos_token is correctly set (e.g. to <|im_end|> or <|endoftext|>)

        outputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for i in range(len(examples["instruction"])):
            system_prompt_text = examples["instruction"][i]
            user_input_text = examples["input"][i]
            assistant_output_text = examples["output"][i]

            # Construct the full text in Qwen chat format
            # <|im_start|>system\n{system_prompt}<|im_end|>
            # <|im_start|>user\n{user_prompt}<|im_end|>
            # <|im_start|>assistant\n{assistant_response}<|im_end|> (or tokenizer.eos_token)
            
            # Parts of the prompt
            system_part = f"<|im_start|>system\n{system_prompt_text}<|im_end|>\n"
            user_part = f"<|im_start|>user\n{user_input_text}<|im_end|>\n"
            assistant_start_tag = "<|im_start|>assistant\n"
            
            # The full text the model sees as input (prompt + start of response)
            prompt_text_for_model = system_part + user_part + assistant_start_tag
            # The part the model generates (response)
            response_text_for_model = f"{assistant_output_text}{tokenizer.eos_token}"

            full_dialogue_text = prompt_text_for_model + response_text_for_model
            
            # Tokenize the full dialogue
            tokenized_dialogue = tokenizer(
                full_dialogue_text,
                truncation=True,
                max_length=args.max_seq_length,
                padding=False, # SFTTrainer or DataCollator will handle padding
                return_attention_mask=True # Ensure attention mask is returned
            )

            # Tokenize the prompt part to know its length for masking labels
            tokenized_prompt = tokenizer(
                prompt_text_for_model,
                truncation=False, # Don't truncate prompt part alone if it's too long; full dialogue truncation handles it
                padding=False,
                return_attention_mask=False
            )
            prompt_token_length = len(tokenized_prompt["input_ids"])

            # Create labels: input_ids are copied, then prompt part is masked with -100
            labels = list(tokenized_dialogue["input_ids"])
            for k in range(min(prompt_token_length, len(labels))): # Ensure we don't go out of bounds if prompt is longer than max_seq_length
                labels[k] = -100
            
            outputs["input_ids"].append(tokenized_dialogue["input_ids"])
            outputs["attention_mask"].append(tokenized_dialogue["attention_mask"])
            outputs["labels"].append(labels)
            
        return outputs

    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True, # Process in batches for efficiency
        remove_columns=dataset.column_names, # Remove original text columns
        desc="Tokenizing dataset for Qwen Persona LoRA",
    )
    logger.info(f"Dataset tokenized. Example of first tokenized input_ids: {tokenized_dataset[0]['input_ids'][:30]}")
    logger.info(f"Example of first tokenized labels: {tokenized_dataset[0]['labels'][:50]}")


    # Data Collator (standard for Causal LM, handles padding)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # --- Training Arguments ---
    save_strategy_val = "steps" if args.save_steps > 0 else "epoch"
    save_steps_val = args.save_steps if args.save_steps > 0 else 0 # Trainer handles 0 as per epoch

    training_args_dict = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "optim": "paged_adamw_8bit" if args.use_qlora else "adamw_torch", # Optimizer based on QLoRA
        "logging_dir": f"{args.output_dir}/logs",
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "save_strategy": save_strategy_val,
        "report_to": "tensorboard", # Or "wandb" if configured
        "bf16": not args.use_qlora and torch.cuda.is_bf16_supported(), # Use bf16 if not QLoRA and supported
        "fp16": not args.use_qlora and not torch.cuda.is_bf16_supported(), # Use fp16 if not QLoRA and bf16 not supported
        "gradient_checkpointing": True, # Saves memory
        "max_steps": args.max_steps if args.max_steps > 0 else -1,
        "warmup_steps": 50, # A small number of warmup steps
        "lr_scheduler_type": "cosine", # Common scheduler
        "weight_decay": 0.01,
        "ddp_find_unused_parameters": False, # Set to False if not using DDP or if it causes issues
    }
    if save_strategy_val == "steps":
        training_args_dict["save_steps"] = save_steps_val
        training_args_dict["save_total_limit"] = 3 # Keep last 3 checkpoints


    train_args_obj = TrainingArguments(**training_args_dict)

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=train_args_obj,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        # SFTTrainer might be an alternative if dataset formatted differently
    )
    
    # --- Start Training ---
    logger.info("Starting LoRA training...")
    try:
        trainer.train()
        logger.info("Training complete.")
    except Exception as e:
        logger.exception(f"An error occurred during training: {e}")
        sys.exit(1)

    # --- Save LoRA adapter ---
    logger.info(f"Saving final LoRA adapter to: {args.output_dir}")
    # model.save_pretrained(args.output_dir) # PEFT model saves adapter_model.bin
    trainer.save_model(args.output_dir) # Trainer's method, also saves adapter
    tokenizer.save_pretrained(args.output_dir) # Save tokenizer for easy loading
    
    # Save training arguments for reference
    with open(Path(args.output_dir) / "training_args.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)

    logger.info(f"LoRA adapter and tokenizer saved to: {args.output_dir}")
    logger.info("To use with Ollama, create a Modelfile pointing to this adapter directory.")

if __name__ == "__main__":
    main()