import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# --- Configuration (UPDATED WITH YOUR INFO) ---

# The official Hugging Face repository for the base model you trained on.
# This is taken directly from your adapter_config.json file.
BASE_MODEL_ID = "Qwen/Qwen3-4B"

# Path to the directory containing your trained LoRA adapter.
# Replace this with the actual path to the folder with adapter_config.json.
LORA_ADAPTER_PATH = os.path.expanduser("~/Desktop/nodes/app/extensions/consciousness_experiment/trained_adapters/oscar_c_persona_qwen3_4b_v1")

# Path where the new, merged model will be saved.
# We'll create this directory.
MERGED_MODEL_OUTPUT_PATH = "./merged_models/oscar-persona-qwen3-4b-merged"

# --- Main Script ---
print("--- Step 1: Loading Base Model ---")
print(f"Model ID: {BASE_MODEL_ID}")
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,  # float16 is standard for merging
        device_map="auto"           # Let transformers handle GPU/CPU placement
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    print("Base model and tokenizer loaded successfully.")
except Exception as e:
    print(f"\n[ERROR] Failed to load the base model. Please check the following:")
    print(f"1. Is the model ID '{BASE_MODEL_ID}' correct?")
    print(f"2. Are you logged into Hugging Face if required? (run 'huggingface-cli login')")
    print(f"3. Do you have a stable internet connection?")
    print(f"Error details: {e}")
    exit()

print("\n--- Step 2: Loading LoRA Adapter ---")
print(f"Adapter Path: {LORA_ADAPTER_PATH}")
if not os.path.exists(LORA_ADAPTER_PATH):
    print(f"\n[ERROR] The LoRA adapter path does not exist: '{LORA_ADAPTER_PATH}'")
    print("Please update the LORA_ADAPTER_PATH variable in this script.")
    exit()

try:
    # Load the LoRA and apply it to the base model
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    print("LoRA adapter loaded successfully.")
except Exception as e:
    print(f"\n[ERROR] Failed to load the LoRA adapter. Please check the following:")
    print(f"1. Is the path '{LORA_ADAPTER_PATH}' correct and does it contain 'adapter_config.json'?")
    print(f"2. Was this LoRA trained on '{BASE_MODEL_ID}'?")
    print(f"Error details: {e}")
    exit()

print("\n--- Step 3: Merging LoRA into Base Model ---")
# This is the magic step. It bakes the LoRA weights into the model's main layers.
model = model.merge_and_unload()
print("Merge and unload complete.")

print(f"\n--- Step 4: Saving Merged Model ---")
print(f"Output Path: {MERGED_MODEL_OUTPUT_PATH}")
try:
    os.makedirs(MERGED_MODEL_OUTPUT_PATH, exist_ok=True)
    model.save_pretrained(MERGED_MODEL_OUTPUT_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_OUTPUT_PATH)
except Exception as e:
    print(f"\n[ERROR] Failed to save the merged model.")
    print(f"Check if you have write permissions for the directory: '{MERGED_MODEL_OUTPUT_PATH}'")
    print(f"Error details: {e}")
    exit()

print("\n\n--- Process Complete! ---")
print(f"Your new, standalone model with the persona baked in is ready at: {MERGED_MODEL_OUTPUT_PATH}")
print("\nNext steps:")
print(f"1. Run llama.cpp's convert.py on the directory: '{MERGED_MODEL_OUTPUT_PATH}'")
print(f"2. Quantize the resulting .gguf file.")
print(f"3. Create a new Modelfile pointing to the final quantized .gguf file.")