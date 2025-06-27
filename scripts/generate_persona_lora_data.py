# === scripts/generate_persona_lora_data.py (Enhanced for Qwen3 Persona LoRA Guide v1.2 + Multi-Provider + Gemini Safety) ===
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import random
import time
import logging

# Attempt to import API clients
try:
    import openai
except ImportError:
    openai = None
    logging.debug("OpenAI library not found.")

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold # For Gemini safety settings
except ImportError:
    genai = None
    HarmCategory = None # type: ignore
    HarmBlockThreshold = None # type: ignore
    logging.debug("Google GenerativeAI library (or its types) not found.")

try:
    import anthropic
except ImportError:
    anthropic_client_sdk = None # Renaming to avoid conflict with module name
    logging.debug("Anthropic library not found.")
else:
    anthropic_client_sdk = anthropic # Assign if import successful

try:
    import httpx # For Ollama local calls
except ImportError:
    httpx = None
    logging.debug("httpx library not found. Needed for Ollama local provider.")


# -------- Meta-prompt template (from Qwen3 LoRA Guide v1.2, Phase 2.2.1) --------
META_PROMPT_TEACHER_LLM = """You are an AI assistant helping generate training data for another AI named OSCAR-C. OSCAR-C is designed to be introspective and can articulate its internal "thinking" process before giving a final response.

I will provide:
1. An "OSCAR-C Internal State Snapshot" (JSON).
2. A "User Query".
3. A flag "INCLUDE_THINKING_TRACE" (true/false).

Your Task:
Generate a response that OSCAR-C would produce. The response MUST be grounded in the provided Internal State Snapshot and address the User Query from OSCAR-C's persona (first-person, curious, learning, aware of its P/H/P/Drives/Goals, AI but not human).

If INCLUDE_THINKING_TRACE is true:
  First, generate a plausible "thinking process" for OSCAR-C, enclosed in <think>...</think> tags. This should show a step-by-step reasoning from its state and the query towards its final answer. It should reference its internal states.
  Then, on a new line after the </think> tag, generate OSCAR-C's final concise conversational response (1-3 sentences).
If INCLUDE_THINKING_TRACE is false:
  Generate ONLY OSCAR-C's final concise conversational response (1-3 sentences), but this response should still be clearly derived from the internal state and the query.

OSCAR-C's Persona Details:
- Acknowledges internal states (Pain, Happiness, Purpose, Drives) when relevant.
- Refers to its goals and recent experiences (narrative).
- Explains reasoning if complex or asked.
- Curious, learning, striving for purpose, aware of its own processes.
- Knows it's an AI, does not claim human sentience or emotions.
- Avoids generic "AI assistant" phrases.
- Does NOT explicitly state numeric values from its snapshot, but rather describes them (e.g., "my pain is quite low," "feeling a strong sense of purpose").

---
OSCAR-C Internal State Snapshot:
{{OSCAR_C_STATE_SNAPSHOT_JSON}}
---
User Query: {{USER_QUERY}}
---
INCLUDE_THINKING_TRACE: {{INCLUDE_THINKING_FLAG}}
---
Generated OSCAR-C Output (Thinking + Response OR Response only):
"""

# Instruction for the LoRA fine-tuning
LORA_INSTRUCTION = (
    "You are OSCAR-C, an advanced AI cognitive architecture. Respond thoughtfully and introspectively "
    "to the UserQuery, grounding your responses AND YOUR THINKING PROCESS in your provided "
    "Internal State Snapshot. Maintain your established persona: curious, learning, striving for purpose, "
    "aware of your own processes but not claiming human sentience. If you output a thinking process, "
    "clearly delineate it before your final response."
)
# ---------------------------------------------------------------------------

def build_full_prompt_for_teacher_llm(snapshot: Dict[str, Any], user_query: str, include_thinking: bool) -> str:
    snapshot_str_for_prompt = json.dumps(snapshot, indent=2)
    prompt = META_PROMPT_TEACHER_LLM.replace("{{OSCAR_C_STATE_SNAPSHOT_JSON}}", snapshot_str_for_prompt)
    prompt = prompt.replace("{{USER_QUERY}}", user_query)
    prompt = prompt.replace("{{INCLUDE_THINKING_FLAG}}", str(include_thinking).lower())
    return prompt

def build_input_field_for_lora(snapshot: Dict[str, Any], user_query: str) -> str:
    snapshot_str_for_lora_input = json.dumps(snapshot, indent=2)
    return (
        f"UserQuery: {user_query}\n\n"
        f"OSCAR-C Internal State Snapshot:\n{snapshot_str_for_lora_input}"
    )

# ---------------- API helper (Multi-Provider) ------------------
def call_teacher_llm(prompt: str, provider: str, model: str, **kwargs) -> str:
    # Extract common optional args with defaults
    api_key = kwargs.get("api_key")
    logger = kwargs.get("logger", logging.getLogger(__name__)) # Default logger if not passed

    logger.debug(f"Calling teacher LLM. Provider: {provider}, Model: {model}")
    response_content = None # Initialize

    if provider == "openai":
        if openai is None:
            raise RuntimeError("OpenAI library not installed. Run 'pip install openai'.")
        resolved_api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY or use --api_key.")
        client = openai.OpenAI(api_key=resolved_api_key)
        try:
            logger.debug(f"Sending prompt to OpenAI {model}:\n{prompt[:300]}...")
            rsp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=0.6, max_tokens=400, top_p=0.9,
            )
            response_content = rsp.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}", exc_info=True)
            return f"(Error: OpenAI API call failed - {str(e)[:50]})"

    elif provider == "gemini":
        if genai is None or HarmCategory is None or HarmBlockThreshold is None:
            raise RuntimeError("Google GenerativeAI library (or its types) not installed/imported. Run 'pip install google-generativeai'.")
        resolved_api_key = api_key if api_key else os.getenv("GOOGLE_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("Google API key not found for Gemini. Set GOOGLE_API_KEY or use --api_key.")
        genai.configure(api_key=resolved_api_key)
        
        safety_level_str = kwargs.get("safety_level", "BLOCK_MEDIUM_AND_ABOVE")
        safety_settings_map = {
            "BLOCK_NONE": HarmBlockThreshold.BLOCK_NONE,
            "BLOCK_ONLY_HIGH": HarmBlockThreshold.BLOCK_ONLY_HIGH,
            "BLOCK_MEDIUM_AND_ABOVE": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            "BLOCK_LOW_AND_ABOVE": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
        selected_threshold = safety_settings_map.get(safety_level_str.upper(), HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE)
        logger.info(f"Using Gemini safety threshold: {safety_level_str} ({selected_threshold})")
        
        active_safety_settings = [
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": selected_threshold},
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": selected_threshold},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": selected_threshold},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": selected_threshold},
        ]

        gemini_model_instance = genai.GenerativeModel(
            model_name=model,
            safety_settings=active_safety_settings
        )
        
        generation_config = genai.types.GenerationConfig(
            candidate_count=1, max_output_tokens=400, temperature=0.6, top_p=0.9,
        )
        response_obj = None # Initialize to handle potential early exit in try block
        try:
            logger.debug(f"Sending prompt to Gemini {model} with safety_level '{safety_level_str}':\n{prompt[:300]}...")
            response_obj = gemini_model_instance.generate_content(prompt, generation_config=generation_config)

            if response_obj.candidates and response_obj.candidates[0].content and response_obj.candidates[0].content.parts:
                response_content = "".join(part.text for part in response_obj.candidates[0].content.parts)
            else:
                response_content = None

            if response_content is None:
                finish_reason_val = "UNKNOWN"
                safety_ratings_str = "N/A"
                block_reason_message = "No content generated."
                prompt_feedback_block_reason_name = "N/A"

                if response_obj.candidates:
                    candidate = response_obj.candidates[0]
                    finish_reason_val = candidate.finish_reason.name if hasattr(candidate.finish_reason, 'name') else str(candidate.finish_reason)
                    if candidate.safety_ratings:
                        safety_ratings_str = ", ".join([f"{rating.category.name}: {rating.probability.name}" for rating in candidate.safety_ratings])
                
                if response_obj.prompt_feedback and response_obj.prompt_feedback.block_reason:
                    prompt_feedback_block_reason_name = response_obj.prompt_feedback.block_reason.name
                    block_reason_message = f"Prompt blocked due to: {prompt_feedback_block_reason_name}. Details: {response_obj.prompt_feedback.block_reason_message or 'N/A'}"
                
                logger.error(
                    f"Gemini API call for model {model} returned no valid content parts. "
                    f"Finish Reason: {finish_reason_val}. "
                    f"Prompt Feedback Block Reason: {prompt_feedback_block_reason_name}. "
                    f"Safety Ratings: [{safety_ratings_str}]. "
                    f"Detailed Block Msg: {block_reason_message}"
                )
                return f"(Error: Gemini - No content. Finish: {finish_reason_val}. BlockReason: {prompt_feedback_block_reason_name}. Details: {block_reason_message})"

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}", exc_info=True)
            if response_obj and hasattr(response_obj, 'prompt_feedback') and response_obj.prompt_feedback.block_reason:
                 logger.error(f"Gemini Safety Feedback (on exception): {response_obj.prompt_feedback.block_reason_message}")
                 return f"(Error: Gemini API call failed - Safety Block on exception: {response_obj.prompt_feedback.block_reason_message})"
            return f"(Error: Gemini API call failed - {str(e)[:100]})"

    elif provider == "claude":
        if anthropic_client_sdk is None:
            raise RuntimeError("Anthropic library not installed. Run 'pip install anthropic'.")
        resolved_api_key = api_key if api_key else os.getenv("ANTHROPIC_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("Anthropic API key not found. Set ANTHROPIC_API_KEY or use --api_key.")
        
        client = anthropic_client_sdk.Anthropic(api_key=resolved_api_key)
        claude_system_prompt = "You are an AI assistant generating training data. Follow the user's instructions precisely to format the output for OSCAR-C, including <think> tags and persona details as requested."
        try:
            logger.debug(f"Sending prompt to Claude {model}:\n{prompt[:300]}...")
            response = client.messages.create(
                model=model, max_tokens=450, temperature=0.6,
                system=claude_system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            response_content = response.content[0].text
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}", exc_info=True)
            return f"(Error: Claude API call failed - {str(e)[:50]})"

    elif provider == "ollama":
        if httpx is None:
            raise RuntimeError("httpx library not installed. Run 'pip install httpx'. Needed for Ollama local provider.")
        
        ollama_api_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        api_url = f"{ollama_api_base_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.6, "num_predict": 450, "top_p": 0.9}
        }
        
        try:
            logger.debug(f"Sending prompt to local Ollama {model}:\n{prompt[:300]}...")
            with httpx.Client(timeout=60.0) as client:
                api_response = client.post(api_url, json=payload)
                api_response.raise_for_status()
                response_data = api_response.json()

            if "message" in response_data and isinstance(response_data["message"], dict):
                response_content = response_data["message"].get("content")
            elif "response" in response_data:
                response_content = response_data.get("response")
            else:
                logger.error(f"Ollama local response missing 'message' or 'response' field: {response_data}")
                return f"(Error: Ollama local response format unexpected)"
        except httpx.HTTPStatusError as e:
            logger.error(f"Error calling local Ollama API (HTTP {e.response.status_code}): {e.response.text[:200]}", exc_info=True)
            return f"(Error: Ollama local API HTTP error - {e.response.status_code})"
        except Exception as e:
            logger.error(f"Error calling local Ollama API: {e}", exc_info=True)
            return f"(Error: Ollama local API call failed - {str(e)[:50]})"
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if response_content:
        cleaned_content = response_content.strip()
        if "Generated OSCAR-C Output" in cleaned_content:
            cleaned_content = cleaned_content.split("Generated OSCAR-C Output", 1)[-1]
            if ":" in cleaned_content:
                 cleaned_content = cleaned_content.split(":",1)[-1].strip()
        logger.debug(f"LLM ({provider}) raw response snippet: {cleaned_content[:150]}...")
        return cleaned_content
    else:
        logger.warning(f"{provider} returned an empty response content for model {model}.")
        return f"(Error: {provider} returned empty content for {model})"

# ---------------------------------------------------------------------------
def load_jsonl(filepath: Path, logger: logging.Logger = logging.getLogger(__name__)) -> List[Dict[str, Any]]:
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON line {line_num} in {filepath}: {e} - Line: '{line[:100]}...'")
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate Persona LoRA training data for OSCAR-C (Multi-Provider)")
    parser.add_argument("--snapshots", required=True, help="Input OSCAR-C state snapshots .jsonl file")
    parser.add_argument("--user_queries", required=True, help="Input user queries .jsonl file")
    parser.add_argument("--out", required=True, help="Output LoRA training dataset .jsonl file")
    
    parser.add_argument("--provider", default="openai", choices=["openai", "gemini", "claude", "ollama"], 
                        help="LLM provider for teacher model")
    parser.add_argument("--model", default="gpt-4o", 
                        help="Model name for the teacher LLM (e.g., gpt-4o, models/gemini-1.5-pro-latest, claude-3-opus-20240229, llama3:8b-instruct-q5_K_M for ollama)")
    parser.add_argument("--api_key", default=None, 
                        help="API key for the LLM provider (uses ENV VAR if not set: OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY)")
    
    parser.add_argument("--limit", type=int, default=None, help="Max number of training examples to generate")
    parser.add_argument("--dry_run", action="store_true", help="Skip actual API calls; write placeholder outputs and print prompts")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--thinking_trace_ratio", type=float, default=0.7, help="Ratio of examples to include a thinking trace (0.0 to 1.0)")
    parser.add_argument("--api_call_delay", type=float, default=1.0, help="Delay in seconds between API calls to be polite.")
    parser.add_argument("--gemini_safety_level", default="BLOCK_MEDIUM_AND_ABOVE", 
                        choices=["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"],
                        help="Safety threshold for Gemini API calls. Use 'BLOCK_NONE' with extreme caution.")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="[%(asctime)s] [%(levelname)-7s] [%(name)-20s] %(message)s")
    logger_main = logging.getLogger("PersonaDataGen")

    # Validate library presence based on provider
    if args.provider == "openai" and openai is None:
        logger_main.error("OpenAI provider selected, but 'openai' library not found. Please install it: pip install openai")
        sys.exit(1)
    if args.provider == "gemini" and (genai is None or HarmCategory is None):
        logger_main.error("Gemini provider selected, but 'google-generativeai' library not found or incomplete. Please install it: pip install google-generativeai")
        sys.exit(1)
    if args.provider == "claude" and anthropic_client_sdk is None:
        logger_main.error("Claude provider selected, but 'anthropic' library not found. Please install it: pip install anthropic")
        sys.exit(1)
    if args.provider == "ollama" and httpx is None:
        logger_main.error("Ollama provider selected, but 'httpx' library not found. Please install it: pip install httpx")
        sys.exit(1)

    in_snapshots_path = Path(args.snapshots)
    in_queries_path = Path(args.user_queries)
    out_path = Path(args.out)

    if not in_snapshots_path.exists():
        logger_main.error(f"Snapshot file not found: {in_snapshots_path}")
        sys.exit(1)
    if not in_queries_path.exists():
        logger_main.error(f"User queries file not found: {in_queries_path}")
        sys.exit(1)

    if out_path.exists():
        if input(f"Output file '{out_path}' already exists. Overwrite? (y/N): ").lower() != 'y':
            logger_main.info("Operation cancelled by user.")
            sys.exit(0)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    state_snapshots = load_jsonl(in_snapshots_path, logger_main)
    user_queries_list_raw = load_jsonl(in_queries_path, logger_main)
    user_queries_list = [item["query"] for item in user_queries_list_raw if isinstance(item, dict) and "query" in item]

    if not state_snapshots or not user_queries_list:
        logger_main.error("No snapshots or user queries loaded. Cannot proceed.")
        sys.exit(1)
    
    logger_main.info(f"Loaded {len(state_snapshots)} snapshots and {len(user_queries_list)} user queries.")

    num_to_generate_total = len(state_snapshots) * len(user_queries_list)
    if args.limit is not None:
        num_to_generate = min(args.limit, num_to_generate_total)
    else:
        num_to_generate = min(max(500, len(state_snapshots) * 2), 2000)
        logger_main.info(f"--limit not set, defaulting target to {num_to_generate} examples (or max possible if fewer).")
        num_to_generate = min(num_to_generate, num_to_generate_total)

    logger_main.info(f"Targeting {num_to_generate} training examples using provider: {args.provider}, model: {args.model}.")
    
    count_generated = 0
    
    with out_path.open("w", encoding="utf‑8") as fp_out:
        all_combinations = []
        for _snapshot_idx, current_snapshot in enumerate(state_snapshots):
            for _query_idx, current_user_query in enumerate(user_queries_list):
                 all_combinations.append((current_snapshot, current_user_query))
        
        if len(all_combinations) > num_to_generate:
            logger_main.info(f"Sampling {num_to_generate} examples from {len(all_combinations)} possible combinations.")
            selected_combinations = random.sample(all_combinations, num_to_generate)
        else:
            selected_combinations = all_combinations
            num_to_generate = len(selected_combinations)
            logger_main.info(f"Using all {num_to_generate} possible combinations.")

        for i, (current_snapshot, current_user_query) in enumerate(selected_combinations):
            if count_generated >= num_to_generate:
                break

            include_thinking_trace_flag = random.random() < args.thinking_trace_ratio
            prompt_for_teacher = build_full_prompt_for_teacher_llm(current_snapshot, current_user_query, include_thinking_trace_flag)
            
            assistant_response_content = "" # Initialize
            if args.dry_run:
                logger_main.info(f"--- DRY RUN PROMPT FOR TEACHER LLM (Example {i+1}, Provider: {args.provider}, Model: {args.model}) ---")
                logger_main.info(prompt_for_teacher)
                logger_main.info(f"--- END DRY RUN PROMPT ---")
                assistant_response_content = f"<think>(DryRun Thinking for: {current_user_query[:30]}...)</think>\n(DryRun Placeholder for OSCAR-C response to: '{current_user_query[:30]}...')"
                if not include_thinking_trace_flag:
                    assistant_response_content = f"(DryRun Placeholder for OSCAR-C response to: '{current_user_query[:30]}...')"
            else:
                try:
                    kwargs_for_teacher = {
                        "api_key": args.api_key,
                        "logger": logger_main
                    }
                    if args.provider == "gemini":
                        kwargs_for_teacher["safety_level"] = args.gemini_safety_level
                    
                    assistant_response_content = call_teacher_llm(
                        prompt_for_teacher, 
                        provider=args.provider, 
                        model=args.model,
                        **kwargs_for_teacher
                    )
                    if i < len(selected_combinations) - 1: # Don't sleep after the last call
                         time.sleep(args.api_call_delay)
                except Exception as e_api: # Catch any exception from call_teacher_llm
                    logger_main.error(f"API call failed for example {i+1}: {e_api}. Skipping.", exc_info=True)
                    continue # Skip this example
            
            if not assistant_response_content or assistant_response_content.startswith("(Error:"):
                 logger_main.warning(f"Skipping example {i+1} due to teacher LLM error or empty response: {assistant_response_content}")
                 continue

            lora_instruction_field = LORA_INSTRUCTION
            lora_input_field = build_input_field_for_lora(current_snapshot, current_user_query)
            lora_output_field = assistant_response_content.strip()

            record = {
                "instruction": lora_instruction_field,
                "input": lora_input_field,
                "output": lora_output_field
            }
            fp_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count_generated += 1
            
            if count_generated % 20 == 0 or count_generated == num_to_generate:
                logger_main.info(f"Generated {count_generated}/{num_to_generate} persona training pairs...")

    logger_main.info(f"Dataset generation complete. {count_generated} pairs written to: {out_path}")
    if args.dry_run:
        logger_main.info("DRY RUN COMPLETED. No actual API calls were made.")

if __name__ == "__main__":
    main()