# --- START OF FILE cognitive_modules.py ---

import asyncio
import logging
import time
import json
import os
import re
import datetime
import random
from typing import Any, Dict, List, Tuple, Optional
import queue

# Import necessary components from other modules
from . import agent_config
from .agent_state import AgentState
from .external_comms import call_ollama, add_document_to_memory, retrieve_from_memory

# Safe Library Imports
try: import psutil; PSUTIL_AVAILABLE = True
except ImportError: psutil = None; PSUTIL_AVAILABLE = False; logging.error("cognitive_modules: psutil missing.")

logger = logging.getLogger(__name__)

# --- Cognitive Step Functions ---

async def perceive(controller: Any):
    """Gathers inputs and updates the blackboard via the controller."""
    # --- Function unchanged ---
    if not controller._is_running_flag.is_set() or not controller._asyncio_loop or not controller._asyncio_loop.is_running():
        logger.debug("Perceive skipped: Agent stopping.")
        return

    logger.debug("Perceive step start...")
    percepts: Dict[str, Any] = {"timestamp": time.time()}

    # 1. Check for user input queue
    try:
        user_input = await asyncio.wait_for(controller._user_input_queue.get(), timeout=0.01)
        if user_input:
            percepts["user_input"] = user_input
            logger.info(f"Perceived input: '{user_input[:50]}...'")
            controller._user_input_queue.task_done()
    except asyncio.TimeoutError: pass
    except asyncio.QueueEmpty: pass
    except Exception as e_queue: logger.error(f"Error getting user input from queue: {e_queue}")

    # 2. Get system state
    if PSUTIL_AVAILABLE:
        try:
            stats = await controller._asyncio_loop.run_in_executor(
                None,
                lambda: {"cpu_percent": psutil.cpu_percent(), "memory_percent": psutil.virtual_memory().percent,}
            )
            percepts["system_state"] = stats
            logger.debug(f"Gathered psutil stats: {stats}")
        except RuntimeError as e:
             if "cannot schedule new futures after shutdown" in str(e): logger.warning("psutil check skipped: Executor shutdown.")
             else: raise
        except Exception as e_psutil: logger.warning(f"Failed to gather psutil stats: {e_psutil}")

    # 3. Update blackboard
    with controller._blackboard_lock:
        controller._blackboard["latest_percepts"] = percepts
        logger.debug(f"Percepts updated: {json.dumps(percepts, default=str)}")


# <<< REVERTED decide_act to previous working version (before Claude's simplification) >>>
async def decide_act(controller: Any):
    """Determines the next action, incorporating drives, self-model, narrative, and few-shot examples."""
    logger.debug("--- Decide/Act START (Reverted Prompt v2) ---") # Note reversion

    # --- Check for RESET_STRATEGY cooldown ---
    with controller._blackboard_lock:
        reset_cooldown_until = controller._blackboard.get('reset_cooldown', 0)
    if time.time() < reset_cooldown_until:
        logger.info(f"Decision cycle paused due to RESET_STRATEGY cooldown (until {reset_cooldown_until:.1f})")
        action_response = "[ACTION: THINKING content=\"Paused briefly after strategy reset.\"]"
        with controller._blackboard_lock:
            controller._blackboard['last_action_response'] = action_response
            controller._add_to_history(assistant_turn={"role": "assistant", "content": action_response})
        try:
             controller.ui_queue.put_nowait(("agent_output", action_response))
        except queue.Full: logger.warning("UI queue full dropping reset cooldown message.")
        except Exception as e: logger.error(f"Error putting cooldown message on UI queue: {e}")
        logger.debug("--- Decide/Act END (RESET Cooldown) ---")
        return
    # --- End Cooldown Check ---

    action_response = "[Action Undecided - Error]"; messages: List[Dict[str, str]] = []; user_input_this_cycle: Optional[str] = None; error: Optional[str] = None; content: Optional[str] = None; retrieved_docs: List[str] = []
    try:
        # --- Define Available Actions ---
        available_actions = agent_config.AVAILABLE_ACTIONS # Use list with placeholders again

        # --- Gather Context from Blackboard ---
        last_action_failed = False
        with controller._blackboard_lock:
            current_goal = controller._blackboard.get("current_goal", "Observe system state and respond.") # Reverted default goal
            recent_history = list(controller._blackboard.get('recent_history', []))
            current_percepts = controller._blackboard.get('latest_percepts', {})
            last_reflection_obj = controller._blackboard.get('last_reflection')
            recent_actions = list(controller._blackboard.get('recent_actions', []))
            last_action_record = recent_actions[-1] if recent_actions else None
            last_action_result_summary = None
            if last_action_record:
                if last_action_record.get("outcome") == "failure":
                     last_action_failed = True
                last_action_result_summary = last_action_record.get("result_summary")

            emotional_valence = controller._blackboard.get("emotional_valence", {"curiosity": 0.5, "satisfaction": 0.5})
            self_model = controller._blackboard.get("self_model", {})
            narrative_history = list(controller._blackboard.get("narrative", []))
            latest_narrative_entry = narrative_history[-1].get("content", "[No narrative yet]") if narrative_history else "[No narrative yet]"
            known_paths = self_model.get("knowledge", {})
            validated_paths = known_paths.get("validated_paths", {})
            invalid_paths = known_paths.get("invalid_paths", [])

        user_input_this_cycle = current_percepts.get('user_input')

        # --- Build Prompt Messages ---
        messages = []

        # 1. Initial System Prompt
        system_prompt_part1 = (f"Your current goal is: '{current_goal}'. Choose ONE action from the list below.\n\nAvailable Actions:\n{available_actions}")
        messages.append({"role": "system", "content": system_prompt_part1})

        # 2. Add Memory Context
        query_text = user_input_this_cycle or (last_reflection_obj.get('immediate') if isinstance(last_reflection_obj, dict) and last_reflection_obj.get('immediate') else f"Goal: {current_goal}")
        if query_text:
            retrieved_docs = await retrieve_from_memory(controller._memory_collection, controller._asyncio_loop, query_text, controller._is_running_flag, agent_config.MEMORY_RETRIEVAL_COUNT)
            if retrieved_docs:
                memory_separator = "\n---\n"; safe_docs = [str(doc) if doc is not None else "[None]" for doc in retrieved_docs]; memory_context = memory_separator.join(safe_docs)[:1000] # Keep reduced length
                messages.append({"role": "system", "content": f"Recent Memory Context:\n```\n{memory_context}\n```"})

        # 3. Add Recent History
        if recent_history: max_history_items = agent_config.MAX_RECENT_TURNS_IN_PROMPT * 2; messages.extend(recent_history[-max_history_items:])

        # 4. Add Current Context Summary
        context_lines = [f"Time: {datetime.datetime.now().isoformat()}"]
        if current_percepts: state = current_percepts.get('system_state'); context_lines.append(f"System: CPU {state.get('cpu_percent', '?')}% Mem {state.get('memory_percent', '?')}%")
        if last_action_result_summary:
            context_lines.append(f"Last Action Result Summary: {last_action_result_summary[:200]}...")
        if last_reflection_obj and isinstance(last_reflection_obj, dict): immediate_reflect = last_reflection_obj.get('immediate'); context_lines.append(f"Last Reflection (Imm): {str(immediate_reflect)[:100]}...")
        context_lines.append(f"Drives: C={emotional_valence.get('curiosity', 0.5):.2f} S={emotional_valence.get('satisfaction', 0.5):.2f}")
        sm_version = self_model.get('version', 0); sm_caps = list(self_model.get('capabilities', {}).keys()); sm_limits = list(self_model.get('limitations', {}).keys())
        context_lines.append(f"Self(v{sm_version}): {len(sm_caps)} Caps, {len(sm_limits)} Limits")
        if validated_paths: context_lines.append(f"Valid Paths: {', '.join(list(validated_paths.keys())[:2])}...")
        if invalid_paths: context_lines.append(f"Invalid Paths: {', '.join(invalid_paths[:2])}...")
        context_lines.append(f"Narrative: ...{latest_narrative_entry[-100:]}")
        context_summary = "\n".join(context_lines); messages.append({"role": "system", "content": f"Current Context:\n{context_summary}"})

        # 5. Enhanced Last Action Feedback
        if last_action_record:
            outcome = last_action_record.get('outcome', 'unknown')
            action_type = last_action_record.get('type', '?')
            action_params = last_action_record.get('params', {})
            action_error = last_action_record.get('error')
            action_result = last_action_record.get('result_summary') # Used in context above

            if outcome == 'failure':
                consecutive_failures = 0
                relevant_recent_actions = recent_actions[-10:]
                for a in reversed(relevant_recent_actions):
                    if a.get('type') == action_type and a.get('outcome') == 'failure':
                        consecutive_failures += 1
                    elif not (a.get('type') == action_type and a.get('outcome') == 'failure'):
                         break

                feedback_prefix = f"!! CRITICAL: Last Action '{action_type}' FAILED {consecutive_failures} {'time' if consecutive_failures == 1 else 'times'}!"
                if consecutive_failures > 1:
                    if len(recent_actions) >= 2 and recent_actions[-1].get('type') == action_type and recent_actions[-2].get('type') == action_type and recent_actions[-1].get('outcome') == 'failure' and recent_actions[-2].get('outcome') == 'failure':
                        feedback_prefix += " (Consecutive)"
                outcome_str = f" Attempted: {action_type} {action_params}"

                if action_error:
                    sanitized_error = str(action_error).replace('\n', ' ').replace('"', "'")[:150]
                    outcome_str += f" Error: {sanitized_error}."
                    error_lower = sanitized_error.lower()
                    if "not a directory" in error_lower or "not directory" in error_lower:
                        outcome_str += f" ADVICE: Path is not a directory! Use PATH_STRATEGY or explore known valid paths."
                    elif "path not absolute" in error_lower:
                        outcome_str += f" ADVICE: Use absolute paths (start with / or drive letter)."
                    elif "not found" in error_lower or "no such file or directory" in error_lower:
                        outcome_str += f" ADVICE: Path/File does not exist! Verify with LIST_FILES/EXPLORE from known location."
                    elif "permission denied" in error_lower:
                         outcome_str += f" ADVICE: Permission denied! Try different location."
                    else:
                        outcome_str += " ADVICE: Analyze error. Do NOT repeat."
                else:
                     outcome_str += " ADVICE: Action failed (unknown reason). Do NOT repeat."

                if consecutive_failures >= 2:
                    outcome_str += f"\n!!! FAILED {consecutive_failures} times! CHANGE APPROACH!"
                    # REMOVED explicit mention of RESET_STRATEGY here, rely on instruction #5

                messages.append({"role": "system", "content": f"{feedback_prefix}{outcome_str}"})
                logger.debug(f"Built prompt: Added CRITICAL error feedback (Failures counted: {consecutive_failures}).")

            elif outcome == 'success':
                 logger.debug(f"Built prompt: Noted last action success ({action_type}).")

            else: # Unknown outcome
                messages.append({"role": "system", "content": f"Feedback: Outcome of Last Action ({action_type} {action_params}) is unknown."})
                logger.debug("Built prompt: Added unknown outcome feedback.")


        # 6. <<< REVERTED Final Instruction (from before simplification) >>>
        format_instruction = (
            "Output Format Instructions (Follow Strictly!):\n"
            "1. Provide concise reasoning considering goal, drives, self-model, narrative, context & **feedback**.\n"
            "2. **CRITICAL: Check known invalid paths before filesystem actions (LIST_FILES, READ_FILE, EXPLORE)!** Avoid paths in 'Invalid Paths' context.\n"
            "3. If you received path errors, use [ACTION: PATH_STRATEGY strategy=\"current_dir\"] or other corrective action.\n"
            "4. **NEVER repeat the exact same failed action if you received specific error feedback.**\n"
            # --- Stricter RESET_STRATEGY Condition ---
            "5. **RESET_STRATEGY Guidance:** This action is NOT in the available list. You MAY ONLY output `[ACTION: RESET_STRATEGY]` IF the CRITICAL feedback explicitly states you failed the *same action 3 or more times CONSECUTIVELY*.\n"
            # --- Encourage Action Diversity After Success ---
            "6. **After a successful action (especially EXPLORE/LIST_FILES):** Analyze the 'Last Action Result Summary' provided in the context. Choose a DIFFERENT follow-up action (e.g., READ_FILE, QUERY_MEMORY about findings, EXPLORE a *sub-directory* listed, SET_GOAL based on findings). Avoid repeating the exact same successful action unless absolutely necessary for the goal.\n"
            "7. Hints: High Curiosity -> EXPLORE (new areas!)/QUERY_MEMORY. Low Satisfaction -> SET_GOAL.\n"
            "8. On a **new line**, output EXACTLY ONE action tag `[ACTION: TYPE param=\"value\"...]` from the 'Available Actions' list (unless using RESET_STRATEGY as per rule #5).\n"
            "9. Use double quotes `\"\"` for parameters. Relative paths are allowed now.\n" # Allow relative paths
            "10. **Example Guidance:** DO NOT use file paths or specific details directly from the examples below. They are for format illustration only.\n\n"
            "Example (Success -> Read File):\nReasoning: Last EXPLORE listed 'config.yaml'. I need to see the config.\n[ACTION: READ_FILE path=\"config.yaml\"]\n\n"
            "Example (Path Error -> Path Strategy):\nReasoning: Failed listing './data' (not found). Need to check current location.\n[ACTION: PATH_STRATEGY strategy=\"current_dir\"]\n\n"
            "Now, provide reasoning and ONE action tag."
        )
        messages.append({"role": "system", "content": format_instruction})
        # <<< END REVERTED Final Instruction >>>

        # 7. Add User Input or Autonomous Trigger
        if user_input_this_cycle: messages.append({"role": "user", "content": user_input_this_cycle})
        else: messages.append({"role": "user", "content": f"Follow instructions precisely. Analyze context and feedback. Provide reasoning and ONE action tag."})

        logger.debug(f"PROMPT TO MODEL (Decide):\nSystem Context:\n{messages[0]['content']}\nUser/Instruction:\n{messages[-1]['content']}") # Debug Log
        logger.debug(f"--- End Decide Prompt ---")

    except Exception as e_prompt:
         logger.exception("!!! ERROR during prompt building !!!"); error = f"Prompt build error: {e_prompt}"; content = None; action_response = f"[Action Error: {error}]"; controller._log_to_ui("error", error)
         try:
             with controller._blackboard_lock: controller._blackboard['last_action_response'] = action_response; controller._add_to_history({"role": "assistant", "content": action_response})
         except Exception as e_hist: logger.error(f"Failed to add prompt build error to history: {e_hist}")
         try: controller.ui_queue.put_nowait(("agent_output", action_response))
         except queue.Full: logger.warning("UI queue full dropping prompt build error.")
         except Exception as e_q: logger.error(f"Error putting prompt build error on UI queue: {e_q}")
         logger.debug("--- Decide/Act END (prompt error) ---"); return

    # --- Call Ollama ---
    temp = agent_config.DECIDE_ACT_TEMPERATURE if hasattr(agent_config, 'DECIDE_ACT_TEMPERATURE') else 0.4 # Use lower temp
    content, error = await call_ollama(controller.selected_ollama_model, messages, temp, controller._asyncio_loop)

    logger.debug(f"MODEL RESPONSE (Decide): {content}") # Debug Log
    if error: logger.error(f"Ollama call error (Decide): {error}") # Debug Log

    if error: action_response = f"[Ollama Action Error: {error}]"; controller._log_to_ui("error", action_response); logger.error(f"Ollama call failed: {error}")
    elif content is not None: action_response = content; logger.info(f"Ollama RAW response received: '{action_response[:200]}...'")
    else: logger.error("Ollama call returned None/None."); action_response = "[Action Error: Unknown Ollama response]"; controller._log_to_ui("error", action_response)

    # --- Update History and UI ---
    with controller._blackboard_lock: controller._blackboard['last_action_response'] = action_response; controller._add_to_history(assistant_turn={"role": "assistant", "content": action_response}, user_turn_content=user_input_this_cycle)
    logger.debug("Blackboard updated with raw action response.")

    try: controller.ui_queue.put_nowait(("agent_output", action_response)); logger.debug("Placed raw agent_output on UI queue.")
    except queue.Full: logger.warning("UI queue full dropping agent output.")
    except Exception as e_queue: logger.exception("Error putting agent_output on queue!")

    logger.debug("--- Decide/Act END ---")


# <<< REVERTED execute_action to restore THINKING fallback and security checks >>>
async def execute_action(controller: Any):
    """Parses action from last raw response and executes it, with THINKING fallback."""
    logger.debug("--- Execute Action START ---")
    action_type = "THINKING"; params = {}; result = None; error = None; parsed_action = None; parse_error = None
    action_tag_found = False
    original_llm_error = None

    with controller._blackboard_lock: raw_response = controller._blackboard.get('last_action_response')
    if not raw_response: logger.warning("No action response found."); logger.debug("--- Execute Action END (No response) ---"); return

    if raw_response.startswith("[Ollama Action Error:") or raw_response.startswith("[Action Error:"):
        original_llm_error = raw_response
        logger.warning(f"Detected LLM error from previous step: {original_llm_error}")
        action_type = "THINKING"
        params = {"content": f"Defaulting due to upstream LLM/Action error: {original_llm_error}"}
        error = original_llm_error
        action_tag_found = False
        parse_error = "Upstream LLM/Action error"
    else:
        # --- Action Parsing Logic ---
        action_line = None; action_match = None
        response_lines = raw_response.strip().split('\n')
        for line in reversed(response_lines):
            clean_line = line.strip()
            if '[ACTION:' in clean_line or '[ACTION :' in clean_line:
                 action_line = clean_line
                 if clean_line == '[ACTION: RESET_STRATEGY]':
                      action_match = re.match(r"\[ACTION:\s*(RESET_STRATEGY)\s*\]", clean_line)
                 else:
                      action_match = re.search(r"\[\s*ACTION\s*:\s*(\w+)\s*(.*?)?\s*\]", action_line, re.IGNORECASE)

                 if action_match:
                     action_tag_found = True
                     logger.debug(f"Found action tag via regex search: '{action_match.group(0)}' in line '{action_line}'")
                     break

        if not action_tag_found:
             last_line = response_lines[-1].strip()
             if last_line.startswith('[ACTION:') or last_line.startswith('[ACTION :'):
                  action_line = last_line
                  logger.warning(f"Potential action tag on last line but regex failed: '{action_line}'. Treating as THINKING.")
                  parse_error = f"Malformed tag suspected: {action_line}"

        if action_tag_found and action_match:
            action_type = action_match.group(1).upper()
            if action_type == "RESET_STRATEGY":
                 params_str = ""
                 params = {}
                 logger.info(f"PARSED ACTION: Type={action_type}, Params={{}}")
            else:
                params_str = (action_match.group(2) or "").strip()
                logger.debug(f"Parsing params for {action_type}: '{params_str}'")
                # Validate against listed actions from config
                known_listed_action_types = re.findall(r"\[ACTION:\s*(\w+)", agent_config.AVAILABLE_ACTIONS)
                if action_type not in known_listed_action_types:
                    parse_error = f"Invalid action type '{action_type}' generated. Known: {known_listed_action_types}"; error = parse_error; action_tag_found = False; action_type = "THINKING"; params = {"content": f"Invalid type '{action_type}'. Known: {known_listed_action_types}. Orig: {raw_response}"}; logger.error(parse_error)
                else:
                     # Parameter parsing logic
                     try:
                         parsed_params_list = re.findall(r'''(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=]+))''', params_str)
                         params = {}
                         for k, v_dq, v_sq, v_uq in parsed_params_list:
                             if v_dq is not None: params[k] = v_dq
                             elif v_sq is not None: params[k] = v_sq
                             else: params[k] = v_uq
                         logger.info(f"PARSED ACTION: Type={action_type}, Params={params}")
                         # Basic validation for required params
                         required_params = {"LIST_FILES": ["path"], "READ_FILE": ["path"], "QUERY_MEMORY": ["query"], "SET_GOAL": ["goal"]}
                         if action_type in required_params:
                              missing = [p for p in required_params[action_type] if p not in params or not params[p]]
                              if missing:
                                   parse_error = f"Missing required param(s) for {action_type}: {missing}"; error = parse_error; action_tag_found = False; action_type = "THINKING"; params = {"content": f"Missing params {missing} for {action_type}. Orig: {raw_response}"}; logger.error(parse_error)
                         if action_type == "EXPLORE" and 'depth' in params:
                              if params['depth'] not in ['1', '2']:
                                   parse_error = f"Invalid 'depth' for EXPLORE: {params['depth']}. Must be '1' or '2'."; error = parse_error; action_tag_found = False; action_type = "THINKING"; params = {"content": f"Invalid depth for EXPLORE. Orig: {raw_response}"}; logger.error(parse_error)

                     except Exception as parse_err: logger.exception(f"Crit err parsing params '{params_str}'"); params = {}; error = f"Param parsing exception: {parse_err}"; parse_error = error; action_tag_found = False; action_type = "THINKING"; params = {"content": f"Param parse exception: {params_str}. Orig: {raw_response}"}
        elif action_line and not action_match:
             logger.warning(f"REGEX FAILED on potential action line: '{action_line}'"); parse_error = f"Malformed tag: {action_line}"; error = parse_error; action_tag_found = False

    # --- Reverted Fallback: Default to THINKING if no valid action parsed ---
    if not action_tag_found and not original_llm_error:
        if not error: error = parse_error if parse_error else "No valid action tag found in response."
        logger.warning(f"Defaulting to THINKING. Reason: {error}")
        action_type = "THINKING" # <<< Default Action Reverted
        default_content = f"Failed action parse ({error}). Orig Response: {raw_response}"
        params = {"content": default_content[:1000]} # <<< Default Params Reverted
        error = error # Keep the parse error for the record

    # Store Parsed Action
    parsed_action = {"type": action_type, "params": params, "parse_error": parse_error}
    with controller._blackboard_lock: controller._blackboard['last_parsed_action'] = parsed_action

    # --- Execute the Action ---
    action_execution_error = None
    if action_type not in ["THINKING", "RESET_STRATEGY"] and error: # Allow RESET execution even if parsing failed
        logger.error(f"Execution skipped for {action_type} due to prior error: {error}")
        result = f"Action {action_type} skipped due to prior error: {error}"
        action_execution_error = error
    else:
        exec_log_msg = f"Agent executing: {action_type} {params if params else ''}"
        if action_type == "THINKING" and error: exec_log_msg += f" (Defaulted due to: {error})"
        controller._log_to_ui("info", exec_log_msg)
        logger.info(f"Executing action: {action_type} with params {params}" + (f" (Defaulted due to: {error})" if error and action_type == "THINKING" else ""))

        try:
            # <<< ACTION EXECUTION LOGIC with security checks reinstated >>>
            if action_type == "THINKING":
                 result = f"Thought process recorded: {str(params.get('content', ''))[:100]}..."
                 logger.info(f"THINKING action executed.")

            elif action_type == "PATH_STRATEGY":
                strategy = params.get('strategy', 'current_dir')
                if strategy == "current_dir":
                    try:
                        current_dir = os.getcwd()
                        result = f"Current working directory: {current_dir}\n"
                        try:
                            entries = os.listdir(current_dir)
                            dirs = [d for d in entries if os.path.isdir(os.path.join(current_dir, d))][:10]
                            files = [f for f in entries if os.path.isfile(os.path.join(current_dir, f))][:10]
                            result += f"Contains {len(entries)} items. First few Dirs: {dirs}. Files: {files}."
                        except Exception as list_err:
                             result += f" (Could not list contents: {list_err})"
                        logger.info(f"Executed PATH_STRATEGY, CWD: {current_dir}")
                    except Exception as cwd_err:
                         action_execution_error = f"Error getting current directory: {str(cwd_err)}"
                else:
                    action_execution_error = f"Unknown PATH_STRATEGY strategy: {strategy}"
                if action_execution_error: logger.error(action_execution_error)


            elif action_type == "LIST_FILES":
                path_param = params.get('path', '.');
                target_path = os.path.abspath(os.path.join(os.getcwd(), path_param))
                logger.info(f"LIST_FILES attempting access to resolved path: {target_path}")
                # --- Security Checks Reinstated ---
                if '..' in target_path.split(os.sep) and target_path != os.path.abspath(target_path):
                     action_execution_error = f"Path traversal potentially disallowed: '{target_path}'"
                elif not os.path.isabs(target_path):
                      action_execution_error = f"Internal Error: Path not absolute after resolving: '{target_path}'"
                # --- End Security Checks ---
                elif not os.path.exists(target_path): action_execution_error = f"Path does not exist: '{target_path}' (Resolved from '{path_param}')"
                elif os.path.isdir(target_path):
                    try:
                        files = os.listdir(target_path)
                        result = f"Files/Dirs in '{target_path}': {files[:agent_config.MAX_FILES_LISTED]}{'...' if len(files) > agent_config.MAX_FILES_LISTED else ''}"
                        logger.info(f"Listed {target_path}")
                    except PermissionError as e: action_execution_error = f"Permission denied listing {target_path}: {str(e)}"
                    except Exception as e: action_execution_error = f"Error listing {target_path}: {str(e)}"
                else: action_execution_error = f"Not a directory: {target_path}"
                if action_execution_error: logger.error(f"LIST_FILES failed: {action_execution_error}")


            elif action_type == "READ_FILE":
                path_param = params.get('path', '');
                if not path_param: action_execution_error = "Missing 'path'."
                else:
                    target_path = os.path.abspath(os.path.join(os.getcwd(), path_param))
                    logger.info(f"READ_FILE attempting access to resolved path: {target_path}")
                    # --- Security Checks Reinstated ---
                    if '..' in target_path.split(os.sep) and target_path != os.path.abspath(target_path):
                         action_execution_error = f"Path traversal potentially disallowed: '{target_path}'"
                    elif not os.path.isabs(target_path):
                         action_execution_error = f"Internal Error: Path not absolute after resolving: '{target_path}'"
                    # --- End Security Checks ---
                    elif not os.path.exists(target_path): action_execution_error = f"File does not exist: {target_path} (Resolved from '{path_param}')"
                    elif os.path.isfile(target_path):
                        try:
                            with open(target_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read(agent_config.MAX_FILE_READ_CHARS)
                            result = f"Content of '{target_path}' ({len(content)} chars read):\n```\n{content}\n{'...' if len(content) == agent_config.MAX_FILE_READ_CHARS else ''}```"; logger.info(f"Read {target_path}")
                        except PermissionError as e: action_execution_error = f"Permission denied reading {target_path}: {str(e)}"
                        except Exception as e: action_execution_error = f"Error reading {target_path}: {str(e)}"
                    else: action_execution_error = f"Not a file: {target_path}"
                if action_execution_error: logger.error(f"READ_FILE failed: {action_execution_error}")


            elif action_type == "QUERY_MEMORY":
                query = params.get('query', '');
                if query:
                     memories = await retrieve_from_memory(controller._memory_collection, controller._asyncio_loop, query, controller._is_running_flag, agent_config.MEMORY_RETRIEVAL_COUNT);
                     result = f"Memory query for '{query}':\n---\n" + "\n---\n".join(memories) if memories else f"Memory query for '{query}': No relevant results found."
                     logger.info(f"Queried memory for '{query}', found {len(memories)} results.")
                else: action_execution_error = "Missing 'query'."
                if action_execution_error: logger.error(f"QUERY_MEMORY failed: {action_execution_error}")


            elif action_type == "SET_GOAL":
                new_goal = params.get('goal', '');
                if new_goal and isinstance(new_goal, str) and len(new_goal) > 5:
                    with controller._blackboard_lock: old_goal = controller._blackboard.get('current_goal', ''); current_set_time = controller._blackboard.get('goal_set_time', time.time()); controller._blackboard['current_goal'] = new_goal; controller._blackboard['goal_set_time'] = time.time()
                    result = f"Goal set: '{new_goal}'"; controller._log_to_ui("info", f"Agent set goal: {new_goal}")
                    goal_age = time.time() - current_set_time
                    with controller._blackboard_lock: archive = controller._blackboard.setdefault("goal_history", []); archive.append({ "goal": old_goal, "duration": goal_age, "ended_reason": "SET_GOAL action", "ended": time.time() })
                else: action_execution_error = f"Invalid or missing 'goal': '{new_goal}'"
                if action_execution_error: logger.error(f"SET_GOAL failed: {action_execution_error}")


            elif action_type == "EXPLORE":
                path_param = params.get('path', '.');
                depth_param = params.get('depth', '1');
                target_path = os.path.abspath(os.path.join(os.getcwd(), path_param))
                logger.info(f"EXPLORE attempting access to resolved path: {target_path}")
                 # --- Security Checks Reinstated ---
                if '..' in target_path.split(os.sep) and target_path != os.path.abspath(target_path):
                     action_execution_error = f"Path traversal potentially disallowed: '{target_path}'"
                elif not os.path.isabs(target_path):
                     action_execution_error = f"Internal Error: Path not absolute after resolving: '{target_path}'"
                # --- End Security Checks ---
                elif not os.path.exists(target_path): action_execution_error = f"Path does not exist: '{target_path}' (Resolved from '{path_param}')"
                elif not os.path.isdir(target_path): action_execution_error = f"Not a directory: '{target_path}'"
                else:
                     try: depth = min(max(int(depth_param), 0), 2)
                     except ValueError: action_execution_error = f"Invalid 'depth': '{depth_param}'. Must be 0, 1 or 2."; depth = 0

                     if not action_execution_error:
                         logger.info(f"Exploring '{target_path}' (depth {depth})")
                         explore_results = []; files_listed_count = 0; dirs_listed_count = 0; max_list_per_dir = 10
                         try:
                             base_level = target_path.rstrip(os.sep).count(os.sep)
                             for root, dirs, files in os.walk(target_path, topdown=True, onerror=lambda e: logger.warning(f"os.walk error: {e}")):
                                 current_level = root.count(os.sep) - base_level
                                 if current_level > depth: dirs[:] = []; continue

                                 rel_path = os.path.relpath(root, target_path) if root != target_path else "."
                                 indent = '  ' * current_level
                                 explore_results.append(f"{indent}📁 {rel_path}/")
                                 dirs_listed_count += 1

                                 sub_indent = '  ' * (current_level + 1)
                                 files_to_show = files[:max_list_per_dir];
                                 for f in files_to_show: explore_results.append(f"{sub_indent}📄 {f}"); files_listed_count+=1
                                 if len(files) > max_list_per_dir: explore_results.append(f"{sub_indent}... ({len(files) - max_list_per_dir} more files)")

                                 dirs_to_show_count = 5
                                 if len(dirs) > dirs_to_show_count: explore_results.append(f"{sub_indent}... ({len(dirs) - dirs_to_show_count} more directories)"); dirs[:] = dirs[:dirs_to_show_count]

                                 if len(explore_results) > 100: explore_results.append("... (Exploration truncated due to length)"); break

                             result = f"Exploration results for '{target_path}' (Depth {depth}, {dirs_listed_count} Dirs, {files_listed_count} Files shown):\n```\n" + "\n".join(explore_results) + "\n```"
                             logger.info(f"Explored {target_path}")
                         except PermissionError as e_explore: action_execution_error = f"Permission denied exploring {target_path}: {str(e_explore)}"
                         except Exception as e_explore: action_execution_error = f"Error exploring {target_path}: {str(e_explore)}"

                if action_execution_error: logger.error(f"EXPLORE failed: {action_execution_error}")

            elif action_type == "RESET_STRATEGY":
                # Logic remains the same, failure count already fixed
                with controller._blackboard_lock:
                    recent_actions_list = list(controller._blackboard.get("recent_actions", []))[-5:]
                    current_goal = controller._blackboard.get("current_goal", "")

                failed_actions = [a for a in recent_actions_list if a.get('outcome') == 'failure']
                num_failed = len(failed_actions)
                failed_types = [a.get('type') for a in failed_actions if a.get('type')]
                most_failed_type = max(set(failed_types), key=failed_types.count) if failed_types else "recent successes"

                result = f"STRATEGY RESET Triggered! ({num_failed}/5 recent actions failed, often '{most_failed_type}').\n"
                result += "Analysis & Suggestions:\n"

                if most_failed_type in ["EXPLORE", "LIST_FILES", "READ_FILE"]:
                    result += "- Filesystem errors detected. Stop repeating failed paths.\n"
                    result += "- Use [ACTION: PATH_STRATEGY strategy=\"current_dir\"] to find your current location.\n"
                    result += "- Use known valid paths from self-model if available.\n"
                    result += "- If consistently failing, maybe [ACTION: SET_GOAL goal=\"...\"] to something not requiring filesystem access?\n"
                elif most_failed_type == "SET_GOAL":
                     result += "- Difficulty setting goals. Ensure goal description is clear and concise.\n"
                     result += "- Maybe try a simpler goal, like observing system state.\n"
                elif num_failed == 0:
                     result += "- Reset triggered despite recent successes. Re-evaluating approach.\n"
                     result += "- Consider focusing on analyzing recent discoveries or setting a more specific goal.\n"
                else:
                    result += "- Multiple/various errors occurred. Re-evaluate the current approach.\n"
                    result += "- Try a completely different action type from the available list.\n"
                    result += "- Consider if the current goal '{current_goal}' is achievable.\n"

                cooldown_duration = 2.0
                with controller._blackboard_lock:
                    controller._blackboard['reset_cooldown'] = time.time() + cooldown_duration
                result += f"- Pausing decisions for {cooldown_duration}s to allow re-orientation."
                logger.info(f"Executed RESET_STRATEGY. Failure count: {num_failed}. Cooldown set for {cooldown_duration}s.")
                if num_failed > 0:
                    controller._log_to_ui("warn", f"Agent triggered RESET_STRATEGY ({num_failed} recent failures, often '{most_failed_type}'). Pausing briefly.")
                else:
                    controller._log_to_ui("info", f"Agent executed RESET_STRATEGY (Note: {num_failed} recent failures). Pausing briefly.")


        except Exception as e_exec:
             logger.exception(f"Unexpected execution error ({action_type})");
             action_execution_error = f"Execution error ({action_type}): {str(e_exec)}"

    # --- Finalize Outcome and Record Action ---
    final_error = action_execution_error if action_execution_error else error
    outcome = "success" if not final_error else "failure"
    if action_type == "THINKING" and error and not action_execution_error:
         outcome = "success"

    result_summary_str = str(result)[:500].replace('\n', ' ') if result else None
    error_str = str(final_error)[:500].replace('\n', ' ') if final_error else None

    if outcome == "failure" and not result_summary_str:
        result_summary_str = f"Action failed: {error_str[:100]}..." if error_str else "Action failed (Unknown reason)"

    action_record = {
        "timestamp": time.time(),
        "type": action_type,
        "params": params,
        "outcome": outcome,
        "result_summary": result_summary_str,
        "error": error_str
    }

    with controller._blackboard_lock:
        actions = controller._blackboard.setdefault("recent_actions", []); actions.append(action_record);
        max_recent_actions = 20;
        if len(actions) > max_recent_actions: controller._blackboard["recent_actions"] = actions[-max_recent_actions:]

    # --- Log Outcome ---
    if outcome == "success":
        log_msg = f"Action '{action_type}' completed."
        if result: log_msg += f" Result: {str(result)[:150]}..."
        if action_type == "THINKING" and original_llm_error: # Check original LLM error for THINKING default
             log_msg += f" (Defaulted due to upstream error: {error_str})"
             logger.info(f"Action {action_type} SUCCEEDED (Defaulted due to upstream error: {error_str})")
        # Removed PATH_STRATEGY fallback logging here, use THINKING fallback logging
        else:
             logger.info(f"Action {action_type} SUCCEEDED.")
        controller._log_to_ui("info", log_msg)
    else: # Failure
         log_msg = f"Action '{action_type}' FAILED: {error_str}"
         controller._log_to_ui("error", log_msg)
         logger.error(f"Action {action_type} FAILED. Error: {error_str}")

    logger.debug(f"--- Execute Action END (Outcome: {outcome}) ---");
    return action_record


# --- Other functions (enhanced_reflect, evaluate_intrinsic_drives, etc.) remain unchanged from the previous version ---
async def enhanced_reflect(controller: Any):
    """Performs combined reflection to save cycle time and focus analysis."""
    if not controller._is_running_flag.is_set() or not controller._asyncio_loop or not controller._asyncio_loop.is_running():
        logger.debug("Reflection skipped: Agent stopping.")
        return {}

    logger.debug("--- Combined Reflection START ---")
    reflection_result = None
    error_log = []
    combined_reflection = {}

    try:
        with controller._blackboard_lock:
            last_action_resp = controller._blackboard.get('last_action_response', "")
            recent_history = list(controller._blackboard.get('recent_history', []))[-6:]
            recent_actions = list(controller._blackboard.get("recent_actions", []))[-5:]
            system_state = controller._blackboard.get('latest_percepts', {}).get('system_state', {})
            last_parsed_action_record = controller._blackboard.get('last_parsed_action')
            last_parsed_action = dict(last_parsed_action_record) if last_parsed_action_record else {"type": "UNKNOWN", "params": {}}
            current_goal = controller._blackboard.get("current_goal", "")
            valence = controller._blackboard.get("emotional_valence", {})
            current_sm_version = controller._blackboard.get("self_model", {}).get("version", 0)

        consecutive_failures = 0
        last_failed_action_type = None
        if recent_actions:
             last_action = recent_actions[-1]
             if last_action.get("outcome") == "failure":
                 last_failed_action_type = last_action.get("type")
                 for a in reversed(recent_actions):
                     if a.get('type') == last_failed_action_type and a.get('outcome') == 'failure':
                         consecutive_failures += 1
                     else:
                         break

        urgency_level = "Normal"
        if consecutive_failures >= 3: urgency_level = "CRITICAL"
        elif consecutive_failures >= 1: urgency_level = "Warning"

        system_prompt = (
            f"You are an internal analysis module for an AI agent. Urgency: {urgency_level}.\n"
            "Analyze the agent's recent performance across three aspects: "
            "1) Immediate Action Quality & Effectiveness, "
            "2) Recent Patterns & Progress, "
            "3) Self-Model Updates (capabilities, limitations, learned paths, identity insights).\n"
            "Be concise, focus on actionable insights. Mention specific errors/successes. "
            "Explicitly mention path success/failures if filesystem actions occurred.\n"
            "**CRITICAL: You MUST format your response using EXACTLY three sections, starting each section on a new line with '1)', '2)', and '3)' respectively.** Do NOT use any other formatting for these section headers."
        )
        user_prompt = (
             f"Current Goal: '{current_goal}'\n"
             f"Drives: Curiosity={valence.get('curiosity', 0.5):.2f}, Satisfaction={valence.get('satisfaction', 0.5):.2f}\n"
             f"Self-Model Version: {current_sm_version}\n"
             f"Last Parsed Action: {last_parsed_action.get('type')} {last_parsed_action.get('params')}\n"
             f"  - Outcome: {recent_actions[-1].get('outcome') if recent_actions else 'N/A'}\n"
             f"  - Error: {recent_actions[-1].get('error') if recent_actions and recent_actions[-1].get('error') else 'None'}\n"
             f"Last Raw Output Snippet:\n```\n{last_action_resp[:300]}...\n```\n"
             f"Recent Actions (up to 5): {[(a.get('type', '?') + '-' + a.get('outcome', '?') + (' ERR' if a.get('error') else '')) for a in recent_actions]}\n"
             f"Recent History (up to 3 turns):\n" + "\n".join([f"  {m.get('role','?')}: {str(m.get('content',''))[:80]}..." for m in recent_history]) + "\n\n"
             f"Provide your analysis in three numbered sections exactly as described in the system prompt (1), 2), 3)):"
        )

        combined_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        reflection_result, error = await call_ollama(
            controller.selected_ollama_model, combined_prompt, 0.4, controller._asyncio_loop
        )

        if error or not reflection_result:
            error_log.append(f"Combined reflection LLM error: {error or 'Empty response'}")
            logger.error(f"Combined reflection failed: {error or 'Empty response'}")
            reflection_result = None
        else:
            logger.debug(f"Combined reflection RAW result before parse:\n---\n{reflection_result}\n---")

            parts = re.split(r"\n\s*(?:\*\*)?[123][).:]?\s+", "\n" + reflection_result)
            parsed_dict = {}
            if len(parts) >= 4:
                parsed_dict["immediate"] = parts[1].strip() if parts[1] else None
                parsed_dict["episodic"] = parts[2].strip() if parts[2] else None
                parsed_dict["self_model"] = parts[3].strip() if parts[3] else None
                logger.info("Successfully parsed reflection into 3 sections using numbered markers.")
            else:
                logger.error(f"Could not parse reflection using numbered markers (found {len(parts)-1} sections). Using crude split. Raw response logged above.")
                lines = reflection_result.strip().split('\n')
                split1 = len(lines) // 3
                split2 = 2 * len(lines) // 3
                parsed_dict["immediate"] = "\n".join(lines[:split1]).strip()
                parsed_dict["episodic"] = "\n".join(lines[split1:split2]).strip()
                parsed_dict["self_model"] = "\n".join(lines[split2:]).strip()

            combined_reflection = {
                "timestamp": time.time(),
                "immediate": parsed_dict.get("immediate"),
                "episodic": parsed_dict.get("episodic"),
                "self_model": parsed_dict.get("self_model"),
                "system_state_at_reflection": system_state,
                "reflection_errors": error_log or None
            }

    except Exception as e_reflect:
        logger.exception("Exception during combined reflection processing:")
        error_log.append(f"Reflection processing exception: {str(e_reflect)}")
        combined_reflection = {
            "timestamp": time.time(),
            "immediate": None, "episodic": None, "self_model": None,
            "system_state_at_reflection": system_state if 'system_state' in locals() else {},
            "reflection_errors": error_log
        }

    with controller._blackboard_lock:
        controller._blackboard['last_reflection'] = combined_reflection
        logger.debug("Blackboard updated with combined reflection.")

    log_summary = "[Reflection Error]"
    if combined_reflection and not combined_reflection.get("reflection_errors"):
        imm = combined_reflection.get('immediate', '')[:80]
        epi = combined_reflection.get('episodic', '')[:80]
        sm = combined_reflection.get('self_model', '')[:80]
        log_summary = f"Reflect: Imm: {imm}... | Epi: {epi}... | SM: {sm}..."

    if urgency_level == "CRITICAL" or random.random() < 0.2:
        controller._log_to_ui("info", log_summary)

    if reflection_result and not error_log and combined_reflection.get("immediate"):
        try:
            reflection_json = json.dumps(combined_reflection, default=str)
            meta = {"type": "structured_reflection", "timestamp": combined_reflection["timestamp"], "urgency": urgency_level}
            await add_document_to_memory(controller._memory_collection, controller._asyncio_loop, reflection_json, meta, controller._is_running_flag)
        except Exception as e_mem:
            logger.error(f"Failed to store structured reflection in memory: {e_mem}")

    logger.debug("--- Combined Reflection END ---")
    return combined_reflection


async def evaluate_intrinsic_drives(controller: Any):
    # --- Function unchanged ---
    """Updates internal motivation metrics based on experiences."""
    logger.debug("--- Evaluate Intrinsic Drives START ---")
    new_valence = {}
    try:
        with controller._blackboard_lock:
            valence = controller._blackboard.get("emotional_valence", {"curiosity": 0.5, "satisfaction": 0.5})
            recent_actions = list(controller._blackboard.get("recent_actions", []))[-10:]

        if not recent_actions: logger.debug("No recent actions to evaluate drives."); return valence

        discovery_actions = ["READ_FILE", "LIST_FILES", "EXPLORE", "QUERY_MEMORY", "PATH_STRATEGY"]
        successful_discoveries = sum(1 for a in recent_actions if a.get("type") in discovery_actions and a.get("outcome") == "success")
        failed_discoveries = sum(1 for a in recent_actions if a.get("type") in discovery_actions and a.get("outcome") == "failure")
        goal_sets = sum(1 for a in recent_actions if a.get("type") == "SET_GOAL" and a.get("outcome") == "success")
        resets = sum(1 for a in recent_actions if a.get("type") == "RESET_STRATEGY")
        success_count = sum(1 for a in recent_actions if a.get("outcome") == "success")
        failure_count = len(recent_actions) - success_count
        non_thinking_actions = [a for a in recent_actions if a.get("type") != "THINKING"]
        non_thinking_success = sum(1 for a in non_thinking_actions if a.get("outcome") == "success")
        success_rate = non_thinking_success / len(non_thinking_actions) if non_thinking_actions else 0.5

        curiosity_decay = 0.02; satisfaction_decay = 0.03
        curiosity_gain_per_discovery = 0.06
        curiosity_loss_per_failed_discovery = 0.04
        satisfaction_gain_per_success = 0.05
        satisfaction_loss_per_failure = 0.06
        satisfaction_gain_on_goal_set = 0.1
        satisfaction_loss_on_reset = 0.2

        current_curiosity = valence.get("curiosity", 0.5); current_satisfaction = valence.get("satisfaction", 0.5)

        new_curiosity = current_curiosity - curiosity_decay * (current_curiosity - 0.5)
        new_curiosity += (curiosity_gain_per_discovery * successful_discoveries)
        new_curiosity -= (curiosity_loss_per_failed_discovery * failed_discoveries)

        new_satisfaction = current_satisfaction - satisfaction_decay * (current_satisfaction - 0.5)
        new_satisfaction += satisfaction_gain_per_success * (success_rate - 0.5)
        new_satisfaction -= (satisfaction_loss_per_failure * failure_count)
        new_satisfaction += (satisfaction_gain_on_goal_set * goal_sets)
        new_satisfaction -= (satisfaction_loss_on_reset * resets)

        new_valence = {"curiosity": min(1.0, max(0.0, new_curiosity)), "satisfaction": min(1.0, max(0.0, new_satisfaction))}

        with controller._blackboard_lock: controller._blackboard["emotional_valence"] = new_valence
        controller._log_to_ui("info", f"Drives updated: Curiosity={new_valence['curiosity']:.2f}, Satisfaction={new_valence['satisfaction']:.2f}")
        logger.debug(f"Valence updated: {new_valence} (Based on {len(recent_actions)} actions, SR: {success_rate:.2f})")

    except Exception as e:
        logger.exception("Error during intrinsic drive evaluation:")
        with controller._blackboard_lock: new_valence = controller._blackboard.get("emotional_valence", {"curiosity": 0.5, "satisfaction": 0.5})

    logger.debug("--- Evaluate Intrinsic Drives END ---"); return new_valence


async def evolve_goals(controller: Any):
    # --- Function unchanged ---
    """Autonomously evolves goals based on experiences, drives, and self-model."""
    if not controller._is_running_flag.is_set() or not controller._asyncio_loop or not controller._asyncio_loop.is_running(): logger.debug("Goal evolution skipped: Agent stopping."); return None
    logger.debug("--- Goal Evolution Check START ---")
    new_goal_generated = None; current_goal = "[Goal Unavailable]"
    try:
        with controller._blackboard_lock:
            current_goal = controller._blackboard.get("current_goal", "Observe and respond.")
            goal_set_time = controller._blackboard.get("goal_set_time", time.time() - 3600)
            recent_actions = list(controller._blackboard.get("recent_actions", []))
            last_reflection = controller._blackboard.get("last_reflection", {})
            goal_history = list(controller._blackboard.get("goal_history", []))
            emotional_valence = controller._blackboard.get("emotional_valence", {"curiosity": 0.5, "satisfaction": 0.5})
            self_model = controller._blackboard.get("self_model", {})

        goal_age = time.time() - goal_set_time;
        actions_since_goal_set = [a for a in recent_actions if a.get("timestamp", 0) >= goal_set_time]
        non_thinking_actions_since_goal = [a for a in actions_since_goal_set if a.get("type") != "THINKING"]
        success_count = sum(1 for a in non_thinking_actions_since_goal if a.get("outcome") == "success");
        total_considered = len(non_thinking_actions_since_goal)
        success_rate = success_count / total_considered if total_considered > 0 else 0.5

        consider_new_goal = False; reason = "";
        goal_stuck_threshold_s = 90
        goal_achieved_threshold_s = 180
        min_success_rate_stuck = 0.25
        min_success_rate_achieved = 0.75
        random_explore_chance = 0.03
        low_satisfaction_threshold = 0.3
        low_satisfaction_boost = 0.1 if emotional_valence.get("satisfaction", 0.5) < low_satisfaction_threshold else 0
        high_curiosity_threshold = 0.8
        high_curiosity_boost = 0.05 if emotional_valence.get("curiosity", 0.5) > high_curiosity_threshold else 0
        min_goal_age_s = 30

        if total_considered > 5 and goal_age > goal_achieved_threshold_s and success_rate >= min_success_rate_achieved: consider_new_goal = True; reason = f"goal likely achieved (SR {success_rate:.2f} over {total_considered} actions, Age {goal_age:.0f}s)"
        elif total_considered > 5 and goal_age > goal_stuck_threshold_s and success_rate <= min_success_rate_stuck: consider_new_goal = True; reason = f"stuck on goal (SR {success_rate:.2f} over {total_considered} actions, Age {goal_age:.0f}s)"
        if random.random() < (random_explore_chance + low_satisfaction_boost + high_curiosity_boost):
             if not consider_new_goal: reason = f"explore chance ({(random_explore_chance+low_satisfaction_boost+high_curiosity_boost)*100:.1f}%)"
             else: reason += " + explore chance"
             consider_new_goal = True

        if goal_age < min_goal_age_s and reason != "explore chance":
            logger.debug(f"Goal Check: Keeping '{current_goal}'. Too new (Age:{goal_age:.0f}s < {min_goal_age_s}s)")
            logger.debug("--- Goal Evolution Check END (Too New) ---"); return current_goal

        if not consider_new_goal:
             logger.debug(f"Goal Check: Keeping '{current_goal}'. Age:{goal_age:.0f}s, SR:{success_rate:.2f} ({total_considered} actions), Drives(C:{emotional_valence.get('curiosity', 0.5):.2f}, S:{emotional_valence.get('satisfaction', 0.5):.2f})")
             logger.debug("--- Goal Evolution Check END (No Change Triggered) ---"); return current_goal

        logger.info(f"Considering new goal. Reason: {reason}"); controller._log_to_ui("info", f"Agent considering new goal ({reason})...")

        reflection_summary = "[No recent reflection]";
        if isinstance(last_reflection, dict):
             imm = last_reflection.get("immediate"); epi = last_reflection.get("episodic"); self_ref = last_reflection.get("self_model");
             parts = [str(p)[:100]+"..." for p in [imm, epi, self_ref] if p];
             reflection_summary = " | ".join(parts) if parts else "[Reflection sections empty/missing]"
             if last_reflection.get("reflection_errors"): reflection_summary += f" (Errors: {last_reflection['reflection_errors']})"

        try:
             recent_actions_str = str([f"{a.get('type')}-{a.get('outcome', '?')}" for a in actions_since_goal_set[-5:]]);
             goal_history_tuples = [f"('{g.get('goal', 'N/A')[:30]}...', {g.get('duration', 0):.0f}s)" for g in goal_history[-2:]];
             goal_history_repr_str = "[" + ", ".join(goal_history_tuples) + "]" if goal_history_tuples else "[No history]"
             sm_version = self_model.get('version', 0)
             sm_caps_count = len(self_model.get('capabilities', {}))
             sm_limits_count = len(self_model.get('limitations', {}))
             sm_knowledge = self_model.get('knowledge', {})
             sm_valid_paths = len(sm_knowledge.get('validated_paths', {}))
             sm_invalid_paths = len(sm_knowledge.get('invalid_paths', []))
             self_model_summary_str = f"Self(v{sm_version}): {sm_caps_count} Caps, {sm_limits_count} Limits, {sm_valid_paths} ValidPaths, {sm_invalid_paths} InvalidPaths"
        except Exception as e: logger.error(f"Err format goal prompt ctx: {e}"); recent_actions_str = "[Err]"; goal_history_repr_str = "[Err]"; self_model_summary_str = "[Err]"

        goal_prompt = [
            {"role": "system", "content": "You are an autonomous agent deciding its next high-level goal. Consider the previous goal, why it's being changed (reason), recent performance (actions/reflections), drives, self-model knowledge (especially limitations/paths), and past goals. Propose ONE specific, achievable, and potentially interesting new goal. Output ONLY the goal description (max 1-2 sentences)."},
            {"role": "user", "content":
                f"Current Goal: '{current_goal}' (Age:{goal_age:.0f}s, SR:{success_rate:.2f})\n"
                f"Reason for Change: {reason}\n"
                f"Recent Actions (current goal): {recent_actions_str}\n"
                f"Latest Reflection Summary: {reflection_summary}\n"
                f"Goal History (Last 2): {goal_history_repr_str}\n"
                f"Current Drives: Curiosity={emotional_valence.get('curiosity', 0.5):.2f}, Satisfaction={emotional_valence.get('satisfaction', 0.5):.2f}\n"
                f"Self-Model Summary: {self_model_summary_str}\n\n"
                f"Propose ONE new goal:"
            }
        ]
        new_goal_response, error = await call_ollama(controller.selected_ollama_model, goal_prompt, 0.7, controller._asyncio_loop)

        if error or not new_goal_response: logger.error(f"Goal generation LLM failed: {error or 'Empty response'}"); controller._log_to_ui("warn", f"Goal gen failed: {error or 'Empty'}")
        else:
            potential_goals = [line.strip().strip('"\'') for line in new_goal_response.strip().split('\n') if line.strip()]
            if potential_goals:
                 new_goal_generated = potential_goals[0]
                 if new_goal_generated and len(new_goal_generated) > 10 and new_goal_generated != current_goal:
                      logger.info(f"LLM proposed new goal: '{new_goal_generated}'")
                      with controller._blackboard_lock:
                          archive = controller._blackboard.setdefault("goal_history", []);
                          archive.append({"goal": current_goal, "duration": goal_age, "success_rate": success_rate, "ended_reason": reason, "ended": time.time()})
                          max_goal_history = 10
                          if len(archive) > max_goal_history: controller._blackboard["goal_history"] = archive[-max_goal_history:]
                          controller._blackboard["current_goal"] = new_goal_generated;
                          controller._blackboard["goal_set_time"] = time.time()
                      controller._log_to_ui("info", f"Agent autonomously set new goal: {new_goal_generated}")
                      await add_document_to_memory(controller._memory_collection, controller._asyncio_loop, f"Goal Change: From '{current_goal}' to '{new_goal_generated}'. Reason: {reason}. LLM Suggestion: {new_goal_response}", {"type": "goal_evolution", "timestamp": time.time()}, controller._is_running_flag)
                 else:
                      logger.info(f"LLM proposed same/invalid goal ('{new_goal_generated}'). Keeping current goal.")
                      new_goal_generated = None
            else:
                logger.info("LLM goal response was empty or whitespace after processing.")
                new_goal_generated = None
    except Exception as e_goal: logger.exception("Err during goal evolution"); controller._log_to_ui("error", f"Goal evolution error: {e_goal}")

    logger.debug("--- Goal Evolution Check END ---")
    with controller._blackboard_lock: return controller._blackboard.get("current_goal", "[Goal Unavailable]")


async def update_self_model(controller: Any):
    # --- Function unchanged ---
    """Updates the agent's understanding of itself based on the self-model part of reflections."""
    logger.debug("--- Update Self-Model START ---")
    updated = False

    try:
        with controller._blackboard_lock:
            current_model = controller._blackboard.get("self_model", {})
            if not all(k in current_model for k in ["capabilities", "limitations", "identity_traits", "knowledge", "version"]):
                 current_model = {"capabilities": {}, "limitations": {}, "identity_traits": {}, "knowledge": {"validated_paths": {}, "invalid_paths": []}, "version": 0}
                 logger.warning("Self-model structure missing or incomplete, re-initialized.")

            original_version = current_model.get("version", 0)
            reflections = controller._blackboard.get("last_reflection", {})
            self_reflection_str = reflections.get("self_model") if isinstance(reflections, dict) else None

        if not self_reflection_str or not isinstance(self_reflection_str, str):
            logger.debug("No valid self-model reflection string found in last reflection.")
            logger.debug("--- Update Self-Model END (No Data) ---")
            return current_model

        logger.info("Processing self-model reflection string...")
        working_model = json.loads(json.dumps(current_model))

        patterns = {
            "capabilities": r"(?i)(?:Capabilities|Caps)[:\s]*\n?(.*?)(?=\n*\b(?:Limitations|Limits|Patterns|Identity|Paths|Validated Paths|Invalid Paths)|$)",
            "limitations": r"(?i)(?:Limitations|Limits)[:\s]*\n?(.*?)(?=\n*\b(?:Capabilities|Caps|Patterns|Identity|Paths|Validated Paths|Invalid Paths)|$)",
            "identity_traits": r"(?i)(?:Identity|Identity traits)[:\s]*\n?(.*?)(?=\n*\b(?:Capabilities|Caps|Limitations|Patterns|Paths|Validated Paths|Invalid Paths)|$)",
            "validated_paths": r"(?i)(?:Validated Paths|Learned Valid Paths|Successful Paths)[:\s]*\n?(.*?)(?=\n*\b(?:Invalid Paths|Capabilities|Limitations|Identity)|$)",
            "invalid_paths": r"(?i)(?:Invalid Paths|Learned Invalid Paths|Failed Paths|Path Errors)[:\s]*\n?(.*?)(?=\n*\b(?:Validated Paths|Capabilities|Limitations|Identity)|$)"
        }
        extracted_texts = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, self_reflection_str, re.DOTALL | re.MULTILINE)
            if match:
                extracted_texts[key] = match.group(1).strip()
                logger.debug(f"Extracted text for '{key}': {extracted_texts[key][:100]}...")
            else:
                 logger.debug(f"No text found for '{key}' using pattern: {pattern}")

        model_update_version_tag = f"v{original_version+1}"

        for key in ["capabilities", "limitations", "identity_traits"]:
            if key in extracted_texts and extracted_texts[key]:
                items = [item.strip('*-• \t').lower() for item in re.split(r'[\n*•-]', extracted_texts[key]) if len(item.strip('*-• \t')) > 3]
                target_dict = working_model.get(key, {})
                for item in items:
                    if item not in target_dict:
                        target_dict[item] = model_update_version_tag
                        updated = True
                        logger.debug(f"Adding new {key}: '{item}'")
                working_model[key] = target_dict

        working_model["knowledge"] = working_model.get("knowledge", {"validated_paths": {}, "invalid_paths": []})
        if "validated_paths" in extracted_texts and extracted_texts["validated_paths"]:
            potential_paths = re.findall(r'([a-zA-Z]:[/\\][\w\-\.\s/\\]+|[/\\\.][\w\-\.\s/\\]+)', extracted_texts["validated_paths"])
            target_dict = working_model["knowledge"].setdefault("validated_paths", {})
            for path in potential_paths:
                 try:
                      abs_path = os.path.abspath(path.strip().strip("'\""))
                      if abs_path not in target_dict:
                           target_dict[abs_path] = model_update_version_tag
                           updated = True
                           logger.debug(f"Adding new validated path: '{abs_path}'")
                 except Exception as path_e: logger.warning(f"Could not process potential valid path '{path}': {path_e}")

        if "invalid_paths" in extracted_texts and extracted_texts["invalid_paths"]:
            potential_paths = re.findall(r'([a-zA-Z]:[/\\][\w\-\.\s/\\]+|[/\\\.][\w\-\.\s/\\]+)', extracted_texts["invalid_paths"])
            target_list = working_model["knowledge"].setdefault("invalid_paths", [])
            for path in potential_paths:
                 try:
                      abs_path = os.path.abspath(path.strip().strip("'\""))
                      if abs_path not in target_list:
                           target_list.append(abs_path)
                           if abs_path in working_model["knowledge"].get("validated_paths", {}):
                                del working_model["knowledge"]["validated_paths"][abs_path]
                                logger.debug(f"Moving path from validated to invalid: '{abs_path}'")
                           updated = True
                           logger.debug(f"Adding new invalid path: '{abs_path}'")
                 except Exception as path_e: logger.warning(f"Could not process potential invalid path '{path}': {path_e}")

            max_invalid_paths = 50
            if len(target_list) > max_invalid_paths:
                 working_model["knowledge"]["invalid_paths"] = target_list[-max_invalid_paths:]

        if updated:
            working_model["version"] = original_version + 1
            logger.info(f"Self-model updated to version {working_model['version']}")
            with controller._blackboard_lock: controller._blackboard["self_model"] = working_model
            try:
                 model_summary = {
                     "v": working_model["version"],
                     "cap": len(working_model.get("capabilities", {})),
                     "lim": len(working_model.get("limitations", {})),
                     "idt": len(working_model.get("identity_traits", {})),
                     "v_paths": len(working_model.get("knowledge", {}).get("validated_paths", {})),
                     "i_paths": len(working_model.get("knowledge", {}).get("invalid_paths", []))
                 }
                 await add_document_to_memory(controller._memory_collection, controller._asyncio_loop, f"SelfModel Update Summary: {json.dumps(model_summary)}", {"type": "self_model_update", "timestamp": time.time()}, controller._is_running_flag)
            except Exception as e_mem: logger.error(f"Failed store self-model update summary memory: {e_mem}")
        else: logger.debug("No significant updates detected in self-model reflection.")

    except Exception as e: logger.exception("Error during self-model update:")
    finally:
        logger.debug("--- Update Self-Model END ---")
        with controller._blackboard_lock: return controller._blackboard.get("self_model", {})


async def update_narrative(controller: Any):
    # --- Function unchanged ---
    """Creates and maintains an autobiographical narrative based on significant events."""
    logger.debug("--- Update Narrative START ---")
    try:
        with controller._blackboard_lock:
            narrative_history = list(controller._blackboard.get("narrative", []))
            last_narrative_time = narrative_history[-1].get("timestamp", 0) if narrative_history else 0
            latest_percepts = controller._blackboard.get("latest_percepts", {})
            recent_actions = [a for a in controller._blackboard.get("recent_actions", []) if a.get("timestamp", 0) > last_narrative_time]
            current_goal = controller._blackboard.get("current_goal", "")
            valence = controller._blackboard.get("emotional_valence", {})

        if not recent_actions: logger.debug("No new actions for narrative."); logger.debug("--- Update Narrative END (No New Actions) ---"); return narrative_history

        significant_events = []
        last_event_time = last_narrative_time
        for action in recent_actions:
            action_type = action.get("type", "UNKNOWN"); outcome = action.get("outcome", "unknown"); params = action.get("params", {}); error = action.get("error"); result_summary = action.get("result_summary")
            timestamp = action.get("timestamp", time.time())
            is_significant = False; description = ""

            if action_type == "SET_GOAL" and outcome == "success": is_significant = True; description = f"Set new goal: '{params.get('goal', '?')[:50]}...'"
            elif action_type == "RESET_STRATEGY": is_significant = True; description = f"Triggered a strategy reset."
            elif outcome == "failure" and action_type != "THINKING": is_significant = True; description = f"Encountered failure performing {action_type}: {error or '?'}"
            elif action_type == "EXPLORE" and outcome == "success" and result_summary and len(result_summary) > 100: is_significant = True; description = f"Explored '{params.get('path', 'area')}' and found numerous items."
            elif action_type == "READ_FILE" and outcome == "success": is_significant = True; description = f"Read content from file '{params.get('path', '?')}'."
            elif action_type == "QUERY_MEMORY" and outcome == "success" and result_summary and "No relevant results" not in result_summary : is_significant = True; description = f"Recalled relevant memories about '{params.get('query', '?')[:30]}...'."
            elif action_type == "THINKING" and "user_input" in latest_percepts and latest_percepts.get("timestamp", 0) > last_event_time:
                user_input = latest_percepts["user_input"]
                is_significant = True; description = f"Processed and responded to user input: '{str(user_input)[:50]}...'"
            elif action_type == "THINKING" and outcome == "success" and error and ("LLM/Action error" in str(params.get("content", "")) or "Timeout" in str(error)):
                 is_significant = True; description = f"Recovered from an internal communication error ({str(error)[:50]}...). Paused to rethink."

            if is_significant:
                significant_events.append({"type": action_type, "outcome": outcome, "description": description, "timestamp": timestamp})
                last_event_time = timestamp

        if not significant_events: logger.debug("No significant events identified since last narrative."); logger.debug("--- Update Narrative END (No Sig. Events) ---"); return narrative_history

        logger.info(f"Identified {len(significant_events)} significant events for narrative.")

        event_summary = [f"- {datetime.datetime.fromtimestamp(evt['timestamp']).strftime('%H:%M')}: {evt['description']}" for evt in significant_events]
        previous_entries_summary = [f"... {entry.get('content', '')[-120:]}" for entry in narrative_history[-2:]]

        narrative_prompt = [
            {"role": "system", "content": "You are the AI agent, writing a brief entry in your first-person autobiographical narrative. Synthesize the *meaning* and *feeling* of the recent significant events listed below into a short, reflective paragraph (2-4 sentences). Continue your ongoing story and sense of self. Focus on learning, progress, setbacks, or shifts in understanding."},
            {"role": "user", "content":
                f"Current Goal: '{current_goal}'\n"
                f"Current Mood (Curiosity/Satisfaction): {valence.get('curiosity', 0.5):.1f}/{valence.get('satisfaction', 0.5):.1f}\n"
                f"Recent Significant Events (since last entry):\n" + "\n".join(event_summary) + "\n\n"
                f"My Previous Narrative Entry Snippets:\n" + "\n".join([f"- \"{s}\"" for s in previous_entries_summary if s]) + "\n\n"
                f"Continue the narrative (write 2-4 sentences from 'I...' perspective):"
            }
        ]

        narrative_content, error = await call_ollama(controller.selected_ollama_model, narrative_prompt, 0.75, controller._asyncio_loop)

        if error or not narrative_content: logger.error(f"Narrative generation failed: {error or 'Empty'}"); logger.debug("--- Update Narrative END (LLM Error) ---"); return narrative_history

        narrative_content = narrative_content.strip()

        new_entry = {
            "timestamp": time.time(),
            "content": narrative_content,
            "triggering_events": significant_events,
            "goal_at_time": current_goal,
            "valence_at_time": valence.copy()
        }
        logger.info(f"Generated Narrative Entry: {new_entry['content'][:150]}...")

        with controller._blackboard_lock:
            current_narrative = controller._blackboard.get("narrative", [])
            current_narrative.append(new_entry)
            max_narrative_entries = 25
            if len(current_narrative) > max_narrative_entries:
                entries_to_archive = current_narrative[:-max_narrative_entries]
                current_narrative = current_narrative[-max_narrative_entries:]
                logger.info(f"Pruned narrative. Archiving {len(entries_to_archive)} entries to memory.")
                for old_entry in entries_to_archive:
                     try:
                         entry_text = f"Archived Narrative Entry ({datetime.datetime.fromtimestamp(old_entry['timestamp']).strftime('%Y-%m-%d %H:%M')})\nGoal: {old_entry.get('goal_at_time', 'N/A')}\nMood: C={old_entry.get('valence_at_time', {}).get('curiosity', 0):.1f}/S={old_entry.get('valence_at_time', {}).get('satisfaction', 0):.1f}\nContent: {old_entry.get('content', '')}\nEvents: {str(old_entry.get('triggering_events', []))[:200]}..."
                         meta = {"type": "archived_narrative", "timestamp": old_entry['timestamp']}
                         await add_document_to_memory(controller._memory_collection, controller._asyncio_loop, entry_text, meta, controller._is_running_flag)
                     except Exception as archive_err: logger.error(f"Error archiving narrative entry: {archive_err}")
            controller._blackboard["narrative"] = current_narrative

        if random.random() < 0.2:
             controller._log_to_ui("info", f"Narrative Update: {new_entry['content']}")

    except Exception as e: logger.exception("Error during narrative update:")
    logger.debug("--- Update Narrative END ---")
    with controller._blackboard_lock: return controller._blackboard.get("narrative", [])

# --- END OF FILE cognitive_modules.py ---