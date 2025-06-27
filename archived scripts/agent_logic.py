# --- START OF FILE agent_logic.py ---

# app/extensions/consciousness_experiment/agent_logic.py
import threading
import asyncio
import queue
import logging
import time
import json
import enum
import os
import re
import datetime
import traceback
import random # Needed for goal evolution chance
from typing import Any, Dict, List, Tuple, Optional

# --- Safe Library Imports ---
try: import psutil; PSUTIL_AVAILABLE = True
except ImportError: psutil = None; PSUTIL_AVAILABLE = False; logging.error("AgentLogic: psutil missing.")
try: import ollama; OLLAMA_AVAILABLE = True
except ImportError: ollama = None; OLLAMA_AVAILABLE = False; logging.error("AgentLogic: ollama missing.")
try: import chromadb; from chromadb.config import Settings as ChromaSettings; CHROMADB_AVAILABLE = True
except ImportError: chromadb = None; CHROMADB_AVAILABLE = False; logging.error("AgentLogic: chromadb missing.")

# --- Constants & Configuration ---
AGENT_CYCLE_INTERVAL_S = 0.1
OLLAMA_CALL_TIMEOUT_S = 120.0
_EXTENSION_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH = os.path.join(_EXTENSION_BASE_DIR, "agent_memory")
MAX_RECENT_TURNS_IN_PROMPT = 5
MEMORY_RETRIEVAL_COUNT = 3
MAX_FILES_LISTED = 25
MAX_FILE_READ_CHARS = 4000
GOAL_EVALUATION_CYCLES = 20 # How often to potentially change goal (e.g., every 20 cycles * 0.1s/cycle = ~2 seconds)
MOTIVATION_EVALUATION_CYCLES = 5 # How often to evaluate intrinsic drives

logger = logging.getLogger(__name__)

class AgentState(enum.Enum): STOPPED=0; STARTING=1; RUNNING=2; STOPPING=3; ERROR=4

class AgentController:
    def __init__(self, ui_queue: queue.Queue, selected_model_name: str):
        if not isinstance(ui_queue, queue.Queue): raise TypeError("ui_queue required")
        if not isinstance(selected_model_name, str) or not selected_model_name: raise ValueError("model_name required")
        self.ui_queue = ui_queue
        self.selected_ollama_model = selected_model_name
        self.agent_thread: threading.Thread | None = None
        self._is_running_flag = threading.Event()
        self._asyncio_loop: asyncio.AbstractEventLoop | None = None
        # --- Initialize Blackboard with Goal Tracking ---
        self._blackboard: Dict[str, Any] = {
            "current_goal": "Initial Goal: Observe system state and respond.",
            "goal_set_time": time.time(), # Track when goal was set
            "goal_history": [], # Archive completed/changed goals
            "recent_history": [],
            "latest_percepts": {},
            "last_reflection": None, # Will become dict after first reflection
            "last_action_response": None,
            "last_parsed_action": None,
            "recent_actions": [],
            # --- Placeholders for Phase 1 additions ---
            "emotional_valence": {"curiosity": 0.5, "satisfaction": 0.5}, # Added placeholder
            "self_model": {"capabilities": {}, "limitations": {}, "identity_traits": {}, "version": 0}, # Added placeholder
            "narrative": [] # Added placeholder
            # --- End Placeholders ---
        }
        # --- End Blackboard Init ---
        self._blackboard_lock = threading.RLock()
        self._user_input_queue = asyncio.Queue()
        self._memory: Optional[chromadb.Client] = None
        self._memory_collection: Optional[chromadb.Collection] = None
        self.agent_state = AgentState.STOPPED
        logger.info("AgentController ready – model=%s", self.selected_ollama_model)

    # --- UI Communication Helpers (_log_to_ui, _update_state_in_ui) ---
    # (Keep as is)
    def _log_to_ui(self, level: str, message: str):
        try: self.ui_queue.put_nowait((f"log_{level.lower()}", message))
        except queue.Full: logger.warning(f"UI queue full. Dropping log: {message}")
        except Exception as e: logger.error(f"Error putting log on UI queue: {e}")
    def _update_state_in_ui(self, new_state: AgentState):
         if self.agent_state != new_state: logger.info(f"Agent state changing: {self.agent_state.name} -> {new_state.name}"); self.agent_state = new_state
         try: self.ui_queue.put_nowait(("state_update", new_state))
         except queue.Full: logger.warning(f"UI queue full. Dropping state update: {new_state.name}")
         except Exception as e: logger.error(f"Error putting state update on UI queue: {e}")
         # Removed 'else: logger.debug...' clause for brevity

    # --- Public Methods (start, stop, send_user_input) ---
    def start(self):
        # (Reset goal/state on start)
        if self.agent_thread and self.agent_thread.is_alive(): logger.warning("Start ignored – thread running"); return
        if not OLLAMA_AVAILABLE: self._log_to_ui("error", "Ollama library missing"); self._update_state_in_ui(AgentState.ERROR); return
        with self._blackboard_lock:
             # Reset key dynamic state
             self._blackboard['current_goal'] = "Observe system state and respond to user." # Reset goal
             self._blackboard['goal_set_time'] = time.time()
             self._blackboard['recent_history'] = []
             self._blackboard['last_reflection'] = None
             self._blackboard['recent_actions'] = []
             self._blackboard['latest_percepts'] = {}
             self._blackboard['last_action_response'] = None
             self._blackboard['last_parsed_action'] = None
             # Reset Phase 1 state as well
             self._blackboard["emotional_valence"] = {"curiosity": 0.5, "satisfaction": 0.5}
             self._blackboard["self_model"] = {"capabilities": {}, "limitations": {}, "identity_traits": {}, "version": 0}
             self._blackboard["narrative"] = []
             # Don't reset goal_history, keep archive
        self._is_running_flag.set(); self._update_state_in_ui(AgentState.STARTING)
        self.agent_thread = threading.Thread(target=self._agent_thread_entry, daemon=True, name="ConsciousnessAgentThread")
        self.agent_thread.start(); logger.info("Agent thread spawned")
    def stop(self):
        # (Keep as is)
        logger.info("Stop requested"); self._is_running_flag.clear(); self._update_state_in_ui(AgentState.STOPPING)
        if self._asyncio_loop and self._asyncio_loop.is_running():
             try: self._asyncio_loop.call_soon_threadsafe(lambda: None) # Poke the loop to help it exit cleanly
             except Exception as e: logger.warning(f"Could not poke loop during stop: {e}")
    def send_user_input(self, text: str):
        # (Keep as is)
        if self.agent_state != AgentState.RUNNING: logger.warning("send_user_input ignored – agent not RUNNING"); return
        if not (self._asyncio_loop and self._asyncio_loop.is_running()): logger.warning("Asyncio loop not active"); return
        try: self._asyncio_loop.call_soon_threadsafe(self._user_input_queue.put_nowait, text)
        except Exception as e: logger.error(f"Failed to queue user input: {e}"); self._log_to_ui("error", f"Failed to send input to agent: {e}")

    # --- Thread Entry and Loop Management ---
    # (Keep as is)
    def _agent_thread_entry(self):
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop); self._asyncio_loop = loop
        logger.debug("Asyncio loop initialised")
        try: loop.run_until_complete(self._initialize_and_run_loop())
        except Exception as exc: logger.exception("Fatal error inside agent loop"); self._log_to_ui("error", f"Agent fatal error: {exc}"); self._update_state_in_ui(AgentState.ERROR)
        finally: self._cleanup_thread_resources(loop); logger.info("Agent thread exited cleanly") # Changed message slightly for confirmation
    def _cleanup_thread_resources(self, loop):
        logger.info("Cleaning up agent thread resources...")
        if loop and loop.is_running():
            tasks = asyncio.all_tasks(loop)
            if tasks:
                logger.info(f"Cancelling {len(tasks)} outstanding tasks...")
                for task in tasks:
                    task.cancel()
                try:
                    # Give tasks a moment to cancel
                    loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                    logger.info("Outstanding tasks cancelled.")
                except Exception as e:
                    logger.error(f"Error during task cancellation: {e}")
            try:
                # Stop the loop before closing
                loop.stop()
                logger.info("Asyncio loop stopped.")
                # Close should be called on a stopped loop
                loop.close()
                logger.info("Asyncio loop closed.")
            except Exception as e: logger.error(f"Error stopping/closing loop: {e}")
        self._asyncio_loop = None; self._memory = None; self._memory_collection = None
        if self.agent_state != AgentState.ERROR: self._update_state_in_ui(AgentState.STOPPED)
    async def _initialize_and_run_loop(self):
        ok = True; self._log_to_ui("info", "Initializing agent…")
        if OLLAMA_AVAILABLE:
            try: await asyncio.get_running_loop().run_in_executor(None, ollama.list); self._log_to_ui("info", "Ollama connection OK.")
            except Exception as e: ok = False; self._log_to_ui("error", f"Ollama connection failed: {e}")
        else: ok = False; self._log_to_ui("error", "Ollama import failed")
        if CHROMADB_AVAILABLE and ok:
            try: os.makedirs(MEMORY_PATH, exist_ok=True); self._memory = chromadb.PersistentClient(path=MEMORY_PATH, settings=ChromaSettings(anonymized_telemetry=False)); self._memory_collection = self._memory.get_or_create_collection("agent_episodic_memory"); self._log_to_ui("info", "Vector memory initialised")
            except Exception as e: logger.exception("Memory init failed"); self._log_to_ui("error", f"Memory init failed: {e}")
        if not ok: self._update_state_in_ui(AgentState.ERROR); return
        self._update_state_in_ui(AgentState.RUNNING); await self._run_agent_loop(); logger.info("Exiting _initialize_and_run_loop.")

    # --- MODIFIED MAIN LOOP (To integrate Phase 1 components later) ---
    async def _run_agent_loop(self):
        """Enhanced cognitive loop integrating internal state components."""
        logger.info("Enhanced autonomous cognitive loop running (Phase 1 Structure)")
        cycle_count = 0
        while self._is_running_flag.is_set():
            start = time.monotonic(); cycle_count += 1; logger.debug(f"--- Cognitive Cycle {cycle_count} START ---")
            try:
                # 1. Perception
                await self._perceive(); logger.debug(f"[Cycle {cycle_count}] Finished _perceive.")

                # 2. Goal Management (Periodic)
                if cycle_count % GOAL_EVALUATION_CYCLES == 0:
                    logger.debug(f"[Cycle {cycle_count}] Evaluating goals...")
                    await self._evolve_goals()
                    logger.debug(f"[Cycle {cycle_count}] Finished goal evaluation.")

                # 3. Decision & Action
                await self._decide_act(); logger.debug(f"[Cycle {cycle_count}] Finished _decide_act.")
                await self._execute_action(); logger.debug(f"[Cycle {cycle_count}] Finished _execute_action.")

                # 4. Reflection & Learning
                await self._enhanced_reflect(); logger.debug(f"[Cycle {cycle_count}] Finished _enhanced_reflect.")

                # --- Integration Points for Phase 1 Steps ---
                # 5. Motivation Update (Periodic)
                if cycle_count % MOTIVATION_EVALUATION_CYCLES == 0: # Evaluate every N cycles
                    await self._evaluate_intrinsic_drives()
                    logger.debug(f"[Cycle {cycle_count}] Finished _evaluate_intrinsic_drives.")

                # 6. Self-Model Update (Periodic)
                # if cycle_count % 10 == 0: await self._update_self_model(); logger.debug(f"[Cycle {cycle_count}] Finished _update_self_model.")

                # 7. Narrative Update (Periodic)
                # if cycle_count % 15 == 0: await self._update_narrative(); logger.debug(f"[Cycle {cycle_count}] Finished _update_narrative.")
                # --- End Integration Points ---

            except asyncio.CancelledError: logger.info("Cognitive loop cancelled"); break
            except Exception as exc: logger.exception(f"Error in cognitive cycle: {exc}"); self._log_to_ui("error", f"Cognitive cycle error: {exc}"); await asyncio.sleep(1.0) # Sleep on error to prevent rapid looping

            # Calculate sleep duration
            elapsed = time.monotonic() - start
            sleep_duration = max(0, AGENT_CYCLE_INTERVAL_S - elapsed)
            if sleep_duration == 0: logger.warning(f"Cycle {cycle_count} overran target interval ({elapsed:.3f}s)")
            await asyncio.sleep(sleep_duration)
            logger.debug(f"Cycle {cycle_count} took {elapsed:.3f}s. Slept {sleep_duration:.3f}s. --- END ---")

        logger.info("Cognitive loop finished.")
    # --- END MODIFIED MAIN LOOP ---

    # --- Core Agent Steps ---
    async def _perceive(self):
        """Gathers inputs and updates the blackboard."""
        # Check if loop is still running before proceeding
        if not self._is_running_flag.is_set() or not self._asyncio_loop or not self._asyncio_loop.is_running():
            logger.debug("Perceive skipped: Agent stopping.")
            return

        logger.debug("Perceive step start...")
        percepts: Dict[str, Any] = {"timestamp": time.time()}

        # 1. Check for user input
        try:
            user_input = await asyncio.wait_for(self._user_input_queue.get(), timeout=0.01)
            if user_input:
                percepts["user_input"] = user_input
                logger.info(f"Perceived input: '{user_input[:50]}...'")
                self._user_input_queue.task_done()
        except asyncio.TimeoutError:
            pass # No user input this cycle is normal
        except asyncio.QueueEmpty: # Should not happen with wait_for, but safe to handle
            pass
        except Exception as e_queue: # Catch other potential queue errors
            logger.error(f"Error getting user input from queue: {e_queue}")

        # 2. Get system state
        if PSUTIL_AVAILABLE:
            try:
                # Use run_in_executor for potentially blocking psutil calls
                stats = await asyncio.get_running_loop().run_in_executor(
                    None, # Use default executor
                    lambda: {"cpu_percent": psutil.cpu_percent(), "memory_percent": psutil.virtual_memory().percent,}
                )
                percepts["system_state"] = stats
                logger.debug(f"Gathered psutil stats: {stats}")
            except RuntimeError as e: # Handle executor shutdown
                 if "cannot schedule new futures after shutdown" in str(e):
                     logger.warning("psutil check skipped: Executor shutdown.")
                 else: raise # Re-raise other runtime errors
            except Exception as e_psutil:
                logger.warning(f"Failed to gather psutil stats: {e_psutil}")
        # End psutil check

        # 3. Update blackboard (outside the psutil check)
        with self._blackboard_lock:
            self._blackboard["latest_percepts"] = percepts
            logger.debug(f"Percepts updated: {json.dumps(percepts, default=str)}")

    # --- UPDATED _decide_act with Enhanced Feedback ---
    async def _decide_act(self):
        """Determines the next action, using few-shot examples to enforce output format."""
        logger.debug("--- Decide/Act START ---");
        try:
            # --- Define Available Actions ---
            available_actions = """
Available Actions:
- [ACTION: THINKING content="<reasoning>"] - Output reasoning/thought process (Default if no other action).
- [ACTION: LIST_FILES path="<abs_path>"] - List directory contents. Use EXPLORE for broader discovery.
- [ACTION: READ_FILE path="<abs_path>"] - Read a text file's content (up to 4k chars). Verify path first if unsure.
- [ACTION: QUERY_MEMORY query="<search>"] - Search internal memory for reflections, past events, etc.
- [ACTION: SET_GOAL goal="<desc>"] - Set a new objective for yourself.
- [ACTION: EXPLORE path="<abs_path>" depth="<1 or 2>"] - Explore directory structure starting at path (defaults to current working dir, depth 1). Useful after path errors.
"""
            # --- Gather Context from Blackboard ---
            with self._blackboard_lock:
                # Gather standard context
                current_goal = self._blackboard.get("current_goal", "Observe and respond.")
                recent_history = self._blackboard.get('recent_history', [])
                current_percepts = self._blackboard.get('latest_percepts', {})
                last_reflection_obj = self._blackboard.get('last_reflection')
                recent_actions = self._blackboard.get('recent_actions', [])
                last_action_record = recent_actions[-1] if recent_actions else None
                # Gather Phase 1 context
                emotional_valence = self._blackboard.get("emotional_valence", {"curiosity": 0.5, "satisfaction": 0.5}) # Get current valence
                # self_model_summary = ... # Add later

            user_input_this_cycle = current_percepts.get('user_input')

            # --- Build Prompt Messages ---
            messages = []

            # 1. Initial System Prompt (Goal + Available Actions)
            system_prompt_part1 = (
                f"Your current goal is: '{current_goal}'. "
                f"Choose ONE action from the list below to take a step towards your goal, or respond to the user if they provided input.\n"
                f"\nAvailable Actions:\n{available_actions}"
            )
            messages.append({"role": "system", "content": system_prompt_part1})

            # 2. Add Memory Context
            query_text = user_input_this_cycle or (last_reflection_obj.get('immediate') if isinstance(last_reflection_obj, dict) and last_reflection_obj.get('immediate') else f"Goal: {current_goal}")
            if query_text:
                retrieved_docs = await self._retrieve_from_memory(query_text, n_results=MEMORY_RETRIEVAL_COUNT)
                if retrieved_docs:
                    memory_separator = "\n---\n"; safe_docs = [str(doc) if doc is not None else "[None]" for doc in retrieved_docs]; memory_context = memory_separator.join(safe_docs)
                    messages.append({"role": "system", "content": f"Memory Context (previous reflections/events):\n```\n{memory_context}\n```"})

            # 3. Add Recent History
            if recent_history:
                 max_history_items = MAX_RECENT_TURNS_IN_PROMPT * 2; messages.extend(recent_history[-max_history_items:])

            # 4. Add Current Context Summary (INCLUDING VALENCE)
            context_lines = [f"Time: {datetime.datetime.now().isoformat()}"]
            if 'system_state' in current_percepts: state = current_percepts['system_state']; context_lines.append(f"System: CPU {state.get('cpu_percent', '?')}% | Mem {state.get('memory_percent', '?')}%") if state else None
            if last_reflection_obj and isinstance(last_reflection_obj, dict): immediate_reflect = last_reflection_obj.get('immediate'); context_lines.append(f"Last Immediate Reflection: {str(immediate_reflect)[:150]}...") if immediate_reflect else None
            # *** ADD VALENCE TO CONTEXT ***
            context_lines.append(f"Current Drives: Curiosity={emotional_valence.get('curiosity', 0.5):.2f}, Satisfaction={emotional_valence.get('satisfaction', 0.5):.2f}")
            # *** END ADD VALENCE ***
            context_summary = "\n".join(context_lines); messages.append({"role": "system", "content": f"Current Context:\n{context_summary}"})


            # 5. Add Explicit Last Action Outcome Feedback
            if last_action_record:
                outcome = last_action_record.get('outcome', 'unknown'); action_type = last_action_record.get('type', '?'); action_params = last_action_record.get('params', {}); action_error = last_action_record.get('error'); action_result = last_action_record.get('result_summary')
                outcome_str = f"Outcome of Last Action ({action_type} {action_params}): {outcome}."; feedback_prefix = "Last Action Feedback: "
                if outcome == 'failure':
                    feedback_prefix = "!! Last Action FAILED: "
                    if action_error: sanitized_error = str(action_error).replace('\n', ' ').replace('"', "'")[:200]; outcome_str += f" Error: {sanitized_error}. "; outcome_str += "Analyze this error. Do NOT repeat the mistake. Consider using EXPLORE if it was a path error."
                    else: outcome_str += " Failure reason unknown."
                elif outcome == 'success':
                     if action_result: sanitized_result = str(action_result).replace('\n', ' ').replace('"', "'")[:200]; outcome_str += f" Result Summary: {sanitized_result}"
                     else: outcome_str += " Action succeeded."
                else: outcome_str += " Outcome unknown."
                messages.append({"role": "system", "content": f"{feedback_prefix}{outcome_str}"})
                logger.debug("Built prompt: Added explicit last action outcome feedback.")


            # --- Final Instruction with Few-Shot Examples ---
            format_instruction = (
                "Output Format Instructions:\n"
                "1. First, provide your reasoning for the action you choose.\n"
                "2. Then, on a **completely new line**, provide EXACTLY ONE action tag using square brackets `[]`.\n"
                "3. The action tag MUST follow this format: `[ACTION: <ACTION_TYPE> param1=\"value1\" param2=\"value2\"...]`\n"
                "4. Use ONLY the `<ACTION_TYPE>` listed in the 'Available Actions' section.\n"
                "5. Ensure all parameter values are enclosed in double quotes `\"\"`.\n\n"
                "Example 1 (Reading a file):\n"
                "Reasoning: I need to check the system log for recent errors.\n"
                "[ACTION: READ_FILE path=\"/var/log/syslog\"]\n\n"
                "Example 2 (Querying Memory):\n"
                "Reasoning: I should recall my previous attempt to list this directory.\n"
                "[ACTION: QUERY_MEMORY query=\"previous attempts to list /data/files\"]\n\n"
                "Example 3 (Thinking):\n"
                "Reasoning: The user's request is ambiguous, I need to ask for clarification.\n"
                "[ACTION: THINKING content=\"Asking user for clarification on their request.\"]\n\n"
                "Now, provide your reasoning and the action tag."
            )
            messages.append({"role": "system", "content": format_instruction})

            # 6. Add User Input or Autonomous Trigger
            if user_input_this_cycle:
                messages.append({"role": "user", "content": user_input_this_cycle})
            else:
                messages.append({"role": "user", "content": f"State reasoning based on goal, drives, context, memory, and last action feedback. Then output ONE action tag in the required format on a new line."})

            logger.debug(f"Final messages count: {len(messages)}. Preparing to call Ollama...")
            # logger.debug(f"Full prompt messages:\n{json.dumps(messages, indent=2)}") # Uncomment for deep debug

        except Exception as e_prompt:
             logger.exception("!!! ERROR during prompt building !!!"); error = f"Prompt build error: {e_prompt}"; content = None; action_response = f"[Action Error: {error}]"; self._log_to_ui("error", error)
             try:
                 with self._blackboard_lock: self._blackboard['last_action_response'] = action_response; self._add_to_history({"role": "assistant", "content": action_response})
             except Exception as e_hist: logger.error(f"Failed to add prompt build error to history: {e_hist}")
             try: self.ui_queue.put_nowait(("agent_output", action_response))
             except queue.Full: logger.warning("UI queue full dropping prompt build error.")
             logger.debug("--- Decide/Act END (prompt error) ---"); return

        content, error = await self._call_ollama(messages, temperature=0.4) # Keep slightly lower temp

        if error: action_response = f"[Ollama Action Error: {error}]"; self._log_to_ui("error", action_response); logger.error(f"Ollama call failed: {error}")
        elif content is not None: action_response = content; logger.info(f"Ollama RAW response received: '{action_response[:200]}...'")
        else: logger.error("Ollama call returned None content and None error."); action_response = "[Action Error: Unknown Ollama response]"; self._log_to_ui("error", action_response)

        with self._blackboard_lock: self._blackboard['last_action_response'] = action_response; self._add_to_history(assistant_turn={"role": "assistant", "content": action_response}, user_turn_content=user_input_this_cycle)
        logger.debug("Blackboard updated with raw action response.")

        try: self.ui_queue.put_nowait(("agent_output", action_response)); logger.debug("Placed raw agent_output on UI queue.")
        except queue.Full: logger.warning("UI queue full dropping agent output.")
        except Exception as e_queue: logger.exception("Error putting agent_output on queue!")

        logger.debug("--- Decide/Act END ---")

    async def _execute_action(self):
        """Parses action from last raw response and executes it, focusing on finding the action tag line."""
        logger.debug("--- Execute Action START ---")
        action_type = "THINKING"; params = {}; result = None; error = None; parsed_action = None; parse_error = None
        action_tag_found = False # Flag to track if we successfully parsed an action tag

        with self._blackboard_lock: raw_response = self._blackboard.get('last_action_response')
        if not raw_response:
             logger.warning("No action response found to execute.");
             logger.debug("--- Execute Action END (No response) ---")
             return # Nothing to execute

        # --- NEW PARSING STRATEGY: Find the ACTION line ---
        action_line = None
        response_lines = raw_response.strip().split('\n')
        for line in reversed(response_lines): # Search from bottom up, assuming action is last
            clean_line = line.strip()
            # Be slightly more flexible with starting space
            if clean_line.startswith('[ACTION:') or clean_line.startswith('[ACTION :'):
                 action_line = clean_line
                 logger.debug(f"Found potential action line: '{action_line}'")
                 # *** ADD CRITICAL LOGGING HERE ***
                 # logger.critical(f"Attempting to parse action line: >>>{action_line}<<<") # Uncomment for deeper debug
                 action_tag_found = True # Tentatively assume found
                 break # Found the likely action line

        if action_tag_found and action_line:
            # Now parse the identified action line more reliably
            # *** RELAXED REGEX: Removed '$' anchor ***
            action_match = re.match(r"\[\s*ACTION\s*:\s*(\w+)\s*(.*?)?\s*\]", action_line, re.IGNORECASE) # Removed $
            if action_match:
                # --- Parsing successful ---
                action_type = action_match.group(1).upper()
                params_str = (action_match.group(2) or "").strip()
                logger.debug(f"REGEX MATCHED! Type='{action_type}', ParamsStr='{params_str}'")
                try:
                    # Use the existing robust parameter parsing
                    parsed_params_list = re.findall(r'''(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)'|(\S+))''', params_str)
                    params = {k: v2 if v2 is not None else (v3 if v3 is not None else v4)
                              for k, v2, v3, v4 in parsed_params_list}
                    logger.info(f"PARSED ACTION: Type={action_type}, Params={params}")

                    # Handle potential edge case: params_str exists but parsing yields nothing
                    if not params and params_str and '=' in params_str:
                         parse_error = f"Parameter format error in '{params_str}'. Expected key=\"value\"."
                         logger.error(parse_error)
                         error = parse_error # Propagate parse error to prevent execution
                         action_type = "THINKING" # Revert to THINKING on error
                         params = {"content": f"Error parsing action parameters: {params_str}. Original response: {raw_response}"}
                         action_tag_found = False # Treat as if tag wasn't successfully processed

                except Exception as parse_err:
                    logger.exception(f"Critical error parsing action params '{params_str}' from line '{action_line}'")
                    params = {}
                    error = f"Parameter parsing exception: {parse_err}"
                    parse_error = error
                    action_type = "THINKING"
                    params = {"content": f"Exception parsing action parameters: {params_str}. Original response: {raw_response}"}
                    action_tag_found = False # Treat as if tag wasn't successfully processed
            else:
                # --- Regex failed on the line ---
                logger.warning(f"REGEX FAILED to parse structure of action line: '{action_line}'")
                parse_error = f"Malformed action tag on line: {action_line}"
                error = parse_error
                action_type = "THINKING" # Revert to THINKING
                params = {"content": f"Malformed action tag found. Original response: {raw_response}"}
                action_tag_found = False # Crucial: Reset flag as parsing failed
        # --- End NEW PARSING STRATEGY ---

        # If no action tag was found OR parsing the tag failed, default to THINKING with the full raw response
        if not action_tag_found:
            # Log the *reason* for defaulting if possible
            if action_line and not action_match: # We found a line but couldn't parse it
                 logger.debug(f"Defaulting to THINKING because regex failed on line: '{action_line}'")
            elif not action_line: # We never even found a line starting with [ACTION:
                 logger.debug("Defaulting to THINKING because no line starting with '[ACTION:' was found.")
            else: # General fallback
                 logger.debug("Defaulting to THINKING.")

            action_type = "THINKING"
            # Use the full raw response for context, unless we had a specific parse error message
            default_content = f"Failed to parse intended action ({parse_error}). Original response: {raw_response}" if parse_error else raw_response
            params = {"content": default_content}
            # Ensure 'error' reflects the situation if a parse error occurred above
            if parse_error and not error: error = parse_error


        # --- Store parsed action (even if defaulted) ---
        parsed_action = {"type": action_type, "params": params, "parse_error": parse_error}
        with self._blackboard_lock: self._blackboard['last_parsed_action'] = parsed_action
        self._log_to_ui("info", f"Agent intends: {action_type} {params if params else ''}") # Logs the *intended* action

        # --- Execute Action ---
        # This outer 'if not error:' checks for PARSING errors propagated from above.
        # Errors during execution are handled inside the 'try' block.
        if not error:
            logger.info(f"Executing action: {action_type} with params {params}")
            self._log_to_ui("info", f"Agent executing: {action_type} {params if params else ''}") # Logs the *executed* action
            try: # Action execution logic
                if action_type == "THINKING":
                     result = f"Thought process recorded: {str(params.get('content', ''))[:100]}..."
                elif action_type == "LIST_FILES":
                    path = params.get('path', '')
                    # --- Path Validation ---
                    if not path: error = "Missing 'path' parameter."
                    elif not os.path.isabs(path): error = f"Path must be absolute: '{path}'"
                    elif '..' in path.split(os.sep): error = f"Path traversal disallowed: '{path}'"
                    # --- Execution ---
                    elif os.path.isdir(path):
                        try: files = os.listdir(path); result = f"Files in '{path}': {files[:MAX_FILES_LISTED]}{'...' if len(files) > MAX_FILES_LISTED else ''}"; logger.info(f"Listed files in {path}")
                        except PermissionError: error = f"Permission denied listing: {path}"
                        except FileNotFoundError: error = f"Directory not found: {path}" # Should be caught by isdir, but belt-and-suspenders
                        except Exception as e: error = f"Error listing {path}: {str(e)}"
                    else: error = f"Path is not a directory: {path}"
                elif action_type == "READ_FILE":
                    path = params.get('path', '')
                     # --- Path Validation ---
                    if not path: error = "Missing 'path' parameter."
                    elif not os.path.isabs(path): error = f"Path must be absolute: '{path}'"
                    elif '..' in path.split(os.sep): error = f"Path traversal disallowed: '{path}'"
                    # --- Execution ---
                    elif os.path.isfile(path):
                        try:
                            with open(path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read(MAX_FILE_READ_CHARS)
                            result = f"Content of '{path}' (first {MAX_FILE_READ_CHARS} chars):\n```\n{content}\n```"; logger.info(f"Read file {path}")
                        except PermissionError: error = f"Permission denied reading: {path}"
                        except FileNotFoundError: error = f"File not found: {path}" # Should be caught by isfile
                        except Exception as e: error = f"Error reading {path}: {str(e)}"
                    else: error = f"Path is not a file or does not exist: {path}"
                elif action_type == "QUERY_MEMORY":
                    query = params.get('query', '')
                    if query: memories = await self._retrieve_from_memory(query, n_results=MEMORY_RETRIEVAL_COUNT); result = f"Memory query results for '{query}': {str(memories)[:300]}..." # Uses corrected _retrieve_from_memory
                    else: error = "Missing or empty 'query' parameter."
                elif action_type == "SET_GOAL":
                    new_goal = params.get('goal', '')
                    if new_goal:
                        with self._blackboard_lock:
                            old_goal = self._blackboard.get('current_goal', '')
                            current_set_time = self._blackboard.get('goal_set_time', time.time()) # Get time BEFORE update
                            self._blackboard['current_goal'] = new_goal
                            self._blackboard['goal_set_time'] = time.time() # Reset timer for new goal
                        result = f"Goal updated to: '{new_goal}'"
                        self._log_to_ui("info", f"Agent autonomously set new goal: {new_goal}")
                        # Archive old goal when SET_GOAL action is used *explicitly*
                        goal_age = time.time() - current_set_time # Calculate age based on old set time
                        with self._blackboard_lock:
                             archive = self._blackboard.setdefault("goal_history", [])
                             archive.append({ "goal": old_goal, "duration": goal_age, "ended_reason": "SET_GOAL action", "ended": time.time() })
                    else: error = "Missing or empty 'goal' parameter."
                elif action_type == "EXPLORE":
                    path_param = params.get('path') # Allow None initially
                    depth_param = params.get('depth', '1')
                    explore_path = path_param if path_param else os.getcwd()
                    # Validate path
                    if not os.path.isabs(explore_path): error = f"EXPLORE path must be absolute: '{explore_path}'"
                    elif '..' in explore_path.split(os.sep): error = f"Path traversal disallowed for EXPLORE: '{explore_path}'"
                    elif not os.path.isdir(explore_path): error = f"EXPLORE path is not a valid directory: '{explore_path}'"
                    else:
                         # Validate depth
                         try: depth = min(max(int(depth_param), 0), 2) # Clamp depth between 0 and 2
                         except ValueError: error = f"Invalid 'depth' parameter: '{depth_param}'. Must be integer."; depth = 0 # Default to 0 depth on error
                         # Proceed only if path and depth are valid
                         if not error:
                             logger.info(f"Exploring path '{explore_path}' to depth {depth}")
                             explore_results = []
                             try:
                                 base_level = explore_path.rstrip(os.sep).count(os.sep)
                                 for root, dirs, files in os.walk(explore_path, topdown=True, onerror=lambda e: logger.warning(f"os.walk error: {e}")):
                                     current_level = root.count(os.sep) - base_level
                                     if current_level > depth: dirs[:] = []; continue # Don't recurse further
                                     indent = '  ' * current_level
                                     explore_results.append(f"{indent}📁 {os.path.basename(root) or root}/") # Handle root dir case
                                     sub_indent = '  ' * (current_level + 1)
                                     files_to_show = files[:MAX_FILES_LISTED]
                                     for f in files_to_show: explore_results.append(f"{sub_indent}📄 {f}")
                                     if len(files) > MAX_FILES_LISTED: explore_results.append(f"{sub_indent}... ({len(files) - MAX_FILES_LISTED} more files)")
                                     dirs_to_show_count = 5 # Limit shown subdirs for brevity in results
                                     if len(dirs) > dirs_to_show_count:
                                          explore_results.append(f"{sub_indent}... ({len(dirs) - dirs_to_show_count} more directories)")
                                          dirs[:] = dirs[:dirs_to_show_count] # Limit recursion for next level
                                 result = f"Exploration results for '{explore_path}' (Depth {depth}):\n```\n" + "\n".join(explore_results) + "\n```"
                             except PermissionError: error = f"Permission denied exploring: {explore_path}"
                             except Exception as e_explore: error = f"Error during exploration of {explore_path}: {str(e_explore)}"
                else:
                     # This case should ideally not be reached if parsing worked and action is known
                     error = f"Attempted to execute unknown action type after parsing: {action_type}"
                     result = f"Default THINKING executed due to unknown action: {action_type}. Original content: {str(params.get('content', ''))[:100]}..."

            except Exception as e_exec: # Catch errors during action execution logic itself
                 error = f"Execution error ({action_type}): {str(e_exec)}"; logger.exception(f"Action execution error")
        elif action_type == "THINKING": # Handle the case where THINKING was defaulted due to parse error/no tag
             logger.info(f"Executing default THINKING action.")
             result = f"Thought process recorded (defaulted): {str(params.get('content', ''))[:100]}..."
             if parse_error: error = parse_error # Keep the parse error as the 'reason'
        else: # Should not happen if logic above is correct, but safety catch
             logger.error(f"Action execution skipped for {action_type} due to prior error: {error}")
             result = f"Action {action_type} skipped. Error: {error}"


        # --- Record action outcome ---
        outcome = "success" if not error else "failure"
        # Ensure there's *some* summary if execution happened but failed setting result
        result_summary_str = str(result)[:500] if result else None
        if outcome == "failure" and not result_summary_str:
             result_summary_str = f"Action failed with error: {str(error)[:100]}..." # Provide error summary if no result

        action_record = {
            "timestamp": time.time(),
            "type": action_type, # Log the action type that was *attempted* or defaulted to
            "params": params,
            "outcome": outcome,
            "result_summary": result_summary_str, # Use the potentially generated summary
            "error": str(error) if error else None
        }
        with self._blackboard_lock:
            actions = self._blackboard.setdefault("recent_actions", [])
            actions.append(action_record);
            max_recent_actions = 20
            if len(actions) > max_recent_actions:
                self._blackboard["recent_actions"] = actions[-max_recent_actions:]

        # --- Corrected UI Logging ---
        # Log to UI based on final outcome (error variable)
        if not error: # Success case
            if action_type == "THINKING" and not action_tag_found:
                 self._log_to_ui("info", f"Action 'THINKING' (defaulted) recorded.")
            elif result: # Log result only if it exists and action was successful
                 self._log_to_ui("info", f"Action '{action_type}' result: {str(result)[:300]}...")
            else: # Success outcome but no specific result string generated
                 self._log_to_ui("info", f"Action '{action_type}' completed successfully.")
        else: # Failure case (error is not None)
             self._log_to_ui("error", f"Action '{action_type}' failed: {error}")
        # --- End Corrected UI Logging ---

        logger.debug(f"--- Execute Action END (Outcome: {outcome}) ---")
        return action_record

    # --- END of _execute_action ---

    async def _enhanced_reflect(self):
        """Performs multi-layered reflection on the agent's state and actions."""
        # Check if loop is still running before proceeding
        if not self._is_running_flag.is_set() or not self._asyncio_loop or not self._asyncio_loop.is_running():
            logger.debug("Enhanced reflection skipped: Agent stopping.")
            return {} # Return empty dict or None

        logger.debug("--- Enhanced Reflection START ---")
        immediate_reflection, episodic_reflection, self_model_reflection = None, None, None
        error_log = []
        combined_reflection = {} # Initialize return value

        try:
            # Gather context for reflection safely
            with self._blackboard_lock:
                last_action_resp = self._blackboard.get('last_action_response', "")
                # Get a *copy* of recent history to avoid modifying blackboard data during iteration
                recent_history = list(self._blackboard.get('recent_history', []))[-6:] # Last 3 turns (user+assist)
                system_state = self._blackboard.get('latest_percepts', {}).get('system_state', {})
                # Get a *copy* of the last parsed action
                last_parsed_action = dict(self._blackboard.get('last_parsed_action', {"type": "UNKNOWN"}))
        except Exception as e_ctx:
            logger.exception("Error getting context for reflection")
            error_log.append(f"Context Error: {e_ctx}")
            # Construct partial reflection object even on context error
            combined_reflection = {"timestamp": time.time(), "reflection_errors": error_log }
            with self._blackboard_lock: self._blackboard['last_reflection'] = combined_reflection
            return combined_reflection # Return early

        # --- Layer 1: Immediate Reflection ---
        try:
            logger.debug("Reflection Layer 1: Immediate...")
            immediate_prompt = [
                {"role": "system", "content": f"Examine the agent's most recent raw output (containing reasoning and action tag) and the parsed action details. Evaluate the quality, relevance, and effectiveness of the chosen action ({last_parsed_action.get('type', '?')}) in relation to the context and goal. Be concise."},
                {"role": "user", "content": f"Last raw output:\n```\n{last_action_resp}\n```\nParsed Action: {last_parsed_action}\n\nYour concise evaluation:"}
            ]
            immediate_reflection, error = await self._call_ollama(immediate_prompt, temperature=0.3)
            if error:
                logger.error(f"Immediate reflection API error: {error}")
                error_log.append(f"Immediate: {error}")
            else:
                logger.info(f"Immediate reflection: '{str(immediate_reflection)[:100]}...'")
            self._log_to_ui("info", f"Reflection (Imm): {str(immediate_reflection)[:100] + '...' if immediate_reflection else f'[Error: {error}]'}")
        except Exception as e_imm:
            logger.exception("Error during immediate reflection")
            error_log.append(f"Immediate Error: {e_imm}")
            self._log_to_ui("error", f"Reflection (Imm) Error: {e_imm}")

        # --- Layer 2: Episodic Reflection ---
        try:
            logger.debug("Reflection Layer 2: Episodic...")
            history_text = "\n".join([f"{msg.get('role','?').capitalize()}: {str(msg.get('content','[empty]'))[:200]}..." for msg in recent_history])
            episodic_prompt = [
                {"role": "system", "content": "Reflect on the recent interaction flow (last ~3 turns). Identify any patterns, progress towards goals, shifts in topic, or emerging needs. Be concise."},
                {"role": "user", "content": f"Recent history snippet:\n```\n{history_text}\n```\n\nYour concise observations on the interaction flow:"}
            ]
            episodic_reflection, error = await self._call_ollama(episodic_prompt, temperature=0.4)
            if error:
                logger.error(f"Episodic reflection API error: {error}")
                error_log.append(f"Episodic: {error}")
            else:
                logger.info(f"Episodic reflection: '{str(episodic_reflection)[:100]}...'")
            self._log_to_ui("info", f"Reflection (Epi): {str(episodic_reflection)[:100] + '...' if episodic_reflection else f'[Error: {error}]'}")
        except Exception as e_epi:
            logger.exception("Error during episodic reflection")
            error_log.append(f"Episodic Error: {e_epi}")
            self._log_to_ui("error", f"Reflection (Epi) Error: {e_epi}")

        # --- Layer 3: Self Model Reflection ---
        try:
            logger.debug("Reflection Layer 3: Self Model...")
            # TODO: Include current self_model summary in prompt when implemented
            self_model_prompt = [
                {"role": "system", "content": "Reflect on the agent's own capabilities, limitations, thinking patterns observed in recent actions/reflections, and how these relate to its identity. Focus on updates or confirmations based on the latest cycle. Be concise."},
                # Provide context from immediate and episodic reflections if available
                {"role": "user", "content": f"Based on recent activity (including immediate reflection: '{str(immediate_reflection)[:100]}...' and episodic reflection: '{str(episodic_reflection)[:100]}...'), update the agent's self-model. Consider: 1) Confirmed/New Capabilities? 2) Observed Limitations/Mistakes? 3) Thinking Patterns? 4) Identity Traits?"}
            ]
            self_model_reflection, error = await self._call_ollama(self_model_prompt, temperature=0.5)
            if error:
                logger.error(f"Self-model reflection API error: {error}")
                error_log.append(f"SelfModel: {error}")
            else:
                logger.info(f"Self-model reflection: '{str(self_model_reflection)[:100]}...'")
            self._log_to_ui("info", f"Reflection (Self): {str(self_model_reflection)[:100] + '...' if self_model_reflection else f'[Error: {error}]'}")
        except Exception as e_self:
            logger.exception("Error during self-model reflection")
            error_log.append(f"SelfModel Error: {e_self}")
            self._log_to_ui("error", f"Reflection (Self) Error: {e_self}")

        # --- Combine and Store Reflection ---
        combined_reflection = {
            "timestamp": time.time(),
            "immediate": immediate_reflection,
            "episodic": episodic_reflection,
            "self_model": self_model_reflection, # Raw reflection string for now
            "system_state_at_reflection": system_state,
            "reflection_errors": error_log or None
        }
        with self._blackboard_lock:
            self._blackboard['last_reflection'] = combined_reflection
            logger.debug("Blackboard updated with structured reflection.")

        # --- Store Reflection in Memory ---
        try:
            reflection_json = json.dumps(combined_reflection, default=str)
            # Add metadata including any errors for context
            meta = {"type": "structured_reflection", "timestamp": time.time(), "has_errors": bool(error_log)}
            await self._add_document_to_memory(reflection_json, meta) # Uses corrected _add_document_to_memory
        except Exception as e_mem:
            logger.exception("Error storing structured reflection to memory")
            self._log_to_ui("error", f"Failed to store reflection in memory: {e_mem}")

        logger.debug("--- Enhanced Reflection END ---")
        return combined_reflection


    async def _evolve_goals(self):
        """Autonomously evolves goals based on experiences and reflections."""
        # Check if loop is still running before proceeding
        if not self._is_running_flag.is_set() or not self._asyncio_loop or not self._asyncio_loop.is_running():
            logger.debug("Goal evolution skipped: Agent stopping.")
            return None # Indicate no goal change

        logger.debug("--- Goal Evolution Check START ---")
        new_goal_generated = None
        current_goal = "[Goal Unavailable]" # Default
        try:
            # --- Gather Context ---
            with self._blackboard_lock:
                current_goal = self._blackboard.get("current_goal", "Observe and respond.")
                goal_set_time = self._blackboard.get("goal_set_time", time.time() - 3600) # Default old if missing
                recent_actions = list(self._blackboard.get("recent_actions", [])) # Get copy
                last_reflection = self._blackboard.get("last_reflection", {})
                goal_history = list(self._blackboard.get("goal_history", [])) # Get copy
                # Get valence
                emotional_valence = self._blackboard.get("emotional_valence", {"curiosity": 0.5, "satisfaction": 0.5})

            # --- Goal Evolution Logic ---
            goal_age = time.time() - goal_set_time
            # Calculate success rate from last N actions (e.g., last 10)
            last_n_actions = recent_actions[-10:]
            success_count = sum(1 for a in last_n_actions if a.get("outcome") == "success")
            total_considered = len(last_n_actions)
            # Avoid division by zero, default 50% if no actions considered yet
            success_rate = success_count / total_considered if total_considered > 0 else 0.5

            # --- Conditions for Considering a New Goal ---
            consider_new_goal = False
            reason = ""
            goal_stuck_threshold = 60 # seconds
            goal_achieved_threshold = 120 # seconds
            min_success_rate_stuck = 0.3
            min_success_rate_achieved = 0.7
            random_explore_chance = 0.05 # 5% chance each evaluation period

            if goal_age > goal_achieved_threshold and success_rate >= min_success_rate_achieved:
                consider_new_goal = True; reason = f"current goal likely achieved (Success Rate {success_rate:.2f} >= {min_success_rate_achieved} over {goal_age:.0f}s > {goal_achieved_threshold}s)"
            elif goal_age > goal_stuck_threshold and success_rate <= min_success_rate_stuck:
                consider_new_goal = True; reason = f"low success rate on current goal (Success Rate {success_rate:.2f} <= {min_success_rate_stuck} over {goal_age:.0f}s > {goal_stuck_threshold}s)"
            elif random.random() < random_explore_chance:
                 consider_new_goal = True; reason = f"random exploration ({(random_explore_chance*100):.0f}% chance)"

            # --- Decision ---
            if not consider_new_goal:
                 logger.debug(f"Goal Check: Keeping current goal '{current_goal}'. Age: {goal_age:.0f}s, Success Rate (last {total_considered}): {success_rate:.2f}")
                 logger.debug("--- Goal Evolution Check END (No Change) ---")
                 return current_goal # Keep current goal

            logger.info(f"Considering new goal. Reason: {reason}")
            self._log_to_ui("info", f"Agent is considering a new goal (Reason: {reason})...")

            # --- Prompt LLM for a New Goal ---
            reflection_summary = "[No reflection yet]"
            if isinstance(last_reflection, dict):
                 imm = last_reflection.get("immediate")
                 epi = last_reflection.get("episodic")
                 self_ref = last_reflection.get("self_model")
                 reflection_parts = [str(p)[:100]+"..." for p in [imm, epi, self_ref] if p]
                 if reflection_parts: reflection_summary = " | ".join(reflection_parts)

            # --- Pre-calculate complex parts for the prompt ---
            try:
                # Format recent actions types
                recent_actions_str = str([a.get('type') for a in last_n_actions[-3:]])

                # Format goal history list representation safely
                goal_history_tuples = []
                for g in goal_history[-2:]:
                    goal_name = g.get('goal', 'N/A')
                    duration_str = f"{g.get('duration', 0):.0f}s"
                    goal_history_tuples.append(f"('{goal_name}', '{duration_str}')") # Create string tuples
                goal_history_repr_str = "[" + ", ".join(goal_history_tuples) + "]"

            except Exception as e:
                 logger.error(f"Error formatting prompt context: {e}")
                 recent_actions_str = "[Error formatting actions]"
                 goal_history_repr_str = "[Error formatting history]"
                 # Valence should still be available from context gathering above

            # --- Construct the goal prompt (INCLUDING VALENCE) ---
            goal_prompt = [
                {"role": "system", "content": "You are an autonomous agent deciding on a new goal. Consider your previous goal, recent performance (success rate), recent actions, reflections, current drives (valence), and system state. Propose ONE specific, achievable, and potentially interesting new goal for yourself. Output ONLY the new goal description as a single line."},
                {"role": "user", "content":
                    f"Previous goal: '{current_goal}' (Age: {goal_age:.0f}s, Success Rate: {success_rate:.2f})\n"
                    f"Reason for change: {reason}\n"
                    f"Recent Actions (last 3 types): {recent_actions_str}\n" # Use pre-formatted string
                    f"Last Reflection Summary: {reflection_summary}\n"
                    f"Goal History (last 2): {goal_history_repr_str}\n" # Use pre-formatted string
                    # *** ADD VALENCE TO GOAL PROMPT ***
                    f"Current Drives: Curiosity={emotional_valence.get('curiosity', 0.5):.2f}, Satisfaction={emotional_valence.get('satisfaction', 0.5):.2f}\n\n"
                    f"Propose ONE new goal:"
                    # *** END ADD VALENCE ***
                }
            ]

            new_goal_response, error = await self._call_ollama(goal_prompt, temperature=0.8) # Higher temp for creativity

            if error or not new_goal_response:
                logger.error(f"New goal generation failed: {error or 'Empty response'}")
                self._log_to_ui("warn", f"Failed to generate new goal: {error or 'Empty response'}")
            else:
                # Process response: take first non-empty line, strip whitespace/quotes
                potential_goals = [line.strip().strip('"\'') for line in new_goal_response.strip().split('\n') if line.strip()]
                if potential_goals:
                     new_goal_generated = potential_goals[0]

                     if new_goal_generated and new_goal_generated != current_goal:
                          logger.info(f"LLM proposed new goal: '{new_goal_generated}'")
                          # Update blackboard
                          with self._blackboard_lock:
                              # Archive old goal
                              archive = self._blackboard.setdefault("goal_history", [])
                              archive.append({
                                  "goal": current_goal,
                                  "duration": goal_age,
                                  "success_rate": success_rate, # Record final success rate
                                  "ended_reason": reason,
                                  "ended": time.time()
                              })
                              # Set new goal
                              self._blackboard["current_goal"] = new_goal_generated
                              self._blackboard["goal_set_time"] = time.time() # Reset timer

                          self._log_to_ui("info", f"Agent autonomously set new goal: {new_goal_generated}")
                          # Store evolution event in memory
                          await self._add_document_to_memory(
                              f"Goal Change: From '{current_goal}' to '{new_goal_generated}'. Reason: {reason}. LLM Suggestion: {new_goal_response}",
                              {"type": "goal_evolution", "timestamp": time.time()}
                          ) # Uses corrected _add_document_to_memory
                     else:
                          logger.info("LLM did not propose a valid *new* goal. Keeping current.")
                          new_goal_generated = None # Ensure no change if invalid/same
                else:
                    logger.info("LLM response for new goal was empty after processing.")
                    new_goal_generated = None

        except Exception as e_goal:
             logger.exception("Error during goal evolution process:")
             self._log_to_ui("error", f"Goal evolution error: {e_goal}")

        logger.debug("--- Goal Evolution Check END ---")
        # Return the goal that is now active (either the new one or the unchanged one)
        with self._blackboard_lock:
            return self._blackboard.get("current_goal", "[Goal Unavailable]")


    async def _call_ollama(self, messages: List[Dict[str, str]], temperature: float) -> Tuple[str | None, str | None]:
        """Helper: Calls Ollama API with timeout, parses response correctly."""
        # Check if loop is still running before proceeding
        if not self._is_running_flag.is_set() or not self._asyncio_loop or not self._asyncio_loop.is_running():
            logger.warning("Ollama call skipped: Agent stopping or loop stopped.")
            return None, "Agent stopped"

        logger.debug(f"Calling Ollama '{self.selected_ollama_model}' ({len(messages)} msgs, T={temperature}, Timeout={OLLAMA_CALL_TIMEOUT_S}s)")
        if not OLLAMA_AVAILABLE:
            return None, "Ollama library missing."

        # Define the synchronous call as a lambda
        sync_call = lambda: ollama.chat(
            model=self.selected_ollama_model,
            messages=messages,
            stream=False,
            options={'temperature': temperature}
        )

        try: # Start of the main try block
            response = await asyncio.wait_for(
                self._asyncio_loop.run_in_executor(None, sync_call),
                timeout=OLLAMA_CALL_TIMEOUT_S
            )

            # Parse response - Check attributes safely
            # Handle potential None response from ollama library itself
            if response is None:
                 logger.error("Ollama API returned None response object.")
                 return None, "Ollama API Error: Received None response"

            # Check structure for successful response (adjust based on actual library)
            # Common structure seems to be dict with response['message']['content']
            if isinstance(response, dict) and 'message' in response and isinstance(response['message'], dict) and 'content' in response['message']:
                content = response['message']['content']
                if isinstance(content, str):
                    content = content.strip()
                    logger.debug(f"Ollama response OK (len {len(content)}).")
                    return content, None # Success case
                else:
                    logger.warning(f"Ollama response 'content' is not a string. Type: {type(content)}")
                    return None, "Unexpected response content type"
            # Add handling for alternative success structures if observed (e.g., older versions?)
            # Example for a Pydantic-like object (unlikely now but safe)
            elif hasattr(response, 'message') and hasattr(response.message, 'content') and isinstance(response.message.content, str):
                content = response.message.content.strip()
                logger.debug(f"Ollama response OK (len {len(content)}, pydantic-like).")
                return content, None # Success case (Pydantic-like)
            else:
                # Log unexpected structure if parsing fails
                logger.warning(f"Unexpected Ollama response structure. Type: {type(response)}, Content: {str(response)[:200]}...")
                # Try to extract content heuristically if possible
                try:
                     maybe_content = str(response) # Last resort
                     logger.warning(f"Attempting fallback content extraction: {maybe_content[:100]}...")
                     return maybe_content, None # Return something, but log warning
                except Exception:
                     return None, "Unexpected response structure (unparseable)"


        except asyncio.TimeoutError: # Correctly indented except block
            logger.error(f"Ollama API call timed out ({OLLAMA_CALL_TIMEOUT_S}s).")
            return None, f"Ollama API Error: Timeout"
        except RuntimeError as e: # Catch the specific executor shutdown error
             if "cannot schedule new futures after shutdown" in str(e):
                 logger.warning(f"Ollama call aborted: Executor shutdown during stop sequence.")
                 return None, "Executor shutdown (stopping)"
             else: # Re-raise other RuntimeErrors
                 logger.exception("Ollama API call failed (RuntimeError):")
                 return None, f"API Error (Runtime): {str(e)}"
        except Exception as e: # Correctly indented except block for other errors
            logger.exception("Ollama API call failed:")
            err=str(e); err_msg=f"API Error: {err}"
            # Simplify common errors
            if "connection refused" in err.lower(): err_msg="API Error: Connection refused?"
            elif "timeout" in err.lower() or "retries" in err.lower(): err_msg="API Error: Connection timeout?"
            elif "failed to connect" in err.lower(): err_msg="API Error: Connection Failed"
            # Check model name within the error message for 'not found'
            elif "not found" in err.lower() and f"model '{self.selected_ollama_model}'" in err.lower():
                 err_msg=f"API Error: Model '{self.selected_ollama_model}' not found?"
            elif "not found" in err.lower(): # Generic not found
                 err_msg="API Error: Resource not found (check model name?)"

            return None, err_msg

    # --- History / Memory Utilities ---
    # (Indentation corrected previously)
    def _add_to_history(self, assistant_turn: Dict[str, str], user_turn_content: Optional[str] = None):
        """Adds turns to the recent history, ensuring it doesn't exceed the limit."""
        with self._blackboard_lock:
            hist = self._blackboard.setdefault("recent_history", [])
            if user_turn_content:
                hist.append({"role": "user", "content": user_turn_content})
                logger.debug(f"Added user turn to history: '{user_turn_content[:50]}...'")
            if assistant_turn and isinstance(assistant_turn, dict) and assistant_turn.get('content'):
                 hist.append(assistant_turn)
                 logger.debug(f"Added assistant turn to history: '{assistant_turn.get('content', '')[:50]}...'")
            else:
                 logger.warning(f"Attempted to add invalid assistant turn to history: {assistant_turn}")

            # Prune history based on number of ITEMS (turns), not pairs
            max_items = MAX_RECENT_TURNS_IN_PROMPT * 2
            if len(hist) > max_items:
                self._blackboard["recent_history"] = hist[-max_items:]
                logger.debug(f"History pruned to last {max_items} items.")

    async def _add_document_to_memory(self, text: str, metadata: Dict[str, Any]):
        """Adds a document to the ChromaDB memory if available and agent is running."""
        if not (CHROMADB_AVAILABLE and self._memory_collection and text): return
        # Prevent adding during shutdown
        if not self._is_running_flag.is_set() or not self._asyncio_loop or not self._asyncio_loop.is_running():
             logger.warning("Add document to memory skipped: Agent not running or loop stopped.")
             return
        # Ensure metadata is suitable for ChromaDB (basic types)
        safe_metadata = {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}
        if not safe_metadata.get("timestamp"): safe_metadata["timestamp"] = time.time() # Ensure timestamp exists
        if not safe_metadata.get("type"): safe_metadata["type"] = "generic_doc" # Ensure type exists

        doc_id = f"{safe_metadata['type']}_{int(safe_metadata['timestamp'] * 1000)}_{random.randint(100,999)}" # More unique ID

        try:
            logger.debug(f"Adding doc to memory: {doc_id} (Type: {safe_metadata['type']}, len {len(text)})")
            # Use a lambda to ensure the collection object is accessed correctly in the executor thread
            add_func = lambda: self._memory_collection.add(documents=[text], metadatas=[safe_metadata], ids=[doc_id])
            await asyncio.get_running_loop().run_in_executor( None, add_func)
            logger.debug(f"Doc {doc_id} added to memory.")
        except RuntimeError as e:
             if "cannot schedule new futures after shutdown" in str(e):
                 logger.warning(f"Add document '{doc_id}' aborted: Executor shutdown during stop.")
             else:
                  logger.exception(f"Failed to add doc ID {doc_id} to ChromaDB (RuntimeError)")
                  self._log_to_ui("error", f"Memory Add Failed: {e}")
        except Exception as e:
            # Catch specific ChromaDB errors if possible, otherwise generic
            logger.exception(f"Failed to add doc ID {doc_id} to ChromaDB")
            self._log_to_ui("error", f"Memory Add Failed: {str(e)[:100]}...") # Avoid overly long UI errors


    async def _retrieve_from_memory(self, query_text: str, n_results: int = 3) -> List[str]:
        """Retrieves documents from ChromaDB memory if available and agent is running."""
        if not (CHROMADB_AVAILABLE and self._memory_collection and query_text): return []
        # Prevent querying during shutdown
        if not self._is_running_flag.is_set() or not self._asyncio_loop or not self._asyncio_loop.is_running():
             logger.warning("Retrieve from memory skipped: Agent not running or loop stopped.")
             return []
        try:
            logger.debug(f"Querying memory (n={n_results}) for: '{query_text[:50]}...'")
            # Use a lambda for the query function
            # Include metadata in retrieval for potential future use/filtering
            query_func = lambda: self._memory_collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=['documents', 'metadatas'] # Retrieve metadata too
            )
            results = await asyncio.get_running_loop().run_in_executor( None, query_func)

            # Safely extract documents
            retrieved_docs = []
            if results and results.get('documents') and isinstance(results['documents'], list) and results['documents']:
                 # results['documents'] is often a list containing one list of actual docs
                 doc_list = results['documents'][0]
                 if isinstance(doc_list, list):
                      retrieved_docs = [str(doc) if doc is not None else "[None]" for doc in doc_list]

            logger.info(f"Retrieved {len(retrieved_docs)} docs from memory.")
            # TODO: Potentially use retrieved metadata if needed
            return retrieved_docs
        except RuntimeError as e:
            if "cannot schedule new futures after shutdown" in str(e):
                 logger.warning(f"Memory query aborted: Executor shutdown during stop.")
                 return []
            else:
                 logger.exception("Failed to query ChromaDB (RuntimeError)")
                 self._log_to_ui("error", f"Memory Query Failed: {e}")
                 return []
        except Exception as e:
            # Catch specific ChromaDB errors if possible
            logger.exception("Failed to query ChromaDB")
            self._log_to_ui("error", f"Memory Query Failed: {str(e)[:100]}...")
            return []

    async def _evaluate_intrinsic_drives(self):
        """Updates internal motivation metrics based on experiences."""
        logger.debug("--- Evaluate Intrinsic Drives START ---")
        new_valence = {}
        try:
            with self._blackboard_lock:
                # Get current valence, default if not present
                valence = self._blackboard.get("emotional_valence", {"curiosity": 0.5, "satisfaction": 0.5})
                # Get copy of recent actions (e.g., last 5-10 for drive calculation)
                recent_actions = list(self._blackboard.get("recent_actions", []))[-10:]
                # We might use reflection quality later, but keep it simple for now
                # reflections = self._blackboard.get("last_reflection", {})

            if not recent_actions: # No actions yet to evaluate
                 logger.debug("No recent actions to evaluate drives.")
                 return valence # Return current valence

            # --- Calculate Drive Modifiers ---
            # 1. Curiosity: Increases with successful non-THINKING actions (exploration/discovery)
            #    Decreases slightly with high success rate (less need to explore?) or maybe just decays
            discovery_actions = ["READ_FILE", "LIST_FILES", "EXPLORE", "QUERY_MEMORY"]
            successful_discoveries = sum(1 for a in recent_actions
                                         if a.get("type") in discovery_actions and a.get("outcome") == "success")
            failed_discoveries = sum(1 for a in recent_actions
                                     if a.get("type") in discovery_actions and a.get("outcome") == "failure")

            # 2. Satisfaction: Increases with overall success rate, decreases over time/with failures
            success_count = sum(1 for a in recent_actions if a.get("outcome") == "success")
            failure_count = len(recent_actions) - success_count
            success_rate = success_count / len(recent_actions) if recent_actions else 0.5

            # --- Update Valence ---
            # Simple model: move towards 0.5 (equilibrium) slightly each time, then apply modifiers
            curiosity_decay = 0.01
            satisfaction_decay = 0.02
            curiosity_gain_per_discovery = 0.05
            curiosity_loss_per_failed_discovery = 0.03 # Penalty for failed exploration
            satisfaction_gain_rate_multiplier = 0.08 # Gain based on success rate
            satisfaction_loss_per_failure = 0.04 # Penalty for failures

            # Apply decay/homeostasis
            current_curiosity = valence.get("curiosity", 0.5)
            current_satisfaction = valence.get("satisfaction", 0.5)
            new_curiosity = current_curiosity - curiosity_decay * (current_curiosity - 0.5)
            new_satisfaction = current_satisfaction - satisfaction_decay * (current_satisfaction - 0.5)

            # Apply modifiers
            new_curiosity += (curiosity_gain_per_discovery * successful_discoveries)
            new_curiosity -= (curiosity_loss_per_failed_discovery * failed_discoveries)
            new_satisfaction += satisfaction_gain_rate_multiplier * (success_rate - 0.5) # Gain above 50% success
            new_satisfaction -= (satisfaction_loss_per_failure * failure_count)

            # Clamp values between 0.1 and 1.0
            new_valence = {
                "curiosity": min(1.0, max(0.1, new_curiosity)),
                "satisfaction": min(1.0, max(0.1, new_satisfaction))
            }

            # Update blackboard
            with self._blackboard_lock:
                self._blackboard["emotional_valence"] = new_valence

            self._log_to_ui("info", f"Drives updated: Curiosity={new_valence['curiosity']:.2f}, Satisfaction={new_valence['satisfaction']:.2f}")
            logger.debug(f"Valence updated: {new_valence}")

        except Exception as e:
            logger.exception("Error during intrinsic drive evaluation:")
            # Return previous valence on error to avoid breaking state
            with self._blackboard_lock:
                 new_valence = self._blackboard.get("emotional_valence", {"curiosity": 0.5, "satisfaction": 0.5})

        logger.debug("--- Evaluate Intrinsic Drives END ---")
        return new_valence

# --- START OF FILE agent_logic.py ---