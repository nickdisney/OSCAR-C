# --- START OF CORRECTED dynamic_self_model.py ---

import asyncio
import logging
import time
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from collections import deque, Counter # Added Counter for _perform_reflection
from dataclasses import dataclass # Not used directly, but often useful with models

# --- Use standard relative imports ---
try:
    from ..protocols import CognitiveComponent
    # Import other types needed if used (e.g., PhenomenalState)
    from ..models.datatypes import PhenomenalState
except ImportError:
    # Fallback for different execution context (e.g., combined script)
    logging.warning("DynamicSelfModel: Relative imports failed, relying on globally defined types.")
    if 'CognitiveComponent' not in globals(): raise ImportError("CognitiveComponent not found via relative import or globally")
    if 'PhenomenalState' not in globals(): logging.warning("PhenomenalState class not found globally.") # Allow if not strictly needed
    CognitiveComponent = globals().get('CognitiveComponent')
    PhenomenalState = globals().get('PhenomenalState') # Might be None


logger_dynamic_self_model = logging.getLogger(__name__) # Use standard module logger name

# Default structure (remains the same)
DEFAULT_SELF_MODEL = {
    "version": 0, 
    "identity_traits": { # Initialize with defaults
        "curiosity": 0.5, "caution": 0.5, 
        "persistence": 0.5, "adaptability": 0.5
    }, 
    "capabilities": {}, "limitations": {},
    "knowledge_meta": {"validated_paths": {}, "invalid_paths": [], "learned_concepts": {}},
    "internal_state_awareness": {"last_valence": 0.0, "last_intensity": 0.0, "recent_error_types": []},
    "learning_rate": 0.1,
    "learning_rate_meta": { # Add this structure
        "capabilities": {
            "fast_learner": [], # List of action_keys
            "slow_learner": []  # List of action_keys
        }
    }
}
# Default reflection interval
DEFAULT_REFLECTION_INTERVAL_CYCLES = 20

# --- Inherit correctly from CognitiveComponent ---
class DynamicSelfModel(CognitiveComponent):
    """
    Maintains and updates the agent's model of its own capabilities,
    limitations, identity, and internal state awareness. Learns from action outcomes.
    """
    def __init__(self):
        self._controller: Optional[Any] = None
        self._config: Dict[str, Any] = {} # Component-specific config
        self.self_model: Dict[str, Any] = json.loads(json.dumps(DEFAULT_SELF_MODEL))
        self.last_update_time: Optional[float] = None
        self._model_path: Optional[Path] = None # Path for saving/loading
        self._max_invalid_paths: int = 50
        
        # --- ADDED FOR REFLECTION ---
        self.learning_events: Deque[Dict[str, Any]] = deque(maxlen=100) # Maxlen can be configured
        self.reflection_interval: int = DEFAULT_REFLECTION_INTERVAL_CYCLES
        self.cycles_since_reflection: int = 0
        # --- END ADDED FOR REFLECTION ---


    async def initialize(self, config: Dict[str, Any], controller: Any) -> bool:
        self._controller = controller
        dsm_config = config.get("dynamic_self_model", {}) # Component-specific config
        self._config = dsm_config
        
        # --- Path Configuration using agent_root_path ---
        path_str_from_config = None
        if controller and hasattr(controller, 'agent_root_path'):
            agent_root = controller.agent_root_path
            agent_data_paths_config = config.get("agent_data_paths", {})
            # Use 'self_model_path' from [agent_data_paths] for both load and save
            path_str_from_config = agent_data_paths_config.get("self_model_path")

            if path_str_from_config:
                # Construct the absolute path
                # If path_str_from_config happens to be absolute, Path() handles it.
                self._model_path = (Path(agent_root) / path_str_from_config).resolve()
                logger_dynamic_self_model.info(f"DynamicSelfModel path for load/save set to: {self._model_path}")
                try:
                    # Ensure parent dir exists for potential saving later
                    self._model_path.parent.mkdir(parents=True, exist_ok=True)
                    # Attempt to load existing model
                    if self._model_path.exists():
                        await self._load_model() # _load_model will use self._model_path
                    else:
                        logger_dynamic_self_model.info(f"Self-model file not found at {self._model_path}. Starting with default model.")
                except OSError as e:
                    logger_dynamic_self_model.error(f"Cannot create/access directory for self-model at {self._model_path.parent}: {e}. Load/save might fail.")
                except Exception as e:
                    logger_dynamic_self_model.error(f"Error setting/loading self-model path {self._model_path}: {e}.")
            else:
                logger_dynamic_self_model.info("self_model_path not specified in [agent_data_paths]. Self-model load/save disabled.")
                self._model_path = None 
        else:
            logger_dynamic_self_model.error("DynamicSelfModel: Controller or agent_root_path not available. Cannot determine model path. Load/save disabled.")
            self._model_path = None
        # --- End Path Configuration ---

        # Ensure default keys exist after potential loading
        self.self_model.setdefault("identity_traits", DEFAULT_SELF_MODEL["identity_traits"].copy())
        self.self_model.setdefault("capabilities", {})
        self.self_model.setdefault("limitations", {})
        km = self.self_model.setdefault("knowledge_meta", {})
        km.setdefault("validated_paths", {}); km.setdefault("invalid_paths", []); km.setdefault("learned_concepts", {})
        isa = self.self_model.setdefault("internal_state_awareness", {})
        isa.setdefault("recent_error_types", [])
        self.self_model.setdefault("learning_rate_meta", DEFAULT_SELF_MODEL["learning_rate_meta"].copy())

        self._max_invalid_paths = dsm_config.get("max_invalid_paths", 50)
        self.self_model["learning_rate"] = dsm_config.get("learning_rate", self.self_model.get("learning_rate", 0.1))
        
        # --- LOAD REFLECTION CONFIG ---
        learning_events_maxlen = dsm_config.get("learning_events_history_size", 100)
        self.learning_events = deque(maxlen=learning_events_maxlen)
        self.reflection_interval = dsm_config.get("reflection_interval_cycles", DEFAULT_REFLECTION_INTERVAL_CYCLES)
        # --- END LOAD REFLECTION CONFIG ---
        
        logger_dynamic_self_model.info(
            f"DynamicSelfModel initialized. Version: {self.self_model.get('version')}, "
            f"LR: {self.self_model.get('learning_rate')}, ReflectInterval: {self.reflection_interval} cycles, "
            f"LearnHistSize: {learning_events_maxlen}"
        )
        return True

    async def _load_model(self):
        """Loads self model from self._model_path."""
        if not (self._model_path and self._model_path.exists()):
            logger_dynamic_self_model.debug(f"Self-model file not found at {self._model_path} or path not set. Using current/default model.")
            return

        logger_dynamic_self_model.info(f"Attempting to load self-model from {self._model_path}")
        try:
            with open(self._model_path, 'r') as f:
                loaded_model = json.load(f)
            
            if isinstance(loaded_model, dict) and "version" in loaded_model:
                default_copy = json.loads(json.dumps(DEFAULT_SELF_MODEL)) 
                default_copy.update(loaded_model)
                self.self_model = default_copy

                self.self_model.setdefault("identity_traits", DEFAULT_SELF_MODEL["identity_traits"].copy())
                self.self_model.setdefault("capabilities", {})
                self.self_model.setdefault("limitations", {})
                km = self.self_model.setdefault("knowledge_meta", {})
                km.setdefault("validated_paths", {}); km.setdefault("invalid_paths", []); km.setdefault("learned_concepts", {})
                isa = self.self_model.setdefault("internal_state_awareness", {})
                isa.setdefault("recent_error_types", [])
                self.self_model.setdefault("learning_rate_meta", DEFAULT_SELF_MODEL["learning_rate_meta"].copy())


                logger_dynamic_self_model.info(f"Successfully loaded self-model v{self.self_model.get('version')} from {self._model_path}")
            else:
                logger_dynamic_self_model.warning(f"Loaded file {self._model_path} is invalid (not dict or no version). Using current/default model.")
        except Exception as e:
            logger_dynamic_self_model.exception(f"Error loading self-model from {self._model_path}: {e}. Using current/default model.")

    def _get_parameter_specific_suffix(self, 
                                     action_type: Optional[str], 
                                     action_params: Dict[str, Any],
                                     action_result_data: Optional[Any], 
                                     action_error: Optional[str] 
                                     ) -> str:
        param_suffix = ""
        if not action_type: # Return early if no action_type
            return ""

        # Parameter-based suffixes (mostly from params or result_data)
        if action_type in ['READ_FILE', 'WRITE_FILE']:
            # This part is fine, relies on action_result_data primarily
            file_size_category = None
            if isinstance(action_result_data, dict):
                file_size_category = action_result_data.get("file_size_category") 
                if "size_bytes" in action_result_data: 
                    try:
                        size_bytes = int(action_result_data["size_bytes"])
                        if size_bytes > 1_000_000: file_size_category = "large"
                        elif size_bytes > 100_000: file_size_category = "medium"
                        # "small" doesn't add a suffix by this logic, which is fine
                    except (ValueError, TypeError):
                        pass
            
            if file_size_category == "large":
                param_suffix = ":large_file"
            elif file_size_category == "medium":
                param_suffix = ":medium_file"

        elif action_type == 'CALL_LLM':
            prompt = action_params.get("prompt", "") if action_params else "" # Ensure action_params is not None
            if isinstance(prompt, str):
                if len(prompt) > 500: 
                    param_suffix = ":complex_prompt"
                elif len(prompt) < 50 and prompt: # Ensure prompt is not empty for "short"
                    param_suffix = ":short_prompt"
        
        # Error-based suffixes (can append to parameter-based suffixes)
        # Make timeout_error specific to CALL_LLM
        if action_type == 'CALL_LLM' and action_error and "timeout" in action_error.lower():
            if param_suffix: # e.g., :complex_prompt_timeout_error
                param_suffix += "_timeout_error" 
            else: # e.g., :timeout_error
                param_suffix = ":timeout_error"
        
        return param_suffix

    async def _update_capability(self, action_key_for_update: Optional[str], outcome: str, params: Dict[str, Any]):
        """Update capability scores based on action results, considering learning rate meta."""
        if not action_key_for_update: # Added safety check
            return False
            
        changes_made = False
        
        # --- Get base learning rate ---
        base_learning_rate = self.self_model.get("learning_rate", 0.1)
        effective_learning_rate = base_learning_rate

        # --- Adjust learning rate based on meta-parameters ---
        lr_meta_caps = self.self_model.get("learning_rate_meta", {}).get("capabilities", {})
        fast_learner_keys = lr_meta_caps.get("fast_learner", [])
        slow_learner_keys = lr_meta_caps.get("slow_learner", [])

        # Define multipliers
        fast_learner_multiplier = 1.5  # Example: learn 50% faster
        slow_learner_multiplier = 0.5  # Example: learn 50% slower

        if action_key_for_update in fast_learner_keys:
            effective_learning_rate *= fast_learner_multiplier
            logger_dynamic_self_model.debug(
                f"Applying FAST learner rate for '{action_key_for_update}'. Base: {base_learning_rate:.3f}, Effective: {effective_learning_rate:.3f}"
            )
        elif action_key_for_update in slow_learner_keys:
            effective_learning_rate *= slow_learner_multiplier
            logger_dynamic_self_model.debug(
                f"Applying SLOW learner rate for '{action_key_for_update}'. Base: {base_learning_rate:.3f}, Effective: {effective_learning_rate:.3f}"
            )
        
        # Ensure learning rate is within a sensible bound, e.g., [0.01, 0.5] after multiplication
        effective_learning_rate = max(0.01, min(0.5, effective_learning_rate))
        # --- End Learning Rate Adjustment ---

        capabilities = self.self_model.setdefault("capabilities", {})
        limitations = self.self_model.setdefault("limitations", {})

        if outcome == "success":
            old_cap_conf = capabilities.get(action_key_for_update, 0.0)
            # Use effective_learning_rate
            new_cap_conf = min(1.0, old_cap_conf + effective_learning_rate * (1.0 - old_cap_conf))
            if abs(new_cap_conf - old_cap_conf) > 1e-9:
                capabilities[action_key_for_update] = new_cap_conf
                changes_made = True
                logger_dynamic_self_model.debug(f"Capability+ '{action_key_for_update}': {new_cap_conf:.3f} (LR: {effective_learning_rate:.3f})")

            if action_key_for_update in limitations:
                old_lim_conf = limitations[action_key_for_update]
                # Use effective_learning_rate for reducing limitation confidence as well
                new_lim_conf = max(0.0, old_lim_conf - effective_learning_rate * old_lim_conf * 0.5) 
                if abs(new_lim_conf - old_lim_conf) > 1e-9:
                    limitations[action_key_for_update] = new_lim_conf
                    changes_made = True
                    logger_dynamic_self_model.debug(f"Limitation- '{action_key_for_update}': {new_lim_conf:.3f} (LR: {effective_learning_rate:.3f})")
                    if new_lim_conf < 0.05:
                        try: del limitations[action_key_for_update]; logger_dynamic_self_model.debug(f"Removed low-confidence limitation '{action_key_for_update}'")
                        except KeyError: pass
        
        elif outcome == "failure":
            if action_key_for_update in capabilities: 
                old_cap_conf = capabilities[action_key_for_update]
                # Use effective_learning_rate
                new_cap_conf = max(0.0, old_cap_conf - effective_learning_rate * old_cap_conf * 0.75)
                if abs(new_cap_conf - old_cap_conf) > 1e-9:
                    capabilities[action_key_for_update] = new_cap_conf
                    changes_made = True
                    logger_dynamic_self_model.debug(f"Capability- '{action_key_for_update}': {new_cap_conf:.3f} (LR: {effective_learning_rate:.3f})")
                    if new_cap_conf < 0.05:
                        try: del capabilities[action_key_for_update]; logger_dynamic_self_model.debug(f"Removed low-confidence capability '{action_key_for_update}'")
                        except KeyError: pass

            old_lim_conf = limitations.get(action_key_for_update, 0.0)
            # Use effective_learning_rate
            new_lim_conf = min(1.0, old_lim_conf + effective_learning_rate * (1.0 - old_lim_conf))
            if abs(new_lim_conf - old_lim_conf) > 1e-9:
                limitations[action_key_for_update] = new_lim_conf
                changes_made = True
                logger_dynamic_self_model.debug(f"Limitation+ '{action_key_for_update}': {new_lim_conf:.3f} (LR: {effective_learning_rate:.3f})")
        
        return changes_made

    async def update_self_model(self, 
                                last_action_type: Optional[str], 
                                action_outcome: str, 
                                action_params: Dict[str, Any], 
                                action_error: Optional[str], 
                                action_result_data: Optional[Any],
                                current_phenomenal_state: Optional['PhenomenalState']
                                ):
        _PhenomenalState = globals().get('PhenomenalState')
        logger_dynamic_self_model.debug(f"Updating self-model: Action={last_action_type} ({action_outcome})")
        initial_update_made_flag = False 
        # learning_rate = self.self_model.get("learning_rate", 0.1) # Base LR is now fetched in _update_capability
        action_key_general = f"action:{last_action_type}" if last_action_type else None
        current_time = time.time()

        param_suffix = self._get_parameter_specific_suffix(
            last_action_type, action_params, action_result_data, action_error
        )
        action_key_specific = f"{action_key_general}{param_suffix}" if action_key_general and param_suffix else None

        if action_key_general:
            keys_to_update = [action_key_general]
            if action_key_specific and action_key_specific != action_key_general:
                keys_to_update.append(action_key_specific)

            for current_action_key_for_update_loop in keys_to_update:
                # Call the refactored _update_capability method
                cap_change = await self._update_capability(current_action_key_for_update_loop, action_outcome, action_params)
                if cap_change:
                    initial_update_made_flag = True
        
        path_update_made_flag = False
        path_involved_str: Optional[str] = None
        is_path_action = last_action_type in ["LIST_FILES", "READ_FILE", "EXPLORE", "WRITE_FILE"]
        if is_path_action:
            if isinstance(action_result_data, dict) and "path" in action_result_data and isinstance(action_result_data["path"], str):
                path_involved_str = action_result_data["path"]
            elif isinstance(action_params, dict) and "path" in action_params and isinstance(action_params["path"], str):
                path_involved_str = action_params["path"]
                if self._controller and hasattr(self._controller, 'agent_root_path'):
                    try:
                        param_path_obj = Path(action_params["path"])
                        if not param_path_obj.is_absolute():
                            path_involved_str = str((self._controller.agent_root_path / param_path_obj).resolve(strict=False))
                        else: 
                            path_involved_str = str(param_path_obj.resolve(strict=False))
                    except Exception as e_resolve_dsm:
                        logger_dynamic_self_model.warning(f"Could not resolve param path '{action_params['path']}' for DSM: {e_resolve_dsm}")
                        path_involved_str = action_params["path"] 
                else:
                     logger_dynamic_self_model.debug(f"Path for DSM update (raw from params, no controller/root_path): {path_involved_str}")
        if path_involved_str:
            path_key = path_involved_str 
            knowledge_meta = self.self_model.setdefault("knowledge_meta", {});
            validated_paths = knowledge_meta.setdefault("validated_paths", {});
            invalid_paths = knowledge_meta.setdefault("invalid_paths", [])
            if action_outcome == "success":
                if path_key not in validated_paths or validated_paths[path_key] != current_time : 
                    validated_paths[path_key] = current_time; path_update_made_flag = True; logger_dynamic_self_model.info(f"SelfModel: Added/updated validated path '{path_key}'")
                if path_key in invalid_paths:
                    try: invalid_paths.remove(path_key); path_update_made_flag = True; logger_dynamic_self_model.info(f"SelfModel: Removed '{path_key}' from invalid paths.")
                    except ValueError: pass 
            elif action_outcome == "failure":
                error_lower = (action_error or "").lower()
                is_invalid_path_error = any(term in error_lower for term in ["not exist", "not found", "no such file", "not a directory", "invalid argument", "permission denied", "path does not exist", "path not exist", "not a file", "is a directory", "securityerror"])
                if is_invalid_path_error:
                    if path_key not in invalid_paths:
                         invalid_paths.append(path_key); path_update_made_flag = True
                         if len(invalid_paths) > self._max_invalid_paths: knowledge_meta["invalid_paths"] = invalid_paths[-self._max_invalid_paths:]
                         logger_dynamic_self_model.info(f"SelfModel: Added invalid path '{path_key}'")
                    if path_key in validated_paths:
                        try: del validated_paths[path_key]; path_update_made_flag = True; logger_dynamic_self_model.info(f"SelfModel: Removed '{path_key}' from validated paths.")
                        except KeyError: pass
        if path_update_made_flag: initial_update_made_flag = True

        internal_state_update_made_flag = False
        internal_state = self.self_model.setdefault("internal_state_awareness", {});
        if _PhenomenalState and current_phenomenal_state and isinstance(current_phenomenal_state, _PhenomenalState):
            if internal_state.get("last_valence") != current_phenomenal_state.valence: internal_state["last_valence"] = current_phenomenal_state.valence; internal_state_update_made_flag = True
            if internal_state.get("last_intensity") != current_phenomenal_state.intensity: internal_state["last_intensity"] = current_phenomenal_state.intensity; internal_state_update_made_flag = True
        elif isinstance(current_phenomenal_state, dict): 
             if "valence" in current_phenomenal_state and internal_state.get("last_valence") != current_phenomenal_state["valence"]: internal_state["last_valence"] = current_phenomenal_state["valence"]; internal_state_update_made_flag = True
             if "intensity" in current_phenomenal_state and internal_state.get("last_intensity") != current_phenomenal_state["intensity"]: internal_state["last_intensity"] = current_phenomenal_state["intensity"]; internal_state_update_made_flag = True
        if action_outcome == "failure" and action_error:
            try: error_type = str(action_error).split(":")[0].strip() 
            except Exception: error_type = "UnknownError"
            recent_errors = internal_state.setdefault("recent_error_types", [])
            if not recent_errors or recent_errors[-1] != error_type:
                recent_errors.append(error_type);
                if len(recent_errors) > 10: internal_state["recent_error_types"] = recent_errors[-10:]
                internal_state_update_made_flag = True
        if internal_state_update_made_flag: initial_update_made_flag = True
        
        # --- Record Learning Event ---
        if last_action_type: 
            self.learning_events.append({
                "timestamp": current_time,
                "action_type": last_action_type, 
                "action_key_specific": action_key_specific if action_key_specific else action_key_general,
                "outcome": action_outcome,
                "params": action_params, 
                "error": action_error
            })
            self.cycles_since_reflection += 1
        
        # --- Periodic Deep Reflection ---
        reflection_performed_flag = False
        if self.cycles_since_reflection >= self.reflection_interval and self.reflection_interval > 0 : # Check reflection_interval > 0
            logger_dynamic_self_model.info(f"Reflection interval ({self.reflection_interval} cycles) reached. Performing deep reflection.")
            await self._perform_reflection()
            self.cycles_since_reflection = 0
            reflection_performed_flag = True 
        
        # --- Final Version/Timestamp Update ---
        if initial_update_made_flag or reflection_performed_flag: 
            self.self_model["version"] = self.self_model.get("version", 0) + 1
            self.last_update_time = current_time
            self.self_model["last_update"] = self.last_update_time
            if initial_update_made_flag and not reflection_performed_flag: # Log direct change only if no reflection also happened
                 logger_dynamic_self_model.info(f"Self-model (direct change) updated to version {self.self_model['version']}")
            elif reflection_performed_flag : # If reflection happened, it's a more significant update type
                 logger_dynamic_self_model.info(f"Self-model (reflection included) updated to version {self.self_model['version']}")


    async def _perform_reflection(self):
        logger_dynamic_self_model.info(f"Performing self-reflection. Analyzing {len(self.learning_events)} recent learning events.")
        if len(self.learning_events) < 5: # Need a minimum number of events for meaningful reflection
            logger_dynamic_self_model.debug("Not enough learning events for deep reflection.")
            return

        traits = self.self_model.setdefault("identity_traits", DEFAULT_SELF_MODEL["identity_traits"].copy())
        lr_meta = self.self_model.setdefault("learning_rate_meta", DEFAULT_SELF_MODEL["learning_rate_meta"].copy())
        lr_meta_caps = lr_meta.setdefault("capabilities", {"fast_learner": [], "slow_learner": []})

        total_events = len(self.learning_events)
        success_count = sum(1 for event in self.learning_events if event.get("outcome") == "success")
        success_rate = success_count / total_events if total_events > 0 else 0.0

        action_keys_tried = set(event.get("action_key_specific", event.get("action_type")) for event in self.learning_events if event.get("action_key_specific") or event.get("action_type"))
        diversity_factor = len(action_keys_tried) / total_events if total_events > 0 else 0.0
        adaptability_change = 0.02 * (diversity_factor - 0.3) * (success_rate - 0.4) 
        traits["adaptability"] = min(1.0, max(0.1, traits.get("adaptability", 0.5) + adaptability_change))
        
        known_high_confidence_caps = { k for k, v in self.self_model.get("capabilities", {}).items() if v > 0.8 }
        exploratory_actions_count = sum(1 for event in self.learning_events if event.get("action_key_specific", event.get("action_type")) not in known_high_confidence_caps)
        exploration_rate = exploratory_actions_count / total_events if total_events > 0 else 0.0
        curiosity_change = 0.04 * (exploration_rate - 0.4) 
        traits["curiosity"] = min(1.0, max(0.1, traits.get("curiosity", 0.5) + curiosity_change))

        # --- REFINED Persistence Calculation ---
        action_key_persistence_score = 0.0
        action_keys_evaluated_for_persistence = set()

        # Iterate through learning events to find patterns of failure then success for same specific action key
        for i, event in enumerate(self.learning_events):
            action_key = event.get("action_key_specific") or event.get("action_type")
            if not action_key or action_key in action_keys_evaluated_for_persistence:
                continue # Skip if no key or already processed this key for persistence patterns

            actions_for_this_key = [e for e in self.learning_events if (e.get("action_key_specific") or e.get("action_type")) == action_key]
            
            if len(actions_for_this_key) < 2: # Need at least two events for a fail-then-succeed pattern
                action_keys_evaluated_for_persistence.add(action_key)
                continue

            failures_before_success = 0
            initial_failure_streak = True
            found_eventual_success_after_streak = False

            for k_event in actions_for_this_key:
                outcome = k_event.get("outcome")
                if initial_failure_streak:
                    if outcome == "failure":
                        failures_before_success += 1
                    elif outcome == "success":
                        initial_failure_streak = False # Streak broken by success
                        if failures_before_success > 0: # Only count if there were prior failures
                            found_eventual_success_after_streak = True
                        break # Stop checking this key's streak once a success breaks it or pattern found
                    # else: other outcomes, streak continues if not success
                # else: already found a success, or streak didn't start with failure
            
            if found_eventual_success_after_streak and failures_before_success > 0:
                # Reward persistence: more failures overcome before success = higher score for this key
                # Normalize by typical number of retries (e.g., 3-5 failures is strong persistence)
                # Max score of 1.0 for this key's persistence contribution.
                # Example: 1 failure then success = 0.2, 2=0.4, ..., 5+=1.0
                key_persistence_value = min(1.0, failures_before_success / 5.0) 
                action_key_persistence_score += key_persistence_value
                logger_dynamic_self_model.debug(
                    f"Persistence pattern for '{action_key}': {failures_before_success} failures then success. Score contribution: {key_persistence_value:.2f}"
                )
            
            action_keys_evaluated_for_persistence.add(action_key)

        # Normalize overall persistence score by number of unique action keys evaluated
        # or by a fixed number to prevent one highly persistent action dominating too much.
        num_distinct_actions_in_events = len(action_keys_evaluated_for_persistence)
        if num_distinct_actions_in_events > 0:
            normalized_persistence_metric = action_key_persistence_score / num_distinct_actions_in_events
        else:
            normalized_persistence_metric = 0.0
        
        # Adjust persistence trait: if metric > 0.5 (average good persistence), increase. If < 0.3 (low persistence), decrease.
        persistence_change_factor = 0.05 # How much to change the trait per reflection
        if normalized_persistence_metric > 0.5: # Threshold for "good" persistence
            persistence_change = persistence_change_factor
        elif normalized_persistence_metric < 0.2 and total_events > len(self.learning_events) * 0.5 : # Threshold for "low" persistence, ensure enough events
            persistence_change = -persistence_change_factor
        else: # Neutral or not enough data to strongly adjust
            persistence_change = 0.0

        traits["persistence"] = min(1.0, max(0.1, traits.get("persistence", 0.5) + persistence_change))
        logger_dynamic_self_model.debug(
            f"Refined Persistence: Metric={normalized_persistence_metric:.2f}, Change={persistence_change:.2f}, NewTraitVal={traits['persistence']:.2f}"
        )
        # --- END REFINED Persistence ---

        caution_change = 0.03 * (0.6 - success_rate) 
        traits["caution"] = min(1.0, max(0.1, traits.get("caution", 0.5) + caution_change))
        logger_dynamic_self_model.debug(f"Reflected Identity Traits: {traits}")

        action_outcomes: Dict[str, List[str]] = {} 
        for event in self.learning_events:
            key = event.get("action_key_specific") or event.get("action_type") # Ensure key is non-None
            if key: # Only process if key is valid
                if key not in action_outcomes: action_outcomes[key] = []
                action_outcomes[key].append(event.get("outcome", "unknown"))
        
        fast_learner_keys = []
        slow_learner_keys = []
        for action_key, outcomes_list in action_outcomes.items():
            if len(outcomes_list) < 3: continue 
            outcome_counts = Counter(outcomes_list)
            if not outcome_counts: continue # Should not happen if outcomes_list is not empty
            most_common_outcome, count = outcome_counts.most_common(1)[0]
            consistency = count / len(outcomes_list)
            if consistency >= 0.85: fast_learner_keys.append(action_key)
            elif consistency <= 0.4: slow_learner_keys.append(action_key)
        
        lr_meta_caps["fast_learner"] = list(set(fast_learner_keys)) 
        lr_meta_caps["slow_learner"] = list(set(slow_learner_keys))
        logger_dynamic_self_model.debug(f"Reflected Learning Rate Meta: Fast={fast_learner_keys}, Slow={slow_learner_keys}")
        self.learning_events.clear() 
        logger_dynamic_self_model.info("Deep reflection complete. Learning events cleared.")

    async def get_self_model(self) -> Dict[str, Any]: return json.loads(json.dumps(self.self_model))

    async def save_model(self):
        if not self._model_path: logger_dynamic_self_model.debug("Self-model saving disabled (no path)."); return False
        logger_dynamic_self_model.info(f"Saving self-model v{self.self_model.get('version')} to {self._model_path}..."); temp_path = None
        try:
            self._model_path.parent.mkdir(parents=True, exist_ok=True); temp_path = self._model_path.with_suffix(".tmp")
            with open(temp_path, 'w') as f: json.dump(self.self_model, f, indent=2, default=str)
            os.replace(temp_path, self._model_path); logger_dynamic_self_model.info("Self-model saved successfully."); return True
        except Exception as e:
            logger_dynamic_self_model.exception(f"Failed to save self-model to {self._model_path}: {e}")
            if temp_path is not None and temp_path.exists():
                try: temp_path.unlink(); logger_dynamic_self_model.debug(f"Cleaned up temp file: {temp_path}")
                except OSError as unlink_e: logger_dynamic_self_model.error(f"Error cleaning temp file {temp_path}: {unlink_e}")
            return False

    async def process(self, input_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not input_state: logger_dynamic_self_model.debug("DSM process: No input."); return None
        action_outcome = input_state.get("action_outcome")
        if not action_outcome: logger_dynamic_self_model.warning("DSM process: Missing 'action_outcome'."); return None
        await self.update_self_model(
            last_action_type=input_state.get("last_action_type"), action_outcome=action_outcome,
            action_params=input_state.get("action_params", {}), action_error=input_state.get("action_error"),
            action_result_data=input_state.get("action_result_data"), current_phenomenal_state=input_state.get("phenomenal_state") )
        return None 

    async def reset(self) -> None:
        logger_dynamic_self_model.warning("Resetting DynamicSelfModel to default state.")
        self.self_model = json.loads(json.dumps(DEFAULT_SELF_MODEL)) 
        
        learning_rate_from_config = DEFAULT_SELF_MODEL["learning_rate"]
        if self._controller and hasattr(self._controller, 'config'):
            controller_config = getattr(self._controller, 'config', {})
            dsm_specific_config = controller_config.get("dynamic_self_model", {})
            learning_rate_from_config = dsm_specific_config.get("learning_rate", DEFAULT_SELF_MODEL["learning_rate"])
        
        self.self_model["learning_rate"] = learning_rate_from_config
        self.last_update_time = None
        self.learning_events.clear()
        self.cycles_since_reflection = 0


    async def get_status(self) -> Dict[str, Any]:
        km = self.self_model.get("knowledge_meta", {}); isa = self.self_model.get("internal_state_awareness", {})
        return { "component": "DynamicSelfModel", "status": "operational", "version": self.self_model.get("version", 0),
                 "last_update_time": self.last_update_time, "num_capabilities": len(self.self_model.get("capabilities", {})),
                 "num_limitations": len(self.self_model.get("limitations", {})), "num_identity": len(self.self_model.get("identity_traits", {})),
                 "num_valid_paths": len(km.get("validated_paths", {})), "num_invalid_paths": len(km.get("invalid_paths", [])),
                 "recent_errors_tracked": len(isa.get("recent_error_types", [])), 
                 "cycles_to_reflection": self.reflection_interval - self.cycles_since_reflection if self.reflection_interval > 0 else -1,
                 "learning_events_count": len(self.learning_events)
                 }

    async def shutdown(self) -> None: logger_dynamic_self_model.info("DynamicSelfModel shutting down."); await self.save_model()

# --- END OF CORRECTED dynamic_self_model.py ---