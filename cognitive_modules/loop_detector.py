# --- START OF CORRECTED loop_detector.py ---

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from collections import deque, Counter

# --- Use standard relative imports ---
try:
    from ..protocols import CognitiveComponent
    # Import specific types needed if interacting with KB results directly
    # from ..models.datatypes import Predicate # Assuming KB returns dicts for now
    # Import KB type hint
    from .knowledge_base import KnowledgeBase
except ImportError:
    # Fallback for different execution context (e.g., combined script)
    logging.warning("LoopDetector: Relative imports failed, relying on globally defined types.")
    if 'CognitiveComponent' not in globals(): raise ImportError("CognitiveComponent not found via relative import or globally")
    if 'KnowledgeBase' not in globals(): raise ImportError("KnowledgeBase not found via relative import or globally")
    CognitiveComponent = globals().get('CognitiveComponent')
    KnowledgeBase = globals().get('KnowledgeBase')


logger_loop_detector = logging.getLogger(__name__) # Use standard module logger name

# Default configuration values
DEFAULT_WINDOW_SIZE = 5
DEFAULT_MAX_CONSECUTIVE_ACTIONS = 3
# --- NEW DEFAULTS ---
DEFAULT_FREQUENCY_THRESHOLD = 0.7  # e.g., 70% of actions in window being the same
DEFAULT_IGNORE_THINKING_ACTIONS = True

# --- Inherit correctly from CognitiveComponent ---
class LoopDetector(CognitiveComponent):
    """
    Detects repetitive action loops based on recent history.
    """

    def __init__(self):
        self._controller: Optional[Any] = None
        self._config: Dict[str, Any] = {}
        self.window_size: int = DEFAULT_WINDOW_SIZE
        self.max_consecutive_actions: int = DEFAULT_MAX_CONSECUTIVE_ACTIONS
        # Use type hint string for forward reference if KB class is complex
        self._kb: Optional['KnowledgeBase'] = None
        
        # --- NEW ATTRIBUTES ---
        self.frequency_threshold: float = DEFAULT_FREQUENCY_THRESHOLD
        self.ignore_thinking_actions: bool = DEFAULT_IGNORE_THINKING_ACTIONS

    async def initialize(self, config: Dict[str, Any], controller: Any) -> bool:
        """Initialize loop detector with configuration and KB reference."""
        self._controller = controller
        loop_config = config.get("loop_detection", {})
        self._config = loop_config # Store component-specific config

        # self.window_size is initially set here
        self.window_size = loop_config.get("window_size", DEFAULT_WINDOW_SIZE) 
        self.max_consecutive_actions = loop_config.get("max_consecutive_actions", DEFAULT_MAX_CONSECUTIVE_ACTIONS)
        
        # --- LOAD NEW CONFIG VALUES ---
        self.frequency_threshold = loop_config.get("frequency_threshold", DEFAULT_FREQUENCY_THRESHOLD)
        self.ignore_thinking_actions = loop_config.get("ignore_thinking_actions", DEFAULT_IGNORE_THINKING_ACTIONS)

        # Validate new config values
        if not (0.0 < self.frequency_threshold <= 1.0):
            logger_loop_detector.warning(
                f"LoopDetector frequency_threshold ({self.frequency_threshold}) invalid. "
                f"Must be (0, 1]. Using default {DEFAULT_FREQUENCY_THRESHOLD}."
            )
            self.frequency_threshold = DEFAULT_FREQUENCY_THRESHOLD
        if not isinstance(self.ignore_thinking_actions, bool):
            logger_loop_detector.warning(
                f"LoopDetector ignore_thinking_actions ({self.ignore_thinking_actions}) invalid. "
                f"Must be boolean. Using default {DEFAULT_IGNORE_THINKING_ACTIONS}."
            )
            self.ignore_thinking_actions = DEFAULT_IGNORE_THINKING_ACTIONS
        # --- END VALIDATION ---

        # (Validation for window_size and max_consecutive_actions remains the same)
        if self.window_size <= 0:
             logger_loop_detector.warning(f"LoopDetector window_size must be positive. Using default {DEFAULT_WINDOW_SIZE}.")
             self.window_size = DEFAULT_WINDOW_SIZE
        if self.max_consecutive_actions <= 0:
             logger_loop_detector.warning(f"LoopDetector max_consecutive_actions must be positive. Using default {DEFAULT_MAX_CONSECUTIVE_ACTIONS}.")
             self.max_consecutive_actions = DEFAULT_MAX_CONSECUTIVE_ACTIONS
        if self.max_consecutive_actions > self.window_size:
             # This is only a warning, not an error that prevents operation.
             # It means the consecutive check might not be very useful if window is too small.
             logger_loop_detector.warning(f"LoopDetector max_consecutive_actions ({self.max_consecutive_actions}) > window_size ({self.window_size}). This might not be effective for consecutive checks.")


        # Get reference to KnowledgeBase component safely
        _KnowledgeBaseClass = globals().get('KnowledgeBase')
        kb_ref = None
        if _KnowledgeBaseClass and hasattr(controller, 'knowledge_base') and isinstance(controller.knowledge_base, _KnowledgeBaseClass):
             kb_ref = controller.knowledge_base

        if kb_ref:
             self._kb = kb_ref
             logger_loop_detector.info(
                 f"LoopDetector initialized. Initial Window: {self.window_size}, MaxConsec: {self.max_consecutive_actions}, " # Log initial
                 f"FreqThresh: {self.frequency_threshold:.2f}, IgnoreThinking: {self.ignore_thinking_actions}. Using KB."
             )
             return True
        else:
             logger_loop_detector.error("LoopDetector initialization failed: KnowledgeBase component not found or invalid.")
             # Fail initialization if KB is essential
             return False


    async def detect_loops(self) -> Optional[Dict[str, Any]]:
        """
        Analyzes recent action history from the Knowledge Base to detect loops.
        Returns loop information if detected, otherwise None.
        """
        if not self._kb:
             logger_loop_detector.error("LoopDetector cannot detect loops: KnowledgeBase not available.")
             return None

        # --- DYNAMICALLY READ window_size FROM CONTROLLER'S CONFIG ---
        current_effective_window_size = self.window_size # Fallback to initialized value
        if self._controller and hasattr(self._controller, 'config') and \
           isinstance(self._controller.config, dict): # type: ignore
            ld_config_live = self._controller.config.get("loop_detection", {}) # type: ignore
            if isinstance(ld_config_live, dict):
                current_effective_window_size = ld_config_live.get("window_size", self.window_size)
                if not isinstance(current_effective_window_size, int) or current_effective_window_size <= 0:
                    logger_loop_detector.warning(
                        f"Invalid window_size ({current_effective_window_size}) from live config. "
                        f"Using initialized value: {self.window_size}"
                    )
                    current_effective_window_size = self.window_size
        # --- END DYNAMIC READ ---

        logger_loop_detector.debug(f"Detecting loops using effective window_size: {current_effective_window_size}.")


        # --- Query recent action history from KnowledgeBase ---
        fetched_actions_raw: List[Dict[str, Any]] = [] # Default to empty list
        try:
            # Fetch slightly more if ignoring thinking, to ensure window_size of relevant actions
            fetch_count = current_effective_window_size * 3 if self.ignore_thinking_actions else current_effective_window_size + 5 # Fetch a bit more buffer
            fetch_count = max(fetch_count, 15) # Fetch at least a decent amount

            kb_state = await self._kb.query_state({"recent_facts": fetch_count })
            action_events_temp = []
            if kb_state and "recent_facts" in kb_state and isinstance(kb_state["recent_facts"], list):
                for fact in kb_state["recent_facts"]:
                     is_event = isinstance(fact, dict) and fact.get("name") == "eventOccurred"
                     args_ok = isinstance(fact.get("args"), (list, tuple)) and len(fact.get("args",[])) == 3
                     is_action_exec = args_ok and fact["args"][0] == "actionExecution"

                     if is_event and is_action_exec:
                         action_events_temp.append({
                             "type": fact["args"][1], # Action type
                             "outcome": fact["args"][2], # Action outcome
                             "timestamp": fact.get("timestamp", 0.0) # Ensure timestamp exists
                         })
                # Ensure they are sorted by timestamp descending if not already (KB should do this for recent_facts)
                action_events_temp.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
                fetched_actions_raw = action_events_temp # Store all fetched, will filter/slice later
                # logger_loop_detector.debug(f"Retrieved {len(fetched_actions_raw)} raw actions for loop detection.")
            elif kb_state and "error" in kb_state:
                 logger_loop_detector.warning(f"LoopDetector: KB query_state returned error: {kb_state['error']}")
            else:
                 logger_loop_detector.warning("LoopDetector: Could not retrieve recent facts from KB status/query (invalid response format).")

        except Exception as e:
            logger_loop_detector.exception(f"LoopDetector: Error querying action history from KB: {e}")
            return None # Fail detection if KB query fails

        # --- FILTER THINKING ACTIONS IF CONFIGURED ---
        if self.ignore_thinking_actions:
            relevant_actions_intermediate = [a for a in fetched_actions_raw if a.get("type") != "THINKING"]
        else:
            relevant_actions_intermediate = fetched_actions_raw
        
        # Now take the most recent self.window_size from the (potentially filtered) list
        recent_actions = relevant_actions_intermediate[:current_effective_window_size]
        
        if self.ignore_thinking_actions:
            logger_loop_detector.debug(
                f"Fetched {len(fetched_actions_raw)} raw, filtered to {len(relevant_actions_intermediate)} non-THINKING, "
                f"using window of {len(recent_actions)} for detection."
            )
        else:
             logger_loop_detector.debug(f"Using window of {len(recent_actions)} actions (THINKING included).")
        # --- END FILTER ---
        
        # Check if enough actions for any kind of loop detection
        if not recent_actions:
            logger_loop_detector.debug("No relevant actions in history for loop detection.")
            return None

        # --- Check for Consecutive Identical Actions ---
        # This check requires at least `max_consecutive_actions` in the `recent_actions` window
        if len(recent_actions) >= self.max_consecutive_actions:
            first_action_type = recent_actions[0].get("type")
            if first_action_type: # Ensure there's a type to check
                is_consecutive_loop = True
                for i in range(1, self.max_consecutive_actions):
                     # The loop runs from i=1 up to max_consecutive_actions-1
                     # So we check recent_actions[0] against recent_actions[1]...recent_actions[max_consecutive_actions-1]
                     if i >= len(recent_actions) or recent_actions[i].get("type") != first_action_type:
                         is_consecutive_loop = False
                         break
                if is_consecutive_loop:
                     loop_info = {
                         "type": "consecutive_action", "action_type": first_action_type,
                         "count": self.max_consecutive_actions, "window_size": len(recent_actions), 
                         "details": f"Action '{first_action_type}' repeated {self.max_consecutive_actions} times consecutively."
                     }
                     logger_loop_detector.warning(f"Loop Detected: {loop_info['details']}")
                     return loop_info
        elif len(recent_actions) > 0: # Not enough for consecutive check, but log it
             logger_loop_detector.debug(
                f"Not enough relevant actions ({len(recent_actions)}) for consecutive check ({self.max_consecutive_actions} needed)."
            )


        # --- Check High Frequency Action ---
        # This check requires at least a few actions in `recent_actions` to be meaningful
        min_actions_for_frequency_check = max(2, int(current_effective_window_size * 0.5)) # e.g., at least half the window, or 2
        if len(recent_actions) < min_actions_for_frequency_check:
            logger_loop_detector.debug(
                f"Not enough relevant actions ({len(recent_actions)}) for frequency check (min {min_actions_for_frequency_check} needed for window {current_effective_window_size})."
            )
            return None

        action_types_in_window = [a.get("type") for a in recent_actions if a.get("type")]
        if not action_types_in_window: return None # Should not happen if len(recent_actions) > 0 and they have types

        counts = Counter(action_types_in_window)
        if not counts: return None # Should not happen

        most_common_action, most_common_count = counts.most_common(1)[0]
        
        current_num_actions_in_window = len(action_types_in_window) # This is len(recent_actions) if all have types
        if current_num_actions_in_window == 0: return None # Avoid division by zero

        current_frequency = most_common_count / current_num_actions_in_window
        
        # The frequency check needs to be against self.frequency_threshold
        # And the count should be significant enough (e.g., more than 1 if window is small)
        min_count_for_high_freq = max(2, self.max_consecutive_actions -1 if self.max_consecutive_actions > 1 else 2)

        if most_common_count >= min_count_for_high_freq and current_frequency >= self.frequency_threshold:
             loop_info = {
                 "type": "high_frequency_action", "action_type": most_common_action,
                 "count": most_common_count, "window_size": current_num_actions_in_window,
                 "frequency": round(current_frequency, 2),
                 "details": (f"Action '{most_common_action}' occurred {most_common_count} times "
                             f"in last {current_num_actions_in_window} relevant actions ({current_frequency*100:.0f}% "
                             f">= threshold {self.frequency_threshold*100:.0f}%).")
             }
             logger_loop_detector.warning(f"Potential Loop Detected (High Frequency): {loop_info['details']}")
             return loop_info

        logger_loop_detector.debug("No obvious loops detected in recent relevant action history.")
        return None # No loop detected


    # --- CognitiveComponent Implementation ---

    async def process(self, input_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Performs loop detection based on history available in the KB.
        Returns loop information if detected.
        """
        loop_info = await self.detect_loops()
        return {"loop_info": loop_info} # Return dict structure always, value is None if no loop


    async def reset(self) -> None:
        """Reset detector state (if any)."""
        # LoopDetector is stateless beyond its config, relies on KB for history.
        logger_loop_detector.info("LoopDetector reset (stateless component, no internal history cleared).")

    async def get_status(self) -> Dict[str, Any]:
        """Return status of the loop detector."""
        kb_status = "operational" if self._kb else "error (KB unavailable)"
        
        # Report the dynamically read window_size if controller is available, else initial
        effective_window_size_for_status = self.window_size
        if self._controller and hasattr(self._controller, 'config') and \
           isinstance(self._controller.config, dict):
            ld_config_live_status = self._controller.config.get("loop_detection", {})
            if isinstance(ld_config_live_status, dict):
                effective_window_size_for_status = ld_config_live_status.get("window_size", self.window_size)
        
        return {
            "component": "LoopDetector", "status": kb_status,
            "window_size": effective_window_size_for_status, # Report effective window size
            "max_consecutive_actions": self.max_consecutive_actions,
            # --- ADD NEW STATUS ITEMS ---
            "frequency_threshold": self.frequency_threshold,
            "ignore_thinking_actions": self.ignore_thinking_actions
        }

    async def shutdown(self) -> None:
        """Perform any necessary cleanup."""
        logger_loop_detector.info("LoopDetector shutting down.")
        self._kb = None # Release reference

# --- END OF CORRECTED loop_detector.py ---