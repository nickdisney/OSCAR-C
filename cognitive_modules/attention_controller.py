# --- START OF FILE cognitive_modules/attention_controller.py (Surprise Score Refinement) ---

import asyncio
import logging
import time
import math 
import re 
from typing import Dict, Any, Optional, List, Deque, Tuple # Added Tuple for Deque
from collections import deque 

try:
    from ..protocols import CognitiveComponent, AttentionMechanism
except ImportError:
    logging.warning("AttentionController: Relative imports for protocols failed. Using global or placeholder.")
    if 'CognitiveComponent' not in globals():
        from typing import Protocol
        class CognitiveComponent(Protocol): # type: ignore
            async def initialize(self, config: Dict[str, Any], controller: Any) -> bool: ...
            async def process(self, input_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]: ...
            async def reset(self) -> None: ...
            async def get_status(self) -> Dict[str, Any]: ...
            async def shutdown(self) -> None: ...
    if 'AttentionMechanism' not in globals():
        from typing import Protocol
        class AttentionMechanism(CognitiveComponent, Protocol): # type: ignore 
             async def allocate_attention(self, candidates: Dict[str, Dict[str, Any]]) -> Dict[str, float]: ...
    CognitiveComponent = globals().get('CognitiveComponent') # type: ignore
    AttentionMechanism = globals().get('AttentionMechanism') # type: ignore

_PainEventTracker_AC: Optional[type] = None
try:
    # Try absolute-like path first, common for test runners finding the package
    from consciousness_experiment.agent_helpers.cognitive_trackers import PainEventTracker as _ImportedPainEventTracker
    _PainEventTracker_AC = _ImportedPainEventTracker
    logging.getLogger(__name__).debug("AttentionController: Successfully imported PainEventTracker via absolute-like path.")
except ImportError:
    try:
        # Fallback to relative path if the first one fails (e.g., if run as part of the package itself)
        from ..agent_helpers.cognitive_trackers import PainEventTracker as _ImportedPainEventTrackerRel
        _PainEventTracker_AC = _ImportedPainEventTrackerRel
        logging.getLogger(__name__).debug("AttentionController: Successfully imported PainEventTracker via relative path.")
    except ImportError:
        logging.getLogger(__name__).error(
            "AttentionController: CRITICAL - Failed to import PainEventTracker from both common paths. "
            "Rumination suppression will be disabled."
        )
        # _PainEventTracker_AC remains None

# Use the potentially imported class or None
PainEventTracker = _PainEventTracker_AC # This makes PainEventTracker available in the module scope
                                        # It will be None if imports failed.


logger_attention_controller = logging.getLogger(__name__) 

DEFAULT_RECENCY_WEIGHT = 0.3
DEFAULT_HINT_WEIGHT = 0.5
DEFAULT_GOAL_RELEVANCE_WEIGHT = 0.2 
DEFAULT_MAX_CANDIDATES = 50 
DEFAULT_SOFTMAX_TEMPERATURE = 0.1
DEFAULT_NOVELTY_WINDOW_SIZE = 10
DEFAULT_NOVELTY_BONUS_WEIGHT = 0.15
DEFAULT_SURPRISE_BONUS_WEIGHT = 0.25
# --- NEW: Default for pain attention control ---
DEFAULT_PAIN_ATTENTION_DISTRACTION_FACTOR = 0.2 # Matches AgentController
DEFAULT_PAIN_RUMINATION_THRESHOLD_CYCLES = 10   # Matches AgentController


class AttentionController(AttentionMechanism): # type: ignore 
    """
    Allocates attention weights to candidate items based on various factors.
    Determines the focus for the Global Workspace.
    """

    def __init__(self):
        self._controller: Optional[Any] = None
        self._config: Dict[str, Any] = {}
        self.recency_weight: float = DEFAULT_RECENCY_WEIGHT
        self.hint_weight: float = DEFAULT_HINT_WEIGHT
        self.goal_relevance_weight: float = DEFAULT_GOAL_RELEVANCE_WEIGHT
        self.max_candidates: int = DEFAULT_MAX_CANDIDATES
        self.softmax_temperature: float = DEFAULT_SOFTMAX_TEMPERATURE

        self.novelty_window_size: int = DEFAULT_NOVELTY_WINDOW_SIZE
        self.novelty_bonus_weight: float = DEFAULT_NOVELTY_BONUS_WEIGHT
        self.surprise_bonus_weight: float = DEFAULT_SURPRISE_BONUS_WEIGHT
        
        self.recent_workspace_content_hashes: Deque[int] = deque(maxlen=self.novelty_window_size)
        self._last_prediction_error_for_surprise: Optional[Dict[str, Any]] = None
        self._current_cycle_active_id: Optional[str] = None

        # --- NEW: Pain Attention Control Attributes ---
        self.pain_attention_distraction_factor: float = DEFAULT_PAIN_ATTENTION_DISTRACTION_FACTOR
        # self.pain_rumination_threshold_cycles: int = DEFAULT_PAIN_RUMINATION_THRESHOLD_CYCLES # Will be passed to tracker

        self.pain_tracker: Optional[PainEventTracker] = None


    async def initialize(self, config: Dict[str, Any], controller: Any) -> bool:
        self._controller = controller
        attn_config = config.get("attention_controller", {})
        self._config = attn_config

        self.recency_weight = attn_config.get("recency_weight", DEFAULT_RECENCY_WEIGHT)
        self.hint_weight = attn_config.get("hint_weight", DEFAULT_HINT_WEIGHT)
        self.goal_relevance_weight = attn_config.get("goal_relevance_weight", DEFAULT_GOAL_RELEVANCE_WEIGHT)
        self.max_candidates = attn_config.get("max_candidates", DEFAULT_MAX_CANDIDATES)
        self.softmax_temperature = attn_config.get("softmax_temperature", DEFAULT_SOFTMAX_TEMPERATURE)

        self.novelty_window_size = attn_config.get("novelty_window_size", DEFAULT_NOVELTY_WINDOW_SIZE)
        self.novelty_bonus_weight = attn_config.get("novelty_bonus_weight", DEFAULT_NOVELTY_BONUS_WEIGHT)
        self.surprise_bonus_weight = attn_config.get("surprise_bonus_weight", DEFAULT_SURPRISE_BONUS_WEIGHT)
        

        # --- Initialize Pain Attention Control Parameters ---
        self.pain_attention_distraction_factor = float(attn_config.get("pain_attention_distraction_factor", DEFAULT_PAIN_ATTENTION_DISTRACTION_FACTOR))
        # self.pain_rumination_threshold_cycles = int(attn_config.get("pain_rumination_threshold_cycles", DEFAULT_PAIN_RUMINATION_THRESHOLD_CYCLES)) # Tracker uses this

        self.recent_workspace_content_hashes = deque(maxlen=self.novelty_window_size)
        
        # Parameters for PainEventTracker
        rum_threshold = int(attn_config.get("pain_rumination_threshold_cycles", DEFAULT_PAIN_RUMINATION_THRESHOLD_CYCLES))
        rum_window_mult = int(attn_config.get("pain_rumination_window_multiplier", 3)) 
        inactive_reset = int(attn_config.get("pain_inactive_reset_cycles", 10))     

        if PainEventTracker is not None: # Explicitly check if the import was successful
            self.pain_tracker = PainEventTracker( # Now this uses the module-level PainEventTracker
                pain_rumination_threshold=rum_threshold,
                rumination_window_multiplier=rum_window_mult,
                inactive_reset_cycles=inactive_reset
            )
            logger_attention_controller.info("PainEventTracker initialized within AttentionController.")
        else:
            self.pain_tracker = None # Explicitly set to None if class wasn't imported
            logger_attention_controller.error("PainEventTracker class not available due to import failure. Pain rumination suppression will not function.")
        
        logger_attention_controller.info(
            f"AttentionController initialized. MaxCand: {self.max_candidates}, Temp: {self.softmax_temperature:.2f}, "
            f"NoveltyWin: {self.novelty_window_size}, NoveltyWt: {self.novelty_bonus_weight:.2f}, SurpriseWt: {self.surprise_bonus_weight:.2f}, "
            f"PainDistractFactor: {self.pain_attention_distraction_factor:.2f}, "
            f"PainRumThresh (for tracker): {rum_threshold}, PainRumWindowMult: {rum_window_mult}, PainInactiveReset: {inactive_reset}, "
            f"Weights (Rec/Hint/GoalRel): {self.recency_weight:.2f}/{self.hint_weight:.2f}/{self.goal_relevance_weight:.2f}"
        )
        return True

    def _calculate_novelty_score(self, item_id: str, item_data: Dict[str, Any]) -> float:
        content_to_check = item_data.get("content")
        if content_to_check is None: 
            return 0.0
        
        try:
            current_hash = hash(str(content_to_check)) 
        except Exception as e:
            logger_attention_controller.warning(f"Could not hash content for novelty check of item '{item_id}': {e}", exc_info=False)
            return 0.0 

        if current_hash not in self.recent_workspace_content_hashes:
            logger_attention_controller.debug(f"Novelty bonus for '{item_id}' (hash: {current_hash} not in recent hashes).")
            return 1.0  
        else:
            return 0.0

    def _calculate_surprise_score(self, item_id: str, item_data: Dict[str, Any]) -> float:
        if not self._last_prediction_error_for_surprise:
            return 0.0

        error_details = self._last_prediction_error_for_surprise
        err_type = error_details.get("type")

        if err_type != "outcome_mismatch":
            return 0.0

        err_source_details = error_details.get("error_source_details", {})
        err_action_type_source = err_source_details.get("action_type_source")
        err_action_params_source = err_source_details.get("params_source", {})
        
        if item_id.startswith("goal_") and err_action_type_source:
            candidate_content_str = str(item_data.get("content", "")).lower() 
            
            if err_action_type_source == "READ_FILE":
                match_goal_read = re.search(r"read file\s*:\s*(.+?)(?:\s*\(|$)", candidate_content_str)
                if match_goal_read:
                    goal_path_param = match_goal_read.group(1).strip().lower() # Normalize path from goal
                    error_path_param = err_action_params_source.get("path", "").lower() # Normalize path from error
                    if error_path_param and goal_path_param == error_path_param:
                        logger_attention_controller.debug(
                            f"Surprise bonus (0.9) for goal '{item_id}' due to direct match on failed READ_FILE for path '{error_path_param}'."
                        )
                        return 0.9 
            
            if err_action_type_source.lower() in candidate_content_str:
                logger_attention_controller.debug(
                    f"Surprise bonus (0.6) for goal '{item_id}' as its description contains failed action type '{err_action_type_source}'."
                )
                return 0.6 

        mispredicted_percept_key_from_error = err_source_details.get("mispredicted_percept_key")
        if mispredicted_percept_key_from_error and item_id.startswith(f"percept_{mispredicted_percept_key_from_error}"):
            error_magnitude = float(error_details.get("error_magnitude", 1.0))
            logger_attention_controller.debug(f"Surprise bonus for mispredicted percept '{item_id}'. Magnitude: {error_magnitude:.2f}")
            return error_magnitude
        
        return 0.0

    async def allocate_attention(self, candidates: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        if not candidates:
            return {}

        current_max_candidates = self.max_candidates 
        if self._controller and hasattr(self._controller, 'config') and \
           isinstance(self._controller.config, dict): 
            ac_config_live = self._controller.config.get("attention_controller", {}) 
            if isinstance(ac_config_live, dict):
                current_max_candidates = ac_config_live.get("max_candidates", self.max_candidates)
                if not isinstance(current_max_candidates, int) or current_max_candidates <= 0:
                    logger_attention_controller.warning(f"Invalid max_candidates ({current_max_candidates}) from live config. Using initialized value: {self.max_candidates}")
                    current_max_candidates = self.max_candidates
        
        logger_attention_controller.debug(
            f"Allocating attention to {len(candidates)} candidates. "
            f"Effective max_candidates: {current_max_candidates}. Active cycle goal ID: {self._current_cycle_active_id}"
        )

        if len(candidates) > current_max_candidates: 
            logger_attention_controller.warning(
                f"Number of attention candidates ({len(candidates)}) exceeds dynamic limit ({current_max_candidates}). Truncating."
            )
            sorted_candidates_by_hint = sorted(candidates.items(), key=lambda x: x[1].get('weight_hint', 0.0), reverse=True)
            candidates = dict(sorted_candidates_by_hint[:current_max_candidates]) 


        weighted_scores: Dict[str, float] = {}
        current_time = time.time()
        active_goal_desc_general_context = ""
        if self._controller and hasattr(self._controller, '_oscar_get_active_goal'):
             active_goal_context_obj = self._controller._oscar_get_active_goal() # type: ignore
             if active_goal_context_obj and hasattr(active_goal_context_obj, 'description'):
                 active_goal_desc_general_context = active_goal_context_obj.description.lower()


        for item_id, item_data in candidates.items():
            content = item_data.get("content", "")
            original_hint_from_gather = float(item_data.get('weight_hint', 0.5))
            
            ## REVISION: Initialize hint_score_for_combined_calc with the clamped original hint.
            ## This will be the value used for non-pain items, or for pain items before suppression.
            hint_score_for_combined_calc = max(0.0, min(1.0, original_hint_from_gather))
            
            timestamp = item_data.get("timestamp", current_time)

            time_delta = max(0, current_time - timestamp)
            recency_score = math.exp(-time_delta / 60.0) 
            
            relevance_score_component = 0.0
            is_this_candidate_a_goal = isinstance(item_id, str) and item_id.startswith("goal_")
            if is_this_candidate_a_goal:
                candidate_goal_id = item_id[len("goal_"):]
                if self._current_cycle_active_id and candidate_goal_id == self._current_cycle_active_id:
                    relevance_score_component = 1.0 
                else: 
                    relevance_score_component = 0.1 
            elif item_id.startswith("pain_event_"): 
                if active_goal_desc_general_context and "pain" in active_goal_desc_general_context:
                    relevance_score_component = 0.7 
                else:
                    relevance_score_component = 0.05 
            else: 
                content_str = str(content).lower() 
                if active_goal_desc_general_context and content_str:
                    try:
                        goal_words = set(active_goal_desc_general_context.split())
                        content_words = set(content_str.split())
                        common_words = goal_words.intersection(content_words)
                        if len(goal_words) > 0:
                             relevance_score_component = len(common_words) / len(goal_words)
                    except Exception: 
                         logger_attention_controller.debug(f"Could not calculate keyword relevance for non-goal item '{item_id}'", exc_info=False)

            
            novelty_score = self._calculate_novelty_score(item_id, item_data)
            surprise_score = self._calculate_surprise_score(item_id, item_data)

            # Initialize suppressed_by_rumination to False for each candidate
            suppressed_by_rumination = False # <<< IMPORTANT: Initialize for each candidate item
            final_hint_for_ac_candidate_log = hint_score_for_combined_calc

            if item_id.startswith("pain_event_") and self.pain_tracker:
                ps_content = item_data.get("content", {})
                ps_intensity = float(ps_content.get("current_intensity", 0.0)) if isinstance(ps_content, dict) else 0.0
                
                ## REVISION: For pain events, hint_score_for_combined_calc is specifically derived from its
                ## intensity and distraction factor (this was the original_hint_from_gather's source).
                ## We re-calculate and clamp it here to be explicit.
                base_pain_hint = ps_intensity * self.pain_attention_distraction_factor
                # This is the specific hint score for this pain event before suppression
                current_pain_specific_hint_clamped = max(0.0, min(1.0, base_pain_hint))
                
                hint_score_for_combined_calc = current_pain_specific_hint_clamped # Override general hint
                final_hint_for_ac_candidate_log = current_pain_specific_hint_clamped
                
                current_cycle_for_check = self._controller.cycle_count if self._controller else 0 

                # Now, check for suppression
                if self.pain_tracker.should_suppress_rumination(item_id, current_cycle_for_check):
                    suppressed_by_rumination = True # Set the flag that will be logged
                    suppression_value = self._config.get("pain_rumination_suppression_factor", 0.1)
                    original_hint_before_suppression_log = final_hint_for_ac_candidate_log # For logging
                    
                    # Apply suppression to the score that will be used
                    hint_score_for_combined_calc = final_hint_for_ac_candidate_log * suppression_value 
                    final_hint_for_ac_candidate_log = hint_score_for_combined_calc # Update log var too
                    
                    logger_attention_controller.info( 
                        f"AC_ALLOCATE_RUMINATE - Pain event '{item_id}' (Intensity: {ps_intensity:.2f}) SUPPRESSED. "
                        f"Hint score adjusted from {original_hint_before_suppression_log:.3f} to {final_hint_for_ac_candidate_log:.3f} (Factor: {suppression_value})."
                    )
            
            logger_attention_controller.debug(
                f"AC_CANDIDATE - ID='{item_id[:30]}...', "
                f"OrigHint(from_AC_gather)={original_hint_from_gather:.2f}, " 
                f"Recency={recency_score:.2f}, Rel={relevance_score_component:.2f}, "
                f"Novelty={novelty_score:.2f}, Surprise={surprise_score:.2f}, "
                f"FinalHintUsedForScoreCalc={final_hint_for_ac_candidate_log:.2f}, " 
                f"SuppressedByRumination={suppressed_by_rumination}"
            )

            ## REVISION: Ensure combined_score uses the potentially modified hint_score_for_combined_calc
            combined_score = ( 
                recency_score * self.recency_weight +
                hint_score_for_combined_calc * self.hint_weight + # Use the potentially suppressed hint     
                relevance_score_component * self.goal_relevance_weight +
                novelty_score * self.novelty_bonus_weight +      
                surprise_score * self.surprise_bonus_weight     
            )
            weighted_scores[item_id] = max(0.0, combined_score)
            
        final_weights = self._normalize_scores(weighted_scores)
        try:
            sorted_weights = sorted(final_weights.items(), key=lambda item: item[1], reverse=True)
            top_n = 5
            logger_attention_controller.debug(f"Top {top_n} attention items: {[(str(k)[:30]+'...', round(v, 4)) for k, v in sorted_weights[:top_n]]}")
        except Exception:
            logger_attention_controller.debug("Could not sort/log top N attention items.")
        return final_weights

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        num_items_to_normalize = len(scores)
        try:
            temp = self.softmax_temperature if self.softmax_temperature > 0 else 0.01
            temp_adjusted_scores = {k: s / temp for k, s in scores.items()}
            max_score = max(temp_adjusted_scores.values()) if temp_adjusted_scores else 0
            exp_scores = {k: math.exp(s - max_score) for k, s in temp_adjusted_scores.items()}
            sum_exp_scores = sum(exp_scores.values())
            if sum_exp_scores > 0:
                 return {k: v / sum_exp_scores for k, v in exp_scores.items()}
            else:
                 return {item_id: 1.0 / num_items_to_normalize for item_id in scores} if num_items_to_normalize > 0 else {}
        except OverflowError:
             logger_attention_controller.error("OverflowError during softmax. Falling back to sum normalization.")
             total_score = sum(scores.values()) 
             return {item_id: score / total_score for item_id, score in scores.items()} if total_score > 0 else \
                    ({item_id: 1.0 / num_items_to_normalize for item_id in scores} if num_items_to_normalize > 0 else {})

    async def process(self, input_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not input_state or "candidates" not in input_state:
            logger_attention_controller.debug("AttentionController process: No 'candidates' in input state.")
            return None

        candidates = input_state["candidates"]
        self._current_cycle_active_id = input_state.get("current_cycle_active_goal_id")
        
        last_gwm_content_for_novelty = input_state.get("last_gwm_content_for_novelty")
        self._last_prediction_error_for_surprise = input_state.get("last_prediction_error")

        if last_gwm_content_for_novelty and isinstance(last_gwm_content_for_novelty, dict):
            for item_id_gwm, content_gwm in last_gwm_content_for_novelty.items():
                try:
                    content_hash = hash(str(content_gwm))
                    if content_hash not in self.recent_workspace_content_hashes:
                        self.recent_workspace_content_hashes.append(content_hash)
                except Exception as e_hash:
                    logger_attention_controller.warning(f"Could not hash GWM content for novelty: {item_id_gwm} - {e_hash}", exc_info=False)

        # --- NEW: Handle Pain Rumination Tracking ---
        if self.pain_tracker:
            pain_ids_in_last_gwm = input_state.get("pain_ids_in_last_gwm", [])
            cycle_of_last_gwm = input_state.get("current_cycle_count", 0) - 1 

            if cycle_of_last_gwm >= 0 and isinstance(pain_ids_in_last_gwm, list):
                for pain_id_from_gwm in pain_ids_in_last_gwm:
                    if isinstance(pain_id_from_gwm, str):
                        self.pain_tracker.record_pain_in_gwm(pain_id_from_gwm, cycle_of_last_gwm)
            elif not isinstance(pain_ids_in_last_gwm, list):
                logger_attention_controller.warning("AC_PROCESS: 'pain_ids_in_last_gwm' was not a list.")
        # --- END NEW ---

        if isinstance(candidates, dict):
            attention_weights = await self.allocate_attention(candidates)
            return {"attention_weights": attention_weights}
        else:
            logger_attention_controller.warning("AttentionController process: 'candidates' is not a dict.")
            return None 

    async def reset(self) -> None:
        self._current_cycle_active_id = None
        self.recent_workspace_content_hashes.clear()
        if self.pain_tracker:
            self.pain_tracker._pain_gwm_history.clear()
            self.pain_tracker._last_recorded_in_gwm.clear()
            logger_attention_controller.info("PainEventTracker history cleared during AttentionController reset.")
        self._last_prediction_error_for_surprise = None
        logger_attention_controller.info("AttentionController reset.")

    async def get_status(self) -> Dict[str, Any]:
        status_dict = {
            "component": "AttentionController",
            "status": "operational",
            "recency_weight": self.recency_weight,
            "hint_weight": self.hint_weight,
            "goal_relevance_weight": self.goal_relevance_weight,
            "novelty_bonus_weight": self.novelty_bonus_weight, 
            "surprise_bonus_weight": self.surprise_bonus_weight, 
            "max_candidates": self.max_candidates,
            "softmax_temperature": self.softmax_temperature,
            "last_active_cycle_goal_id_processed": self._current_cycle_active_id,
            "novelty_hash_buffer_size": len(self.recent_workspace_content_hashes),
            "pain_tracker_status": self.pain_tracker.get_status_summary() if self.pain_tracker else "Not Initialized"
        }
        return status_dict

    async def shutdown(self) -> None:
        logger_attention_controller.info("AttentionController shutting down.")

# --- END OF FILE cognitive_modules/attention_controller.py ---