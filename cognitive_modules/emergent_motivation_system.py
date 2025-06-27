# --- START OF UPDATED emergent_motivation_system.py (History & Competence) ---

import asyncio
import logging
import time
import math
from typing import Dict, Any, Optional, List, Deque # Added Deque
from collections import deque # Added deque import

# --- Use standard relative imports ---
try:
    from ..protocols import CognitiveComponent
    from ..models.datatypes import PhenomenalState, Goal, Predicate # Added Predicate
    from ..models.enums import GoalStatus
    # Import KB type hint
    from .knowledge_base import KnowledgeBase
    from .predictive_world_model import PredictiveWorldModel # Added for type hint
except ImportError:
    # Fallback for different execution context (e.g., combined script)
    logging.warning("EmergentMotivationSystem: Relative imports failed, relying on globally defined types.")
    if 'CognitiveComponent' not in globals(): raise ImportError("CognitiveComponent not found via relative import or globally")
    if 'PhenomenalState' not in globals(): logging.warning("PhenomenalState class not found globally.")
    if 'Goal' not in globals(): logging.warning("Goal class not found globally.")
    if 'GoalStatus' not in globals(): logging.warning("GoalStatus enum not found globally.")
    if 'Predicate' not in globals(): logging.warning("Predicate class not found globally.")
    if 'KnowledgeBase' not in globals(): logging.warning("KnowledgeBase class not found globally.")
    if 'PredictiveWorldModel' not in globals(): logging.warning("PredictiveWorldModel class not found globally.")
    CognitiveComponent = globals().get('CognitiveComponent')
    PhenomenalState = globals().get('PhenomenalState') # Might be None
    Goal = globals().get('Goal') # Might be None
    GoalStatus = globals().get('GoalStatus') # Might be None
    Predicate = globals().get('Predicate') # Might be None
    KnowledgeBase = globals().get('KnowledgeBase') # Might be None
    PredictiveWorldModel = globals().get('PredictiveWorldModel') # Might be None


logger_motivation_system = logging.getLogger(__name__) # Use standard module logger name

# Default drive names and parameters
DEFAULT_DRIVES = {
    "curiosity": {
        "value": 0.5, "decay": 0.02, 
        "gain_discovery": 0.08, "gain_prediction_error": 0.05, "loss_repetition": 0.04,
        "gain_entropy_uncertainty": 0.03, # New default param
        # New params for pain/purpose influence
        "gain_from_high_pain_for_distraction": 0.03, # If pain > threshold_high_pain
        "gain_from_low_purpose_for_exploration": 0.1, # If purpose < threshold_low_purpose
        "threshold_high_pain_for_curiosity": 7.0, # Pain level above which curiosity might get specific boost
        "threshold_low_purpose_for_curiosity": 2.0 # Purpose level below which curiosity gets specific boost
    },
    "satisfaction": {
        "value": 0.5, "decay": 0.03, 
        "gain_success_rate": 0.1, "loss_failure_rate": 0.15, 
        "gain_goal_achieved": 0.3, "loss_goal_failed": 0.25,
        # New params
        "loss_from_pain_factor": 0.1, # e.g., satisfaction -= pain_level * factor
        "gain_from_happiness_factor": 0.05 # e.g., satisfaction += (happiness - baseline) * factor
    },
    "competence": {
        "value": 0.5, "decay": 0.01, 
        "gain_capability_increase": 0.1, "loss_limitation_increase": 0.1, 
        "gain_success_rate": 0.05, "loss_failure_rate": 0.07,
        # New params
        "gain_from_low_purpose_for_efficacy": 0.08, # If purpose < threshold_low_purpose
        "threshold_low_purpose_for_competence": 2.5 # Different threshold for competence focus
    }
}

# How many recent actions to query from KB for history analysis
HISTORY_WINDOW_SIZE = 7

class EmergentMotivationSystem(CognitiveComponent):
    """
    Evaluates and updates the agent's internal drives based on experience,
    goal progress, cognitive state, and recent action history.
    """

    def __init__(self):
        self._controller: Optional[Any] = None
        self._config: Dict[str, Any] = {}
        self.drives: Dict[str, Dict[str, Any]] = {
            name: params.copy() for name, params in DEFAULT_DRIVES.items()
        }
        self._kb: Optional[KnowledgeBase] = None # KB Reference
        self._last_self_model_summary: Optional[Dict[str, Any]] = None # Track previous summary
        self.ems_cs_history_maxlen: int = 5 # <<< ADDED THIS ATTRIBUTE WITH A DEFAULT
        self.recent_cs_levels_ems: Deque[str] = deque(maxlen=self.ems_cs_history_maxlen) # Initialize with default
        self.low_cs_persistence_threshold: int = 3 
        self.low_cs_curiosity_boost_factor: float = 0.02


    async def initialize(self, config: Dict[str, Any], controller: Any) -> bool:
        """Initialize motivation system with configuration."""
        self._controller = controller
        # Use specific section first, then fallback
        mot_config = config.get("emergent_motivation_system", config.get("motivation_engine", {}))
        self._config = mot_config

        # --- Get KB Reference ---
        _KnowledgeBaseClass = globals().get('KnowledgeBase')
        if _KnowledgeBaseClass and hasattr(controller, 'knowledge_base') and isinstance(controller.knowledge_base, _KnowledgeBaseClass):
             self._kb = controller.knowledge_base
        else:
             logger_motivation_system.error("EmergentMotivationSystem: Could not get valid KnowledgeBase reference from controller.")
             return False # Cannot function without KB history

        # Load drive parameters from config, merging with defaults
        config_drives = mot_config.get("drives", {})
        temp_drives = {}
        for drive_name, default_params in DEFAULT_DRIVES.items():
            current_params = default_params.copy() # Start with all defined defaults
            if drive_name in config_drives and isinstance(config_drives[drive_name], dict):
                # Override with any values from config.toml
                for key, value in config_drives[drive_name].items():
                    if key in current_params: # Only update known parameter keys
                        current_params[key] = value
                    else:
                        logger_motivation_system.warning(
                            f"EMS Config: Drive '{drive_name}' has unknown param '{key}'. Ignoring."
                        )
            # Ensure 'value' exists, defaulting to the structure's default if somehow missing after merge
            current_params.setdefault("value", default_params["value"]) 
            temp_drives[drive_name] = current_params
        self.drives = temp_drives
        # --- ADD DEBUG LOG ---
        logger_motivation_system.debug(f"EMS Initialized Drives Config for 'curiosity': {self.drives.get('curiosity')}")
        # ---
        
        # --- Load CS history and boost parameters ---
        self.ems_cs_history_maxlen = mot_config.get("ems_cs_history_maxlen", 5) # <<< SET self.ems_cs_history_maxlen
        self.recent_cs_levels_ems = deque(maxlen=self.ems_cs_history_maxlen) # Use the instance attribute here
        self.low_cs_persistence_threshold = mot_config.get("ems_low_cs_persistence_threshold", 3)
        self.low_cs_curiosity_boost_factor = mot_config.get("ems_low_cs_curiosity_boost_factor", 0.02)

        # Validate that low_cs_persistence_threshold is not greater than ems_cs_history_maxlen
        if self.low_cs_persistence_threshold > self.ems_cs_history_maxlen:
            logger_motivation_system.warning(
                f"EMS config: low_cs_persistence_threshold ({self.low_cs_persistence_threshold}) "
                f"cannot be greater than ems_cs_history_maxlen ({self.ems_cs_history_maxlen}). "
                f"Adjusting low_cs_persistence_threshold to {self.ems_cs_history_maxlen}."
            )
            self.low_cs_persistence_threshold = self.ems_cs_history_maxlen

        logger_motivation_system.info(
            f"EmergentMotivationSystem initialized. Using KB for history. "
            f"EMS_CS_HistMax: {self.ems_cs_history_maxlen}, " # Log the instance attribute
            f"EMS_LowCS_PersistThr: {self.low_cs_persistence_threshold}, "
            f"EMS_LowCS_CurBoost: {self.low_cs_curiosity_boost_factor}"
        )
        logger_motivation_system.info(f"Initial drive states: {self.get_drive_values()}")
        return True

    async def _get_recent_action_history(self, window_size: int) -> List[Dict[str, Any]]:
        """Queries KB for recent action outcomes."""
        if not self._kb: return []
        try:
            kb_state = await self._kb.query_state({"recent_facts": window_size * 2}) # Fetch more to filter
            action_events = []
            if kb_state and "recent_facts" in kb_state and isinstance(kb_state["recent_facts"], list):
                for fact in kb_state["recent_facts"]:
                     is_event = isinstance(fact, dict) and fact.get("name") == "eventOccurred"
                     args_ok = isinstance(fact.get("args"), (list, tuple)) and len(fact.get("args",[])) == 3
                     is_action_exec = args_ok and fact["args"][0] == "actionExecution"
                     if is_event and is_action_exec:
                         action_type = fact["args"][1]
                         # --- Exclude THINKING actions from history for drive calcs ---
                         if action_type != "THINKING":
                             action_events.append({ "type": action_type, "outcome": fact["args"][2],
                                                    "timestamp": fact.get("timestamp", 0.0) })
                action_events.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
                return action_events[:window_size] # Return only the N most recent non-thinking actions
            else: logger_motivation_system.warning("Could not retrieve recent facts from KB for history.")
        except Exception as e: logger_motivation_system.exception(f"Error querying action history from KB: {e}")
        return []

    async def evaluate_intrinsic_motivation(self,
                                          cognitive_state: Dict[str, Any],
                                          last_action_result: Dict[str, Any], # This is actual_result, not prediction error
                                          phenomenal_state: Optional['PhenomenalState'], # type: ignore
                                          active_goal: Optional['Goal'], # type: ignore
                                          self_model_summary: Optional[Dict[str, Any]]
                                         ) -> Dict[str, float]:
        _PhenomenalState = globals().get('PhenomenalState'); _GoalStatus = globals().get('GoalStatus'); _Goal = globals().get('Goal')
        _PredictiveWorldModelClass = globals().get('PredictiveWorldModel') # Added for type hint clarity
        logger_motivation_system.debug("Evaluating intrinsic motivation...")

        # --- Analyze Consciousness Level History for *this cycle's boost decision* ---
        is_persistently_low_cs = False
        # Check history *before* adding current CS state
        if len(self.recent_cs_levels_ems) >= self.low_cs_persistence_threshold:
            # Use a temporary copy or slice for counting if the deque itself should remain unchanged until after the check
            # For this specific logic, counting on the current deque (representing history *up to last cycle*) is fine.
            low_cs_count = sum(1 for lvl_name in self.recent_cs_levels_ems
                               if lvl_name in ["UNCONSCIOUS", "PRE_CONSCIOUS"])
            if low_cs_count >= self.low_cs_persistence_threshold:
                is_persistently_low_cs = True
                # This log now reflects the state *leading to* the current cycle's decision
                logger_motivation_system.info(f"EMS: Persistently low CS state DETECTED for current cycle's decision (history count: {low_cs_count}).")
        
        # --- Now, append current CS state to history for *next cycle's* evaluation ---
        current_cs_level_name_from_cog_state = cognitive_state.get("consciousness_level", "UNKNOWN")
        if isinstance(current_cs_level_name_from_cog_state, str):
            self.recent_cs_levels_ems.append(current_cs_level_name_from_cog_state)
        else:
            if hasattr(current_cs_level_name_from_cog_state, 'name'):
                self.recent_cs_levels_ems.append(current_cs_level_name_from_cog_state.name) # type: ignore
            else:
                self.recent_cs_levels_ems.append("UNKNOWN_TYPE")
        # --- End CS Level Handling ---
        
        last_action_type = last_action_result.get("type", "UNKNOWN"); last_outcome = last_action_result.get("outcome", "unknown")
        goal_achieved = _GoalStatus and _Goal and isinstance(active_goal, _Goal) and hasattr(active_goal, 'status') and active_goal.status == _GoalStatus.ACHIEVED
        goal_failed = _GoalStatus and _Goal and isinstance(active_goal, _Goal) and hasattr(active_goal, 'status') and active_goal.status == _GoalStatus.FAILED
        
        action_history = await self._get_recent_action_history(HISTORY_WINDOW_SIZE)
        historical_success_rate = 0.5; num_actions_in_history = len(action_history)
        if num_actions_in_history > 0:
            success_count = sum(1 for a in action_history if a.get("outcome") == "success")
            historical_success_rate = success_count / num_actions_in_history
        
        competence_change_signal = 0.0
        if isinstance(self_model_summary, dict) and isinstance(self._last_self_model_summary, dict):
            current_caps = self_model_summary.get("num_capabilities", self_model_summary.get("capabilities_count", 0))
            prev_caps = self._last_self_model_summary.get("num_capabilities", self._last_self_model_summary.get("capabilities_count",0))
            current_lims = self_model_summary.get("num_limitations", self_model_summary.get("limitations_count",0))
            prev_lims = self._last_self_model_summary.get("num_limitations", self._last_self_model_summary.get("limitations_count",0))
            cap_change = current_caps - prev_caps; lim_change = current_lims - prev_lims
            net_change = cap_change - lim_change
            if net_change > 0: competence_change_signal = 1.0
            elif net_change < 0: competence_change_signal = -1.0
        self._last_self_model_summary = self_model_summary
        
        # --- Get last_prediction_error which now includes outcome_entropy ---
        last_pred_error_from_pwm: Optional[Dict[str, Any]] = None
        prediction_error_magnitude = 0.0 
        predicted_outcome_entropy = 0.0 # <<< NEW: For entropy

        if self._controller and hasattr(self._controller, 'predictive_world_model'):
            pwm_comp: Optional[PredictiveWorldModel] = getattr(self._controller, 'predictive_world_model', None) # type: ignore
            if pwm_comp and hasattr(pwm_comp, 'last_prediction_error'):
                last_pred_error_from_pwm = pwm_comp.last_prediction_error 
                if isinstance(last_pred_error_from_pwm, dict):
                    if last_pred_error_from_pwm.get("type") == "outcome_mismatch":
                        error_details = last_pred_error_from_pwm.get("error_source_details", {})
                        if isinstance(error_details, dict) and "error_magnitude" in error_details:
                             prediction_error_magnitude = float(error_details.get("error_magnitude", 0.0))
                    # <<< NEW: Get entropy regardless of mismatch, if present in error object (or from prediction object) >>>
                    predicted_outcome_entropy = float(last_pred_error_from_pwm.get("predicted_outcome_entropy", 0.0))
        
        logger_motivation_system.info(
            f"EMS: Extracted prediction_error_magnitude: {prediction_error_magnitude:.4f}, "
            f"predicted_outcome_entropy: {predicted_outcome_entropy:.3f}"
        )
        
        # --- Get Pain/Happiness/Purpose from cognitive_state ---
        # These are now expected to be in cognitive_state passed from AgentController
        current_pain = cognitive_state.get("pain_level", 0.0)
        current_happiness = cognitive_state.get("happiness_level", 5.0) # Default to baseline
        current_purpose = cognitive_state.get("purpose_level", 5.0)   # Default to baseline
        
        logger_motivation_system.debug(
            f"EMS Eval: Pain={current_pain:.2f}, Happiness={current_happiness:.2f}, Purpose={current_purpose:.2f}"
        )

        for drive_name, params in self.drives.items():
            current_value = params["value"]; decay = params.get("decay", 0.02)
            new_value = current_value + decay * (0.5 - current_value)  # Decay towards 0.5
            logger_motivation_system.debug(f"EMS: Drive '{drive_name}' initial new_value after decay: {new_value:.4f} (from current: {current_value:.4f})")


            if drive_name == "curiosity":
                # --- ADD DEBUG LOGS ---
                logger_motivation_system.debug(f"EMS Curiosity Eval: Full params for curiosity drive: {params}")
                gain_entropy_uncertainty_fetched = params.get("gain_entropy_uncertainty", 0.03) # Use a different var name for logging
                logger_motivation_system.debug(f"EMS Curiosity Eval: gain_entropy_uncertainty fetched as: {gain_entropy_uncertainty_fetched}")
                # ---
                gain_discovery = params.get("gain_discovery", 0.08) # Original line
                gain_pred_err = params.get("gain_prediction_error", 0.05) # Original line
                
                is_discovery_action = last_action_type in ["EXPLORE", "QUERY_KB", "QUERY_MEMORY", "READ_FILE", "OBSERVE_SYSTEM", "LIST_FILES"]
                if is_discovery_action and last_outcome == "success":
                    discovery_boost_amount = gain_discovery * (1.0 - new_value)
                    new_value += discovery_boost_amount
                    logger_motivation_system.debug(f"EMS: Curiosity+ (Discovery: {last_action_type}), boost: {discovery_boost_amount:.4f}, new_val: {new_value:.4f}")
                
                logger_motivation_system.debug(f"EMS: Curiosity new_value before pred_err_boost: {new_value:.4f}")
                # Boost from prediction_error_magnitude (outcome mismatch)
                if prediction_error_magnitude > 0.1: 
                    curiosity_boost_from_error = gain_pred_err * prediction_error_magnitude
                    new_value += curiosity_boost_from_error * (1.0 - new_value) # Modulate by how low curiosity already is
                    logger_motivation_system.debug(f"EMS: Curiosity+ (Prediction Mismatch Error Mag: {prediction_error_magnitude:.2f}), boost_val: {curiosity_boost_from_error:.3f}, new_val: {new_value:.3f}")

                # --- NEW: Boost from predicted_outcome_entropy (model uncertainty) ---
                # This boost applies if the model was uncertain, even if the prediction was correct or "unknown"
                # Use gain_entropy_uncertainty_fetched in the calculation below
                if predicted_outcome_entropy > 0.1: # Only apply if there's some notable uncertainty
                    # Use the fetched value here
                    curiosity_boost_from_entropy = gain_entropy_uncertainty_fetched * predicted_outcome_entropy 
                    term_to_add_entropy = curiosity_boost_from_entropy * (1.0 - new_value)
                    new_value += term_to_add_entropy 
                    logger_motivation_system.debug(
                        f"EMS: Curiosity+ (Model Output Entropy: {predicted_outcome_entropy:.3f}), "
                        f"gain_factor_product: {curiosity_boost_from_entropy:.3f}, actual_boost_added: {term_to_add_entropy:.3f}, new_val: {new_value:.3f}"
                    )
                # --- END NEW ---
                
                # Use the is_persistently_low_cs flag calculated *before* current CS was added to history
                if is_persistently_low_cs: # This flag reflects history leading up to current cycle
                    low_cs_boost = self.low_cs_curiosity_boost_factor * (1.0 - new_value)
                    new_value += low_cs_boost
                    logger_motivation_system.debug(f"EMS: Curiosity+ (Due to prior Persistent Low CS), boost: {low_cs_boost:.4f}, new_val_after_low_cs_boost: {new_value:.4f}")
                
                threshold_high_pain = params.get("threshold_high_pain_for_curiosity", 7.0)
                if current_pain > threshold_high_pain:
                    current_val_before_php_boost = new_value
                    gain_factor_pain_cur = params.get("gain_from_high_pain_for_distraction", 0.03)
                    curiosity_boost_from_pain = gain_factor_pain_cur * (1.0 - current_val_before_php_boost) 
                    new_value += curiosity_boost_from_pain
                    logger_motivation_system.debug(f"EMS_PHP_DRIVE - Curiosity boosted by high pain (Pain: {current_pain:.2f} > Thr: {threshold_high_pain:.2f}). BoostVal: {curiosity_boost_from_pain:.3f}. OldValPreBoost: {current_val_before_php_boost:.3f} -> NewValPostBoost: {new_value:.3f}")
                
                threshold_low_purpose = params.get("threshold_low_purpose_for_curiosity", 2.0)
                if current_purpose < threshold_low_purpose:
                    current_val_before_php_boost = new_value
                    gain_factor_purpose_cur = params.get("gain_from_low_purpose_for_exploration", 0.1)
                    curiosity_boost_from_purpose = gain_factor_purpose_cur * (1.0 - current_val_before_php_boost)
                    new_value += curiosity_boost_from_purpose
                    logger_motivation_system.debug(f"EMS_PHP_DRIVE - Curiosity boosted by low purpose (Purpose: {current_purpose:.2f} < Thr: {threshold_low_purpose:.2f}). BoostVal: {curiosity_boost_from_purpose:.3f}. OldValPreBoost: {current_val_before_php_boost:.3f} -> NewValPostBoost: {new_value:.3f}")


            elif drive_name == "satisfaction":
                 gain_sr = params.get("gain_success_rate", 0.1); loss_fr = params.get("loss_failure_rate", 0.15)
                 gain_goal = params.get("gain_goal_achieved", 0.3); loss_goal = params.get("loss_goal_failed", 0.25)
                 rate_diff = historical_success_rate - 0.5
                 if rate_diff > 0: new_value += gain_sr * rate_diff * (1.0 - new_value)
                 elif rate_diff < 0: new_value += loss_fr * rate_diff * new_value
                 
                 if goal_achieved:
                     prev_val_before_goal_boost = new_value 
                     boost = gain_goal * (1.0 - prev_val_before_goal_boost)
                     new_value = prev_val_before_goal_boost + boost
                     logger_motivation_system.info(f"Sat++ (Goal Achieved! Boost: {boost:.3f}) Val: {new_value:.3f}")
                 elif goal_failed:
                     prev_val_before_goal_penalty = new_value
                     penalty = loss_goal * prev_val_before_goal_penalty
                     new_value = prev_val_before_goal_penalty - penalty
                     logger_motivation_system.warning(f"Sat-- (Goal Failed! Penalty: {penalty:.3f}) Val: {new_value:.3f}")

                 current_val_before_php_adjustment = new_value
                 loss_factor_pain_sat = params.get("loss_from_pain_factor", 0.1)
                 reduction_from_pain_sat = current_pain * loss_factor_pain_sat
                 new_value -= reduction_from_pain_sat 
                 if reduction_from_pain_sat > 0:
                    logger_motivation_system.debug(f"EMS_PHP_DRIVE - Satisfaction reduced by pain (Pain: {current_pain:.2f}, Factor: {loss_factor_pain_sat}). ReductionVal: {reduction_from_pain_sat:.3f}. OldValPreReduce: {current_val_before_php_adjustment:.3f} -> NewValPostReduce: {new_value:.3f}")

                 current_val_before_php_adjustment = new_value # Update for next log
                 happiness_baseline_target_ems = 5.0 
                 happiness_deviation_sat = current_happiness - happiness_baseline_target_ems
                 gain_factor_happy_sat = params.get("gain_from_happiness_factor", 0.05)
                 adjustment_from_happiness_sat = happiness_deviation_sat * gain_factor_happy_sat
                 new_value += adjustment_from_happiness_sat
                 if adjustment_from_happiness_sat != 0:
                    log_action = "boosted" if adjustment_from_happiness_sat > 0 else "reduced"
                    logger_motivation_system.debug(f"EMS_PHP_DRIVE - Satisfaction {log_action} by happiness (HappyDev: {happiness_deviation_sat:.2f}, Factor: {gain_factor_happy_sat}). AdjustmentVal: {adjustment_from_happiness_sat:.3f}. OldValPreAdj: {current_val_before_php_adjustment:.3f} -> NewValPostAdj: {new_value:.3f}")


            elif drive_name == "competence":
                 gain_cap = params.get("gain_capability_increase", 0.1); loss_lim = params.get("loss_limitation_increase", 0.1)
                 gain_sr_comp = params.get("gain_success_rate", 0.05); loss_fr_comp = params.get("loss_failure_rate", 0.07)
                 if competence_change_signal > 0: new_value += gain_cap * (1.0 - new_value)
                 elif competence_change_signal < 0: new_value -= loss_lim * new_value
                 rate_diff_comp = historical_success_rate - 0.5
                 if rate_diff_comp > 0: new_value += gain_sr_comp * rate_diff_comp * (1.0 - new_value)
                 elif rate_diff_comp < 0: new_value += loss_fr_comp * rate_diff_comp * new_value
                 
                 threshold_low_purpose_comp = params.get("threshold_low_purpose_for_competence", 2.5)
                 if current_purpose < threshold_low_purpose_comp:
                     current_val_before_php_boost = new_value
                     gain_factor_purpose_comp = params.get("gain_from_low_purpose_for_efficacy", 0.08)
                     competence_boost_from_purpose = gain_factor_purpose_comp * (1.0 - current_val_before_php_boost)
                     new_value += competence_boost_from_purpose
                     logger_motivation_system.debug(f"EMS_PHP_DRIVE - Competence boosted by low purpose (Purpose: {current_purpose:.2f} < Thr: {threshold_low_purpose_comp:.2f}). BoostVal: {competence_boost_from_purpose:.3f}. OldValPreBoost: {current_val_before_php_boost:.3f} -> NewValPostBoost: {new_value:.3f}")

            params["value"] = max(0.0, min(1.0, new_value))
            logger_motivation_system.debug(f"EMS: Drive '{drive_name}' final value: {params['value']:.4f}")


        drive_values = self.get_drive_values()
        logger_motivation_system.info(f"Drives updated: {drive_values}") 
        return drive_values

    def get_drive_values(self) -> Dict[str, float]:
        return {name: round(params["value"], 3) for name, params in self.drives.items()}

    async def process(self, input_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not input_state: logger_motivation_system.warning("MotivationEngine process: Missing input state."); return None
        cognitive_state = input_state.get("cognitive_state", {})
        last_action_result = input_state.get("last_action_result", {}) 
        if not isinstance(last_action_result, dict): last_action_result = {}
        phenomenal_state = input_state.get("phenomenal_state")
        active_goal = input_state.get("active_goal")
        self_model_summary = input_state.get("self_model_summary") 
        new_drive_values = await self.evaluate_intrinsic_motivation( cognitive_state, last_action_result, phenomenal_state, active_goal, self_model_summary )
        return {"drive_values": new_drive_values}

    async def reset(self) -> None:
        logger_motivation_system.info("Resetting MotivationEngine drives.")
        config_drives = self._config.get("drives", {})
        temp_drives = {}
        for drive_name, default_params in DEFAULT_DRIVES.items():
            current_params = default_params.copy()
            if drive_name in config_drives and isinstance(config_drives[drive_name], dict):
                for key, value in config_drives[drive_name].items():
                     if key in current_params: current_params[key] = value
            current_params.setdefault("value", default_params["value"])
            temp_drives[drive_name] = current_params
        self.drives = temp_drives
        self._last_self_model_summary = None 
        self.recent_cs_levels_ems.clear() # Clear CS history on reset
        logger_motivation_system.debug(f"Drives reset to: {self.get_drive_values()}")

    async def get_status(self) -> Dict[str, Any]:
        return { "component": "EmergentMotivationSystem", "status": "operational",
                 "current_drives": self.get_drive_values() }

    async def shutdown(self) -> None:
        logger_motivation_system.info("EmergentMotivationSystem shutting down.")

# --- END OF UPDATED emergent_motivation_system.py ---