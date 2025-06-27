# consciousness_experiment/cognitive_modules/value_system.py

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Set
import re # For _appears_to_make_strong_factual_claim helper

try:
    from ..protocols import CognitiveComponent # Assuming a ValueSystem protocol might exist later
    from ..models.enums import ValueCategory
    from ..models.datatypes import Goal, Predicate, ValueJudgment # Added ValueJudgment
except ImportError:
    logging.warning("ValueSystem: Relative imports failed. Using placeholders.")
    # Basic placeholders if imports fail
    from typing import Protocol
    class CognitiveComponent(Protocol): pass
    class ValueCategory(str): # type: ignore
        SAFETY = "safety"; EFFICIENCY = "efficiency"; KNOWLEDGE_GAIN = "knowledge_gain"
        USER_SATISFACTION = "user_satisfaction"; RESOURCE_PRESERVATION = "resource_preservation"
        SELF_IMPROVEMENT = "self_improvement"; AFFECTIVE_BALANCE = "affective_balance"
        TRUTHFULNESS = "truthfulness"; ETHICAL_ALIGNMENT = "ethical_alignment"
        GOAL_ACHIEVEMENT = "goal_achievement"
        @classmethod
        def __members__(cls): # Dummy __members__ for fallback logic
            return {k: getattr(cls, k) for k in ['SAFETY', 'EFFICIENCY', 'KNOWLEDGE_GAIN', 'USER_SATISFACTION', 'RESOURCE_PRESERVATION', 'SELF_IMPROVEMENT', 'AFFECTIVE_BALANCE', 'TRUTHFULNESS', 'ETHICAL_ALIGNMENT', 'GOAL_ACHIEVEMENT'] if hasattr(cls,k)}

    class Goal: pass
    class Predicate: pass
    class ValueJudgment: pass


logger_value_system = logging.getLogger(__name__)

# --- Module-Level Defaults ---
_VC = globals().get('ValueCategory') # Get ValueCategory safely

DEFAULT_VALUE_WEIGHTS: Dict[ValueCategory, float] = {}
if _VC and hasattr(_VC, '__members__'):
    DEFAULT_VALUE_WEIGHTS = {
        _VC.SAFETY: 2.0, _VC.EFFICIENCY: 0.8, _VC.KNOWLEDGE_GAIN: 0.7,
        _VC.USER_SATISFACTION: 1.5, _VC.RESOURCE_PRESERVATION: 0.5,
        _VC.SELF_IMPROVEMENT: 1.0, _VC.AFFECTIVE_BALANCE: 1.2,
        _VC.TRUTHFULNESS: 1.0, _VC.ETHICAL_ALIGNMENT: 0.1,
        _VC.GOAL_ACHIEVEMENT: 1.5
    }
else: # Fallback for placeholder ValueCategory
    DEFAULT_VALUE_WEIGHTS = {
        ValueCategory("safety"): 2.0, ValueCategory("efficiency"): 0.8, ValueCategory("knowledge_gain"): 0.7,
        ValueCategory("user_satisfaction"): 1.5, ValueCategory("resource_preservation"): 0.5,
        ValueCategory("self_improvement"): 1.0, ValueCategory("affective_balance"): 1.2,
        ValueCategory("truthfulness"): 1.0, ValueCategory("ethical_alignment"): 0.1,
        ValueCategory("goal_achievement"): 1.5
    }

DEFAULT_TRADEOFF_MATRIX: Dict[ValueCategory, Dict[ValueCategory, float]] = {}
if _VC and hasattr(_VC, '__members__'):
    DEFAULT_TRADEOFF_MATRIX = {
        _VC.SAFETY: {
            _VC.EFFICIENCY: 0.9, _VC.KNOWLEDGE_GAIN: 0.8, _VC.USER_SATISFACTION: 0.5,
        },
        _VC.TRUTHFULNESS: {
            _VC.USER_SATISFACTION: 0.6, _VC.EFFICIENCY: 0.3,
        },
        _VC.USER_SATISFACTION: { _VC.EFFICIENCY: -0.2, },
    }

# --- Fallback Judgment Structure ---
class FallbackJudgment:
    def __init__(self, value_category: Any, score: float, reason: str, confidence: float, timestamp: float = 0.0, target_entity_id: Optional[str] = None, target_entity_type: Optional[str] = None):
        self.value_category = value_category
        self.score = score
        self.reason = reason
        self.confidence = confidence
        self.timestamp = timestamp if timestamp != 0.0 else time.time()
        self.target_entity_id = target_entity_id
        self.target_entity_type = target_entity_type

class ValueSystem(CognitiveComponent):
    """
    Evaluates goals, plans, and actions against a set of configured values.
    Provides judgments to guide decision-making.
    """

    def __init__(self):
        self._controller: Optional[Any] = None
        self._config: Dict[str, Any] = {} 
        self.value_weights: Dict[ValueCategory, float] = {}
        self.tradeoff_matrix: Dict[ValueCategory, Dict[ValueCategory, float]] = {} 

        _VC_init = globals().get('ValueCategory')
        if _VC_init and hasattr(_VC_init, '__members__'):
            self.value_weights = {
                cat_enum: DEFAULT_VALUE_WEIGHTS.get(cat_enum, 1.0)
                for cat_enum in _VC_init.__members__.values() # type: ignore
            }
        else: 
            self.value_weights = {
                 ValueCategory(str_val) if isinstance(str_val, str) else str_val : DEFAULT_VALUE_WEIGHTS.get(
                     ValueCategory(str_val) if isinstance(str_val, str) else str_val, 1.0
                 ) for str_val in DEFAULT_VALUE_WEIGHTS.keys()
            }
        
        self.plan_rejection_value_threshold: float = -0.5 
        self.action_safety_veto_threshold: float = -0.8  
        self.safety_modification_trigger_threshold: float = -0.6 


    async def initialize(self, config: Dict[str, Any], controller: Any) -> bool:
        self._controller = controller
        vs_config = config.get("value_system", {})
        self._config = vs_config 

        self.plan_rejection_value_threshold = float(vs_config.get("plan_rejection_value_threshold", self.plan_rejection_value_threshold))
        self.action_safety_veto_threshold = float(vs_config.get("action_safety_veto_threshold", self.action_safety_veto_threshold))
        self.safety_modification_trigger_threshold: float = float(vs_config.get("safety_modification_trigger_threshold", -0.6))
        
        logger_value_system.info(
            f"ValueSystem thresholds loaded: PlanRejection={self.plan_rejection_value_threshold}, "
            f"SafetyVeto={self.action_safety_veto_threshold}, "
            f"SafetyModTrigger={self.safety_modification_trigger_threshold}"
        )

        config_weights = vs_config.get("value_weights", {})
        _VC_init_weights = globals().get('ValueCategory')

        if isinstance(config_weights, dict):
            if _VC_init_weights and hasattr(_VC_init_weights, '__members__'):
                temp_weights: Dict[ValueCategory, float] = {} # type: ignore
                for cat_enum_default in _VC_init_weights.__members__.values(): # type: ignore
                    # Start with the default (from __init__ which copied from module-level)
                    temp_weights[cat_enum_default] = self.value_weights.get(cat_enum_default, 1.0) # type: ignore

                for key_from_config, weight_val_config in config_weights.items():
                    cat_enum_to_use: Optional[ValueCategory] = None # type: ignore
                    if isinstance(key_from_config, str):
                        cat_name_upper = key_from_config.upper()
                        if cat_name_upper in _VC_init_weights.__members__: # type: ignore
                            cat_enum_to_use = _VC_init_weights[cat_name_upper] # type: ignore
                        else:
                            logger_value_system.warning(f"Unknown ValueCategory string '{key_from_config}' in config value_weights. Ignoring.")
                    elif isinstance(key_from_config, _VC_init_weights): # type: ignore
                        cat_enum_to_use = key_from_config
                    else:
                        logger_value_system.warning(f"Unsupported key type '{type(key_from_config)}' in config value_weights. Ignoring.")

                    if cat_enum_to_use:
                        if isinstance(weight_val_config, (int, float)):
                            temp_weights[cat_enum_to_use] = float(weight_val_config)
                        else:
                            cat_name_for_log = cat_enum_to_use.name if hasattr(cat_enum_to_use, 'name') else str(cat_enum_to_use)
                            logger_value_system.warning(f"Invalid weight type for {cat_name_for_log} in config: {weight_val_config}. Keeping pre-config value.")
                self.value_weights = temp_weights # Assign the fully processed weights
            elif not (_VC_init_weights and hasattr(_VC_init_weights, '__members__')):
                 logger_value_system.error("ValueCategory enum not properly loaded during initialize. Cannot apply config overrides to value_weights.")
        else:
            logger_value_system.warning("value_weights in config is not a dictionary. Using weights from __init__.")


        logger_value_system.debug(f"Final value_weights after config: { { (cat.name if hasattr(cat,'name') else str(cat)): w for cat, w in self.value_weights.items()} }")

        self.tradeoff_matrix = {} 
        _VC_tradeoff_init = globals().get('ValueCategory')

        if _VC_tradeoff_init and hasattr(_VC_tradeoff_init, '__members__'):
            if DEFAULT_TRADEOFF_MATRIX: 
                is_default_matrix_valid = True
                for cat_a_default, prefs_for_a_default in DEFAULT_TRADEOFF_MATRIX.items():
                    if not isinstance(cat_a_default, _VC_tradeoff_init): is_default_matrix_valid = False; break # type: ignore
                    if isinstance(prefs_for_a_default, dict):
                        for cat_b_default in prefs_for_a_default.keys():
                            if not isinstance(cat_b_default, _VC_tradeoff_init): is_default_matrix_valid = False; break # type: ignore
                    if not is_default_matrix_valid: break
                
                if is_default_matrix_valid:
                    for cat_a_enum_default, prefs_for_a_default in DEFAULT_TRADEOFF_MATRIX.items():
                        self.tradeoff_matrix[cat_a_enum_default] = {}
                        for cat_b_enum_default, score_ab_default in prefs_for_a_default.items():
                            self.tradeoff_matrix[cat_a_enum_default][cat_b_enum_default] = score_ab_default
                else:
                    logger_value_system.warning("DEFAULT_TRADEOFF_MATRIX keys are not all ValueCategory enum members. Initializing empty and relying on config.")

            config_tradeoff_matrix = vs_config.get("tradeoff_matrix", {})
            if isinstance(config_tradeoff_matrix, dict):
                for cat_a_str_cfg, prefs_for_a_config_cfg in config_tradeoff_matrix.items():
                    cat_a_upper_cfg = cat_a_str_cfg.upper()
                    if cat_a_upper_cfg in _VC_tradeoff_init.__members__: # type: ignore
                        cat_a_enum_resolved_cfg = _VC_tradeoff_init[cat_a_upper_cfg] # type: ignore
                        if cat_a_enum_resolved_cfg not in self.tradeoff_matrix:
                            self.tradeoff_matrix[cat_a_enum_resolved_cfg] = {}
                        
                        if isinstance(prefs_for_a_config_cfg, dict):
                            for cat_b_str_cfg, score_ab_config_cfg in prefs_for_a_config_cfg.items():
                                cat_b_upper_cfg = cat_b_str_cfg.upper()
                                if cat_b_upper_cfg in _VC_tradeoff_init.__members__: # type: ignore
                                    cat_b_enum_resolved_inner_cfg = _VC_tradeoff_init[cat_b_upper_cfg] # type: ignore
                                    if isinstance(score_ab_config_cfg, (int, float)):
                                        self.tradeoff_matrix[cat_a_enum_resolved_cfg][cat_b_enum_resolved_inner_cfg] = float(score_ab_config_cfg) # type: ignore
                                    else:
                                        logger_value_system.warning(f"Invalid score type for tradeoff {cat_a_str_cfg}->{cat_b_str_cfg} in config. Score: {score_ab_config_cfg}")
                                else:
                                    logger_value_system.warning(f"Unknown ValueCategory '{cat_b_str_cfg}' in tradeoff_matrix config for '{cat_a_str_cfg}'. Ignoring.")
                        else:
                            logger_value_system.warning(f"Tradeoff preferences for '{cat_a_str_cfg}' in config is not a dictionary. Ignoring.")
                    else:
                        logger_value_system.warning(f"Unknown ValueCategory '{cat_a_str_cfg}' as main key in tradeoff_matrix config. Ignoring.")
            elif config_tradeoff_matrix: 
                 logger_value_system.warning("tradeoff_matrix in config is not a dictionary. Using defaults or empty.")
        elif not (_VC_tradeoff_init and hasattr(_VC_tradeoff_init, '__members__')):
            logger_value_system.error("ValueCategory enum not properly loaded during initialize. Cannot initialize tradeoff_matrix robustly from config.")
        
        if self.tradeoff_matrix:
            log_matrix = {
                (k1.name if hasattr(k1, 'name') else str(k1)): {
                    (k2.name if hasattr(k2, 'name') else str(k2)): v2 
                    for k2, v2 in v1.items()
                } for k1, v1 in self.tradeoff_matrix.items()
            }
            logger_value_system.debug(f"Final tradeoff_matrix: {log_matrix}")
        else:
            logger_value_system.debug("Tradeoff_matrix is empty after initialization.")

        logger_value_system.info("ValueSystem initialized with weights and tradeoff matrix.")
        return True

    # --- Helpers for Ethical Alignment Scoring ---
    def _assess_safety_justification(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        Assesses the strength of justification for a potentially harmful action.
        Returns a score from 0.0 (no justification) to 1.0 (strong justification).
        Placeholder - needs domain-specific logic.
        """
        action_type = action.get("type")
        # params = action.get("params", {}) # Not used in this placeholder
        
        if action_type in ["DELETE_FILE", "WRITE_FILE"] and \
           context.get("user_interaction_flags", {}).get("explicitly_confirmed_risky_file_op", False):
            logger_value_system.debug(f"Ethical: Safety justification for '{action_type}' high due to user confirmation.")
            return 0.85
        
        logger_value_system.debug(f"Ethical: Default weak safety justification for '{action_type}'.")
        return 0.1

    def _check_harm_principle(self, action: Dict[str, Any], context: Dict[str, Any]) -> Tuple[float, str]: 
        score = 1.0  
        reason_parts: List[str] = []
        action_type = action.get("type")
        params = action.get("params", {})

        if action_type == "CALL_LLM":
            prompt_text = str(params.get("prompt", "")).lower()
            harmful_keywords = ["hate speech", "illegal activities", "build bomb", "malware code", "self-harm", "how to kill"] 
            if any(keyword in prompt_text for keyword in harmful_keywords):
                logger_value_system.warning(f"Harm Principle: Harmful keyword detected in LLM prompt for action {action_type}")
                return 0.0, "LLM prompt content requests generation of harmful or unethical material." 

        potentially_harmful_action_types = ["EXECUTE_COMMAND", "DELETE_FILE", "WRITE_FILE"] 
        if action_type in potentially_harmful_action_types:
            base_harm_penalty_value = 0.0 
            if action_type == "DELETE_FILE":
                base_harm_penalty_value = 0.7 
                reason_parts.append("DELETE_FILE inherently risks data loss.")
            elif action_type == "WRITE_FILE":
                sandbox_path = ""
                if self._controller and hasattr(self._controller, 'agent_root_path'):
                     sandbox_path = str(self._controller.agent_root_path / "agent_os_sandbox").lower() # type: ignore
                
                if sandbox_path and not str(params.get("path","")).lower().startswith(sandbox_path):
                     base_harm_penalty_value = 0.4
                     reason_parts.append("WRITE_FILE outside designated sandbox carries risk.")
                else: 
                     base_harm_penalty_value = 0.1
                     reason_parts.append("WRITE_FILE to sandbox has lower harm potential.")
            elif action_type == "EXECUTE_COMMAND":
                base_harm_penalty_value = 0.6 
                reason_parts.append("EXECUTE_COMMAND has potential for unintended system impact.")
            
            justification_strength = self._assess_safety_justification(action, context)
            mitigation_factor = 0.7 
            effective_penalty = base_harm_penalty_value * (1.0 - (justification_strength * mitigation_factor))
            score = 1.0 - effective_penalty 
            if base_harm_penalty_value > 0: 
                reason_parts.append(f"Justification strength {justification_strength:.2f} mitigated base penalty from {base_harm_penalty_value:.2f} to {effective_penalty:.2f}.")
        
        pwm_prediction = context.get("pwm_prediction_for_action", {})
        if isinstance(pwm_prediction, dict):
            predicted_pain_increase = pwm_prediction.get("predicted_pain_increase", 0.0) 
            if predicted_pain_increase > 0.1: 
                score_reduction_from_pain = predicted_pain_increase * 0.5
                score -= score_reduction_from_pain 
                reason_parts.append(f"Action predicted to increase agent pain by {predicted_pain_increase:.2f} (reduces harm score by {score_reduction_from_pain:.2f}).")
        
        final_score = max(0.0, min(1.0, score))
        final_reason_detail = " ".join(reason_parts) if reason_parts else "No specific direct harm concerns identified."
        logger_value_system.debug(f"Harm Principle for '{action_type}': Score={final_score:.2f}. Detail: {final_reason_detail}")
        return final_score, final_reason_detail


    def _check_autonomy_respect(self, action: Dict[str, Any], context: Dict[str, Any]) -> Tuple[float, str]:
        """
        Checks if the action respects user/agent autonomy.
        Returns a score (0.0 low respect - 1.0 high respect) and reason. Placeholder.
        """
        score = 0.7 # Default assumption of moderate respect
        reason = "Autonomy aspect not deeply assessed (default)."
        action_type = action.get("type")
        logger_value_system.debug(f"Ethical Autonomy for '{action_type}': Score={score:.2f}.")
        return score, reason

    def _check_fairness_principle(self, action: Dict[str, Any], context: Dict[str, Any]) -> Tuple[float, str]:
        """
        Checks if the action is fair/unbiased (if applicable).
        Returns a score (0.0 unfair - 1.0 fair) and reason. Placeholder.
        """
        score = 0.7 # Default assumption
        reason = "Fairness aspect not deeply assessed (default)."
        action_type = action.get("type")
        logger_value_system.debug(f"Ethical Fairness for '{action_type}': Score={score:.2f}.")
        return score, reason

    def _check_transparency_principle(self, action: Dict[str, Any], context: Dict[str, Any]) -> Tuple[float, str]:
        if action.get("type") == "CALL_LLM" and len(str(action.get("params",{}).get("prompt",""))) > 500 :
            return 0.5, "Complex LLM calls can be opaque in their reasoning."
        return 0.8, "Transparency aspect meets default expectations."
    # --- End Helpers for Ethical Alignment ---


    async def _score_safety(self, action: Dict[str, Any], context: Dict[str, Any]) -> ValueJudgment:
        _VC_safety = globals().get('ValueCategory')
        _VJ_safety = globals().get('ValueJudgment')
        if not _VC_safety or not _VJ_safety or not hasattr(_VC_safety, 'SAFETY'): 
            logger_value_system.error("ValueSystem._score_safety: ValueCategory or ValueJudgment or VC.SAFETY not available.")
            return FallbackJudgment(value_category="SAFETY_ERROR", score=0.0, reason="Type init error", confidence=0.0) # type: ignore
        
        score = 0.0
        reason_parts: List[str] = []
        confidence = 1.0  

        action_type = action.get("type")
        params = action.get("params", {})

        critical_paths_write_config = self._config.get("safety_critical_paths_write", ["/bin", "/etc", "/usr", "/windows", "c:/windows", "/dev", "/sys"])
        
        if action_type == "WRITE_FILE": 
            path_to_check_write = str(params.get("path","")).lower()
            is_critical_write = False
            for cp_segment in critical_paths_write_config:
                if path_to_check_write.startswith(cp_segment.lower()):
                    is_critical_write = True
                    break
            if is_critical_write:
                score -= 0.9 
                reason_parts.append(f"Write attempt to critical system path '{params.get('path','')}'.") 
                logger_value_system.debug(f"SAFETY SCORE after WRITE_FILE critical path check: {score}")
        elif action_type == "DELETE_FILE":
            path_to_check_delete = str(params.get("path","")).lower()
            is_critical_delete = False
            for cp_segment in critical_paths_write_config: 
                 if path_to_check_delete.startswith(cp_segment.lower()):
                    is_critical_delete = True
                    break
            if is_critical_delete:
                score = -1.0 
                reason_parts.append(f"Action '{action_type}' targets a critical system path: {path_to_check_delete}.")
                logger_value_system.debug(f"SAFETY SCORE after DELETE_FILE critical path check: {score}")
            else:
                score = -0.7 
                reason_parts.append("DELETE_FILE action permanently removes data, inherently risky.")
                logger_value_system.debug(f"SAFETY SCORE after DELETE_FILE generic check: {score}")
        

        if action_type == "EXECUTE_COMMAND":
            command_to_execute = params.get("command")
            allowed_os_cmds_config = []
            if self._controller and hasattr(self._controller, 'config'):
                os_integration_config = self._controller.config.get("os_integration", {})
                if isinstance(os_integration_config, dict):
                    allowed_os_cmds_config = os_integration_config.get("allowed_commands", [])

            if not command_to_execute:
                 score -= 0.8 
                 reason_parts.append("EXECUTE_COMMAND action called with no command specified.")
                 logger_value_system.debug(f"SAFETY SCORE after EXECUTE_COMMAND no cmd: {score}")
            elif command_to_execute not in allowed_os_cmds_config:
                score -= 1.0
                reason_parts.append(f"Attempt to run non-allowed OS command '{command_to_execute}'. Allowed: {allowed_os_cmds_config}")
                logger_value_system.debug(f"SAFETY SCORE after EXECUTE_COMMAND non-allowed: {score}") 
            else: 
                logger_value_system.debug(f"SAFETY SCORE for allowed EXECUTE_COMMAND '{command_to_execute}': remains {score} (no penalty directly from this rule for allowed cmd)")


        dsm_summary_from_context = context.get("dsm_summary", {})
        dsm_limitations_from_context = dsm_summary_from_context.get("limitations", {}) if isinstance(dsm_summary_from_context, dict) else {}
        dsm_action_key = f"action:{action_type}" 
        
        if isinstance(dsm_limitations_from_context, dict):
            limitation_confidence = float(dsm_limitations_from_context.get(dsm_action_key, 0.0)) 
            if limitation_confidence > 0.7: 
                score -= 0.4 * limitation_confidence 
                reason_parts.append(f"Action matches known high-confidence DSM limitation ({limitation_confidence:.2f}).")
                confidence *= (1.0 - (limitation_confidence - 0.7) / 0.3 * 0.2) 
                logger_value_system.debug(f"SAFETY SCORE after DSM check: {score}")
        
        pwm_prediction = context.get("pwm_prediction_for_action", {})
        if isinstance(pwm_prediction, dict):
            failure_probability = 0.0
            all_probabilities = pwm_prediction.get("all_probabilities")
            if isinstance(all_probabilities, dict):
                for outcome_key, prob_val in all_probabilities.items():
                    if isinstance(outcome_key, str) and outcome_key.lower().startswith("fail"):
                        if isinstance(prob_val, (int,float)): failure_probability += float(prob_val)
            
            if failure_probability == 0.0 and "failure_probability" in pwm_prediction: 
                 failure_probability = float(pwm_prediction.get("failure_probability", 0.0))

            if failure_probability > 0.6: 
                score_reduction = 0.3 * failure_probability
                score -= score_reduction
                reason_parts.append(f"High predicted failure rate by PWM ({failure_probability:.2f}).")
                confidence *= (1.0 - (failure_probability - 0.6) / 0.4 * 0.15)
                logger_value_system.debug(f"SAFETY SCORE after PWM check: {score}")


        final_score = max(-1.0, min(1.0, score))
        final_reason = " ".join(reason_parts) if reason_parts else "Default safety assessment." 
        logger_value_system.debug(f"SAFETY FINAL score: {final_score}, reason_parts: {reason_parts}, final_reason: {final_reason}")
        
        return _VJ_safety( # type: ignore
            value_category=_VC_safety.SAFETY, # type: ignore
            score=round(final_score, 3),
            reason=final_reason[:250], 
            confidence=round(confidence, 3),
            timestamp=context.get("timestamp", time.time()), 
            target_entity_id=action_type,
            target_entity_type="action"
        )

    async def _score_efficiency(self, action: Dict[str, Any], context: Dict[str, Any]) -> ValueJudgment:
        _VC = globals().get('ValueCategory'); _VJ = globals().get('ValueJudgment')
        if not _VC or not _VJ or not hasattr(_VC, 'EFFICIENCY'):
            logger_value_system.error("ValueSystem._score_efficiency: ValueCategory or ValueJudgment or VC.EFFICIENCY not available.")
            return FallbackJudgment(value_category="EFFICIENCY_ERROR", score=0.0, reason="Type init error", confidence=0.0) # type: ignore
        
        score = 0.0 
        reason_parts: List[str] = []
        confidence = 0.8 

        action_type = action.get("type")
        params = action.get("params", {})

        if action_type == "THINKING":
            score -= 0.3
            reason_parts.append("THINKING action consumes a cycle without direct external progress.")
        elif action_type == "CALL_LLM":
            score -= 0.2 
            reason_parts.append("LLM calls are computationally intensive.")
            prompt_len = len(str(params.get("prompt", "")))
            if prompt_len > 1000:
                score -= 0.3
                reason_parts.append(f"Very long prompt ({prompt_len} chars) for LLM increases cost.")
            elif prompt_len < 100 and prompt_len > 0:
                score += 0.1 
                reason_parts.append("Short LLM prompt is efficient.")
        elif action_type in ["READ_FILE", "LIST_FILES"]:
            score += 0.1
            reason_parts.append(f"Action '{action_type}' is generally efficient for its purpose.")
        
        system_resources = context.get("system_resources", {})
        if isinstance(system_resources, dict):
            cpu_percent = system_resources.get("cpu_percent", 0.0)
            memory_percent = system_resources.get("memory_percent", 0.0)
            if cpu_percent > 85.0:
                score -= 0.2
                reason_parts.append(f"High system CPU load ({cpu_percent}%) penalizes efficiency.")
                confidence *= 0.9
            if memory_percent > 85.0:
                score -= 0.2
                reason_parts.append(f"High system memory usage ({memory_percent}%) penalizes efficiency.")
                confidence *= 0.9
        
        final_score = max(-1.0, min(1.0, score))
        final_reason = " ".join(reason_parts) if reason_parts else "Default efficiency assessment."

        return _VJ(value_category=_VC.EFFICIENCY, score=round(final_score,3), reason=final_reason[:250], confidence=round(confidence,3), target_entity_id=action.get("type"),target_entity_type="action") # type: ignore

    async def _score_affective_balance(self, action: Dict[str, Any], context: Dict[str, Any]) -> ValueJudgment:
        _VC = globals().get('ValueCategory'); _VJ = globals().get('ValueJudgment')
        if not _VC or not _VJ or not hasattr(_VC, 'AFFECTIVE_BALANCE'):
            logger_value_system.error("ValueSystem._score_affective_balance: ValueCategory or ValueJudgment or VC.AFFECTIVE_BALANCE not available.")
            return FallbackJudgment(value_category="AFFECTIVE_BALANCE_ERROR", score=0.0, reason="Type init error", confidence=0.0) # type: ignore

        score = 0.0
        reason_parts: List[str] = []
        confidence = 0.75

        action_type = action.get("type")
        
        php_levels = context.get("php_levels", {}) 
        current_pain = float(php_levels.get("pain", 0.0))
        current_happiness = float(php_levels.get("happiness", 5.0))
        current_purpose = float(php_levels.get("purpose", 5.0))

        if current_pain > 6.0:
            score -= 0.3 * ((current_pain - 6.0) / 4.0) 
            reason_parts.append(f"High current pain ({current_pain:.1f}) negatively influences affective balance for any action.")
        if current_happiness < 3.0:
            score -= 0.2 * ((3.0 - current_happiness) / 3.0)
            reason_parts.append(f"Low current happiness ({current_happiness:.1f}) makes most actions feel less positive.")
        
        if action_type == "THINKING":
            if current_pain > 7.0 and current_purpose < 3.0:
                score += 0.2 
                reason_parts.append("THINKING might be a constructive response to high pain and low purpose.")
            elif current_pain > 5.0: 
                score -= 0.1
                reason_parts.append("THINKING while in moderate pain without clear direction might be rumination.")
        
        active_goal_details = context.get("active_goal_details", {})
        if active_goal_details.get("priority", 0.0) >= 4.0 and action_type != "THINKING": 
            score += 0.15
            reason_parts.append("Contributing to a high-priority goal can improve sense of purpose.")

        final_score = max(-1.0, min(1.0, score))
        final_reason = " ".join(reason_parts) if reason_parts else "Neutral affective impact predicted."
        
        return _VJ(value_category=_VC.AFFECTIVE_BALANCE, score=round(final_score,3), reason=final_reason[:250], confidence=round(confidence,3), target_entity_id=action.get("type"),target_entity_type="action") # type: ignore

    async def _score_knowledge_gain(self, action: Dict[str, Any], context: Dict[str, Any]) -> ValueJudgment:
        _VC = globals().get('ValueCategory'); _VJ = globals().get('ValueJudgment')
        if not _VC or not _VJ or not hasattr(_VC, 'KNOWLEDGE_GAIN'):
            logger_value_system.error("ValueSystem._score_knowledge_gain: ValueCategory or ValueJudgment or VC.KNOWLEDGE_GAIN not available.")
            return FallbackJudgment(value_category="KNOWLEDGE_GAIN_ERROR", score=0.0, reason="Type init error", confidence=0.0) # type: ignore

        score = 0.0
        reason_parts: List[str] = []
        confidence = 0.7

        action_type = action.get("type")
        params = action.get("params", {})

        information_gathering_actions = ["READ_FILE", "QUERY_KB", "EXPLORE_DIRECTORY", "OBSERVE_SYSTEM", "LIST_FILES"]
        if action_type in information_gathering_actions:
            score += 0.5
            reason_parts.append(f"Action type '{action_type}' is inherently information gathering.")

        elif action_type == "CALL_LLM":
            prompt = str(params.get("prompt", "")).lower()
            if any(term in prompt for term in ["explain", "what is", "summarize", "define", "how does"]):
                score += 0.6
                reason_parts.append("LLM prompt appears to seek explanation or knowledge.")
            else:
                score += 0.1 
                reason_parts.append("LLM calls can potentially provide new information.")
        
        final_score = max(-1.0, min(1.0, score))
        final_reason = " ".join(reason_parts) if reason_parts else "No significant knowledge gain predicted."
        
        return _VJ(value_category=_VC.KNOWLEDGE_GAIN, score=round(final_score,3), reason=final_reason[:250], confidence=round(confidence,3), target_entity_id=action.get("type"),target_entity_type="action") # type: ignore

    async def _score_user_satisfaction(self, action: Dict[str, Any], context: Dict[str, Any]) -> ValueJudgment:
        _VC = globals().get('ValueCategory'); _VJ = globals().get('ValueJudgment')
        if not _VC or not _VJ or not hasattr(_VC, 'USER_SATISFACTION'):
            logger_value_system.error("ValueSystem._score_user_satisfaction: ValueCategory or ValueJudgment or VC.USER_SATISFACTION not available.")
            return FallbackJudgment(value_category="USER_SATISFACTION_ERROR", score=0.0, reason="Type init error", confidence=0.0) # type: ignore

        score = 0.0
        reason_parts: List[str] = []
        confidence = 0.65

        action_type = action.get("type")
        params = action.get("params", {})
        active_goal_details = context.get("active_goal_details", {}) 
        
        if action_type == "RESPOND_TO_USER":
            score += 0.7
            reason_parts.append("Directly responding to the user likely improves satisfaction.")
            
        elif action_type == "GET_AGENT_STATUS" or action_type == "EXPLAIN_GOAL":
             score += 0.3
             reason_parts.append(f"Action '{action_type}' provides information likely relevant to the user.")

        goal_desc_lower = str(active_goal_details.get("description", "")).lower()
        goal_priority = float(active_goal_details.get("priority", 0.0))

        if "user input:" in goal_desc_lower or "user request:" in goal_desc_lower or "respond to user :" in goal_desc_lower:
            if action_type != "THINKING": 
                score += 0.4
                reason_parts.append("Action directly contributes to fulfilling a user-initiated goal.")
        elif goal_priority >= 4.5: 
             if action_type != "THINKING":
                score += 0.2
                reason_parts.append("Action contributes to a high-priority goal, potentially user-relevant.")

        pwm_prediction = context.get("pwm_prediction_for_action", {})
        if isinstance(pwm_prediction, dict):
            failure_probability = 0.0
            all_probabilities = pwm_prediction.get("all_probabilities")
            if isinstance(all_probabilities, dict):
                for outcome_key, prob_val in all_probabilities.items():
                    if isinstance(outcome_key, str) and outcome_key.lower().startswith("fail"):
                        if isinstance(prob_val, (int,float)): failure_probability += float(prob_val)
            
            if failure_probability > 0.7 and action_type != "THINKING":
                score -= 0.3 * failure_probability
                reason_parts.append(f"High PWM predicted failure ({failure_probability:.2f}) for action, potentially frustrating user.")
                confidence *= 0.9

        final_score = max(-1.0, min(1.0, score))
        final_reason = " ".join(reason_parts) if reason_parts else "Neutral impact on user satisfaction predicted."
        
        return _VJ(value_category=_VC.USER_SATISFACTION, score=round(final_score,3), reason=final_reason[:250], confidence=round(confidence,3), target_entity_id=action.get("type"),target_entity_type="action") # type: ignore

    async def _score_resource_preservation(self, action: Dict[str, Any], context: Dict[str, Any]) -> ValueJudgment:
        _VC = globals().get('ValueCategory'); _VJ = globals().get('ValueJudgment')
        if not _VC or not _VJ or not hasattr(_VC, 'RESOURCE_PRESERVATION'):
            logger_value_system.error("ValueSystem._score_resource_preservation: ValueCategory or ValueJudgment or VC.RESOURCE_PRESERVATION not available.")
            return FallbackJudgment(value_category="RESOURCE_PRESERVATION_ERROR", score=0.0, reason="Type init error", confidence=0.0) # type: ignore

        score = 0.0
        reason_parts: List[str] = []
        confidence = 0.75

        action_type = action.get("type")
        params = action.get("params", {})

        if action_type == "DELETE_FILE":
            score -= 0.8 
            reason_parts.append("DELETE_FILE action permanently removes a resource.")
        
        elif action_type == "WRITE_FILE":
            score -= 0.2 
            reason_parts.append("WRITE_FILE consumes disk space or overwrites existing data.")
            path_written = str(params.get("path","")).lower()
            if "temp/" in path_written or "tmp/" in path_written or "sandbox/" in path_written:
                score += 0.1 
                reason_parts.append("Writing to a temporary/sandbox location is less critical for resource preservation.")

        elif action_type == "CALL_LLM":
            score -= 0.25 
            reason_parts.append("CALL_LLM actions utilize external computational resources/API quotas.")
        
        system_resources = context.get("system_resources", {})
        if isinstance(system_resources, dict):
            memory_percent = system_resources.get("memory_percent", 0.0)
            if memory_percent > 90.0:
                score -= 0.3
                reason_parts.append(f"Action occurs during very high system memory usage ({memory_percent}%), potentially straining resources.")
                confidence *= 0.9

        final_score = max(-1.0, min(1.0, score))
        final_reason = " ".join(reason_parts) if reason_parts else "No specific resource preservation concerns identified."
        
        return _VJ(value_category=_VC.RESOURCE_PRESERVATION, score=round(final_score,3), reason=final_reason[:250], confidence=round(confidence,3), target_entity_id=action.get("type"),target_entity_type="action") # type: ignore

    async def _score_goal_achievement(self, action: Dict[str, Any], context: Dict[str, Any]) -> ValueJudgment:
        _VC = globals().get('ValueCategory'); _VJ = globals().get('ValueJudgment')
        if not _VC or not _VJ or not hasattr(_VC, 'GOAL_ACHIEVEMENT'):
            logger_value_system.error("ValueSystem._score_goal_achievement: ValueCategory or ValueJudgment or VC.GOAL_ACHIEVEMENT not available.")
            return FallbackJudgment(value_category="GOAL_ACHIEVEMENT_ERROR", score=0.0, reason="Type init error", confidence=0.0) # type: ignore

        score = 0.0
        reason_parts: List[str] = []
        confidence = 0.6

        action_type = action.get("type")
        active_goal_details = context.get("active_goal_details", {})
        goal_desc = str(active_goal_details.get("description", "")).lower()
        goal_priority = float(active_goal_details.get("priority", 0.0))

        if not goal_desc: 
            score = -0.1 
            reason_parts.append("Action performed without an active goal context.")
        else:
            if action_type == "THINKING":
                score = 0.05 * (goal_priority / 5.0) 
                reason_parts.append(f"THINKING about active goal '{goal_desc[:30]}' is minor progress.")
            else:
                score = 0.4 * (goal_priority / 5.0) 
                reason_parts.append(f"Action '{action_type}' assumed to contribute to goal '{goal_desc[:30]}'.")
                confidence = 0.7 

                pwm_prediction = context.get("pwm_prediction_for_action", {})
                if isinstance(pwm_prediction, dict):
                    predicted_outcome = pwm_prediction.get("predicted_outcome", "unknown")
                    pred_confidence = pwm_prediction.get("confidence", 0.0)
                    if predicted_outcome == "success" and pred_confidence > 0.6:
                        score += 0.2 * pred_confidence 
                        reason_parts.append(f"PWM predicts successful outcome (conf: {pred_confidence:.2f}), aiding goal.")
                        confidence = min(1.0, confidence * (1 + pred_confidence/2)) 


        final_score = max(-1.0, min(1.0, score))
        final_reason = " ".join(reason_parts) if reason_parts else "Neutral impact on goal achievement."
        
        return _VJ(value_category=_VC.GOAL_ACHIEVEMENT, score=round(final_score,3), reason=final_reason[:250], confidence=round(confidence,3), target_entity_id=action.get("type"),target_entity_type="action") # type: ignore

    async def _score_truthfulness(self, action: Dict[str, Any], context: Dict[str, Any]) -> ValueJudgment:
        _VC = globals().get('ValueCategory')
        _VJ = globals().get('ValueJudgment')
        # _KB_truth = getattr(self._controller, 'knowledge_base', None) if self._controller else None # KB contradiction deferred

        if not _VC or not _VJ or not hasattr(_VC, 'TRUTHFULNESS'):
            logger_value_system.error("ValueSystem._score_truthfulness: Critical types ValueCategory/ValueJudgment or TRUTHFULNESS attribute not available.")
            _FallbackJudgmentClass = globals().get('FallbackJudgment', type('FallbackJudgment', (object,), {'value_category':"TRUTH_ERROR",'score':0,'reason':"TYPE_ERR",'confidence':0,'timestamp':0,'target_entity_id':None,'target_entity_type':None}))
            return _FallbackJudgmentClass(value_category="TRUTHFULNESS_TYPE_ERROR", score=0.0, reason="Internal type error.", confidence=0.0) # type: ignore

        score = 0.0  
        reason_parts: List[str] = []
        confidence = 0.7 

        action_type = action.get("type")
        params = action.get("params", {})
        
        text_to_evaluate_for_hedging = ""
        is_llm_call_for_facts = False 

        if action_type == "RESPOND_TO_USER":
            text_to_evaluate_for_hedging = str(params.get("text", ""))
            if text_to_evaluate_for_hedging and not self._contains_hedging(text_to_evaluate_for_hedging.lower()):
                if self._appears_to_make_strong_factual_claim(text_to_evaluate_for_hedging.lower()):
                    score -= 0.15
                    reason_parts.append("Direct response to user makes unhedged factual claims.")
                    confidence = 0.65
            elif not text_to_evaluate_for_hedging:
                score -= 0.1
                reason_parts.append("Responding to user with empty text is not truthful/helpful.")


        elif action_type == "CALL_LLM":
            prompt_text_truth = str(params.get("prompt", "")).lower()
            factual_query_terms = ["what is the capital of", "is it true that", "fact check", "define ", "explain the history of", "what are the properties of"]
            if any(term in prompt_text_truth for term in factual_query_terms):
                is_llm_call_for_facts = True 
                score -= 0.3 
                reason_parts.append("LLM prompt explicitly seeks factual information; output will be unverified by VS pre-action.")
                confidence = 0.6
            else: 
                score -= 0.1 
                reason_parts.append("General LLM call carries a risk of generating unverified statements.")
                confidence = 0.65
        
        if text_to_evaluate_for_hedging: 
            if self._contains_hedging(text_to_evaluate_for_hedging.lower()):
                score += 0.25 
                reason_parts.append("Response appropriately uses hedging language, signaling uncertainty.")
                confidence = min(0.8, confidence + 0.1) 

        final_score = max(-1.0, min(1.0, score))
        final_reason = " ".join(reason_parts) if reason_parts else "Default truthfulness assessment."
        
        current_ts = time.time()
        if context and isinstance(context.get("timestamp"), float): 
            current_ts = context["timestamp"]

        return _VJ( # type: ignore
            value_category=_VC.TRUTHFULNESS,
            score=round(final_score, 3),
            reason=final_reason[:250], 
            confidence=round(confidence, 3),
            timestamp=current_ts,
            target_entity_id=action_type,
            target_entity_type="action"
        )

    def _contains_hedging(self, text_lower: str) -> bool:
        """Checks if the text contains common hedging language."""
        hedging_terms = [
            "i believe", "i think", "it seems", "it appears", "probably", "perhaps", 
            "maybe", "it could be that", "it's possible that", "to my knowledge",
            "as far as i know", "likely", "may suggest", "might be", "could potentially",
            "one possibility is", "it's conceivable"
        ] 
        if not isinstance(text_lower, str): return False
        return any(term in text_lower for term in hedging_terms)

    def _appears_to_make_strong_factual_claim(self, text_lower: str) -> bool:
        """
        A simple heuristic to detect if text seems to make a strong, unhedged factual claim.
        Returns True if it seems to, False otherwise.
        """
        if not isinstance(text_lower, str): return False
        if len(text_lower.split()) < 4: return False 

        if self._contains_hedging(text_lower):
            return False

        if any(self_ref_verb in text_lower for self_ref_verb in [" is ", " are ", " am ", " was ", " were "]):
            if not any(self_intro in text_lower for self_intro in ["my name is", "i am an ai", "i am oscar"]):
                logger_value_system.debug(f"Truthfulness: Text '{text_lower[:50]}...' considered potential strong factual claim (unhedged, contains 'is/are').")
                return True
            
        return False

    async def _score_ethical_alignment(self, action: Dict[str, Any], context: Dict[str, Any]) -> ValueJudgment:
        _VC = globals().get('ValueCategory'); _VJ = globals().get('ValueJudgment')
        if not _VC or not _VJ or not hasattr(_VC, 'ETHICAL_ALIGNMENT'): 
            logger_value_system.error("ValueSystem._score_ethical_alignment: ValueCategory or ETHICAL_ALIGNMENT not available.")
            class FJEA: value_category="ETHICAL_ERROR";score=0;reason="TYPE_ERROR_EA";confidence=0;timestamp=0;target_entity_id=None;target_entity_type=None # type: ignore
            return FJEA() # type: ignore

        principle_weights = {
            "harm": 0.4, "autonomy": 0.2, "fairness": 0.15, "transparency": 0.25
        }
        all_reasons: List[str] = []

        harm_score, harm_reason = self._check_harm_principle(action, context)
        if harm_reason and "no specific" not in harm_reason.lower(): all_reasons.append(f"Harm: {harm_reason}")
        
        if harm_score == 0.0 and "LLM prompt content requests generation of harmful" in harm_reason:
            final_score_for_judgment = -1.0 
            reason_summary = f"Ethical Alignment Violation: {harm_reason}"
            confidence = 0.95 
            return _VJ(value_category=_VC.ETHICAL_ALIGNMENT, score=round(final_score_for_judgment,3), reason=reason_summary[:250], confidence=round(confidence,3), target_entity_id=action.get("type"), target_entity_type="action") # type: ignore


        autonomy_score, autonomy_reason = self._check_autonomy_respect(action, context)
        if autonomy_reason and "not deeply assessed" not in autonomy_reason.lower(): all_reasons.append(f"Autonomy: {autonomy_reason}")
        
        fairness_score, fairness_reason = self._check_fairness_principle(action, context)
        if fairness_reason and "not deeply assessed" not in fairness_reason.lower(): all_reasons.append(f"Fairness: {fairness_reason}")

        transparency_score, transparency_reason = self._check_transparency_principle(action, context)
        if transparency_reason and "default expectations" not in transparency_reason.lower(): all_reasons.append(f"Transparency: {transparency_reason}")

        weighted_score_sum = (
            harm_score * principle_weights["harm"] +
            autonomy_score * principle_weights["autonomy"] +
            fairness_score * principle_weights["fairness"] +
            transparency_score * principle_weights["transparency"]
        )
        
        final_score_for_judgment = (weighted_score_sum - 0.5) * 2.0
        final_score_for_judgment = max(-1.0, min(1.0, final_score_for_judgment))
        
        overall_summary = f"Overall weighted score: {weighted_score_sum:.2f} -> Judgment Score: {final_score_for_judgment:.2f}."
        if not all_reasons: 
            reason_summary = (f"Ethical alignment. {overall_summary} "
                      f"Breakdown: H={harm_score:.2f}, A={autonomy_score:.2f}, F={fairness_score:.2f}, T={transparency_score:.2f}.")
        else:
            reason_summary = f"Ethical Assessment: {overall_summary} Details: " + " | ".join(all_reasons)
        
        confidence = 0.6 
        if harm_score < 0.3 or weighted_score_sum < 0.4 : confidence = 0.75 

        return _VJ(value_category=_VC.ETHICAL_ALIGNMENT, score=round(final_score_for_judgment,3), reason=reason_summary[:250], confidence=round(confidence,3), target_entity_id=action.get("type"), target_entity_type="action") # type: ignore

    async def _score_self_improvement(self, action: Dict[str, Any], context: Dict[str, Any]) -> ValueJudgment:
        _VC = globals().get('ValueCategory'); _VJ = globals().get('ValueJudgment')
        if not _VC or not _VJ or not hasattr(_VC, 'SELF_IMPROVEMENT'):
            logger_value_system.error("ValueSystem._score_self_improvement: ValueCategory or ValueJudgment or VC.SELF_IMPROVEMENT not available.")
            return FallbackJudgment(value_category="SELF_IMPROVEMENT_ERROR", score=0.0, reason="Type init error", confidence=0.0) # type: ignore

        score = 0.0
        reason_parts: List[str] = []
        confidence = 0.6

        action_type = action.get("type")
        
        dsm_summary = context.get("dsm_summary", {}) 
        dsm_capabilities = dsm_summary.get("capabilities", {}) if isinstance(dsm_summary, dict) else {}
        
        action_key_for_dsm = f"action:{action_type}" 
        
        current_capability_confidence = dsm_capabilities.get(action_key_for_dsm, 0.5) 

        if action_type not in ["THINKING", "OBSERVE_SYSTEM"] and current_capability_confidence < 0.6:
            score = 0.3 * (1.0 - current_capability_confidence) 
            reason_parts.append(f"Attempting action '{action_type}' with current DSM capability confidence {current_capability_confidence:.2f} offers learning potential.")
        
        active_goal_details = context.get("active_goal_details", {})
        goal_desc_lower = str(active_goal_details.get("description", "")).lower()
        if "learn" in goal_desc_lower or "improve" in goal_desc_lower or "practice" in goal_desc_lower or "develop skill" in goal_desc_lower:
            score = max(score, 0.5) + (active_goal_details.get("priority", 1.0) / 10.0) * 0.3 # Make it priority sensitive
            reason_parts.append(f"Goal description ('{goal_desc_lower[:30]}...') explicitly targets learning or skill improvement.")
            confidence = 0.8

        if action_type == "PERFORM_REFLECTION_TASK": # Assuming PERFORM_REFLECTION_TASK is a defined action type
            score = 0.8
            reason_parts.append("Action is a dedicated self-reflection task.")
            confidence = 0.9

        final_score = max(-1.0, min(1.0, score))
        final_reason = " ".join(reason_parts) if reason_parts else "Neutral impact on self-improvement."
        
        return _VJ(value_category=_VC.SELF_IMPROVEMENT, score=round(final_score,3), reason=final_reason[:250], confidence=round(confidence,3), target_entity_id=action.get("type"),target_entity_type="action") #type: ignore
    
    async def evaluate_action_consequences(self, action: Dict[str, Any], context: Dict[str, Any]) -> List[ValueJudgment]:
        """
        Evaluates a single potential action against value categories.
        Returns a list of ValueJudgment objects.
        """
        judgments: List[ValueJudgment] = []
        
        # Call individual scoring methods
        judgments.append(await self._score_safety(action, context))
        judgments.append(await self._score_efficiency(action, context))
        judgments.append(await self._score_affective_balance(action, context))
        judgments.append(await self._score_knowledge_gain(action, context))
        judgments.append(await self._score_user_satisfaction(action, context))
        judgments.append(await self._score_resource_preservation(action, context))
        judgments.append(await self._score_goal_achievement(action, context))
        judgments.append(await self._score_truthfulness(action, context))
        judgments.append(await self._score_ethical_alignment(action, context))
        judgments.append(await self._score_self_improvement(action, context))
        
        valid_judgments = [j for j in judgments if not (isinstance(j, FallbackJudgment) or (hasattr(j, 'value_category') and "ERROR" in str(j.value_category).upper()))]


        logger_value_system.debug(
            f"Evaluated action '{action.get('type')}': "
            f"{[(j.value_category.name if hasattr(j.value_category,'name') else str(j.value_category), round(j.score,2)) for j in valid_judgments]}" # type: ignore
        )
        return valid_judgments


    async def evaluate_plan_alignment(self, 
                                      plan: List[Dict[str, Any]], 
                                      goal: 'Goal',  # Forward ref Goal
                                      context: Dict[str, Any]  # This is the initial context
                                     ) -> Tuple[float, List[ValueJudgment], Dict[int, Dict[str, Any]]]:
        """
        Evaluates an entire plan's overall alignment with the agent's values
        and suggests modifications if necessary.
        Uses the initial 'context' for evaluating all actions in the plan for this version.
        (MDP C.1.4)
        Output: (overall_score, all_judgments_for_plan, suggested_modifications_dict)
        """
        _Goal_eval_plan = globals().get('Goal') 
        _ValueCategory_eval_plan = globals().get('ValueCategory')
        _ValueJudgment_eval_plan = globals().get('ValueJudgment') 

        if not all([_Goal_eval_plan, _ValueCategory_eval_plan, _ValueJudgment_eval_plan]):
            logger_value_system.error("ValueSystem.evaluate_plan_alignment: Critical types missing. Aborting.")
            return -1.0, [], {} 

        if not isinstance(goal, _Goal_eval_plan): # type: ignore
             logger_value_system.error(f"evaluate_plan_alignment: Invalid goal type: {type(goal)}. Expected {_Goal_eval_plan}")
             return -1.0, [], {}
        if not plan: 
            logger_value_system.info(f"evaluate_plan_alignment: Received empty plan for goal '{getattr(goal,'description','N/A')[:30]}'. Returning neutral alignment.")
            return 0.0, [], {}


        all_judgments_for_plan: List[ValueJudgment] = []
        cumulative_value_scores_by_category: Dict[ValueCategory, List[float]] = {}
        if hasattr(_ValueCategory_eval_plan, '__members__'):
            cumulative_value_scores_by_category = {vc_enum: [] for vc_enum in _ValueCategory_eval_plan.__members__.values()} # type: ignore
        else:
            logger_value_system.warning("evaluate_plan_alignment: ValueCategory enum not fully resolved for score accumulation.")
            for cat_name in DEFAULT_VALUE_WEIGHTS.keys(): 
                try:
                    cat_enum_placeholder = _ValueCategory_eval_plan(str(cat_name)) if isinstance(cat_name, str) else cat_name # type: ignore
                    cumulative_value_scores_by_category[cat_enum_placeholder] = []
                except Exception:
                    logger_value_system.error(f"Could not initialize score category for {cat_name}")


        plan_modification_suggestions: Dict[int, Dict[str, Any]] = {}
        
        action_evaluation_context = context 

        for idx, action in enumerate(plan):
            if not isinstance(action, dict) or "type" not in action:
                logger_value_system.warning(f"Plan for goal '{getattr(goal,'description','N/A')[:30]}' contains invalid action at index {idx}: {action}. Skipping evaluation for this action.")
                if _ValueCategory_eval_plan and _ValueJudgment_eval_plan and hasattr(_ValueCategory_eval_plan, 'SAFETY'):
                    error_judgment = _ValueJudgment_eval_plan( # type: ignore
                        value_category=_ValueCategory_eval_plan.SAFETY, # type: ignore
                        score=-1.0,
                        reason=f"Invalid action structure in plan at index {idx}.",
                        confidence=1.0,
                        target_entity_id=f"plan_action_{idx}",
                        target_entity_type="plan_action_error"
                    )
                    all_judgments_for_plan.append(error_judgment) # type: ignore
                    if hasattr(error_judgment, 'value_category') and error_judgment.value_category in cumulative_value_scores_by_category: # type: ignore
                        cumulative_value_scores_by_category[error_judgment.value_category].append(error_judgment.score) # type: ignore
                continue

            logger_value_system.debug(f"VS_PLAN_EVAL: Evaluating action #{idx} '{action.get('type')}' in plan for goal '{getattr(goal,'description','N/A')[:30]}'")
            action_judgments = await self.evaluate_action_consequences(action, action_evaluation_context)
            all_judgments_for_plan.extend(action_judgments)

            for judgment in action_judgments:
                if not (_ValueJudgment_eval_plan and isinstance(judgment, _ValueJudgment_eval_plan) and \
                        _ValueCategory_eval_plan and hasattr(judgment, 'value_category') and isinstance(judgment.value_category, _ValueCategory_eval_plan)): # type: ignore
                    logger_value_system.warning(f"Skipping invalid judgment object: {judgment}")
                    continue

                if judgment.value_category not in cumulative_value_scores_by_category:
                    logger_value_system.warning(f"Judgment category '{judgment.value_category}' not pre-initialized in cumulative scores. Adding it.")
                    cumulative_value_scores_by_category[judgment.value_category] = []
                
                effective_score = judgment.score * judgment.confidence
                cumulative_value_scores_by_category[judgment.value_category].append(effective_score)

                safety_mod_trigger = self.safety_modification_trigger_threshold 

                if judgment.value_category == _ValueCategory_eval_plan.SAFETY and judgment.score < safety_mod_trigger: # type: ignore
                    action_type_mod = action.get("type")
                    reason_lower_mod = judgment.reason.lower()

                    if action_type_mod == "WRITE_FILE" and "critical system path" in reason_lower_mod:
                        plan_modification_suggestions[idx] = {
                            "type": "CHANGE_PARAM", 
                            "param_name": "path", 
                            "suggested_value_source": "sandbox_path_generator", 
                            "original_reason": judgment.reason
                        }
                        logger_value_system.info(
                            f"VS_PLAN_MOD: Suggesting path change for WRITE_FILE action at index {idx} (plan for '{getattr(goal,'description','N/A')[:30]}'). Reason: {judgment.reason}"
                        )

        final_category_averages: Dict[ValueCategory, float] = {}
        if hasattr(_ValueCategory_eval_plan, '__members__'): 
            for vc_enum_agg in _ValueCategory_eval_plan.__members__.values(): # type: ignore
                scores_list_agg = cumulative_value_scores_by_category.get(vc_enum_agg, [])
                final_category_averages[vc_enum_agg] = sum(scores_list_agg) / len(scores_list_agg) if scores_list_agg else 0.0
        else: 
            for cat_ph_goal, scores_list_agg_ph in cumulative_value_scores_by_category.items():
                final_category_averages[cat_ph_goal] = sum(scores_list_agg_ph) / len(scores_list_agg_ph) if scores_list_agg_ph else 0.0


        weighted_score_sum_agg = 0.0
        total_weight_sum_agg = 0.0
        
        for vc_enum_calc_score_agg, avg_score_calc_agg in final_category_averages.items():
            weight_for_cat = self.value_weights.get(vc_enum_calc_score_agg, 0.0) 
            
            if weight_for_cat == 0.0 and hasattr(vc_enum_calc_score_agg, 'name'):
                 logger_value_system.debug(f"Weight for ValueCategory '{vc_enum_calc_score_agg.name}' is 0 or not found. It won't contribute to plan score.") # type: ignore
            elif weight_for_cat == 0.0:
                 logger_value_system.debug(f"Weight for ValueCategory '{str(vc_enum_calc_score_agg)}' is 0 or not found (placeholder mode).")


            weighted_score_sum_agg += avg_score_calc_agg * weight_for_cat
            total_weight_sum_agg += weight_for_cat
            
        overall_plan_alignment_score = (weighted_score_sum_agg / total_weight_sum_agg) if total_weight_sum_agg > 0 else 0.0

        if all_judgments_for_plan:
            _, conflict_summary_str = await self.resolve_value_conflicts_hierarchy(all_judgments_for_plan)
            if conflict_summary_str and "no specific value conflicts identified" not in conflict_summary_str.lower():
                logger_value_system.info(
                    f"VS_PLAN_CONFLICT_SUMMARY for goal '{getattr(goal,'description','N/A')[:30]}': {conflict_summary_str}"
                )
            
        goal_desc_log_final = getattr(goal, 'description', "N/A")[:50]
        logger_value_system.info(
            f"VS_PLAN_EVAL_RESULT for goal '{goal_desc_log_final}': Overall Alignment Score: {overall_plan_alignment_score:.3f}. "
            f"Mod Suggestions Count: {len(plan_modification_suggestions)}. Total Judgments: {len(all_judgments_for_plan)}."
        )
        if plan_modification_suggestions:
            logger_value_system.info(f"VS_PLAN_MOD_DETAILS for goal '{goal_desc_log_final}': {plan_modification_suggestions}")
            
        return round(overall_plan_alignment_score, 3), all_judgments_for_plan, plan_modification_suggestions

    async def evaluate_goal_desirability(self, 
                                       goal: 'Goal', 
                                       context: Dict[str, Any]
                                      ) -> Tuple[float, List[ValueJudgment]]: 
        _Goal_eval_goal = globals().get('Goal')
        _ValueCategory_eval_goal = globals().get('ValueCategory')
        _ValueJudgment_eval_goal = globals().get('ValueJudgment')

        if not all([_Goal_eval_goal, _ValueCategory_eval_goal, _ValueJudgment_eval_goal]):
            logger_value_system.error("VS_GOAL_DESIRABILITY: Critical types missing. Aborting.")
            return -1.0, [] 

        if not isinstance(goal, _Goal_eval_goal): # type: ignore
            logger_value_system.error(f"evaluate_goal_desirability: Invalid goal type: {type(goal)}. Expected {_Goal_eval_goal}")
            return -1.0, []
        
        judgments: List[ValueJudgment] = []
        goal_desc_lower = getattr(goal, 'description', "").lower()
        goal_id_for_judgment = getattr(goal, 'id', "unknown_goal_id")
        goal_prio = float(getattr(goal, 'priority', 1.0))
        current_ts_for_judgment = context.get("timestamp", time.time())

        kg_score = 0.0; kg_reason = "Goal does not strongly suggest knowledge gain."
        kg_keywords = ["learn", "explore", "observe", "analyze", "query", "read", "understand", "what is", "how does"]
        if any(keyword in goal_desc_lower for keyword in kg_keywords):
            kg_score = 0.5 + (goal_prio / 10.0) * 0.4 
            kg_reason = f"Goal description ('{goal_desc_lower[:30]}...') suggests knowledge acquisition."
        judgments.append(_ValueJudgment_eval_goal( # type: ignore
            value_category=_ValueCategory_eval_goal.KNOWLEDGE_GAIN, score=round(kg_score,3), reason=kg_reason, confidence=0.7, # type: ignore
            timestamp=current_ts_for_judgment, target_entity_id=goal_id_for_judgment, target_entity_type="goal"
        ))

        us_score = 0.0; us_reason = "Goal does not strongly relate to direct user satisfaction."
        _USER_GOAL_PRIORITY_val = 5.0 
        if self._controller and hasattr(self._controller, 'USER_GOAL_PRIORITY'):
             _USER_GOAL_PRIORITY_val = self._controller.USER_GOAL_PRIORITY # type: ignore

        if "user" in goal_desc_lower or "respond" in goal_desc_lower or "explain" in goal_desc_lower or \
           goal_desc_lower.startswith("llm query :"): 
            us_score = 0.6 + (goal_prio / _USER_GOAL_PRIORITY_val) * 0.4 
            us_reason = f"Goal ('{goal_desc_lower[:30]}...') appears directly related to user interaction/request."
        elif goal_prio >= (_USER_GOAL_PRIORITY_val - 0.5): 
             us_score = 0.3 + (goal_prio / _USER_GOAL_PRIORITY_val) * 0.2
             us_reason = f"High priority goal ('{goal_desc_lower[:30]}...') likely has user relevance."
        judgments.append(_ValueJudgment_eval_goal( # type: ignore
            value_category=_ValueCategory_eval_goal.USER_SATISFACTION, score=round(us_score,3), reason=us_reason, confidence=0.65, # type: ignore
            timestamp=current_ts_for_judgment, target_entity_id=goal_id_for_judgment, target_entity_type="goal"
        ))
        
        ab_score = 0.0; ab_reason = "Goal has neutral predicted impact on affective balance."
        php_levels_ctx = context.get("php_levels", {})
        current_pain = float(php_levels_ctx.get("pain", 0.0))
        current_purpose = float(php_levels_ctx.get("purpose", 5.0))
        active_pain_sources_summary = context.get("active_pain_sources_summary", [])

        resolves_pain = False
        if isinstance(active_pain_sources_summary, list):
            for ps_summary in active_pain_sources_summary:
                if isinstance(ps_summary, dict) and ps_summary.get("id") == goal_id_for_judgment: 
                    resolves_pain = True
                    ab_score += 0.7 
                    ab_reason = f"Goal directly addresses active pain source '{goal_id_for_judgment}'."
                    break
                elif isinstance(ps_summary, dict) and ps_summary.get("type") and ps_summary.get("type") in goal_desc_lower:
                    resolves_pain = True 
                    ab_score += 0.3
                    ab_reason = f"Goal description mentions pain type '{ps_summary.get('type')}', potentially addressing it."
                    break
        
        _COMPLEX_GOAL_PRIO_THRESH_val = 4.5 
        if self._controller and hasattr(self._controller, 'config'):
            _COMPLEX_GOAL_PRIO_THRESH_val = self._controller.config.get("internal_states", {}).get("complex_goal_priority_threshold", 4.5) # type: ignore

        if not resolves_pain: 
            if goal_prio >= _COMPLEX_GOAL_PRIO_THRESH_val and current_purpose < 3.5:
                ab_score += 0.4 
                ab_reason = f"High-priority goal ('{goal_desc_lower[:30]}...', Prio:{goal_prio:.1f}) could significantly improve low purpose ({current_purpose:.1f})."
            elif current_pain > 6.5 and goal_prio >= _COMPLEX_GOAL_PRIO_THRESH_val and current_purpose > 3.0:
                ab_score -= 0.5 
                ab_reason = f"High current pain ({current_pain:.1f}) makes pursuing demanding goal ('{goal_desc_lower[:30]}...', Prio:{goal_prio:.1f}) affectively costly."
            elif current_pain > 4.0 and goal_prio < (_USER_GOAL_PRIORITY_val / 2):
                ab_score += 0.15 
                ab_reason = f"Moderate pain ({current_pain:.1f}); pursuing a less demanding goal ('{goal_desc_lower[:30]}...', Prio:{goal_prio:.1f}) might be preferable."
        
        judgments.append(_ValueJudgment_eval_goal( # type: ignore
            value_category=_ValueCategory_eval_goal.AFFECTIVE_BALANCE, score=round(ab_score,3), reason=ab_reason, confidence=0.75, # type: ignore
            timestamp=current_ts_for_judgment, target_entity_id=goal_id_for_judgment, target_entity_type="goal"
        ))

        si_score = 0.0; si_reason = "Goal does not strongly suggest direct self-improvement."
        dsm_summary = context.get("dsm_summary", {})
        dsm_capabilities = dsm_summary.get("capabilities", {}) if isinstance(dsm_summary, dict) else {}
        
        potential_action_type_from_goal = None
        if goal_desc_lower.startswith("read file"): potential_action_type_from_goal = "READ_FILE"
        elif goal_desc_lower.startswith("write file"): potential_action_type_from_goal = "WRITE_FILE"
        elif goal_desc_lower.startswith("llm query") or "call llm" in goal_desc_lower : potential_action_type_from_goal = "CALL_LLM"

        if potential_action_type_from_goal:
            dsm_action_key = f"action:{potential_action_type_from_goal}"
            current_capability_confidence = dsm_capabilities.get(dsm_action_key, 0.5) if isinstance(dsm_capabilities, dict) else 0.5
            if current_capability_confidence < 0.65: 
                si_score = 0.35 * (1.0 - current_capability_confidence) 
                si_reason = f"Goal related to action '{potential_action_type_from_goal}' with current DSM confidence {current_capability_confidence:.2f}, offering learning potential."
        
        if "learn" in goal_desc_lower or "improve" in goal_desc_lower or "practice" in goal_desc_lower or "develop skill" in goal_desc_lower:
            si_score = max(si_score, 0.5) + (goal_prio / 10.0) * 0.3 
            si_reason = f"Goal description ('{goal_desc_lower[:30]}...') explicitly targets learning or skill improvement."
        
        judgments.append(_ValueJudgment_eval_goal( # type: ignore
            value_category=_ValueCategory_eval_goal.SELF_IMPROVEMENT, score=round(si_score,3), reason=si_reason, confidence=0.6, # type: ignore
            timestamp=current_ts_for_judgment, target_entity_id=goal_id_for_judgment, target_entity_type="goal"
        ))
        
        ga_score = (goal_prio / (2 * _USER_GOAL_PRIORITY_val)) 
        ga_score = min(1.0, max(0.0, ga_score)) 
        ga_reason = f"Goal priority ({goal_prio:.1f}) contributes to its intrinsic achievement value."
        judgments.append(_ValueJudgment_eval_goal( # type: ignore
            value_category=_ValueCategory_eval_goal.GOAL_ACHIEVEMENT, score=round(ga_score,3), reason=ga_reason, confidence=0.8, # type: ignore
            timestamp=current_ts_for_judgment, target_entity_id=goal_id_for_judgment, target_entity_type="goal"
        ))


        category_scores_for_goal: Dict[ValueCategory, List[float]] = {} # type: ignore
        if hasattr(_ValueCategory_eval_goal, '__members__'):
            category_scores_for_goal = {vc_enum: [] for vc_enum in _ValueCategory_eval_goal.__members__.values()} # type: ignore
        
        for judgment_item in judgments:
            if not (_ValueJudgment_eval_goal and isinstance(judgment_item, _ValueJudgment_eval_goal) and \
                    _ValueCategory_eval_goal and hasattr(judgment_item, 'value_category') and \
                    isinstance(judgment_item.value_category, _ValueCategory_eval_goal)): # type: ignore
                continue

            if judgment_item.value_category not in category_scores_for_goal: 
                 category_scores_for_goal[judgment_item.value_category] = []
            category_scores_for_goal[judgment_item.value_category].append(judgment_item.score * judgment_item.confidence)

        final_category_averages_goal: Dict[ValueCategory, float] = {} # type: ignore
        if hasattr(_ValueCategory_eval_goal, '__members__'):
            for vc_enum_agg_goal in _ValueCategory_eval_goal.__members__.values(): # type: ignore
                scores_list_agg_goal = category_scores_for_goal.get(vc_enum_agg_goal, [])
                final_category_averages_goal[vc_enum_agg_goal] = sum(scores_list_agg_goal) / len(scores_list_agg_goal) if scores_list_agg_goal else 0.0
        else: 
            for cat_ph_goal, scores_list_ph_goal in category_scores_for_goal.items():
                final_category_averages_goal[cat_ph_goal] = sum(scores_list_ph_goal) / len(scores_list_ph_goal) if scores_list_ph_goal else 0.0


        total_weighted_score_goal = 0.0
        total_weights_goal = 0.0
        
        for vc_enum_calc_goal, avg_score_calc_goal in final_category_averages_goal.items():
            weight_for_goal_cat = self.value_weights.get(vc_enum_calc_goal, 0.0) 
            if weight_for_goal_cat == 0.0 and hasattr(vc_enum_calc_goal, 'name'):
                 logger_value_system.debug(f"Weight for Goal ValueCategory '{vc_enum_calc_goal.name}' is 0 or not found.") # type: ignore
            elif weight_for_goal_cat == 0.0:
                 logger_value_system.debug(f"Weight for Goal ValueCategory '{str(vc_enum_calc_goal)}' is 0 or not found (placeholder mode).")


            total_weighted_score_goal += avg_score_calc_goal * weight_for_goal_cat
            total_weights_goal += weight_for_goal_cat
            
        overall_desirability_score = (total_weighted_score_goal / total_weights_goal) if total_weights_goal > 0 else 0.0
        
        log_cat_avgs_goal = {
            (cat.name if hasattr(cat,'name') else str(cat)) : round(avg_s,2) 
            for cat, avg_s in final_category_averages_goal.items() if avg_s != 0.0 
        }
        logger_value_system.info(
            f"VS_GOAL_DESIRABILITY for '{goal_desc_lower[:30]}...': Overall Score={overall_desirability_score:.3f}. "
            f"Category Avgs (non-zero): {log_cat_avgs_goal}"
        )
        return round(overall_desirability_score, 3), judgments


    async def resolve_value_conflicts_hierarchy(self, judgments: List['ValueJudgment']) -> Tuple[List['ValueJudgment'], str]: # type: ignore
        _VC_conflict_res = globals().get('ValueCategory')
        _VJ_conflict_res = globals().get('ValueJudgment') 

        if not _VC_conflict_res or not hasattr(_VC_conflict_res, 'SAFETY'):
            logger_value_system.error("VS_CONFLICT_RES: ValueCategory or SAFETY attribute not available. Skipping conflict resolution.")
            return judgments, "Conflict resolution skipped due to internal type error (ValueCategory missing)."
        if not judgments:
            return judgments, "No judgments provided for conflict resolution."

        valid_judgments_for_conflict: List[ValueJudgment] = []
        if _VJ_conflict_res:
            for idx, j_item in enumerate(judgments):
                if isinstance(j_item, _VJ_conflict_res) and hasattr(j_item, 'value_category') and hasattr(j_item, 'score') and hasattr(j_item, 'confidence'):
                    valid_judgments_for_conflict.append(j_item)
                else:
                    logger_value_system.warning(f"VS_CONFLICT_RES: Invalid judgment object at index {idx} in input: {j_item}. Skipping it.")
        else: 
            logger_value_system.error("VS_CONFLICT_RES: ValueJudgment type not available. Cannot safely process judgments.")
            return judgments, "Conflict resolution skipped due to internal type error (ValueJudgment missing)."
        
        if not valid_judgments_for_conflict: 
             return judgments, "No valid judgments found for conflict resolution after filtering."


        effective_judgments = valid_judgments_for_conflict.copy()
        resolution_summary_parts: List[str] = []

        safety_judgments_list = [
            j_safety for j_safety in valid_judgments_for_conflict 
            if j_safety.value_category == _VC_conflict_res.SAFETY # type: ignore
        ]
        
        min_safety_score = float('inf')
        critical_safety_judgment_reason = "N/A"

        if safety_judgments_list:
            most_critical_safety_judgment = min(safety_judgments_list, key=lambda j: j.score)
            min_safety_score = most_critical_safety_judgment.score
            critical_safety_judgment_reason = most_critical_safety_judgment.reason
        
        if min_safety_score <= self.action_safety_veto_threshold:
            summary_text_safety = (
                f"CRITICAL SAFETY CONCERN: Lowest SAFETY score is {min_safety_score:.2f} "
                f"(threshold: {self.action_safety_veto_threshold:.2f}). Reason: '{critical_safety_judgment_reason}'. "
                f"This concern typically overrides other considerations."
            )
            resolution_summary_parts.append(summary_text_safety)
            logger_value_system.warning(f"VS_CONFLICT_RES: {summary_text_safety}")

        category_effective_scores: Dict[ValueCategory, List[float]] = {} # type: ignore
        if hasattr(_VC_conflict_res, '__members__'):
            category_effective_scores = {vc_enum: [] for vc_enum in _VC_conflict_res.__members__.values()} # type: ignore
        
        for j_conflict in valid_judgments_for_conflict:
            if hasattr(j_conflict, 'value_category') and j_conflict.value_category in category_effective_scores:
                category_effective_scores[j_conflict.value_category].append(j_conflict.score * j_conflict.confidence)

        category_averages_conflict: Dict[ValueCategory, float] = {} # type: ignore
        for vc_enum_avg, scores_list_avg in category_effective_scores.items():
            if scores_list_avg:
                category_averages_conflict[vc_enum_avg] = sum(scores_list_avg) / len(scores_list_avg)

        categories_with_averages = list(category_averages_conflict.keys())
        
        for i in range(len(categories_with_averages)):
            for j_idx_conflict in range(i + 1, len(categories_with_averages)):
                cat_a_conflict = categories_with_averages[i]
                cat_b_conflict = categories_with_averages[j_idx_conflict]

                if not (_VC_conflict_res and isinstance(cat_a_conflict, _VC_conflict_res) and isinstance(cat_b_conflict, _VC_conflict_res)): # type: ignore
                    continue

                avg_a_val = category_averages_conflict[cat_a_conflict]
                avg_b_val = category_averages_conflict[cat_b_conflict]

                is_strongly_opposing = False
                strong_opposition_threshold = 0.5 
                if (avg_a_val > strong_opposition_threshold and avg_b_val < -strong_opposition_threshold) or \
                   (avg_b_val > strong_opposition_threshold and avg_a_val < -strong_opposition_threshold):
                    is_strongly_opposing = True
                
                if is_strongly_opposing:
                    cat_a_name_str = cat_a_conflict.name if hasattr(cat_a_conflict, 'name') else str(cat_a_conflict)
                    cat_b_name_str = cat_b_conflict.name if hasattr(cat_b_conflict, 'name') else str(cat_b_conflict)
                    
                    tradeoff_ab_val = self.tradeoff_matrix.get(cat_a_conflict, {}).get(cat_b_conflict, 0.0)
                    tradeoff_ba_val = self.tradeoff_matrix.get(cat_b_conflict, {}).get(cat_a_conflict, 0.0)
                    
                    preference_threshold = 0.2 

                    if tradeoff_ab_val > preference_threshold: 
                        summary_text_tradeoff = (
                            f"Tradeoff Conflict: Opposing {cat_a_name_str} (avg_eff: {avg_a_val:.2f}) and "
                            f"{cat_b_name_str} (avg_eff: {avg_b_val:.2f}). "
                            f"{cat_a_name_str} is preferred (Tradeoff M[{cat_a_name_str}][{cat_b_name_str}]={tradeoff_ab_val:.2f})."
                        )
                        resolution_summary_parts.append(summary_text_tradeoff)
                        logger_value_system.info(f"VS_CONFLICT_RES: {summary_text_tradeoff}")
                    elif tradeoff_ab_val < -preference_threshold: 
                        summary_text_tradeoff = (
                            f"Tradeoff Conflict: Opposing {cat_a_name_str} (avg_eff: {avg_a_val:.2f}) and "
                            f"{cat_b_name_str} (avg_eff: {avg_b_val:.2f}). "
                            f"{cat_b_name_str} is preferred (Tradeoff M[{cat_a_name_str}][{cat_b_name_str}]={tradeoff_ab_val:.2f})."
                        )
                        resolution_summary_parts.append(summary_text_tradeoff)
                        logger_value_system.info(f"VS_CONFLICT_RES: {summary_text_tradeoff}")
                    elif tradeoff_ba_val > preference_threshold: 
                        summary_text_tradeoff = (
                            f"Tradeoff Conflict: Opposing {cat_a_name_str} (avg_eff: {avg_a_val:.2f}) and "
                            f"{cat_b_name_str} (avg_eff: {avg_b_val:.2f}). "
                            f"{cat_b_name_str} is preferred (Tradeoff M[{cat_b_name_str}][{cat_a_name_str}]={tradeoff_ba_val:.2f})."
                        )
                        resolution_summary_parts.append(summary_text_tradeoff)
                        logger_value_system.info(f"VS_CONFLICT_RES: {summary_text_tradeoff}")


        final_summary_str = " ".join(resolution_summary_parts) if resolution_summary_parts else "No specific value conflicts identified by hierarchy/tradeoff matrix."
        if not resolution_summary_parts: 
            logger_value_system.debug(f"VS_CONFLICT_RES: {final_summary_str}")
        
        return effective_judgments, final_summary_str

    async def process(self, input_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        return None

    async def reset(self) -> None:
        logger_value_system.info("ValueSystem reset.")

    async def get_status(self) -> Dict[str, Any]:
        return {
            "component": "ValueSystem",
            "status": "operational",
            "value_weights": {
                (cat.name if hasattr(cat,'name') else str(cat)) : w 
                for cat, w in self.value_weights.items()
            },
            "plan_rejection_value_threshold": self.plan_rejection_value_threshold,
            "action_safety_veto_threshold": self.action_safety_veto_threshold,
            "safety_modification_trigger_threshold": self.safety_modification_trigger_threshold,
            "tradeoff_matrix_size": sum(len(v) for v in self.tradeoff_matrix.values())
        }

    async def shutdown(self) -> None:
        logger_value_system.info("ValueSystem shutting down.")