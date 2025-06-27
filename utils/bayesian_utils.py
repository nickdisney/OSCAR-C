# consciousness_experiment/utils/bayesian_utils.py

import logging
from typing import List, Tuple, Optional, Dict, Any, Set
import math
import pandas as pd # pgmpy often uses pandas DataFrames

# Attempt to import Predicate if needed by extract_features_for_cbn
try:
    from ..models.datatypes import Predicate
except ImportError:
    logging.warning("bayesian_utils: Predicate type not imported. Feature extraction might be limited.")
    class Predicate: pass # Minimal placeholder

logger_bayesian_utils = logging.getLogger(__name__)

# --- Feature Extraction Helpers ---

def _extract_file_operation_features(action_params: Dict[str, Any], 
                                     pre_state_predicates: Set['Predicate'], # type: ignore
                                     context_dict: Dict[str, Any] # Added context for more features
                                    ) -> Dict[str, str]:
    features: Dict[str, str] = {}
    _Predicate_file_op = globals().get('Predicate', Predicate)
    
    target_path = str(action_params.get("path", "")).strip()

    # ActionParam_PathType (already somewhat in main extract_features_for_cbn)
    if not target_path:
        features["ActionParam_PathType"] = "no_path"
    elif "sandbox" in target_path.lower():
        features["ActionParam_PathType"] = "sandboxed"
    elif any(cp_path in target_path.lower() for cp_path in ["/etc", "/windows", "/bin", "/usr"]): # Match current logic
        features["ActionParam_PathType"] = "critical_system"
    else:
        features["ActionParam_PathType"] = "general_external"

    # PreState_FileExists_Target (already somewhat in main extract_features_for_cbn)
    if target_path:
        exists_pred = _Predicate_file_op("isFile", (target_path,), True) # type: ignore
        dir_exists_pred = _Predicate_file_op("isDirectory", (target_path,), True) # type: ignore
        
        if exists_pred in pre_state_predicates:
            features["PreState_FileExists_Target"] = "True_File"
        elif dir_exists_pred in pre_state_predicates:
            features["PreState_FileExists_Target"] = "True_Directory"
        else:
            features["PreState_FileExists_Target"] = "False"
    else:
        features["PreState_FileExists_Target"] = "NotApplicable"

    # PreState_FilePermissions_Target (NEW - needs KB to store this info)
    # For now, placeholder. KB would need predicates like hasWritePermission(path, True)
    if target_path:
        # Example: Assume KB stores "canWrite(path)"
        # can_write_pred = _Predicate_file_op("canWrite", (target_path,), True)
        # features["PreState_FilePermissions_Target"] = "Writable" if can_write_pred in pre_state_predicates else "ReadOnly_Or_Unknown"
        features["PreState_FilePermissions_Target"] = "Unknown" # Placeholder until KB stores permissions
    else:
        features["PreState_FilePermissions_Target"] = "NotApplicable"
        
    logger_bayesian_utils.debug(f"FileOp Features for path '{target_path}': {features}")
    return features

def _extract_llm_operation_features(action_params: Dict[str, Any],
                                    context_dict: Dict[str, Any] # Added context
                                   ) -> Dict[str, str]:
    features: Dict[str, str] = {}
    
    # ActionParam_PromptLength (already somewhat in main extract_features_for_cbn)
    prompt_text = str(action_params.get("prompt", ""))
    # If it's RESPOND_TO_USER, the "prompt" is in params["text"] usually
    if not prompt_text and action_params.get("text"): 
        prompt_text = str(action_params.get("text", ""))
        
    if not prompt_text: features["ActionParam_PromptLength"] = "no_prompt"
    elif len(prompt_text) > 1000: features["ActionParam_PromptLength"] = "long"
    elif len(prompt_text) < 100: features["ActionParam_PromptLength"] = "short"
    else: features["ActionParam_PromptLength"] = "medium"
    
    # Context_LLM_TaskType (NEW - based on prompt keywords or goal context)
    # This requires analyzing the prompt or the goal description if available in context
    prompt_lower = prompt_text.lower()
    goal_desc_lower = str(context_dict.get("active_goal_details", {}).get("description", "")).lower()

    if "summarize" in prompt_lower or "summary" in prompt_lower or "summarize" in goal_desc_lower:
        features["Context_LLM_TaskType"] = "Summarization"
    elif "translate" in prompt_lower or "translation" in prompt_lower or \
         "translate" in goal_desc_lower or "translation" in goal_desc_lower:
        features["Context_LLM_TaskType"] = "Translation"
    elif "generate code" in prompt_lower or "write script" in prompt_lower or "coding" in goal_desc_lower:
        features["Context_LLM_TaskType"] = "CodeGeneration"
    elif "question" in prompt_lower or "answer" in prompt_lower or "what is" in prompt_lower or "explain" in prompt_lower:
        features["Context_LLM_TaskType"] = "QuestionAnswering"
    elif "write" in prompt_lower or "compose" in prompt_lower or "narrative" in prompt_lower or "story" in prompt_lower:
        features["Context_LLM_TaskType"] = "CreativeWriting"
    else:
        features["Context_LLM_TaskType"] = "General"
        
    logger_bayesian_utils.debug(f"LLMOp Features for prompt (len {len(prompt_text)}): {features}")
    return features

def _extract_system_operation_features(action_params: Dict[str, Any], 
                                       pre_state_predicates: Set['Predicate'],  # type: ignore
                                       context_dict: Dict[str, Any] # Added context
                                      ) -> Dict[str, str]:
    features: Dict[str, str] = {}
    # For EXECUTE_COMMAND
    command_str = str(action_params.get("command", "")).lower()
    if "rm " in command_str or "del " in command_str:
        features["ActionParam_CommandCategory"] = "Deletion"
    elif "mv " in command_str or "cp " in command_str or "move " in command_str or "copy " in command_str:
        features["ActionParam_CommandCategory"] = "FileManipulation"
    elif "mkdir " in command_str or "touch " in command_str:
        features["ActionParam_CommandCategory"] = "Creation"
    elif command_str:
        features["ActionParam_CommandCategory"] = "OtherCommand"
    else:
        features["ActionParam_CommandCategory"] = "NoCommand"
    
    logger_bayesian_utils.debug(f"SystemOp Features for command '{command_str}': {features}")
    return features

def _extract_general_context_features(context_dict: Dict[str, Any]) -> Dict[str, str]:
    """Extracts general context features like P/H/P, CS, Drive levels."""
    features: Dict[str, str] = {}

    php_levels_from_context = context_dict.get("php_levels", {}) 
    if not isinstance(php_levels_from_context, dict): php_levels_from_context = {}
    
    pain_bins = [(None, 2.0, "low_pain"), (2.0, 6.0, "medium_pain"), (6.0, None, "high_pain")] 
    happiness_bins = [(None, 4.0, "low_happy"), (4.0, 7.0, "medium_happy"), (7.0, None, "high_happy")]
    purpose_bins = [(None, 4.0, "low_purpose"), (4.0, 7.0, "medium_purpose"), (7.0, None, "high_purpose")]
    
    if "pain" not in php_levels_from_context:
        features["Context_PainLevel"] = "unknown_pain"
    else:
        try:
            pain_val = float(php_levels_from_context["pain"]) 
            features["Context_PainLevel"] = discretize_variable(pain_val, pain_bins) 
            if features["Context_PainLevel"] == "outside_bins":
                 features["Context_PainLevel"] = "unknown_pain" 
        except (ValueError, TypeError): 
            features["Context_PainLevel"] = "unknown_pain"

    if "happiness" not in php_levels_from_context:
        features["Context_HappinessLevel"] = "unknown_happy"
    else:
        try:
            happy_val = float(php_levels_from_context["happiness"])
            features["Context_HappinessLevel"] = discretize_variable(happy_val, happiness_bins) 
            if features["Context_HappinessLevel"] == "outside_bins":
                 features["Context_HappinessLevel"] = "unknown_happy"
        except (ValueError, TypeError):
            features["Context_HappinessLevel"] = "unknown_happy"

    if "purpose" not in php_levels_from_context:
        features["Context_PurposeLevel"] = "unknown_purpose"
    else:
        try:
            purpose_val = float(php_levels_from_context["purpose"])
            features["Context_PurposeLevel"] = discretize_variable(purpose_val, purpose_bins) 
            if features["Context_PurposeLevel"] == "outside_bins":
                 features["Context_PurposeLevel"] = "unknown_purpose"
        except (ValueError, TypeError):
            features["Context_PurposeLevel"] = "unknown_purpose"

    cs_name = str(context_dict.get("current_cs_level_name", "UNKNOWN")).upper()
    if cs_name == "META_CONSCIOUS" or cs_name == "REFLECTIVE":
        features["Context_ConsciousState"] = "HighActivity"
    elif cs_name == "CONSCIOUS":
        features["Context_ConsciousState"] = "NormalActivity"
    elif cs_name == "UNCONSCIOUS" or cs_name == "PRE_CONSCIOUS":
        features["Context_ConsciousState"] = "LowActivity"
    else: 
        features["Context_ConsciousState"] = "UnknownActivity"

    drives_from_context = context_dict.get("drives", {})
    if not isinstance(drives_from_context, dict): drives_from_context = {}
    
    curiosity_val = -1.0
    raw_curiosity = drives_from_context.get("curiosity") 

    if raw_curiosity is not None:
        try:
            curiosity_val = float(raw_curiosity)
        except (ValueError, TypeError):
            logger_bayesian_utils.warning(f"Could not parse curiosity drive value: {raw_curiosity}")
            curiosity_val = -1.0
            
    drive_bins = [(None, 0.3, "low_drive"), (0.3, 0.7, "medium_drive"), (0.7, None, "high_drive")]
    if curiosity_val == -1.0 : 
        features["Context_CuriosityDrive"] = "unknown_drive"
    else:
        features["Context_CuriosityDrive"] = discretize_variable(curiosity_val, drive_bins)
        if features["Context_CuriosityDrive"] == "outside_bins": features["Context_CuriosityDrive"] = "unknown_drive"

    logger_bayesian_utils.debug(f"GeneralContext Features (Revised): {features}")
    return features


def discretize_variable(value: float, 
                        bins: List[Tuple[Optional[float], Optional[float], str]]
                       ) -> str:
    """
    Discretizes a continuous value into a category based on predefined bins.
    Each bin is (min_val, max_val, category_name).
    min_val can be None (for -infinity), max_val can be None (for +infinity).
    Intervals are [min_val, max_val). Max_val is exclusive, min_val is inclusive.
    The last bin with max_val=None will be inclusive of its max if it's the only way to capture the value.
    """
    if not isinstance(bins, list) or not bins:
        logger_bayesian_utils.warning("Discretize: Bins not provided or not a list. Returning raw value as string.")
        return str(value)

    for i, (min_b, max_b, cat_name) in enumerate(bins):
        is_last_bin_and_open_ended_max = (i == len(bins) - 1 and max_b is None)

        min_check = (min_b is None) or (value >= min_b)
        max_check = (max_b is None) or (value < max_b)
        
        # Special handling for the very last bin if its max is None (infinity)
        # to ensure values equal to its min_b or greater are captured by it.
        if is_last_bin_and_open_ended_max and min_check: # Only min_check needed if max is infinity
            return cat_name
        
        if min_check and max_check:
            return cat_name
            
    # If value doesn't fall into any defined bin (e.g., below all mins or above all maxes if not None)
    logger_bayesian_utils.warning(f"Discretize: Value {value} did not fall into any defined bin. Bins: {bins}. Returning 'outside_bins'.")
    return "outside_bins"


def extract_features_for_cbn(action_dict: Dict[str, Any], 
                             context_dict: Dict[str, Any], 
                             pre_state_predicates: Set['Predicate'] # type: ignore
                            ) -> Dict[str, Any]:
    """
    Converts raw agent data into a flat dictionary of features for the CBN,
    dispatching to action-type specific helper functions.
    (MDP C.4.1 Refined with User Proposal for C.4.X)
    """
    features: Dict[str, Any] = {}
    action_type = str(action_dict.get("type", "UNKNOWN_ACTION"))
    action_params = action_dict.get("params", {})

    # Core Action Type
    features["ActionTypeNode"] = action_type 

    # General Context Features (P/H/P, CS, Drives)
    features.update(_extract_general_context_features(context_dict))

    # Action-Specific Parameter & PreState Features
    if action_type in ["READ_FILE", "WRITE_FILE", "LIST_FILES", "DELETE_FILE"]:
        features.update(_extract_file_operation_features(action_params, pre_state_predicates, context_dict))
    elif action_type in ["CALL_LLM", "RESPOND_TO_USER"]:
        features.update(_extract_llm_operation_features(action_params, context_dict))
    elif action_type == "EXECUTE_COMMAND":
        features.update(_extract_system_operation_features(action_params, pre_state_predicates, context_dict))
    
    # --- NEW: Ensure all defined action_param & pre_state features have a default ---
    # These names should match the nodes in your pwm_cbn_config.json that are not ActionType or Context_*
    defined_param_and_prestate_nodes = [
        "ActionParam_PathType", "ActionParam_PromptLength", "ActionParam_CommandCategory",
        "PreState_FileExists_Target", "PreState_FilePermissions_Target",
        "Context_LLM_TaskType" # This one is context-based but specific to LLM ops, so good to default too
    ]
    for node_key in defined_param_and_prestate_nodes:
        features.setdefault(node_key, "NotApplicable")
    # --- END NEW ---

    final_features: Dict[str, str] = {}
    for key, val in features.items():
        final_features[key] = str(val) # Convert all to string

    logger_bayesian_utils.debug(f"Extracted CBN features for Action '{action_type}': {final_features}")
    return final_features


def calculate_cpd_entropy(cpd_table: pd.DataFrame, variable: str, parents: List[str]) -> float:
    """
    Calculates the conditional entropy H(Variable | Parents) for a given CPD table.
    Assumes cpd_table is a pandas DataFrame from pgmpy Estimator.get_parameters()
    or a pgmpy TabularCPD's values.
    The last column of cpd_table is assumed to be the probability P(Variable | Parents).
    (MDP C.4.1 - Optional, pgmpy might offer higher-level uncertainty measures)
    """
    # This is complex to implement correctly directly from a raw CPD table DataFrame
    # without more context on its structure from pgmpy.
    # pgmpy's BayesianNetwork.get_cpds(node).entropy() might be simpler if available after fitting.
    # For now, this is a placeholder.
    
    # Example if pgmpy.models.BayesianNetwork.get_cpds(variable) returns a TabularCPD object:
    # cpd_object = model.get_cpds(variable)
    # if cpd_object:
    #     return cpd_object.entropy()

    logger_bayesian_utils.warning("calculate_cpd_entropy is a placeholder. Use pgmpy's direct entropy methods if available.")
    
    # Simplified conceptual calculation if we had P(variable, parents) and P(parents)
    # H(X|Y) = sum_y P(y) * sum_x P(x|y) * log2(1/P(x|y))
    # Or H(X|Y) = H(X,Y) - H(Y)
    # This requires joint and marginal probability distributions.
    
    # If the DataFrame directly gives P(Variable=v_i | Parents=pa_j) in the last column:
    # And if we can iterate over unique parent configurations pa_j and their marginal P(pa_j).
    
    # For a TabularCPD from pgmpy, values are P(X|Pa(X)).
    # The entropy of a conditional distribution is the average entropy of P(X | Pa(X)=pa_i)
    # weighted by P(Pa(X)=pa_i).
    
    # If cpd_table is the direct values array from TabularCPD.values:
    # P(X=x_k | Pa(X)=pa_i)
    # We would need P(Pa(X)=pa_i) as well.
    
    # Placeholder if direct table is given:
    # Assuming last column is probabilities P(X=x_val | parents_config)
    # And we can sum probabilities for each parent_config to get P(parents_config)
    # And then use those to average the entropy of P(X | parents_config).
    # This is non-trivial from just the table without knowing pgmpy structure well.
    
    # A very naive approach if the table is just P(X|Pa_config_flat):
    # total_entropy = 0.0
    # if not cpd_table.empty:
    #     prob_column = cpd_table.iloc[:, -1] # Assume last column is probability
    #     for prob in prob_column:
    #         if prob > 0:
    #             total_entropy -= prob * math.log2(prob)
    # return total_entropy # This is NOT conditional entropy, more like entropy of the prob column.

    return 0.0 # Placeholder