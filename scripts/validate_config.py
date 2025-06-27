# --- START OF FILE scripts/validate_config.py ---

"""Validates OSCAR-C configuration file against requirements."""

import sys
import logging # Ensure logging is imported if used by other parts of the file
from pathlib import Path
import toml
from typing import Dict, Any, List, Optional, Tuple, Union # Added Union
# Import psutil conditionally at the top if hardware checks are to be re-enabled
# try:
#     import psutil
#     PSUTIL_AVAILABLE = True
# except ImportError:
#     psutil = None
#     PSUTIL_AVAILABLE = False
#     logging.warning("psutil not found. Hardware requirement checks skipped.")


# --- Configuration Requirements ---
REQUIRED_CONFIG_KEYS = {
    "filesystem": {
        "max_list_items": int,
        "max_read_chars": int,
        "allow_file_write": bool,
        "allow_overwrite": bool,
    },
    "agent": {
        "pid_file_name": str,
        "ui_meter_update_interval_s": float,
        "goal_re_evaluation_interval_cycles": int,
        "default_goal_cooldown_cycles": int,
        "min_curiosity_for_observe": float,
        "max_consecutive_planning_failures": int,
        "max_execution_failures_per_goal": int,
    },
    "agent_data_paths": {
        "pid_directory": str,
        "kb_db_path": str,
        "narrative_log_path": str,
        "self_model_path": str,
        "predictive_model_path": str, # Keep for now, even if "OLD"
        "predictive_model_learned_data_path": str, # Added
        "performance_adjustments_path": str,
        "htn_library_path": str, # Added
    },
    "global_workspace": {
        "capacity": int,
        "broadcast_threshold": float,
        "min_items_if_any_attended": int,
    },
    "attention_controller": {
        "recency_weight": float,
        "hint_weight": float,
        "goal_relevance_weight": float,
        "max_candidates": int,
        "softmax_temperature": float,
        "novelty_window_size": int,
        "novelty_bonus_weight": float,
        "surprise_bonus_weight": float,
        "pain_attention_distraction_factor": float,
        "pain_rumination_threshold_cycles": int,
        "pain_rumination_window_multiplier": int,
        "pain_inactive_reset_cycles": int,
        "pain_rumination_suppression_factor": float,
    },
    "consciousness_assessor": {
        "meta_conscious_threshold": float,
        "conscious_threshold": float,
        "pre_conscious_threshold": float,
        "unconscious_threshold": float,
        "diff_weight_sources": float,
        "diff_weight_lexical": float,
        "int_weight_shared_concepts": float,
        "phi_contrib_diff": float,
        "phi_contrib_int": float,
        "global_workspace_capacity_for_norm": int,
        "phi_calculator_integration_weight": float, # Added
        "cla_stopwords": list, # Added
        "cla_goal_keywords": list, # Added
        "cla_coherence_edge_threshold": float, # Added
        "differentiation_norm_factor": float, # Added
    },
    "performance": {
        "target_cycle_time": float,
        "max_planning_depth": int,
        "profiler_history_size": int,
    },
    "loop_detection": {
        "window_size": int,
        "max_consecutive_actions": int,
        "frequency_threshold": float,
        "ignore_thinking_actions": bool,
    },
    "knowledge_base": {
        "default_context_retrieval_count": int,
    },
    "cognitive_cache": {
        "default_ttl": float,
    },
    "telemetry": {
        "enabled": bool,
        "host": str,
        "port": int,
    },
    "error_recovery": {
        "max_error_history": int,
        "frequency_window": int,
        "frequency_threshold": int,
    },
    "dynamic_self_model": {
        "learning_rate": float,
        "max_invalid_paths": int,
        "reflection_interval_cycles": int, # Added
        "learning_events_history_size": int, # Added
    },
    "emergent_motivation_system": {
        "detailed_evaluation_interval_cycles": int,
        "ems_cs_history_maxlen": int,
        "ems_low_cs_persistence_threshold": int,
        "ems_low_cs_curiosity_boost_factor": float,
        "drives": dict, # Specify that "drives" should be a dictionary
    },
    # Structure for emergent_motivation_system.drives.DRIVE_NAME
    # is handled by _custom_validate_ems_drives
    "narrative_constructor": {
        "max_length": int,
        "valence_change_threshold": float,
        "intensity_threshold": float,
        "save_interval_s": (int, float), # Can be int or float
        "drive_change_threshold": float,
        # "significance_threshold": float, # This was in old config, seems replaced by specific thresholds
        "pain_change_threshold_sig": float,
        "happiness_change_threshold_sig": float,
        "purpose_change_threshold_sig": float,
    },
    "predictive_world_model": {
        "initial_version": int,
        "learning_rate": float,
        "memory_length": int,
        "save_interval_versions": int, # Added
        "cpd_reestimation_trigger_count_frequent": int, # Added
        "cbn_config_file": str, # Added
        "cbn_query_cache_ttl_s": float, # Added
        "cbn_target_outcome_node": str, # Added
    },
    "performance_optimizer": {
        "history_size": int,
        "auto_apply_adjustments": bool, # Added
    },
    "experience_stream": {
        "intensity_factor": float,
        "valence_decay": float,
        "custom_stopwords": list, # Added
    },
    "meta_cognition": {
        "stagnation_threshold_s": (int, float),
        "failure_rate_threshold": float,
        "low_consciousness_threshold_s": (int, float),
        "history_size": int,
        "reflection_trigger_frequency_cycles": int,
    },
    "llm_settings": {
        "default_llm_model": str, # Added
        "default_timeout_s": float,
        "default_enable_thinking": bool, # Added
        "action_selection_temperature": float,
        "max_recent_turns_for_prompt": int,
        "intent_mapping_temperature": float,
    },
    # New section for specialized LLMs
    "oscar_specialized_llms": { # This section itself is optional, but if present, its keys are checked
        # Keys inside this are dynamic based on what's configured.
        # We'll check for common sub-keys like _model, _enable_thinking, _temperature, _timeout_s
    },
    "htn_planner": {
        "plan_cache_ttl_s": (int, float),
        "min_planning_depth_on_low_cs": int,
        "max_planning_depth_on_low_cs": int,
        "low_cs_simplicity_penalty_factor": float,
        # llm_sketch settings are now part of oscar_specialized_llms
    },
    # htn_planner.htn_planner_pruning is a sub-table
    "internal_states": {
        "baseline_pain_age_factor": float,
        "max_baseline_pain_from_age": float,
        "acute_pain_goal_fail_priority_scale_factor": float,
        "pain_from_planning_failure_scale_factor": float, # Added
        "pain_from_execution_failure_scale_factor": float, # Added
        "pain_event_max_initial_intensity": float,
        "default_pain_event_decay_rate_per_cycle": float,
        "pain_event_min_intensity_to_retain": float,
        "happiness_from_goal_priority_scale_factor": float,
        "happiness_decay_to_baseline_factor": float,
        "happiness_baseline_target": float,
        "pain_impact_on_happiness_scale_factor": float,
        "purpose_from_capability_gain_factor": float,
        "purpose_from_high_priority_goal_factor": float,
        "purpose_decay_rate_per_cycle": float,
        "complex_goal_priority_threshold": float,
        "max_pain_shutdown_threshold": float,
        "min_purpose_shutdown_threshold": float,
    },
    "value_system": {
        "plan_rejection_value_threshold": float,
        "action_safety_veto_threshold": float,
        "safety_critical_paths_write": list, # Added
        "safety_modification_trigger_threshold": float, # Added
        # value_weights and tradeoff_matrix are optional sub-tables
    },
    "phi_calculator": { # Added
        "max_partitions_to_evaluate": int,
    },
    "computation_budgets": { # Added
        "phi_calculator_max_time_ms_per_cycle": int,
        "pwm_cbn_fit_max_time_ms_overall": int,
        "htn_method_learning_max_time_ms": int,
    },
    "os_integration": { # Added
        "pain_threshold_cpu": float,
        "pain_threshold_memory": float,
        "allowed_commands": list,
        "sandbox_dir": str,
        "allowed_http_domains_for_curl": list,
    },
}


# Optional: Define range checks or specific value checks
VALUE_CHECKS: Dict[str, Any] = { # Allow Any for the lambda type with config access
    "filesystem.max_list_items": lambda x: x > 0,
    "filesystem.max_read_chars": lambda x: x > 0,
    "agent.ui_meter_update_interval_s": lambda x: x > 0,
    "agent.default_goal_cooldown_cycles": lambda x: x >=0,
    "agent.min_curiosity_for_observe": lambda x: 0.0 <= x <= 1.0,
    "agent.max_consecutive_planning_failures": lambda x: x >= 1,
    "agent.max_execution_failures_per_goal": lambda x: x >= 1,
    "global_workspace.capacity": lambda x: x > 0,
    "global_workspace.broadcast_threshold": lambda x: 0.0 <= x <= 1.0,
    "global_workspace.min_items_if_any_attended": lambda x: x >= 0,
    "attention_controller.softmax_temperature": lambda x: x > 0,
    "attention_controller.max_candidates": lambda x: x > 0,
    "attention_controller.novelty_window_size": lambda x: x > 0,
    "attention_controller.pain_attention_distraction_factor": lambda x: 0.0 <= x <= 1.0,
    "attention_controller.pain_rumination_threshold_cycles": lambda x: x >= 1,
    "attention_controller.pain_rumination_window_multiplier": lambda x: x >= 1,
    "attention_controller.pain_inactive_reset_cycles": lambda x: x >= 1,
    "attention_controller.pain_rumination_suppression_factor": lambda x: 0.0 <= x <= 1.0,
    "performance.target_cycle_time": lambda x: x > 0,
    "performance.max_planning_depth": lambda x: x >= 1,
    "performance.profiler_history_size": lambda x: x > 0,
    "loop_detection.window_size": lambda x: x > 0,
    "loop_detection.max_consecutive_actions": lambda x: x > 0,
    "loop_detection.frequency_threshold": lambda x: 0.0 < x <= 1.0,
    "knowledge_base.default_context_retrieval_count": lambda x: x >= 0,
    "cognitive_cache.default_ttl": lambda x: x >= 0,
    "telemetry.port": lambda x: 1 <= x <= 65535, # Port 0 is valid but usually not intended
    "error_recovery.max_error_history": lambda x: x > 0,
    "error_recovery.frequency_window": lambda x: x > 0,
    "error_recovery.frequency_threshold": lambda x: x > 0,
    "dynamic_self_model.learning_rate": lambda x: 0.0 < x <= 1.0,
    "dynamic_self_model.max_invalid_paths": lambda x: x >= 0,
    "dynamic_self_model.reflection_interval_cycles": lambda x: x > 0,
    "dynamic_self_model.learning_events_history_size": lambda x: x > 0,
    "emergent_motivation_system.detailed_evaluation_interval_cycles": lambda x: x > 0,
    "emergent_motivation_system.ems_cs_history_maxlen": lambda x: x > 0,
    "emergent_motivation_system.ems_low_cs_persistence_threshold": lambda x: x > 0,
    "emergent_motivation_system.ems_low_cs_curiosity_boost_factor": lambda x: 0.0 <= x <= 1.0, # Adjusted range
    "consciousness_assessor.meta_conscious_threshold": lambda x: 0.0 < x <= 1.0,
    "consciousness_assessor.conscious_threshold": lambda x: 0.0 < x <= 1.0,
    "consciousness_assessor.pre_conscious_threshold": lambda x: 0.0 < x <= 1.0,
    "consciousness_assessor.unconscious_threshold": lambda x: 0.0 <= x < 1.0, # Can be 0
    "consciousness_assessor.global_workspace_capacity_for_norm": lambda x: x > 0,
    "consciousness_assessor.phi_calculator_integration_weight": lambda x: 0.0 <= x <= 1.0,
    "consciousness_assessor.cla_coherence_edge_threshold": lambda x: 0.0 <= x <= 1.0,
    "consciousness_assessor.differentiation_norm_factor": lambda x: x > 0.0,
    "narrative_constructor.max_length": lambda x: x > 0,
    "narrative_constructor.valence_change_threshold": lambda x: x >= 0,
    "narrative_constructor.intensity_threshold": lambda x: x >= 0,
    "narrative_constructor.save_interval_s": lambda x: x >= 0,
    "narrative_constructor.drive_change_threshold": lambda x: x >=0,
    # "narrative_constructor.significance_threshold": lambda x: x >= 0, # Removed, more specific now
    "narrative_constructor.pain_change_threshold_sig": lambda x: 0.0 <= x <= 10.0, # Can be 0 if no change needed
    "narrative_constructor.happiness_change_threshold_sig": lambda x: 0.0 <= x <= 10.0,
    "narrative_constructor.purpose_change_threshold_sig": lambda x: 0.0 <= x <= 10.0,
    "predictive_world_model.learning_rate": lambda x: 0.0 < x <= 1.0,
    "predictive_world_model.memory_length": lambda x: x > 0,
    "predictive_world_model.save_interval_versions": lambda x: x >= 0,
    "predictive_world_model.cpd_reestimation_trigger_count_frequent": lambda x: x >= 1,
    "predictive_world_model.cbn_query_cache_ttl_s": lambda x: x >= 0,
    "performance_optimizer.history_size": lambda x: x > 0,
    "experience_stream.intensity_factor": lambda x: x > 0,
    "experience_stream.valence_decay": lambda x: 0.0 <= x <= 1.0,
    "meta_cognition.stagnation_threshold_s": lambda x: x > 0,
    "meta_cognition.failure_rate_threshold": lambda x: 0.0 < x <= 1.0,
    "meta_cognition.low_consciousness_threshold_s": lambda x: x > 0,
    "meta_cognition.history_size": lambda x: x > 0,
    "meta_cognition.reflection_trigger_frequency_cycles": lambda x: x > 0,
    "llm_settings.default_timeout_s": lambda x: x > 0,
    "llm_settings.action_selection_temperature": lambda x: 0.0 <= x <= 2.0,
    "llm_settings.max_recent_turns_for_prompt": lambda x: x >= 0,
    "llm_settings.intent_mapping_temperature": lambda x: 0.0 <= x <= 2.0,
    "htn_planner.plan_cache_ttl_s": lambda x: x >= 0,
    "htn_planner.min_planning_depth_on_low_cs": lambda x: x >= 1,
    "htn_planner.max_planning_depth_on_low_cs": lambda x, cfg=config: x >= 1 and x >= cfg.get("htn_planner", {}).get("min_planning_depth_on_low_cs", 1),
    "htn_planner.low_cs_simplicity_penalty_factor": lambda x: x >= 0.0,
    "internal_states.baseline_pain_age_factor": lambda x: 0.0 <= x <= 0.1,
    "internal_states.max_baseline_pain_from_age": lambda x: 0.0 <= x <= 5.0,
    "internal_states.acute_pain_goal_fail_priority_scale_factor": lambda x: 0.0 <= x <= 1.0,
    "internal_states.pain_from_planning_failure_scale_factor": lambda x: 0.0 <= x <= 1.0,
    "internal_states.pain_from_execution_failure_scale_factor": lambda x: 0.0 <= x <= 1.0,
    "internal_states.pain_event_max_initial_intensity": lambda x: 0.1 <= x <= 10.0, # Adjusted from 5.0
    "internal_states.default_pain_event_decay_rate_per_cycle": lambda x: 0.0 <= x <= 0.1,
    "internal_states.pain_event_min_intensity_to_retain": lambda x: 0.0 <= x <= 1.0, # Adjusted from 0.1
    "internal_states.happiness_from_goal_priority_scale_factor": lambda x: 0.0 <= x <= 1.0,
    "internal_states.happiness_decay_to_baseline_factor": lambda x: 0.0 <= x <= 0.1,
    "internal_states.happiness_baseline_target": lambda x: 0.0 <= x <= 10.0,
    "internal_states.pain_impact_on_happiness_scale_factor": lambda x: 0.0 <= x <= 1.0,
    "internal_states.purpose_from_capability_gain_factor": lambda x: 0.0 <= x <= 0.5,
    "internal_states.purpose_from_high_priority_goal_factor": lambda x: 0.0 <= x <= 1.0,
    "internal_states.purpose_decay_rate_per_cycle": lambda x: 0.0 <= x <= 0.01,
    "internal_states.complex_goal_priority_threshold": lambda x: x > 0,
    "internal_states.max_pain_shutdown_threshold": lambda x: 1.0 <= x <= 10.0,
    "internal_states.min_purpose_shutdown_threshold": lambda x: 0.0 <= x <= 5.0,
    "value_system.plan_rejection_value_threshold": lambda x: -1.0 <= x <= 1.0,
    "value_system.action_safety_veto_threshold": lambda x: -1.0 <= x <= 0.0,
    "value_system.safety_modification_trigger_threshold": lambda x: -1.0 <= x <= 0.0,
    "phi_calculator.max_partitions_to_evaluate": lambda x: x > 0,
    "computation_budgets.phi_calculator_max_time_ms_per_cycle": lambda x: x >= 0,
    "computation_budgets.pwm_cbn_fit_max_time_ms_overall": lambda x: x >= 0,
    "computation_budgets.htn_method_learning_max_time_ms": lambda x: x >= 0,
    "os_integration.pain_threshold_cpu": lambda x: 0.0 <= x <= 100.0,
    "os_integration.pain_threshold_memory": lambda x: 0.0 <= x <= 100.0,
}

config: Optional[Dict[str, Any]] = None # Global for VALUE_CHECKS lambda access

# --- Custom Validation Functions ---
def _custom_validate_ems_drives(config_section: Dict[str, Any], errors: List[str], warnings: List[str]):
    """Custom validation for the structure of [emergent_motivation_system.drives]."""
    drives_config = config_section.get("drives")
    if not isinstance(drives_config, dict):
        errors.append("Invalid type for [emergent_motivation_system.drives]. Expected a dictionary.")
        return

    expected_drive_names = {"curiosity", "satisfaction", "competence"}
    for drive_name in expected_drive_names:
        if drive_name not in drives_config:
            warnings.append(f"Optional drive '{drive_name}' not configured in [emergent_motivation_system.drives]. EMS will use defaults.")
            continue
        
        if not isinstance(drives_config[drive_name], dict):
            errors.append(f"Invalid type for [emergent_motivation_system.drives.{drive_name}]. Expected a dictionary of parameters.")
            continue
        
        drive_params = drives_config[drive_name]
        # Check specific parameters within each drive
        # Example for curiosity (can be expanded for others)
        if drive_name == "curiosity":
            for param_key, expected_param_type in {
                "gain_prediction_error": float,
                "gain_from_high_pain_for_distraction": float,
                "gain_from_low_purpose_for_exploration": float,
                "threshold_high_pain_for_curiosity": float,
                "threshold_low_purpose_for_curiosity": float,
                "gain_entropy_uncertainty": float,
                "decay": float, # from DEFAULT_DRIVES
                "gain_discovery": float # from DEFAULT_DRIVES
            }.items():
                if param_key in drive_params and not isinstance(drive_params[param_key], expected_param_type) and \
                   not (expected_param_type is float and isinstance(drive_params[param_key], int)): # Allow int for float
                    errors.append(f"Invalid type for '{param_key}' in [emergent_motivation_system.drives.{drive_name}]. Expected {expected_param_type.__name__}.")
                elif param_key not in drive_params:
                     warnings.append(f"Optional parameter '{param_key}' not found in [emergent_motivation_system.drives.{drive_name}]. Component will use default.")


def _custom_validate_specialized_llms(config_section: Dict[str, Any], errors: List[str], warnings: List[str]):
    """Custom validation for [oscar_specialized_llms] sub-tables."""
    if not isinstance(config_section, dict): # Should be caught by main validator
        return

    for llm_key, llm_config in config_section.items():
        if not isinstance(llm_config, dict):
            errors.append(f"Invalid type for entry '{llm_key}' in [oscar_specialized_llms]. Expected a dictionary of parameters.")
            continue

        # Check for common parameters within each specialized LLM config
        # These are typically optional, so we issue warnings if missing, errors for wrong type.
        common_params = {
            f"{llm_key}_model": str,
            f"{llm_key}_enable_thinking": bool,
            f"{llm_key}_temperature": float,
            f"{llm_key}_timeout_s": float,
        }
        # However, the keys in config.toml are like "persona_dialogue_model", not "persona_dialogue_persona_dialogue_model"
        # The keys in the TOML are the full names like "persona_dialogue_model".
        # So, the logic needs to check for these exact keys if they are present.

        model_key_actual = f"{llm_key}" # The key for the model name is just the llm_key itself in the sub-table
        if model_key_actual in llm_config:
            if not isinstance(llm_config[model_key_actual], str):
                 errors.append(f"Invalid type for '{model_key_actual}' in [oscar_specialized_llms.{llm_key}]. Expected string (Ollama model name).")
        else:
            # If the primary key (e.g., "persona_dialogue_model") itself is missing, it's an error
            # as that defines the model to use. Other sub-keys are optional overrides.
            # However, the structure given by the user is [oscar_specialized_llms].<key_name_from_code> = "ollama_model_tag"
            # And then <key_name_from_code>_enable_thinking, etc.
            # This means `llm_key` here is like "persona_dialogue", "narrative_llm", etc.
            # So, the model name key is `llm_key + "_model"`.

            # Corrected logic based on provided config structure:
            # config has:
            # [oscar_specialized_llms]
            # persona_dialogue_model = "oscar_persona_dialogue_qwen3_4b"
            # persona_dialogue_enable_thinking = true

            # Here, `llm_key` would be e.g. "persona_dialogue_model" if iterating config_section.items().
            # This custom validator is called with config['oscar_specialized_llms'] as config_section.
            # So, llm_key will be "persona_dialogue_model", "persona_dialogue_enable_thinking", etc.
            # This approach for a custom validator is not ideal for this flat structure.

            # Let's adjust. The main validator handles the type of these flat keys.
            # This custom validator should iterate over *logical groups* of specialized LLMs.
            # The current main validator will check `persona_dialogue_model: str`, `persona_dialogue_enable_thinking: bool`, etc.
            # So, this custom validator for `oscar_specialized_llms` might not be strictly needed if all keys are flat.

            # If the intention was to group them like:
            # [oscar_specialized_llms.persona_dialogue]
            # model = "..."
            # enable_thinking = true
            # Then the original custom validator logic would be correct.
            # Given the TOML provided, the main validator handles it.
            # I'll keep this placeholder to show how it *would* work for nested structures.
            pass # For now, let the main validator handle the flat keys in oscar_specialized_llms.


# --- Main Validation Function ---
def validate_config(config_filepath: str = "config.toml") -> bool:
    global config
    config_path = Path(config_filepath)
    is_valid = True
    errors: List[str] = []
    warnings: List[str] = []

    if not config_path.exists() or not config_path.is_file():
        errors.append(f"Configuration file not found or is not a file: {config_path.resolve()}")
        for error in errors: print(f"❌ ERROR: {error}")
        return False

    try:
        loaded_config = toml.load(config_path)
        config = loaded_config # Assign to global for VALUE_CHECKS
        print(f"ℹ️ Successfully parsed config file: {config_path.resolve()}")
    except toml.TomlDecodeError as e:
        errors.append(f"Failed to parse TOML configuration file: {e}")
        is_valid = False
    except Exception as e:
        errors.append(f"An unexpected error occurred during config parsing: {e}")
        is_valid = False

    if not is_valid or config is None:
        for error in errors: print(f"❌ ERROR: {error}")
        return False

    # --- Key and Type Validation ---
    for section, keys_config in REQUIRED_CONFIG_KEYS.items():
        if section not in config:
            # Check if the section itself is optional (e.g., oscar_specialized_llms)
            if section == "oscar_specialized_llms":
                warnings.append(f"Optional configuration section '[{section}]' not found. Specialized LLMs will not be configured beyond defaults.")
                continue # Skip checking keys for an optional missing section
            elif section == "htn_planner.htn_planner_pruning" or section == "value_system.value_weights" or section == "value_system.tradeoff_matrix":
                 warnings.append(f"Optional configuration sub-table '[{section}]' not found. Component will use defaults.")
                 continue
            
            errors.append(f"Missing required configuration section: '[{section}]'")
            is_valid = False
            continue

        if not isinstance(config[section], dict):
             errors.append(f"Configuration section '[{section}]' is not a valid table/dictionary.")
             is_valid = False
             continue

        # --- For oscar_specialized_llms, keys are dynamic. ---
        if section == "oscar_specialized_llms":
            # Check common patterns for each specialized LLM entry
            for key_name in config[section]:
                # Example: key_name could be "persona_dialogue_model", "persona_dialogue_enable_thinking", etc.
                # Determine the base prefix (e.g., "persona_dialogue")
                base_key_parts = key_name.split('_')
                if len(base_key_parts) > 1:
                    suffix = base_key_parts[-1]
                    # prefix = "_".join(base_key_parts[:-1]) # Not needed with current flat structure
                    
                    expected_type: Optional[Union[type, Tuple[type, ...]]] = None
                    if suffix == "model": expected_type = str
                    elif suffix == "thinking": expected_type = bool
                    elif suffix == "temperature": expected_type = float
                    elif suffix == "timeout" or suffix == "s": # Catches _timeout_s
                         if key_name.endswith("_timeout_s"): expected_type = float # More specific
                    
                    if expected_type:
                        current_value = config[section][key_name]
                        is_expected = False
                        if isinstance(expected_type, tuple): is_expected = isinstance(current_value, expected_type)
                        else: is_expected = isinstance(current_value, expected_type)
                        
                        if not is_expected and expected_type is float and isinstance(current_value, int):
                            warnings.append(f"Key '{key_name}' in section '[{section}]' is int, but float expected. Will be treated as float.")
                            is_expected = True

                        if not is_expected:
                            errors.append(f"Invalid type for key '{key_name}' in section '[{section}]'. Expected {expected_type.__name__ if not isinstance(expected_type, tuple) else ' or '.join(t.__name__ for t in expected_type)}, found {type(current_value).__name__}.")
                            is_valid = False
            continue # Done with specialized_llms section

        # --- Regular key checking for other sections ---
        for key, expected_type_or_tuple in keys_config.items():
            if key not in config[section]:
                # Handle genuinely optional keys within a required section
                # (These are less common now that many have defaults in code)
                if section == "predictive_world_model" and key == "save_interval_versions": # Example if it became optional
                     warnings.append(f"Optional key '{key}' not found in section '[{section}]'. Component will use defaults.")
                     continue

                errors.append(f"Missing required key '{key}' in section '[{section}]'")
                is_valid = False
                continue

            current_value = config[section][key]
            is_expected_type = False
            actual_expected_type = expected_type_or_tuple
            
            if isinstance(actual_expected_type, tuple):
                is_expected_type = isinstance(current_value, actual_expected_type)
            elif actual_expected_type is dict: # Special handling for dict type
                is_expected_type = isinstance(current_value, dict)
            elif actual_expected_type is list: # Special handling for list type
                 is_expected_type = isinstance(current_value, list)
            else: # Standard type
                is_expected_type = isinstance(current_value, actual_expected_type) # type: ignore

            if not is_expected_type and actual_expected_type is float and isinstance(current_value, int):
                warnings.append(f"Key '{key}' in section '[{section}]' is an integer, but float expected. Will be treated as float.")
                is_expected_type = True

            if not is_expected_type:
                expected_type_names = ""
                if isinstance(actual_expected_type, tuple):
                    expected_type_names = " or ".join([t.__name__ for t in actual_expected_type])
                else:
                    expected_type_names = actual_expected_type.__name__ if hasattr(actual_expected_type, "__name__") else str(actual_expected_type)
                errors.append(f"Invalid type for key '{key}' in section '[{section}]'. Expected {expected_type_names}, found {type(current_value).__name__}.")
                is_valid = False

    # --- Value Validation ---
    if config: # Ensure config was loaded
        for key_path, validation_func_or_val in VALUE_CHECKS.items():
            section, key = key_path.split('.', 1)
            if section in config and isinstance(config.get(section), dict) and key in config[section]:
                value = config[section][key]
                try:
                    if callable(validation_func_or_val):
                        # If the validation function takes config, pass it
                        if "cfg=config" in str(validation_func_or_val): # Simple check
                            if not validation_func_or_val(value, cfg=config):
                                errors.append(f"Invalid value for '{key_path}': {value}. Failed check: {validation_func_or_val.__doc__ or validation_func_or_val.__name__}")
                                is_valid = False
                        elif not validation_func_or_val(value): # Regular lambda
                            errors.append(f"Invalid value for '{key_path}': {value}. Failed check: {validation_func_or_val.__doc__ or validation_func_or_val.__name__}")
                            is_valid = False
                    # Else, if it was a direct value to check against (not used in this template)
                    # elif value != validation_func_or_val:
                    #    errors.append(f"Invalid value for '{key_path}': {value}. Expected {validation_func_or_val}")
                    #    is_valid = False
                except Exception as e:
                    errors.append(f"Error validating value for '{key_path}' ({value}): {e}")
                    is_valid = False
    else:
        errors.append("Config object is None, cannot perform value checks.")
        is_valid = False

    if config:
        # --- Custom Validations ---
        if "emergent_motivation_system" in config and isinstance(config["emergent_motivation_system"], dict):
            _custom_validate_ems_drives(config["emergent_motivation_system"], errors, warnings)
        
        # No specific custom validation needed for [oscar_specialized_llms] with flat structure,
        # as individual keys like "persona_dialogue_model" are handled by the main loop.
        # If it were nested, _custom_validate_specialized_llms would be useful.

        if "emergent_motivation_system" in config and isinstance(config["emergent_motivation_system"], dict):
            ems_config = config["emergent_motivation_system"]
            ems_cs_hist_maxlen = ems_config.get("ems_cs_history_maxlen")
            ems_low_cs_persist_thresh = ems_config.get("ems_low_cs_persistence_threshold")
            if isinstance(ems_cs_hist_maxlen, int) and isinstance(ems_low_cs_persist_thresh, int):
                if ems_low_cs_persist_thresh > ems_cs_hist_maxlen:
                    errors.append(f"Invalid emergent_motivation_system config: ems_low_cs_persistence_threshold ({ems_low_cs_persist_thresh}) > ems_cs_history_maxlen ({ems_cs_hist_maxlen}).")
                    is_valid = False
            elif "ems_cs_history_maxlen" in ems_config and "ems_low_cs_persistence_threshold" in ems_config:
                 if not isinstance(ems_cs_hist_maxlen, int): errors.append("ems_cs_history_maxlen must be int for cross-validation.")
                 if not isinstance(ems_low_cs_persist_thresh, int): errors.append("ems_low_cs_persistence_threshold must be int for cross-validation.")


        if "consciousness_assessor" in config and isinstance(config["consciousness_assessor"], dict):
             thresholds = config["consciousness_assessor"]
             req_cs_thresh = ["unconscious_threshold", "pre_conscious_threshold", "conscious_threshold", "meta_conscious_threshold"]
             if all(k in thresholds and isinstance(thresholds[k], (int, float)) for k in req_cs_thresh):
                 uncon, pre, con, meta = (thresholds[k] for k in req_cs_thresh)
                 if not (0 <= uncon < pre < con < meta <= 1.0):
                     errors.append(f"Invalid consciousness thresholds order/range: 0 <= u({uncon}) < p({pre}) < c({con}) < m({meta}) <= 1.0 not met.")
                     is_valid = False
             else:
                missing_or_invalid_cs_thresh = [k for k in req_cs_thresh if k not in thresholds or not isinstance(thresholds.get(k), (int,float))]
                if missing_or_invalid_cs_thresh:
                    errors.append(f"Missing or non-numeric consciousness thresholds in [consciousness_assessor] for order check: {missing_or_invalid_cs_thresh}")
                    is_valid = False

    # --- Print Results ---
    if warnings:
         print("\n--- Configuration Warnings ---")
         for warning in warnings: print(f"⚠️ WARNING: {warning}")
    if errors:
        print("\n--- Configuration Errors ---")
        for error in errors: print(f"❌ ERROR: {error}")
        print("\nConfiguration is INVALID.")
    elif warnings: print("\nConfiguration is VALID with warnings.")
    else: print("\n✅ Configuration is VALID.")

    return is_valid

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    script_dir = Path(__file__).resolve().parent
    # Try to find config.toml relative to the script's parent directory (project root)
    # This assumes validate_config.py is in a 'scripts' subdirectory of the project.
    default_config_path = script_dir.parent / "config.toml"

    config_to_validate_path_str = sys.argv[1] if len(sys.argv) > 1 else str(default_config_path)

    if not Path(config_to_validate_path_str).exists():
        print(f"❌ ERROR: Config file '{config_to_validate_path_str}' not found. Please specify a valid path or place config.toml in the project root.")
        sys.exit(2) # Different exit code for file not found

    if validate_config(config_to_validate_path_str):
        sys.exit(0)
    else:
        sys.exit(1)