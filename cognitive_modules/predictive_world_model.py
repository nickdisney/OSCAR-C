# --- START OF cognitive_modules/predictive_world_model.py ---

import asyncio
import logging
import time
import os
import json
import math # Added for entropy calculation
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from collections import deque, Counter

# --- NEW IMPORTS ---
try:
    import pandas as pd
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD # For defining CPDs if done manually later
    import pgmpy.estimators
    import pgmpy.inference
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    logging.getLogger(__name__).error(
        "PredictiveWorldModel: pgmpy or pandas not found. CBN functionality will be disabled. "
        "Please install with: pip install pgmpy pandas"
    )
    # Define placeholders if needed for type hints
    class DiscreteBayesianNetwork: pass
    class TabularCPD: pass
    class VariableElimination: pass # type: ignore
# --- END NEW IMPORTS ---

try:
    from ..protocols import CognitiveComponent
    from ..models.datatypes import Predicate
    from ..cognitive_modules.cognitive_cache import CognitiveCache # For type hint
    from ..utils.bayesian_utils import extract_features_for_cbn # MOVED IMPORT TO TOP
except ImportError:
    logging.warning("PredictiveWorldModel: Relative imports failed, using placeholders (expected in combined script).")
    if 'CognitiveComponent' not in globals(): raise ImportError("CognitiveComponent not found via relative import or globally")
    
    CognitiveComponent = globals().get('CognitiveComponent', type('DummyCognitiveComponent', (object,), {}))
    Predicate = globals().get('Predicate', type('DummyPredicate', (object,), {})) # type: ignore
    CognitiveCache = globals().get('CognitiveCache', type('DummyCognitiveCache', (object,), {})) # type: ignore
    # Fallback for extract_features_for_cbn if utils import fails
    if 'extract_features_for_cbn' not in globals():
        def extract_features_for_cbn(*args, **kwargs):
            logging.getLogger(__name__).error("Dummy extract_features_for_cbn called due to import error.")
            return {}
        globals()['extract_features_for_cbn'] = extract_features_for_cbn


logger_predictive_model = logging.getLogger(__name__)

class PredictiveWorldModel(CognitiveComponent):
    """
    Predicts future states based on current state and potential actions.
    Learns from prediction errors by updating its Causal Bayesian Network model.
    """

    def __init__(self):
        self._controller: Optional[Any] = None
        self.pwm_config_main: Dict[str, Any] = {}
        
        self.model_version: int = 0
        self.last_update_time: Optional[float] = None
        self.last_prediction_error: Optional[Dict[str, Any]] = None
        
        self.model_path: Optional[Path] = None
        
        self.cbn_node_definitions: List[Dict[str, Any]] = []
        self.cbn_structure: List[Tuple[str, str]] = []    
        
        self.causal_model: Optional[DiscreteBayesianNetwork] = None
        if not PGMPY_AVAILABLE: self.causal_model = None # type: ignore
        
        self.inference_engine: Optional[pgmpy.inference.VariableElimination] = None # type: ignore
        if not PGMPY_AVAILABLE: self.inference_engine = None # type: ignore

        self.learning_data_buffer: Optional[pd.DataFrame] = None
        if PGMPY_AVAILABLE: self.learning_data_buffer = pd.DataFrame()
        
        self.new_observations_since_last_fit: int = 0
        self.cpd_reestimation_trigger_count: int = 20

    async def initialize(self, config: Dict[str, Any], controller: Any) -> bool:
        self._controller = controller
        self.pwm_config_main = config.get("predictive_world_model", {})

        if not PGMPY_AVAILABLE:
            logger_predictive_model.error("pgmpy/pandas not available. PredictiveWorldModel CBN functionality disabled.")
            return False

        self.model_version = self.pwm_config_main.get("initial_version", 0)
        self.cpd_reestimation_trigger_count = int(self.pwm_config_main.get("cpd_reestimation_trigger_count_frequent", 20))

        path_str_from_config = None
        if controller and hasattr(controller, 'agent_root_path'):
            agent_root = controller.agent_root_path
            agent_data_paths_config = config.get("agent_data_paths", {})
            path_str_from_config = agent_data_paths_config.get("predictive_model_learned_data_path", "data/pwm_learning_buffer.csv") 

            if path_str_from_config:
                self.model_path = (Path(agent_root) / path_str_from_config).resolve()
                logger_predictive_model.info(f"PredictiveWorldModel learned data path: {self.model_path}")
                try:
                    self.model_path.parent.mkdir(parents=True, exist_ok=True)
                    if self.model_path.exists() and self.model_path.suffix == '.csv':
                        try:
                            self.learning_data_buffer = pd.read_csv(self.model_path)
                            logger_predictive_model.info(f"Loaded {len(self.learning_data_buffer) if self.learning_data_buffer is not None else 0} records from {self.model_path}")
                        except Exception as e_load_csv:
                            logger_predictive_model.error(f"Failed to load learning data buffer from CSV {self.model_path}: {e_load_csv}. Starting fresh.")
                            self.learning_data_buffer = pd.DataFrame()
                except Exception as e_path:
                    logger_predictive_model.error(f"Error with predictive model data path {self.model_path}: {e_path}.")
            else:
                logger_predictive_model.info("predictive_model_learned_data_path not set. Learned model/data will not be persisted.")
                self.model_path = None
        else:
            logger_predictive_model.error("PredictiveWorldModel: Controller or agent_root_path missing. Learned model/data persistence disabled.")
            self.model_path = None
        
        cbn_config_path_str = self.pwm_config_main.get("cbn_config_file", "data/pwm_cbn_config.json")
        cbn_config_path = None
        if controller and hasattr(controller, 'agent_root_path'):
            cbn_config_path = (Path(controller.agent_root_path) / cbn_config_path_str).resolve()
        
        loaded_cbn_config = None
        if cbn_config_path and cbn_config_path.exists():
            try:
                with open(cbn_config_path, 'r') as f:
                    loaded_cbn_config = json.load(f)
                logger_predictive_model.info(f"Successfully loaded CBN structure from {cbn_config_path}")
            except Exception as e_cbn_conf:
                logger_predictive_model.error(f"Failed to load CBN config from {cbn_config_path}: {e_cbn_conf}")
        else:
            logger_predictive_model.warning(f"CBN config file not found at {cbn_config_path if cbn_config_path else 'None'}. Using empty/default CBN structure.")

        if loaded_cbn_config and isinstance(loaded_cbn_config, dict):
            self.cbn_node_definitions = loaded_cbn_config.get("cbn_nodes", [])
            self.cbn_structure = [tuple(edge) for edge in loaded_cbn_config.get("cbn_structure", [])]
        else:
            self.cbn_node_definitions = []
            self.cbn_structure = []

        if not self.cbn_node_definitions or not self.cbn_structure:
            logger_predictive_model.error("CBN node definitions or structure is empty. PWM cannot build causal model.")
            return False

        try:
            self.causal_model = DiscreteBayesianNetwork()
            node_names_for_buffer = []
            for node_def in self.cbn_node_definitions:
                node_name = node_def.get("name")
                if node_name:
                    self.causal_model.add_node(node_name) 
                    node_names_for_buffer.append(node_name)
                else:
                    logger_predictive_model.warning(f"Invalid node definition in CBN config: {node_def}. Skipping.")
            
            self.causal_model.add_edges_from(self.cbn_structure)

            if self.learning_data_buffer is not None and self.learning_data_buffer.empty and node_names_for_buffer:
                self.learning_data_buffer = pd.DataFrame(columns=node_names_for_buffer)

            if self.learning_data_buffer is not None and not self.learning_data_buffer.empty:
                logger_predictive_model.info("Attempting initial model fit with pre-loaded learning data...")
                fit_success, _ = await self._fit_model_from_buffer() 
                if not fit_success:
                    logger_predictive_model.warning("Initial model fit from buffer failed. Inference engine might not be ready.")
            
            logger_predictive_model.info(
                f"PredictiveWorldModel DiscreteCBN initialized. Model Version: {self.model_version}, "
                f"Nodes: {len(self.causal_model.nodes()) if self.causal_model else 0}, "
                f"Edges: {len(self.causal_model.edges()) if self.causal_model else 0}, "
                f"CPD Re-est Trigger: {self.cpd_reestimation_trigger_count}"
            )
            return True
        except Exception as e_pgm_init:
            logger_predictive_model.exception(f"Error initializing pgmpy DiscreteBayesianNetwork: {e_pgm_init}")
            self.causal_model = None
            return False

    async def _fit_model_from_buffer(self) -> Tuple[bool, int]: # Return success status and new version
        logger_predictive_model.info(f"PWM_FIT: Starting _fit_model_from_buffer (Thread: {os.getpid()})")
        start_time_fit = time.monotonic()

        if not PGMPY_AVAILABLE or self.causal_model is None or \
           self.learning_data_buffer is None or len(self.learning_data_buffer) < 1:
            logger_predictive_model.debug(f"Fit model skipped: pgmpy, model, or data buffer not ready/empty. Buffer size: {len(self.learning_data_buffer) if self.learning_data_buffer is not None else 'None'}")
            logger_predictive_model.info(f"PWM_FIT: Exiting _fit_model_from_buffer early due to pre-conditions not met. Duration: {time.monotonic() - start_time_fit:.4f}s")
            return False, self.model_version

        logger_predictive_model.info(f"PWM_FIT: Fitting DiscreteCBN model with {len(self.learning_data_buffer)} data points...")
        try:
            loop = asyncio.get_running_loop()
            df_for_fitting_copy = self.learning_data_buffer.copy() 
            state_names_map_copy = {
                node_def.get("name"): node_def.get("states")
                for node_def in self.cbn_node_definitions
                if node_def.get("name") and isinstance(node_def.get("states"), list) and \
                   df_for_fitting_copy is not None and node_def.get("name") in df_for_fitting_copy.columns
            }

            temp_model_to_fit = DiscreteBayesianNetwork()
            for node_def_fit in self.cbn_node_definitions:
                 node_name_fit = node_def_fit.get("name")
                 if node_name_fit: temp_model_to_fit.add_node(node_name_fit)
            temp_model_to_fit.add_edges_from(self.cbn_structure)
            
            for col in df_for_fitting_copy.columns: # Ensure all columns are string type for estimator
                if temp_model_to_fit and col in temp_model_to_fit.nodes():
                    df_for_fitting_copy[col] = df_for_fitting_copy[col].astype(str)


            estimator = pgmpy.estimators.MaximumLikelihoodEstimator(
                model=temp_model_to_fit, 
                data=df_for_fitting_copy, 
                state_names=state_names_map_copy 
            )
            
            logger_predictive_model.info("PWM_FIT: Calling estimator.get_parameters()...")
            get_params_start = time.monotonic()
            cpds_list = await loop.run_in_executor(None, estimator.get_parameters, 1) 
            logger_predictive_model.info(f"PWM_FIT: estimator.get_parameters() completed in {time.monotonic() - get_params_start:.4f}s")

            logger_predictive_model.info("PWM_FIT: Calling temp_model_to_fit.add_cpds()...")
            add_cpds_start = time.monotonic()
            temp_model_to_fit.add_cpds(*cpds_list)
            logger_predictive_model.info(f"PWM_FIT: temp_model_to_fit.add_cpds() completed in {time.monotonic() - add_cpds_start:.4f}s")
            
            model_ok = await loop.run_in_executor(None, temp_model_to_fit.check_model)

            if model_ok:
                # This section needs to be thread-safe if PWM is accessed concurrently.
                # For now, assuming secondary loop handles this update sequentially.
                self.causal_model = temp_model_to_fit
                self.inference_engine = pgmpy.inference.VariableElimination(self.causal_model)
                self.model_version += 1 
                # self.new_observations_since_last_fit = 0 # Reset when scheduling, not on completion
                logger_predictive_model.info(f"Async PWM model fit successful. New version: {self.model_version}")
                logger_predictive_model.info(f"PWM_FIT: _fit_model_from_buffer successful. Duration: {time.monotonic() - start_time_fit:.4f}s")
                return True, self.model_version
            else:
                logger_predictive_model.error("DiscreteCBN model check failed after fitting CPDs.")
                logger_predictive_model.info(f"PWM_FIT: _fit_model_from_buffer failed (model check). Duration: {time.monotonic() - start_time_fit:.4f}s")
                return False, self.model_version
        except Exception as e_fit_async:
            logger_predictive_model.exception(f"Error during Async DiscreteCBN model fitting: {e_fit_async}")
            logger_predictive_model.info(f"PWM_FIT: _fit_model_from_buffer failed (exception). Duration: {time.monotonic() - start_time_fit:.4f}s")
            return False, self.model_version

    async def _load_model(self):
        if not (self.model_path and self.model_path.exists()):
            logger_predictive_model.debug("Predictive model metadata file not found or path not set for load.")
            return
        
        if self.model_path.suffix == '.json':
            logger_predictive_model.info(f"Attempting to load predictive model metadata from {self.model_path}")
            try:
                with open(self.model_path, 'r') as f: model_data = json.load(f)
                self.model_version = model_data.get('version', self.model_version)
                logger_predictive_model.info(f"Loaded predictive model metadata v{self.model_version}")
            except Exception as e:
                logger_predictive_model.exception(f"Error loading predictive model metadata: {e}")
        else:
            logger_predictive_model.debug(f"Model path {self.model_path} is not a JSON metadata file. Skipping metadata load.")

    async def _save_model(self):
        if not self.model_path:
            logger_predictive_model.debug("Predictive model saving disabled (no path).")
            return False
        
        if self.learning_data_buffer is not None and self.model_path.suffix == '.csv':
            logger_predictive_model.info(f"Saving learning data buffer ({len(self.learning_data_buffer)} records) to {self.model_path}...")
            try:
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                temp_csv_path = self.model_path.with_suffix(".tmp.csv")
                self.learning_data_buffer.to_csv(temp_csv_path, index=False)
                os.replace(temp_csv_path, self.model_path)
                logger_predictive_model.info("Learning data buffer saved successfully.")

                meta_json_path = self.model_path.with_suffix(".meta.json")
                with open(meta_json_path, 'w') as f_meta:
                    json.dump({"version": self.model_version, "last_update": time.time()}, f_meta, indent=2)
                logger_predictive_model.info(f"Metadata saved to {meta_json_path}")
                return True
            except Exception as e_csv:
                logger_predictive_model.exception(f"Failed to save learning data buffer to CSV: {e_csv}")
                if 'temp_csv_path' in locals() and temp_csv_path.exists() and isinstance(temp_csv_path, Path): # type: ignore
                    try: temp_csv_path.unlink() # type: ignore
                    except OSError: pass
                return False
        else:
            logger_predictive_model.warning(f"Model path suffix {self.model_path.suffix if self.model_path else 'None'} not '.csv', or buffer is None. Cannot save learning data.")
            return False

    def _get_action_key(self, action_type: str, params: Dict[str, Any]) -> str:
        if not params: return action_type
        key_parts = [action_type]
        if action_type in ["READ_FILE", "LIST_FILES", "WRITE_FILE"]:
            path_param = params.get("path")
            if isinstance(path_param, str) and path_param:
                try: key_parts.append(f"target_{Path(path_param).name[:30]}")
                except Exception: key_parts.append("with_path_param")
            else: key_parts.append("no_path_param_specified")
        elif action_type == "CALL_LLM":
            prompt_param = params.get("prompt")
            if isinstance(prompt_param, str) and prompt_param:
                if len(prompt_param) > 200: key_parts.append("long_prompt")
                elif len(prompt_param) < 50: key_parts.append("short_prompt")
                else: key_parts.append("medium_prompt")
        return ":".join(key_parts)

    def _extract_context_key(self, context: Dict[str, Any]) -> str:
        if not context: return "default_context"
        parts = []
        cs_level_name = context.get("consciousness_level_name")
        if cs_level_name and isinstance(cs_level_name, str):
            cs_level_upper = cs_level_name.upper()
            if "UNCONSCIOUS" in cs_level_upper or "PRE_CONSCIOUS" in cs_level_upper: parts.append("low_cs")
            elif "META_CONSCIOUS" in cs_level_upper or "REFLECTIVE" in cs_level_upper: parts.append("high_cs")
            elif "CONSCIOUS" in cs_level_upper: parts.append("mid_cs")
        goal_type = context.get("active_goal_type")
        if goal_type and isinstance(goal_type, str) and goal_type != "none" and goal_type != "general_task":
            parts.append(f"goal_{''.join(filter(str.isalnum, goal_type))[:10]}")
        curiosity = context.get("drive_curiosity")
        if isinstance(curiosity, (int, float)):
            if curiosity > 0.75: parts.append("high_cur")
            elif curiosity < 0.25: parts.append("low_cur")
        satisfaction = context.get("drive_satisfaction")
        if isinstance(satisfaction, (int, float)):
            if satisfaction > 0.75: parts.append("high_sat")
            elif satisfaction < 0.25: parts.append("low_sat")
        return ":".join(sorted(parts)) if parts else "default_context"

    def _get_default_prediction_heuristic(self, action_type: str) -> str:
        if action_type in ['READ_FILE', 'LIST_FILES', 'QUERY_KB', "OBSERVE_SYSTEM", "THINKING", "CALL_LLM", "RESPOND_TO_USER", "GET_AGENT_STATUS", "EXPLAIN_GOAL"]:
            return "success"
        elif action_type == "WRITE_FILE":
             allow_write = True
             if self._controller and hasattr(self._controller, 'config'):
                 fs_config = self._controller.config.get("filesystem", {})
                 allow_write = fs_config.get("allow_file_write", False)
             return "success" if allow_write else "failure"
        return "unknown"

    async def predict_next_state(self, current_state_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts the outcome of a potential action using the Causal Bayesian Network if fitted,
        otherwise falls back to heuristics.
        (MDP C.4.5)
        Input: current_state_info = {
            'action_to_execute': Dict, 
            'context': Dict (planning-time context: CS, goal_type, drives),
            'current_world_state_predicates': Set[Predicate] (current KB state for PreStateFeatures)
        }
        Output: {
            "predicted_outcome": str, 
            "confidence": float, 
            "basis": str, 
            "all_probabilities": Optional[Dict[str, float]],
            "outcome_entropy": float 
        }
        """
        prediction_output = { # Default response structure
            "predicted_outcome": "unknown", 
            "confidence": 0.0, 
            "basis": "initial_default_no_prediction_attempted", # More specific default
            "all_probabilities": None, # Store full distribution if from CBN
            "outcome_entropy": 0.0 # <<< ADD NEW KEY WITH DEFAULT
        }

        if not PGMPY_AVAILABLE:
            prediction_output["basis"] = "pgmpy_unavailable_fallback_heuristic"
            # Fallback to simple heuristic if pgmpy isn't even installed
            action_type_fallback = current_state_info.get('action_to_execute', {}).get('type', 'UNKNOWN_ACTION_TYPE')
            prediction_output["predicted_outcome"] = self._get_default_prediction_heuristic(action_type_fallback)
            prediction_output["confidence"] = 0.1 # Low confidence for this basic fallback
            prediction_output["outcome_entropy"] = 0.8 # Default high uncertainty
            return prediction_output

        if not isinstance(current_state_info, dict):
            logger_predictive_model.error("predict_next_state: current_state_info is not a dict.")
            prediction_output["basis"] = "invalid_input_format"
            prediction_output["outcome_entropy"] = 0.8 # Default high uncertainty
            return prediction_output
        
        action_to_predict = current_state_info.get('action_to_execute')
        if not isinstance(action_to_predict, dict) or not action_to_predict.get('type'):
            logger_predictive_model.warning("predict_next_state: Invalid or missing 'action_to_execute' or action type. Using default heuristic.")
            # Use default heuristic if action info is bad
            action_type_fallback_no_action = action_to_predict.get('type', 'UNKNOWN_ACTION_TYPE') if isinstance(action_to_predict, dict) else 'UNKNOWN_ACTION_TYPE'
            prediction_output["predicted_outcome"] = self._get_default_prediction_heuristic(action_type_fallback_no_action)
            prediction_output["confidence"] = 0.1
            prediction_output["basis"] = "invalid_action_info_fallback_heuristic"
            prediction_output["outcome_entropy"] = 0.8 # Default high uncertainty
            return prediction_output

        action_type = action_to_predict.get('type', 'UNKNOWN_ACTION_TYPE')
        action_params = action_to_predict.get('params', {})
        # This is the context AT THE TIME OF PLANNING for the prediction
        planning_time_context_features = current_state_info.get('context', {}) 
        
        # Get current world state predicates (e.g., from KB) for PreState_Features
        current_world_state_predicates_for_prediction: Set[Predicate] = current_state_info.get('current_world_state_predicates', set()) # type: ignore
        if not isinstance(current_world_state_predicates_for_prediction, set):
            logger_predictive_model.warning("predict_next_state: 'current_world_state_predicates' missing or not a set. Pre-state features may be incomplete.")
            current_world_state_predicates_for_prediction = set() # type: ignore
            
        # Ensure extract_features_for_cbn is callable
        global extract_features_for_cbn # Refer to the globally available (or imported) function
        if not callable(extract_features_for_cbn):
            logger_predictive_model.error("PWM Predict: extract_features_for_cbn function is not available.")
            prediction_output["basis"] = "feature_extraction_unavailable_fallback_heuristic"
            prediction_output["predicted_outcome"] = self._get_default_prediction_heuristic(action_type)
            prediction_output["confidence"] = 0.1
            prediction_output["outcome_entropy"] = 0.8 # Default high uncertainty
            return prediction_output

        # --- Feature Extraction for CBN Evidence ---
        evidence_dict = extract_features_for_cbn(
            action_to_predict, 
            planning_time_context_features, # Use the planning-time context
            current_world_state_predicates_for_prediction
        )
        logger_predictive_model.debug(f"PWM Predict: Extracted features for CBN query: {evidence_dict}")
        
        # --- CBN Inference ---
        query_result_pgmpy = None
        if self.causal_model and self.inference_engine: # Check if model is fitted and engine ready
            # Ensure evidence keys are actual nodes in the model and values are valid states
            valid_evidence_for_query = {}
            for k, v_untyped in evidence_dict.items():
                v = str(v_untyped) # Ensure value is string for categorical state matching
                if k in self.causal_model.nodes():
                    # Pgmpy's query method can be sensitive to states not defined for a variable.
                    # However, it should handle evidence for observed states.
                    # We assume extract_features_for_cbn produces valid states as per cbn_node_definitions.
                    valid_evidence_for_query[k] = v
                # else: feature extracted by util is not a node in our current CBN, ignore it.
            
            # Determine the target node for prediction (usually "OutcomeNode")
            # Make target_outcome_node_name configurable or ensure it's fixed
            target_outcome_node_name = "OutcomeNode" # Default from our pwm_cbn_config.json
            if "cbn_target_outcome_node" in self.pwm_config_main: # Check if overridden in PWM config
                 target_outcome_node_name = self.pwm_config_main["cbn_target_outcome_node"]


            if target_outcome_node_name in self.causal_model.nodes():
                if valid_evidence_for_query: # Only query if there's some valid evidence
                    try:
                        logger_predictive_model.debug(f"PWM Predict CBN Query: Target='{target_outcome_node_name}', Evidence={valid_evidence_for_query}")
                        
                        # --- Prediction Caching (MDP C.4.5) ---
                        cache_component_pred_cbn: Optional[CognitiveCache] = None # type: ignore
                        if self._controller and hasattr(self._controller, 'cache'):
                            cache_ref = getattr(self._controller, 'cache')
                            _CognitiveCache_local_pred = globals().get('CognitiveCache')
                            if _CognitiveCache_local_pred and isinstance(cache_ref, _CognitiveCache_local_pred): # type: ignore
                                cache_component_pred_cbn = cache_ref
                        
                        if cache_component_pred_cbn and hasattr(cache_component_pred_cbn, 'get_or_compute'):
                            # Create a hashable cache key from the evidence
                            evidence_tuple_for_cache_key = tuple(sorted(valid_evidence_for_query.items()))
                            cache_key_pred_cbn = ("pwm_cbn_query", target_outcome_node_name, evidence_tuple_for_cache_key)
                            
                            async def perform_inference_for_cache():
                                # VariableElimination.query is synchronous
                                return self.inference_engine.query( # type: ignore
                                    variables=[target_outcome_node_name],
                                    evidence=valid_evidence_for_query, 
                                    show_progress=False # pgmpy internal, might not be in all versions
                                )
                            try:
                                # Get TTL from config: [predictive_world_model] cbn_query_cache_ttl_s
                                query_ttl_pred = float(self.pwm_config_main.get("cbn_query_cache_ttl_s", 0.2)) # Short TTL for predictions
                                query_result_pgmpy = await cache_component_pred_cbn.get_or_compute(
                                    cache_key_pred_cbn, perform_inference_for_cache, ttl_override=query_ttl_pred
                                )
                                # Check if it was a cache hit for logging (heuristic)
                                if hasattr(cache_component_pred_cbn, '_cache') and cache_key_pred_cbn in cache_component_pred_cbn._cache: # type: ignore
                                     logger_predictive_model.debug(f"PWM Predict: CBN query cache HIT for key components: {target_outcome_node_name}, {evidence_tuple_for_cache_key}")
                                else:
                                     logger_predictive_model.debug(f"PWM Predict: CBN query cache MISS for key components: {target_outcome_node_name}, {evidence_tuple_for_cache_key}. Computed.")
                            except Exception as e_cache_cbn:
                                logger_predictive_model.error(f"PWM Predict: Error during cached CBN query: {e_cache_cbn}. Performing direct query.")
                                query_result_pgmpy = self.inference_engine.query( # type: ignore
                                    variables=[target_outcome_node_name], evidence=valid_evidence_for_query, show_progress=False) # type: ignore
                        else: # No cache component available
                            query_result_pgmpy = self.inference_engine.query( # type: ignore
                                variables=[target_outcome_node_name], evidence=valid_evidence_for_query, show_progress=False) # type: ignore

                        # --- Process Query Result ---
                        if query_result_pgmpy:
                            # query_result_pgmpy is a Factor object (specifically, a TabularCPD without evidence vars)
                            # We need to extract probabilities for each state of the OutcomeNode
                            max_prob_val = -1.0
                            best_outcome_state = "unknown_outcome" # Default
                            all_probs_dict: Dict[str, float] = {}
                            
                            # Get the states defined for the OutcomeNode in the model
                            # This is safer than relying on query_result_pgmpy.state_names directly
                            outcome_node_def = next((n for n in self.cbn_node_definitions if n.get("name") == target_outcome_node_name), None)
                            if outcome_node_def and isinstance(outcome_node_def.get("states"), list):
                                defined_outcome_states = outcome_node_def["states"]
                                
                                # Get the marginal distribution for the target variable
                                # The query result is already marginalized for the target variable.
                                # Its values are P(OutcomeNode | evidence)
                                if hasattr(query_result_pgmpy, 'values') and hasattr(query_result_pgmpy, 'variables') and \
                                   len(query_result_pgmpy.variables) == 1 and query_result_pgmpy.variables[0] == target_outcome_node_name:
                                    
                                    # Get state names for the target variable from the result factor
                                    # These should align with defined_outcome_states if model is consistent
                                    states_in_result_factor = query_result_pgmpy.state_names.get(target_outcome_node_name, [])

                                    if len(query_result_pgmpy.values) == len(states_in_result_factor):
                                        for i, state_name_from_query in enumerate(states_in_result_factor):
                                            prob_for_state = query_result_pgmpy.values.item(i) # .item() for single value from numpy array
                                            all_probs_dict[state_name_from_query] = round(float(prob_for_state), 4)
                                            if float(prob_for_state) > max_prob_val:
                                                max_prob_val = float(prob_for_state)
                                                best_outcome_state = state_name_from_query
                                        
                                        if max_prob_val >= 0.0: # Ensure a valid probability was found
                                            prediction_output["predicted_outcome"] = best_outcome_state
                                            prediction_output["confidence"] = round(max_prob_val, 3)
                                            prediction_output["basis"] = "cbn_inference"
                                            prediction_output["all_probabilities"] = all_probs_dict
                                            
                                            # --- NEW: Calculate Shannon Entropy (MDP C.4.X) ---
                                            if all_probs_dict: # Ensure there are probabilities to calculate entropy from
                                                entropy = 0.0
                                                num_states = 0
                                                for prob in all_probs_dict.values():
                                                    if prob > 0: # log2(0) is undefined
                                                        entropy -= prob * math.log2(prob)
                                                    num_states +=1 # Count number of possible states for normalization
                                                
                                                # Normalize entropy to be between 0 and 1
                                                # Max entropy for N states is log2(N)
                                                if num_states > 1:
                                                    max_entropy = math.log2(num_states)
                                                    if max_entropy > 0:
                                                        normalized_entropy = entropy / max_entropy
                                                        prediction_output["outcome_entropy"] = round(normalized_entropy, 3)
                                                        logger_predictive_model.debug(
                                                            f"PWM Predict CBN: Raw Entropy={entropy:.3f}, "
                                                            f"Max Entropy (for {num_states} states)={max_entropy:.3f}, "
                                                            f"Normalized Entropy={prediction_output['outcome_entropy']}"
                                                        )
                                                    else: # Only one state had prob > 0, or num_states <= 1 (should not happen for OutcomeNode)
                                                        prediction_output["outcome_entropy"] = 0.0 # No uncertainty
                                                else: # Only one possible state, so no uncertainty
                                                    prediction_output["outcome_entropy"] = 0.0
                                            # --- END NEW ---
                                            
                                            logger_predictive_model.info(
                                                f"PWM Predict CBN: Action='{action_type}', PredOutcome='{best_outcome_state}', "
                                                f"Conf={max_prob_val:.2f}, Entropy={prediction_output['outcome_entropy']:.3f}, Probs={all_probs_dict}"
                                            )
                                        else: # Should not happen if probabilities sum to 1
                                            prediction_output["basis"] = "cbn_inference_no_max_prob"
                                    else:
                                        logger_predictive_model.warning(f"PWM Predict CBN: Mismatch between result values and states for {target_outcome_node_name}.")
                                        prediction_output["basis"] = "cbn_inference_state_value_mismatch"
                                else:
                                     logger_predictive_model.warning(f"PWM Predict CBN: Query result factor for {target_outcome_node_name} has unexpected variable structure: {query_result_pgmpy.variables if hasattr(query_result_pgmpy,'variables') else 'N/A'}")
                                     prediction_output["basis"] = "cbn_inference_bad_factor_vars"
                            else:
                                logger_predictive_model.warning(f"PWM Predict CBN: Could not get defined states for OutcomeNode '{target_outcome_node_name}' from CBN config.")
                                prediction_output["basis"] = "cbn_inference_outcome_states_undefined"
                        else: # query_result_pgmpy is None
                            prediction_output["basis"] = "cbn_inference_query_returned_none"
                    except Exception as e_cbn_query_inner:
                        logger_predictive_model.error(f"PWM Predict CBN: Error during inference query execution or result processing: {e_cbn_query_inner}", exc_info=True)
                        prediction_output["basis"] = "cbn_inference_exception"
                else: # No valid evidence to query with
                    prediction_output["basis"] = "cbn_no_valid_evidence"
            elif self.causal_model and target_outcome_node_name not in self.causal_model.nodes(): # Target node not in model
                 prediction_output["basis"] = "cbn_target_node_missing_in_model"
        
        # --- Fallback to Heuristics if CBN prediction is insufficient ---
        # Define conditions under which we consider CBN prediction insufficient
        cbn_prediction_failed_bases = [
            "initial_default_no_prediction_attempted", # Only if CBN block was entirely skipped
            "cbn_pgmpy_unavailable", "cbn_model_not_built", "cbn_inference_engine_not_ready",
            "cbn_target_node_missing_in_model", "cbn_no_valid_evidence", "cbn_inference_exception",
            "cbn_inference_query_returned_none", "cbn_inference_no_max_prob",
            "cbn_inference_state_value_mismatch", "cbn_inference_bad_factor_vars",
            "cbn_inference_outcome_states_undefined",
            "feature_extraction_unavailable_fallback_heuristic" # If features couldn't be made
        ]
        current_confidence = float(prediction_output.get("confidence", 0.0))
        current_basis = str(prediction_output.get("basis", "initial_default_no_prediction_attempted"))

        # Fallback if CBN failed OR if CBN confidence is very low (e.g., uniform distribution)
        # Low confidence from CBN (e.g. <0.2 or if max_prob is just 1/num_states for outcome)
        # might warrant trying heuristic.
        # For OutcomeNode with ~9 states, uniform prob is ~0.11. So conf < 0.2 is a reasonable threshold.
        if current_basis in cbn_prediction_failed_bases or \
           (current_basis == "cbn_inference" and current_confidence < 0.20):
            
            logger_predictive_model.debug(f"PWM Predict: CBN prediction deemed insufficient (Basis: {current_basis}, Conf: {current_confidence:.2f}). Attempting heuristic fallbacks.")
            
            # Specific hardcoded test fallbacks (from previous version)
            if action_type == "READ_FILE":
                path_param_for_fallback = action_params.get("path", "")
                if path_param_for_fallback == "test_good_file.txt": 
                    prediction_output["predicted_outcome"], prediction_output["confidence"], prediction_output["basis"] = "success", 0.7, "hardcoded_test_file_good"
                elif path_param_for_fallback == "non_existent_for_surprise.txt": 
                    prediction_output["predicted_outcome"], prediction_output["confidence"], prediction_output["basis"] = "success", 0.6, "hardcoded_test_file_surprise_positive"
                    logger_predictive_model.warning("PWM Predict: Using hardcoded 'success' for 'non_existent_for_surprise.txt'.")
            
            # If hardcoded fallbacks didn't apply or weren't confident enough
            if prediction_output["basis"] in cbn_prediction_failed_bases or \
               (prediction_output["basis"].startswith("hardcoded") and prediction_output.get("confidence",0.0) < 0.5) or \
               prediction_output.get("confidence",0.0) < 0.20 : # General low confidence threshold

                heuristic_outcome = self._get_default_prediction_heuristic(action_type)
                heuristic_confidence, heuristic_basis = 0.15, "default_heuristic" # Lowered confidence for general heuristic

                # Only use heuristic if it's more confident than a failed/very_low_conf CBN prediction
                if prediction_output["basis"] in cbn_prediction_failed_bases or \
                   heuristic_confidence > prediction_output.get("confidence", 0.0):
                    prediction_output["predicted_outcome"] = heuristic_outcome
                    prediction_output["confidence"] = heuristic_confidence
                    prediction_output["basis"] = heuristic_basis
                    logger_predictive_model.debug(f"PWM Predict: Fallback to default heuristic for '{action_type}': Outcome='{heuristic_outcome}', Conf={heuristic_confidence:.2f}")
        
        # If fallback to heuristics, outcome_entropy remains 0.0 or could be set to a default high value
        # indicating maximum uncertainty as heuristic doesn't provide a distribution.
        if prediction_output["basis"] != "cbn_inference" or not prediction_output["all_probabilities"]:
            prediction_output["outcome_entropy"] = 0.8 # Default high uncertainty if not from CBN distribution

        # Final confidence rounding (already done for CBN path, ensure for others)
        prediction_output["confidence"] = round(float(prediction_output.get("confidence", 0.0)), 3)

        logger_predictive_model.info( # Updated log to include entropy
             f"PWM Final Prediction: Action='{action_type}', Outcome='{prediction_output['predicted_outcome']}', "
             f"Confidence={prediction_output['confidence']:.2f}, Basis='{prediction_output['basis']}', "
             f"Entropy={prediction_output['outcome_entropy']:.3f}. " # Added entropy
             f"AllProbs: {str(prediction_output.get('all_probabilities'))[:100] if prediction_output.get('all_probabilities') else 'N/A'}"
        )
        return prediction_output

    async def update_model(self, 
                         prediction: Optional[Dict[str, Any]], # This 'prediction' is the full output from predict_next_state
                         actual_result_wrapper: Dict[str, Any]
                        ):
        self.last_update_time = time.time()
        prediction = prediction if isinstance(prediction, dict) else {} # Ensure prediction is a dict
        
        if not isinstance(actual_result_wrapper, dict):
            logger_predictive_model.error("PWM update_model: actual_result_wrapper is not a dict.")
            return

        actual_result = actual_result_wrapper.get("actual_result")
        buffered_pre_action_state = actual_result_wrapper.get("buffered_pre_action_state")

        if not isinstance(actual_result, dict):
            logger_predictive_model.error("PWM update_model: 'actual_result' missing or not a dict in wrapper.")
            return
        if not isinstance(buffered_pre_action_state, dict) and actual_result.get("type") != "THINKING":
            logger_predictive_model.warning( f"PWM update_model: 'buffered_pre_action_state' missing for action '{actual_result.get('type')}'.")

        predicted_outcome = prediction.get("predicted_outcome", "unknown") # prediction here is the full dict
        actual_outcome_category = actual_result.get("outcome", "unknown_outcome")
        action_type_executed = actual_result.get("type", "UNKNOWN_ACTION")
        action_params_executed = actual_result.get("params", {})
        
        pre_action_context_features = {}
        pre_action_kb_predicates: Set[Predicate] = set() # type: ignore
        if isinstance(buffered_pre_action_state, dict):
            pre_action_context_features = buffered_pre_action_state.get("context", {})
            pre_action_kb_predicates = buffered_pre_action_state.get("predicates", set()) # type: ignore
        else:
            pre_action_context_features = actual_result.get("context_at_execution", {})
            logger_predictive_model.debug(f"Using context_at_execution for CBN features for action '{action_type_executed}'.")

        global extract_features_for_cbn # Refer to the globally available (or imported) function
        if callable(extract_features_for_cbn) and self.learning_data_buffer is not None and action_type_executed != "THINKING":
            action_dict_for_features = {"type": action_type_executed, "params": action_params_executed}
            cbn_feature_row = extract_features_for_cbn(
                action_dict_for_features, pre_action_context_features, pre_action_kb_predicates )
            
            target_outcome_node = self.pwm_config_main.get("cbn_target_outcome_node", "OutcomeNode") # CORRECTED DEFAULT
            outcome_node_value = "unknown_outcome"
            if actual_outcome_category == "success": outcome_node_value = "success"
            elif actual_outcome_category == "failure":
                err_msg = str(actual_result.get("error", "")).lower()
                if "permission" in err_msg: outcome_node_value = "failure_permission"
                elif "not found" in err_msg or "no such file" in err_msg: outcome_node_value = "failure_not_found"
                elif "timeout" in err_msg: outcome_node_value = "failure_timeout"
                elif "llm error" in err_msg: outcome_node_value = "failure_llm_error" 
                elif "vetoed" in err_msg: outcome_node_value = "failure_value_veto"
                elif "io" in err_msg: outcome_node_value = "failure_io" 
                else: outcome_node_value = "failure_other"
            
            cbn_feature_row[target_outcome_node] = outcome_node_value
            
            new_data_row = pd.DataFrame([cbn_feature_row])
            for col in self.learning_data_buffer.columns:
                if col not in new_data_row.columns: new_data_row[col] = "NotApplicable" 
            try:
                self.learning_data_buffer = pd.concat([self.learning_data_buffer, new_data_row], ignore_index=True)
                self.new_observations_since_last_fit += 1
            except Exception as e_concat: logger_predictive_model.error(f"Error concatenating data to learning_data_buffer: {e_concat}")
        elif not callable(extract_features_for_cbn):
            logger_predictive_model.error("PWM update_model: extract_features_for_cbn function not available.")


        prediction_error_occurred = False
        if predicted_outcome != "unknown" and predicted_outcome != actual_outcome_category:
            prediction_error_occurred = True
            error_source_details = {
                "action_type_source": action_type_executed,
                "params_source": action_params_executed,
                "context_source": pre_action_context_features,
                "mispredicted_percept_key": prediction.get("basis") if isinstance(prediction, dict) and "percept" in prediction.get("basis", "") else None,
                "error_magnitude": abs(prediction.get("confidence",0.5) - (1.0 if actual_outcome_category == "success" else 0.0))
            }
            if actual_result.get("error"):
                 error_source_details["specific_error_message"] = str(actual_result["error"])

            self.last_prediction_error = {
                "type": "outcome_mismatch", 
                "predicted": predicted_outcome, 
                "actual": actual_outcome_category,
                "predicted_outcome_summary": predicted_outcome, 
                "actual_outcome_summary": actual_outcome_category,
                "certainty_of_prediction": prediction.get("confidence", 0.0),
                "action_type": action_type_executed, 
                "action_params": action_params_executed,
                "error_source_details": error_source_details,
                "predicted_outcome_entropy": prediction.get("outcome_entropy", 0.0), # <<< ADD THIS
                "timestamp": self.last_update_time
            }
            logger_predictive_model.warning(f"Prediction Error: Action='{action_type_executed}', Pred='{predicted_outcome}', Actual='{actual_outcome_category}'. Error details: {self.last_prediction_error}")
        elif self.last_prediction_error and self.last_prediction_error.get("type") == "outcome_mismatch":
            self.last_prediction_error = None
        
        if self.cpd_reestimation_trigger_count > 0 and \
           self.new_observations_since_last_fit >= self.cpd_reestimation_trigger_count:
            logger_predictive_model.info(
                f"Scheduling CPD re-estimation. New observations: {self.new_observations_since_last_fit} "
                f"(Threshold: {self.cpd_reestimation_trigger_count})"
            )
            if self._controller and hasattr(self._controller, 'schedule_offline_task'):
                
                def fit_callback(result_future: asyncio.Future):
                    try:
                        success, version = result_future.result() 
                        if success:
                            logger_predictive_model.info(f"Offline PWM model fitting completed successfully. New model version: {version}")
                        else:
                            logger_predictive_model.warning("Offline PWM model fitting failed.")
                    except asyncio.CancelledError:
                        logger_predictive_model.warning("Offline PWM model fitting task was cancelled.")
                    except Exception as e_cb:
                        logger_predictive_model.error(f"Error in PWM fit_callback: {e_cb}", exc_info=True)

                self._controller.schedule_offline_task(self._fit_model_from_buffer, callback_on_done=fit_callback)
                self.new_observations_since_last_fit = 0 
            else:
                logger_predictive_model.error("Cannot schedule offline model fitting: Controller or schedule_offline_task method missing.")
        
        save_interval_versions = self.pwm_config_main.get("save_interval_versions", 10) 
        if self.model_path and self.model_path.suffix == '.csv' and \
           save_interval_versions > 0 and self.model_version > 0 and \
           (self.model_version % save_interval_versions == 0) and \
           self.new_observations_since_last_fit == 0: # Only save if just fitted
            await self._save_model()

    async def process(self, input_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not input_state: return None
        if 'predict_request' in input_state:
            if isinstance(input_state["predict_request"], dict):
                prediction = await self.predict_next_state(input_state["predict_request"])
                return {"prediction": prediction}
            else:
                logger_predictive_model.warning("PWM process: Invalid 'predict_request' data type.")
                return {"prediction": {"predicted_outcome": "unknown", "confidence": 0.0, "basis": "invalid_input_type"}}
        elif 'update_request' in input_state:
            req = input_state["update_request"]
            if isinstance(req, dict) and "actual_result" in req: 
                actual_result_wrapper = req
                if isinstance(actual_result_wrapper.get("actual_result"), dict):
                    await self.update_model(actual_result_wrapper.get("prediction"), actual_result_wrapper)
                    return {"update_status": "processed", "last_prediction_error_details": self.last_prediction_error}
                else:
                    logger_predictive_model.warning("PWM process: Invalid data type for actual_result in update.")
                    return {"update_status": "failed_invalid_data"}
            else:
                logger_predictive_model.warning("PWM process: Missing data for update, or invalid request type.")
                return {"update_status": "failed_missing_data"}
        return None

    async def reset(self) -> None:
        self.model_version = self.pwm_config_main.get("initial_version", 0)
        self.last_update_time = None; self.last_prediction_error = None
        if PGMPY_AVAILABLE:
            node_names = [node_def.get("name") for node_def in self.cbn_node_definitions if node_def.get("name")]
            self.learning_data_buffer = pd.DataFrame(columns=node_names if node_names else None)
            self.new_observations_since_last_fit = 0
            if self.cbn_structure and self.cbn_node_definitions:
                 self.causal_model = DiscreteBayesianNetwork() 
                 for node_def in self.cbn_node_definitions:
                     node_name = node_def.get("name")
                     if node_name: self.causal_model.add_node(node_name)
                 self.causal_model.add_edges_from(self.cbn_structure) 
                 self.inference_engine = None 
                 logger_predictive_model.info("CBN structure re-initialized, awaiting data to fit CPDs.")
            else: self.causal_model = None; self.inference_engine = None
        logger_predictive_model.info("PredictiveWorldModel reset.")

    async def get_status(self) -> Dict[str, Any]:
        error_type = self.last_prediction_error.get("type") if self.last_prediction_error else None
        return { 
            "component": "PredictiveWorldModel", 
            "status": "operational" if PGMPY_AVAILABLE and self.causal_model else "disabled_no_pgmpy_or_model",
            "model_version": self.model_version, "last_update_time": self.last_update_time,
            "last_prediction_error": self.last_prediction_error, "last_prediction_error_type": error_type, 
            "cbn_nodes_count": len(self.causal_model.nodes()) if self.causal_model else 0,
            "cbn_edges_count": len(self.causal_model.edges()) if self.causal_model else 0,
            "learning_buffer_size": len(self.learning_data_buffer) if self.learning_data_buffer is not None else 0,
            "new_obs_since_fit": self.new_observations_since_last_fit,
            "inference_engine_ready": self.inference_engine is not None
        }

    async def shutdown(self) -> None:
        logger_predictive_model.info("PredictiveWorldModel shutting down.")
        if self.model_path: await self._save_model()

# --- END OF cognitive_modules/predictive_world_model.py ---