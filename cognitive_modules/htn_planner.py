# --- START OF cognitive_modules/htn_planner.py (Corrected ImportError Fallback) ---

import asyncio
import logging
import re
import time
from typing import Dict, Any, Optional, List, Set, Callable, Tuple, Union
from dataclasses import dataclass, field
from typing import Protocol
import json
from pathlib import Path

# Attempt standard relative imports
try:
    from ..models.datatypes import Goal as ActualGoal
    from ..models.datatypes import Predicate as ActualPredicate
    from ..protocols import CognitiveComponent as ActualCognitiveComponent
    from ..protocols import Planner as ActualPlanner
    from ..cognitive_modules.cognitive_cache import CognitiveCache as ActualCognitiveCache
    from ..models.enums import ConsciousState as ActualConsciousState
    from ..external_comms import call_ollama
    from ..cognitive_modules.predictive_world_model import PredictiveWorldModel as ActualPredictiveWorldModel
    from ..cognitive_modules.knowledge_base import KnowledgeBase as ActualKnowledgeBase # Import KB

    # Assign to module-level names that the rest of the file expects
    Goal = ActualGoal
    Predicate = ActualPredicate
    CognitiveComponent = ActualCognitiveComponent
    Planner = ActualPlanner
    CognitiveCache = ActualCognitiveCache
    ConsciousState = ActualConsciousState
    PredictiveWorldModel = ActualPredictiveWorldModel
    KnowledgeBase = ActualKnowledgeBase # Make KB available at module level

    HTN_DEPENDENCIES_AVAILABLE = True
    logging.info("HTNPlanner: Successfully imported dependencies via relative paths.")

except ImportError as e:
    logging.warning(f"HTNPlanner: Relative imports failed ({e}). Using placeholders.")
    HTN_DEPENDENCIES_AVAILABLE = False

    # Define or get MockDataclass and MockEnum if not already available
    # This assumes MockDataclass and MockEnum might be defined in a higher-level context
    # or need to be defined here if truly standalone. For this fix, let's assume
    # agent_controller's MockDataclass could be a reference if this were combined,
    # but it's better to define simple ones here for true fallback.

    class MockDataclassHTN:
        def __init__(self, *args, **kwargs): pass
        @classmethod
        def _is_dummy(cls): return True

    class MockEnumHTN(str):
        def __new__(cls, value, *args, **kwargs): return super().__new__(cls, value)
        def __getattr__(self, name: str) -> str: return f"DUMMY_{name}"
        @classmethod
        def _is_dummy(cls): return True
    
    # Assign placeholders to the module-level names
    if 'Goal' not in globals(): Goal = MockDataclassHTN
    if 'Predicate' not in globals(): Predicate = MockDataclassHTN
    if 'CognitiveComponent' not in globals():
        class CognitiveComponentProtocolPlaceholder(Protocol):
            async def initialize(self, config: Dict[str, Any], controller: Any) -> bool: ...
            async def process(self, input_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]: ...
            async def reset(self) -> None: ...
            async def get_status(self) -> Dict[str, Any]: ...
            async def shutdown(self) -> None: ...
        CognitiveComponent = CognitiveComponentProtocolPlaceholder # type: ignore
    
    if 'Planner' not in globals():
        _CCP_Planner = globals().get('CognitiveComponent', Protocol)
        class PlannerProtocolPlaceholder(_CCP_Planner, Protocol): # type: ignore
             async def plan(self, goal: Goal, current_state: Set[Predicate], modification_hints: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]: ...
        Planner = PlannerProtocolPlaceholder # type: ignore

    if 'CognitiveCache' not in globals(): CognitiveCache = MockDataclassHTN
    if 'ConsciousState' not in globals(): ConsciousState = MockEnumHTN("ConsciousState_HTN_Fallback")
    if 'PredictiveWorldModel' not in globals(): PredictiveWorldModel = MockDataclassHTN
    if 'KnowledgeBase' not in globals(): KnowledgeBase = MockDataclassHTN # Placeholder for KB

    if 'call_ollama' not in globals():
        async def call_ollama_placeholder(*args, **kwargs):
            logging.getLogger(__name__).error("call_ollama function not available for HTNPlanner LLM sketch (using placeholder).")
            return None, "call_ollama unavailable"
        call_ollama = call_ollama_placeholder


logger = logging.getLogger(__name__)

# --- Type Aliases ---
TaskType = Union[str, Tuple[str, ...]]
PlanResultType = List[Tuple[str, Dict[str, Any]]]

# --- Helper: Binding Function ---
def bind_value(value: Any, bindings: Dict[str, Any]) -> Any:
    """Binds a single value if it's a variable string ('?var')."""
    if isinstance(value, str) and value.startswith('?'):
        var_name = value[1:]
        if var_name in bindings:
            return bindings[var_name]
        else:
            # It's okay for variables in effects/preconds to be unbound initially
            # The check happens later if needed. Log debug instead of warning.
            logger.debug(f"Variable '{value}' not in current bindings.")
            return value
    return value

# --- Planning Structures ---
@dataclass
class Operator:
    name: str
    parameters: List[str] = field(default_factory=list)
    preconditions: Set['Predicate'] = field(default_factory=set) # Forward ref
    effects: Set['Predicate'] = field(default_factory=set)       # Forward ref
    estimated_cost: float = 1.0 # <<< ADDED: Default cost for an operator

    def _bind_predicate(self, pred: 'Predicate', bindings: Dict[str, Any]) -> Optional['Predicate']: # type: ignore
        _PredicateClass = globals().get('Predicate')
        if not _PredicateClass: logger.error("Predicate class missing for binding."); return None
        try:
            bound_args = tuple(bind_value(arg, bindings) for arg in pred.args) # type: ignore
            # Check if any *result* of binding is still an unbound variable
            if any(isinstance(a, str) and a.startswith('?') for a in bound_args):
                 logger.debug(f"Pred binding failed for {pred.name}: Unbound var remains in args {bound_args}.") # type: ignore
                 return None
            ts = pred.timestamp if hasattr(pred, 'timestamp') and isinstance(pred.timestamp, float) else time.time() # type: ignore
            # Correctly handle the case where Predicate might not be fully defined
            try:
                return _PredicateClass(name=pred.name, args=bound_args, value=pred.value, timestamp=ts) # type: ignore
            except TypeError: # Handle potential issues if Predicate is just 'Any'
                logger.warning("Predicate class might be Any type, attempting basic dict return.")
                return {"name":pred.name, "args": bound_args, "value": pred.value, "timestamp":ts} # type: ignore
        except Exception as e: logger.error(f"Error binding predicate {pred}: {e}"); return None

    def is_applicable(self, state: Set['Predicate'], bindings: Dict[str, Any]) -> bool: # type: ignore
        _PredicateClass = globals().get('Predicate')
        if not _PredicateClass: return False
        for precond in self.preconditions:
            bound_precond = self._bind_predicate(precond, bindings)
            if bound_precond is None:
                logger.debug(f"Op '{self.name}' precond '{precond.name}{precond.args}' failed: Unbound variable.") # type: ignore
                return False
            # Handle Predicate object or dict fallback
            precond_value = getattr(bound_precond, 'value', bound_precond.get('value') if isinstance(bound_precond, dict) else None) # type: ignore
            precond_name = getattr(bound_precond, 'name', bound_precond.get('name') if isinstance(bound_precond, dict) else None) # type: ignore
            precond_args = getattr(bound_precond, 'args', bound_precond.get('args') if isinstance(bound_precond, dict) else None) # type: ignore

            if precond_value is None or precond_name is None or precond_args is None:
                logger.error(f"Could not get properties from bound precond: {bound_precond}")
                return False

            if precond_value is True:
                 # Comparison needs to handle dict vs object
                 if isinstance(bound_precond, dict):
                      # Check if a predicate object matching the dict exists in state
                      match_found = any(isinstance(p, _PredicateClass) and
                                        p.name == precond_name and p.args == precond_args and p.value is True # type: ignore
                                        for p in state)
                      if not match_found:
                           logger.debug(f"Op '{self.name}' precond '{precond_name}{str(precond_args)}=True' failed: Not found in current state (size: {len(state)}).")
                           return False
                 elif bound_precond not in state: # type: ignore
                     logger.debug(f"Op '{self.name}' precond '{precond_name}{str(precond_args)}=True' failed: Not found in current state (size: {len(state)}).")
                     return False
            else: # precond_value is False (negative precondition)
                 # Construct the positive version (object or dict)
                 if isinstance(bound_precond, dict):
                      # positive_version_dict = {"name": precond_name, "args": precond_args, "value": True} # Not used
                      # Check if a predicate object matching the positive dict exists
                      match_found = any(isinstance(p, _PredicateClass) and
                                        p.name == precond_name and p.args == precond_args and p.value is True # type: ignore
                                        for p in state)
                      if match_found:
                           logger.debug(f"Op '{self.name}' precond '{precond_name}{str(precond_args)}=False' failed: Positive version found in current state (size: {len(state)}).")
                           return False
                 else:
                      positive_version = _PredicateClass(precond_name, precond_args, True, getattr(bound_precond, 'timestamp', time.time())) # type: ignore
                      if positive_version in state:
                           logger.debug(f"Op '{self.name}' precond '{precond_name}{str(precond_args)}=False' failed: Positive version found in current state (size: {len(state)}).")
                           return False
        return True


    def apply(self, state: Set['Predicate'], bindings: Dict[str, Any]) -> Optional[Set['Predicate']]: # type: ignore
        _PredicateClass = globals().get('Predicate')
        if not _PredicateClass: return None
        new_state = state.copy()
        for effect in self.effects:
            bound_effect = self._bind_predicate(effect, bindings)
            if bound_effect is None: logger.error(f"Cannot apply effect for {self.name}: unbound in {effect}"); return None

            effect_name = getattr(bound_effect, 'name', bound_effect.get('name') if isinstance(bound_effect, dict) else None) # type: ignore
            effect_args = getattr(bound_effect, 'args', bound_effect.get('args') if isinstance(bound_effect, dict) else None) # type: ignore
            effect_value = getattr(bound_effect, 'value', bound_effect.get('value') if isinstance(bound_effect, dict) else None) # type: ignore
            effect_ts = getattr(bound_effect, 'timestamp', bound_effect.get('timestamp') if isinstance(bound_effect, dict) else time.time()) # type: ignore

            if effect_name is None or effect_args is None or effect_value is None:
                 logger.error(f"Could not get properties from bound effect: {bound_effect}"); return None

            # Remove the opposite state before adding the new one
            # Need to handle comparison correctly if Predicate is Any
            opposite_value = not effect_value
            preds_to_remove_opposite = {p for p in new_state if isinstance(p, _PredicateClass) and
                                        p.name == effect_name and p.args == effect_args and p.value == opposite_value} # type: ignore
            new_state -= preds_to_remove_opposite

            # Also remove any previous version of the same predicate (different timestamp) before adding
            preds_to_remove_same = {p for p in new_state if isinstance(p, _PredicateClass) and
                                    p.name == effect_name and p.args == effect_args} # type: ignore
            new_state -= preds_to_remove_same

            # Add the new effect (as a Predicate object)
            try:
                 new_state.add(_PredicateClass(name=effect_name, args=effect_args, value=effect_value, timestamp=effect_ts)) # type: ignore
            except Exception as add_e:
                 logger.error(f"Failed to add new predicate to state: {add_e}")
                 return None # State update failed

        return new_state


@dataclass
class Method:
    name: str
    task_signature: Tuple[str, ...] 
    preconditions: Set['Predicate'] = field(default_factory=set)
    subtasks: List[TaskType] = field(default_factory=list)
    heuristic_score: Optional[float] = None 
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.heuristic_score is None: 
            precondition_complexity_factor = len(self.preconditions) * 0.1
            self.heuristic_score = float(len(self.subtasks)) + precondition_complexity_factor
        
        # Ensure essential metadata keys exist with defaults
        self.metadata.setdefault("confidence", 0.5) 
        self.metadata.setdefault("success_rate", 0.0)
        self.metadata.setdefault("usage_count", 0)
        self.metadata.setdefault("total_recorded_successes", 0) 
        self.metadata.setdefault("learned_via", "hardcoded") 
        self.metadata.setdefault("creation_timestamp", time.time())
        self.metadata.setdefault("last_used_ts", None) 
        self.metadata.setdefault("last_successful_use_ts", None) 


    def _bind_predicate(self, pred: 'Predicate', bindings: Dict[str, Any]) -> Optional['Predicate']: # type: ignore
        # Same logic as Operator's method
        _PredicateClass = globals().get('Predicate')
        if not _PredicateClass: return None
        try:
            bound_args = tuple(bind_value(arg, bindings) for arg in pred.args) # type: ignore
            if any(isinstance(a, str) and a.startswith('?') for a in bound_args): return None
            ts = pred.timestamp if hasattr(pred, 'timestamp') and isinstance(pred.timestamp, float) else time.time() # type: ignore
            # Correctly handle the case where Predicate might not be fully defined
            try:
                return _PredicateClass(name=pred.name, args=bound_args, value=pred.value, timestamp=ts) # type: ignore
            except TypeError: # Handle potential issues if Predicate is just 'Any'
                logger.warning("Predicate class might be Any type, attempting basic dict return.")
                return {"name":pred.name, "args": bound_args, "value": pred.value, "timestamp":ts} # type: ignore
        except Exception as e: logger.error(f"Error binding predicate {pred}: {e}"); return None


    def is_applicable(self, state: Set['Predicate'], bindings: Dict[str, Any]) -> bool: # type: ignore
        # Same logic as Operator's method, adjusted for potential dict type
        _PredicateClass = globals().get('Predicate')
        if not _PredicateClass: return False
        for precond in self.preconditions:
            bound_precond = self._bind_predicate(precond, bindings)
            if bound_precond is None:
                logger.debug(f"Method '{self.name}' precond '{precond.name}{precond.args}' failed: Unbound variable.") # type: ignore
                return False
            # Extract properties safely
            precond_value = getattr(bound_precond, 'value', bound_precond.get('value') if isinstance(bound_precond, dict) else None) # type: ignore
            precond_name = getattr(bound_precond, 'name', bound_precond.get('name') if isinstance(bound_precond, dict) else None) # type: ignore
            precond_args = getattr(bound_precond, 'args', bound_precond.get('args') if isinstance(bound_precond, dict) else None) # type: ignore
            if precond_value is None or precond_name is None or precond_args is None:
                 logger.error(f"Could not get properties from bound precond (method): {bound_precond}")
                 return False

            if precond_value is True:
                 if isinstance(bound_precond, dict):
                      match_found = any(isinstance(p, _PredicateClass) and
                                        p.name == precond_name and p.args == precond_args and p.value is True # type: ignore
                                        for p in state)
                      if not match_found:
                          logger.debug(f"Method '{self.name}' precond '{precond_name}{str(precond_args)}=True' failed: Not found in current state (size: {len(state)}).")
                          return False
                 elif bound_precond not in state: # type: ignore
                      logger.debug(f"Method '{self.name}' precond '{precond_name}{str(precond_args)}=True' failed: Not found in current state (size: {len(state)}).")
                      return False
            else:
                 if isinstance(bound_precond, dict):
                      # positive_version_dict = {"name": precond_name, "args": precond_args, "value": True} # Not used
                      match_found = any(isinstance(p, _PredicateClass) and
                                        p.name == precond_name and p.args == precond_args and p.value is True # type: ignore
                                        for p in state)
                      if match_found:
                          logger.debug(f"Method '{self.name}' precond '{precond_name}{str(precond_args)}=False' failed: Positive version found in current state (size: {len(state)}).")
                          return False
                 else:
                      positive_version = _PredicateClass(precond_name, precond_args, True, getattr(bound_precond, 'timestamp', time.time())) # type: ignore
                      if positive_version in state:
                          logger.debug(f"Method '{self.name}' precond '{precond_name}{str(precond_args)}=False' failed: Positive version found in current state (size: {len(state)}).")
                          return False
        return True


    def get_parameter_bindings(self, task: TaskType) -> Optional[Dict[str, Any]]:
        """Attempts to bind method signature parameters to task arguments."""
        # Logic remains the same
        task_name = task[0] if isinstance(task, tuple) else task
        sig_name = self.task_signature[0]

        if task_name != sig_name:
            return None

        task_args = task[1:] if isinstance(task, tuple) else ()
        sig_params = self.task_signature[1:]

        if len(task_args) != len(sig_params):
            logger.debug(f"Method '{self.name}' arg count mismatch: Task {len(task_args)}, Sig {len(sig_params)}")
            return None

        bindings = {}
        try:
            for sig_param, task_arg in zip(sig_params, task_args):
                if isinstance(sig_param, str) and sig_param.startswith('?'):
                    var_name = sig_param[1:]
                    if var_name in bindings and bindings[var_name] != task_arg:
                        logger.warning(f"Method '{self.name}': Inconsistent binding for {sig_param}. Current: {bindings[var_name]}, New: {task_arg}")
                        return None
                    bindings[var_name] = task_arg
                elif sig_param != task_arg:
                    logger.debug(f"Method '{self.name}': Constant mismatch. Sig: {sig_param}, Task: {task_arg}")
                    return None
            return bindings
        except Exception as e:
            logger.error(f"Error during parameter binding for method {self.name}: {e}")
            return None


    def bind_subtask(self, subtask_template: TaskType, bindings: Dict[str, Any]) -> Optional[TaskType]:
         """Binds variables in a subtask template using the provided bindings."""
         # Logic remains the same
         if isinstance(subtask_template, str):
             bound_val = bind_value(subtask_template, bindings)
             if isinstance(bound_val, str) and bound_val.startswith('?'):
                 logger.error(f"Unbound variable '{bound_val}' in primitive subtask template '{subtask_template}' of method '{self.name}'")
                 return None
             return bound_val
         elif isinstance(subtask_template, tuple):
              task_name = subtask_template[0]
              bound_args = []
              try:
                   for arg_template in subtask_template[1:]:
                        bound_arg = bind_value(arg_template, bindings)
                        if isinstance(bound_arg, str) and bound_arg.startswith('?'):
                            logger.error(f"Unbound variable '{bound_arg}' remains after binding subtask '{task_name}' in method '{self.name}'")
                            return None
                        bound_args.append(bound_arg)
                   return (task_name,) + tuple(bound_args)
              except Exception as e:
                   logger.error(f"Error binding arguments for subtask {subtask_template} in method {self.name}: {e}")
                   return None
         else:
             logger.error(f"Invalid subtask template type in method '{self.name}': {type(subtask_template)}")
             return None


class HTNPlanner(Planner): # Correct inheritance
    """Hierarchical Task Network (HTN) Planner with parameter handling."""
    def __init__(self):
        self._controller: Optional[Any] = None
        self._config: Dict[str, Any] = {}
        self.operators: Dict[str, Operator] = {}
        self.methods: Dict[str, List[Method]] = {}
        self.max_depth: int = 5
        
        # --- CORRECTED: Get Goal and Predicate classes from module globals ---
        # These will be either the actual imported classes or their fallbacks
        self._GoalClass = globals().get('Goal')
        self._PredicateClass = globals().get('Predicate')
        # --- END CORRECTION ---

        self._cache: Optional[CognitiveCache] = None
        self._plan_cache_ttl: float = 300.0
        self._planning_session_pwm_cache: Dict[Tuple, Any] = {}
        self._last_top_method_for_goal: Dict[str, Tuple[str, str]] = {}
        
        if not self._GoalClass or (hasattr(self._GoalClass, '_is_dummy') and self._GoalClass._is_dummy()): # type: ignore
            logger.warning("HTNPlanner __init__: Goal class not found or is a dummy.")
        if not self._PredicateClass or (hasattr(self._PredicateClass, '_is_dummy') and self._PredicateClass._is_dummy()): # type: ignore
            logger.warning("HTNPlanner __init__: Predicate class not found or is a dummy.")
            
        self._pending_llm_sketches: Dict[str, Dict[str, Any]] = {}
        self._llm_sketch_task_id_counter: int = 0
        self._kb: Optional[KnowledgeBase] = None # type: ignore


# --- Helper methods for Predicate serialization ---
    def _predicate_to_dict(self, pred: 'Predicate') -> Optional[Dict[str, Any]]: # type: ignore
        _Predicate_p2d = globals().get('Predicate')
        if not _Predicate_p2d or not isinstance(pred, _Predicate_p2d): # type: ignore
            logger.error(f"Cannot serialize non-Predicate object: {pred}")
            return None
        return {
            "name": pred.name, # type: ignore
            "args": list(pred.args),  # Convert tuple to list for JSON compatibility # type: ignore
            "value": pred.value, # type: ignore
            "timestamp": pred.timestamp # type: ignore
        }

    def _dict_to_predicate(self, pred_dict: Dict[str, Any]) -> Optional['Predicate']: # type: ignore
        _Predicate_d2p = globals().get('Predicate')
        if not _Predicate_d2p:
            logger.error("Cannot deserialize to Predicate, Predicate class not available.")
            return None
        try:
            return _Predicate_d2p( # type: ignore
                name=pred_dict["name"],
                args=tuple(pred_dict["args"]), # Convert list back to tuple
                value=pred_dict["value"],
                timestamp=pred_dict.get("timestamp", time.time()) # Add default for timestamp
            )
        except KeyError as e:
            logger.error(f"Missing key {e} in predicate dictionary for deserialization: {pred_dict}")
            return None
        except Exception as e:
            logger.error(f"Error deserializing predicate from dict {pred_dict}: {e}")
            return None

    async def initialize(self, config: Dict[str, Any], controller: Any) -> bool:
        self._controller = controller
        htn_planner_config = config.get("htn_planner", {}) # Specific config for HTNPlanner
        perf_config = config.get("performance", {})
        cache_config = config.get("cognitive_cache", {}) # Get cache config if needed for TTL
        # --- ADDED: Get agent_data_paths for learned library path ---
        agent_data_paths_config = config.get("agent_data_paths", {})
        # --- END ADDED ---

        self._config = {
            **perf_config, 
            **htn_planner_config, 
            **cache_config,
            **agent_data_paths_config # Merge agent_data_paths into self._config
        }

        self.max_depth = self._config.get("max_planning_depth", 5)
        
        # --- GET CACHE REFERENCE ---
        _CognitiveCache_local = globals().get('CognitiveCache') # Get local ref to class for isinstance
        if _CognitiveCache_local and hasattr(controller, 'cache') and \
           isinstance(controller.cache, _CognitiveCache_local): # type: ignore
            self._cache = controller.cache # type: ignore
            logger.info("HTNPlanner: Successfully obtained CognitiveCache reference.")
            # Get TTL for plan cache from HTNPlanner's own config, fallback to cache default
            self._plan_cache_ttl = htn_planner_config.get(
                "plan_cache_ttl_s", 
                cache_config.get("default_ttl", 300.0) # Fallback to cache's default TTL
            )
            logger.info(f"HTNPlanner: Plan cache TTL set to {self._plan_cache_ttl}s.")
        else:
            logger.warning("HTNPlanner: CognitiveCache component not found or invalid. Plan caching will be disabled.")
            self._cache = None
        # --- END GET CACHE REFERENCE ---

        # --- REVISED KB REFERENCE ACQUISITION WITH DETAILED LOGGING ---
        logger.info("HTNPlanner.initialize: Attempting to get KnowledgeBase reference...")
        _KnowledgeBase_htn_scope = globals().get('KnowledgeBase') # KB class as HTNPlanner sees it
        logger.debug(f"HTNPlanner: KnowledgeBase class in my (HTNPlanner) scope: {_KnowledgeBase_htn_scope}")

        kb_instance_from_controller = None
        if not controller:
            logger.error("HTNPlanner: Controller object is None! Cannot get KB.")
        elif hasattr(controller, 'knowledge_base'):
            kb_instance_from_controller = controller.knowledge_base
            logger.info(f"HTNPlanner: Found 'controller.knowledge_base'. Type: {type(kb_instance_from_controller)}")
        elif hasattr(controller, 'components') and isinstance(controller.components, dict) and "knowledge_base" in controller.components:
            kb_instance_from_controller = controller.components.get("knowledge_base")
            logger.info(f"HTNPlanner: Found 'controller.components[\"knowledge_base\"]'. Type: {type(kb_instance_from_controller)}")
        else:
            logger.error("HTNPlanner: Controller has no 'knowledge_base' attribute and no 'knowledge_base' key in 'components'.")
            if hasattr(controller, 'components'):
                 logger.debug(f"HTNPlanner: Available components on controller: {list(controller.components.keys())}")
            else:
                 logger.debug(f"HTNPlanner: Controller object does not have 'components' attribute.")


        if kb_instance_from_controller:
            if _KnowledgeBase_htn_scope: # If HTNPlanner knows what a KnowledgeBase class is
                if isinstance(kb_instance_from_controller, _KnowledgeBase_htn_scope):
                    self._kb = kb_instance_from_controller
                    logger.info("HTNPlanner: Successfully obtained and validated KnowledgeBase reference.")
                else:
                    logger.error(
                        f"HTNPlanner: KnowledgeBase instance from controller is of WRONG TYPE. "
                        f"Actual type: {type(kb_instance_from_controller)}, "
                        f"Expected type (HTNPlanner scope): {_KnowledgeBase_htn_scope}."
                    )
                    self._kb = None # Do not use it if type is wrong
            else: # HTNPlanner doesn't even have a definition for KnowledgeBase
                logger.error(
                    "HTNPlanner: KnowledgeBase class definition not found in HTNPlanner's own scope. "
                    "Cannot validate type of controller's KB. Assuming it's invalid."
                )
                self._kb = None
        else: # kb_instance_from_controller was None from the start
            logger.error("HTNPlanner: No KnowledgeBase instance could be retrieved from controller.")
            self._kb = None
        # --- END REVISED KB REFERENCE ACQUISITION ---


        self._define_example_plan_library() # Load hardcoded library first

        # --- NEW: Load learned methods ---
        self._learned_methods_path: Optional[Path] = None
        learned_lib_path_str = self._config.get("htn_library_path") # From agent_data_paths
        
        if learned_lib_path_str and hasattr(self._controller, 'agent_root_path'):
            self._learned_methods_path = (Path(self._controller.agent_root_path) / learned_lib_path_str).resolve()
            logger.info(f"HTNPlanner learned methods library path set to: {self._learned_methods_path}")
            if self._learned_methods_path.exists():
                await self._load_learned_methods()
            else:
                logger.info(f"No existing learned methods library found at {self._learned_methods_path}.")
        else:
            logger.warning("HTNPlanner: htn_library_path not configured or controller missing agent_root_path. Learned methods will not be persisted/loaded.")
        # --- END NEW ---

        logger.info(
            f"HTNPlanner initialized. Max depth: {self.max_depth}. "
            f"Ops: {len(self.operators)}, Methods: {sum(len(m) for m in self.methods.values())}. "
            f"Plan Caching: {'ENABLED' if self._cache else 'DISABLED'}. "
            f"Learned Lib Path: {self._learned_methods_path}. "
            f"KB Access: {'ENABLED' if self._kb else 'DISABLED'}." 
        )
        return True 

    def _define_example_plan_library(self):
        """Defines a basic library of operators and methods."""
        if not self._PredicateClass or (hasattr(self._PredicateClass, '_is_dummy') and self._PredicateClass._is_dummy()): # type: ignore
            logger.error("HTNPlanner: Predicate class missing or dummy for plan library setup."); return

        # --- Operators with estimated_cost ---
        self.operators["THINKING"] = Operator(
            name="THINKING", parameters=["content"], 
            effects={self._PredicateClass("thoughtAbout", ("?content",), True)},
            estimated_cost=0.1 # Very cheap
        )
        self.operators["QUERY_KB"] = Operator(
            name="QUERY_KB", parameters=["name", "args", "value"], 
            effects={self._PredicateClass("queriedKB", ("?name", "?args"), True)},
            estimated_cost=0.5 
        )
        self.operators["OBSERVE_SYSTEM"] = Operator(
            name="OBSERVE_SYSTEM", parameters=[], 
            effects={self._PredicateClass("systemStateObserved", (), True)},
            estimated_cost=0.3
        )
        self.operators["LIST_FILES"] = Operator(
            name="LIST_FILES", parameters=["path"], 
            preconditions={self._PredicateClass("isDirectory", ("?path",), True)}, 
            effects={self._PredicateClass("listedDirectoryContents", ("?path",), True)},
            estimated_cost=1.0 
        )
        self.operators["READ_FILE"] = Operator(
            name="READ_FILE", parameters=["path"], 
            preconditions={self._PredicateClass("isFile", ("?path",), True)}, 
            effects={self._PredicateClass("readFileContent", ("?path",), True)},
            estimated_cost=1.2 # Slightly more than list
        )
        self.operators["RESPOND_TO_USER"] = Operator( # Assumes LLM call for generation
            name="RESPOND_TO_USER", parameters=["text"], 
            effects={self._PredicateClass("userAcknowledged", (), True)},
            estimated_cost=2.0 # LLM calls can be expensive
        )
        self.operators["GET_AGENT_STATUS"] = Operator(
            name="GET_AGENT_STATUS", parameters=[], 
            effects={self._PredicateClass("statusReported", (), True)},
            estimated_cost=0.2
        )
        self.operators["EXPLAIN_GOAL"] = Operator( # Assumes internal logic or simple LLM
            name="EXPLAIN_GOAL", parameters=[], 
            effects={self._PredicateClass("goalExplained", (), True)},
            estimated_cost=1.5 
        )
        self.operators["WRITE_FILE"] = Operator(
            name="WRITE_FILE", parameters=["path", "content"],
            effects={self._PredicateClass("fileWritten", ("?path",), True)},
            estimated_cost=1.0
        )
        # --- ADD NEW OPERATOR ---
        self.operators["CALL_LLM"] = Operator(
            name="CALL_LLM", parameters=["prompt"], # Parameter name changed to "prompt"
            effects={self._PredicateClass("llmQueried", ("?prompt",), True)},
            estimated_cost=2.5 
        )
        # --- END NEW OPERATOR ---


        # --- Methods (heuristic_score on Method dataclass will be deprecated/ignored for now) ---
        # The heuristic will be calculated dynamically in _decompose
        self.methods["task_observe_and_learn"] = [Method(
            name="method_observe_basic", task_signature=("task_observe_and_learn",), 
            subtasks=["OBSERVE_SYSTEM", ("THINKING", "Observed system state.")]
        )] 
        self.methods["task_analyze_and_break_loop"] = [Method(
            name="method_query_and_think_loop", task_signature=("task_analyze_and_break_loop",), 
            subtasks=[("QUERY_KB", "actionFailed", (), True), ("THINKING", "Queried recent failures.")]
        )] 
        self.methods["task_explore_directory"] = [Method(
            name="method_list_dir", task_signature=("task_explore_directory", "?dirpath"), 
            preconditions={self._PredicateClass("isDirectory", ("?dirpath",), True)}, 
            subtasks=[("LIST_FILES", "?dirpath"), ("THINKING", "Listed files in ?dirpath")]
        )] 
        self.methods["task_read_file"] = [Method(
            name="method_read_this_file", task_signature=("task_read_file", "?filepath"), 
            preconditions={self._PredicateClass("isFile", ("?filepath",), True)}, 
            subtasks=[("READ_FILE", "?filepath")]
        )] 

        self.methods["task_respond_simple"] = [Method(
            name="method_respond_simple", task_signature=("task_respond_simple", "?text"), 
            subtasks=[("RESPOND_TO_USER", "?text")]
        )] 
        self.methods["task_report_status"] = [Method(
            name="method_report_status", task_signature=("task_report_status",), 
            subtasks=["GET_AGENT_STATUS"]
        )] 
        self.methods["task_explain_current_goal"] = [Method(
            name="method_explain_current_goal", task_signature=("task_explain_current_goal",), 
            subtasks=["EXPLAIN_GOAL"]
        )]
        # --- ADD NEW METHOD FOR DIRECT LLM QUERY ---
        # CORRECTED DEFINITION:
        self.methods["task_call_llm_direct_query"] = [Method(
            name="method_execute_direct_llm_query",
            task_signature=("task_call_llm_direct_query", "?prompt"), # <-- CHANGED HERE
            preconditions=set(), 
            subtasks=[("CALL_LLM", "?prompt")] # <-- AND HERE
        )]
        # --- END NEW METHOD ---
        # --- ADD NEW METHOD FOR DIRECT RESPONSE CONTENT ---
        self.methods["task_direct_response_content"] = [Method(
            name="method_execute_direct_response",
            task_signature=("task_direct_response_content", "?response_text_to_send"),
            preconditions=set(),
            subtasks=[("RESPOND_TO_USER", "?response_text_to_send")]
        )]
        # --- END NEW METHOD ---


    def _get_task_name_from_tasktype(self, task: TaskType) -> str:
        """Extracts the task name string from a TaskType."""
        if isinstance(task, tuple):
            return task[0]
        elif isinstance(task, str):
            return task
        else:
            logger.error(f"Invalid TaskType encountered: {task}. Cannot extract name.")
            return "UNKNOWN_TASK_TYPE"

    async def _estimate_method_properties(self, 
                                          method: Method, 
                                          bindings: Dict[str, Any],
                                          current_state_for_heuristic: Set['Predicate'], # type: ignore
                                          full_cognitive_context_for_pwm: Dict[str, Any]
                                         ) -> Tuple[float, float, int]:
        _ConsciousState_est = globals().get('ConsciousState') 
        
        estimated_duration_cost = 0.01 * (1 + len(method.preconditions)) 
        estimated_overall_success_prob = 1.0
        estimated_subtask_count = len(method.subtasks)
        
        for subtask_template in method.subtasks:
            concrete_subtask = method.bind_subtask(subtask_template, bindings)
            if concrete_subtask is None:
                return float('inf'), 0.0, estimated_subtask_count 

            subtask_name = concrete_subtask[0] if isinstance(concrete_subtask, tuple) else concrete_subtask
            
            if subtask_name in self.operators:
                op_def = self.operators[subtask_name]
                estimated_duration_cost += getattr(op_def, 'estimated_cost', 1.0) 

                op_success_prob = 0.5 
                if self._controller and hasattr(self._controller, 'predictive_world_model'):
                    pwm: Optional[PredictiveWorldModel] = getattr(self._controller, 'predictive_world_model') # type: ignore
                    _Predicate_est_pwm = globals().get('Predicate')


                    if pwm and hasattr(pwm, 'process') and _Predicate_est_pwm:
                        op_params_dict_for_pwm: Dict[str, Any] = {}
                        if isinstance(concrete_subtask, tuple) and len(concrete_subtask) > 1:
                            task_args_for_pwm = concrete_subtask[1:]
                            if len(task_args_for_pwm) == len(op_def.parameters):
                                op_params_dict_for_pwm = {p: v for p, v in zip(op_def.parameters, task_args_for_pwm)}
                            else: 
                                 logger.warning(f"Heuristic PWM: Param count mismatch for {subtask_name}. Expected {len(op_def.parameters)}, got {len(task_args_for_pwm)}. Using empty params for PWM query.") 
                        
                        prediction_context_for_pwm = {
                            "current_cs_level_name": full_cognitive_context_for_pwm.get("current_cs_level_name", "UNKNOWN"),
                            "active_goal_type": full_cognitive_context_for_pwm.get(
                                "current_goal_type", 
                                self._controller._get_active_goal_type() if hasattr(self._controller, '_get_active_goal_type') else "unknown" # type: ignore
                            ),
                            "php_levels": { 
                                "pain": full_cognitive_context_for_pwm.get("php_levels", {}).get("pain", 0.0),
                                "happiness": full_cognitive_context_for_pwm.get("php_levels", {}).get("happiness", 5.0),
                                "purpose": full_cognitive_context_for_pwm.get("php_levels", {}).get("purpose", 5.0),
                            },
                            "drives": full_cognitive_context_for_pwm.get("drives", {}) 
                        }
                        
                        pred_input_for_pwm = {
                            "predict_request": {
                                "action_to_execute": {"type": subtask_name, "params": op_params_dict_for_pwm},
                                "context": prediction_context_for_pwm,
                                "current_world_state_predicates": current_state_for_heuristic 
                            }
                        }
                        
                        params_tuple_for_cache = tuple(sorted(op_params_dict_for_pwm.items()))
                        
                        hashable_context_items = []
                        for k_ctx, v_ctx in sorted(prediction_context_for_pwm.items()):
                            if isinstance(v_ctx, dict):
                                hashable_context_items.append((k_ctx, tuple(sorted(v_ctx.items()))))
                            elif isinstance(v_ctx, list):
                                try:
                                    hashable_context_items.append((k_ctx, tuple(v_ctx)))
                                except TypeError: 
                                    hashable_context_items.append((k_ctx, str(v_ctx))) 
                            else:
                                hashable_context_items.append((k_ctx, v_ctx))
                        
                        context_tuple_for_cache = tuple(hashable_context_items)
                        cache_key_pwm = ("pwm_heuristic_pred", subtask_name, params_tuple_for_cache, context_tuple_for_cache)


                        pwm_output = None
                        if hasattr(self, '_planning_session_pwm_cache') and cache_key_pwm in self._planning_session_pwm_cache:
                            pwm_output = self._planning_session_pwm_cache[cache_key_pwm]
                            logger.debug(f"Heuristic PWM: Cache hit for {subtask_name} with params {op_params_dict_for_pwm}")
                        else:
                            logger.debug(f"Heuristic PWM: Querying PWM for op '{subtask_name}' with params {op_params_dict_for_pwm}")
                            process_method_pwm = getattr(pwm, 'process')
                            if asyncio.iscoroutinefunction(process_method_pwm):
                                pwm_output_raw = await process_method_pwm(pred_input_for_pwm)
                                if isinstance(pwm_output_raw, dict): pwm_output = pwm_output_raw.get("prediction")
                            else: 
                                logger.warning(f"PWM process method for {subtask_name} not async.")
                            
                            if hasattr(self, '_planning_session_pwm_cache') and pwm_output is not None:
                                self._planning_session_pwm_cache[cache_key_pwm] = pwm_output


                        if pwm_output and isinstance(pwm_output, dict) and pwm_output.get("all_probabilities"):
                            all_probs = pwm_output.get("all_probabilities")
                            if isinstance(all_probs, dict):
                                op_success_prob = float(all_probs.get("success", 0.1)) 
                            else: 
                                op_success_prob = float(pwm_output.get("confidence", 0.1)) if pwm_output.get("predicted_outcome") == "success" else 0.1
                        elif pwm_output and isinstance(pwm_output, dict) and pwm_output.get("predicted_outcome") == "success":
                            op_success_prob = float(pwm_output.get("confidence", 0.5)) 
                        else:
                            op_success_prob = 0.4 
                            logger.debug(f"Heuristic PWM: PWM for '{subtask_name}' did not predict success or provide probabilities. Defaulting op_success_prob to {op_success_prob}. PWM_output: {str(pwm_output)[:100]}")

                estimated_overall_success_prob *= op_success_prob
            
            elif subtask_name in self.methods:
                estimated_duration_cost += 1.5 
                estimated_overall_success_prob *= 0.75 
                logger.debug(f"Heuristic: Complex subtask '{subtask_name}' contributes static cost 1.5, success factor 0.75.")
            else:
                estimated_duration_cost += 10.0 
                estimated_overall_success_prob *= 0.01 
                logger.warning(f"Heuristic: Unknown subtask '{subtask_name}' in method '{method.name}'. Penalizing heavily.")

            if estimated_overall_success_prob < 1e-6:
                estimated_overall_success_prob = 1e-6
                
        logger.debug(f"Method '{method.name}' estimated props: DurationCost={estimated_duration_cost:.2f}, SuccessProb={estimated_overall_success_prob:.3f}, Subtasks={estimated_subtask_count}")
        return estimated_duration_cost, estimated_overall_success_prob, estimated_subtask_count

    async def plan(self, 
                   goal: 'Goal', 
                   current_state: Set['Predicate'], 
                   modification_hints: Optional[Dict[str, Any]] = None # NEW PARAMETER
                  ) -> Optional[List[Dict[str, Any]]]: # type: ignore
        if hasattr(self, '_planning_session_pwm_cache'): self._planning_session_pwm_cache.clear()
        else: self._planning_session_pwm_cache = {}

        if hasattr(self, '_last_top_method_for_goal'): self._last_top_method_for_goal.clear()
        else: self._last_top_method_for_goal = {}

        if not self._GoalClass or (hasattr(self._GoalClass, '_is_dummy') and self._GoalClass._is_dummy()): # type: ignore
            logger.error("HTNPlanner: Goal class definition is missing or a dummy for planning."); return None
        if not isinstance(goal, self._GoalClass): # type: ignore
            logger.error(f"HTNPlanner: Invalid Goal object (type: {type(goal)})."); return None
        if not isinstance(current_state, set):
             logger.error("HTNPlanner: Invalid current_state."); return None

        initial_task = self._goal_to_task(goal)
        if not initial_task: 
            goal_desc_log = goal.description if hasattr(goal,'description') else 'Unknown Goal'
            logger.warning(f"Could not map goal: {goal_desc_log}") # type: ignore
            return None
        
        if modification_hints:
            goal_id_log = goal.id if hasattr(goal,'id') else 'Unknown ID'
            logger.info(f"HTNPlanner: Received modification_hints for goal '{goal_id_log}', task '{initial_task}': {modification_hints}")

        # --- Calculate Cache Key (used for both plan cache and LLM sketch tracking) ---
        cache_key_for_this_problem: Optional[str] = None
        _Predicate_plan_cache = globals().get('Predicate')
        if self._PredicateClass and (not current_state or isinstance(next(iter(current_state), None), self._PredicateClass)): # type: ignore
            try:
                goal_id_for_cache = goal.id if hasattr(goal, 'id') else goal.description
                state_repr_for_hash = frozenset(
                    sorted([f"{p.name}:{str(p.args)}:{p.value}" for p in current_state if isinstance(p, self._PredicateClass)])
                )
                hints_repr_for_hash = ""
                if modification_hints:
                    try:
                        hints_repr_for_hash = str(hash(frozenset(sorted(modification_hints.items()))))
                    except TypeError: 
                        hints_repr_for_hash = str(modification_hints)

                cache_key_tuple = ("htn_problem", goal_id_for_cache, state_repr_for_hash, hints_repr_for_hash)
                cache_key_for_this_problem = str(hash(cache_key_tuple))
                logger.debug(f"HTNPlanner: Cache key for current planning problem (with hints): {cache_key_for_this_problem}")

            except Exception as e_cache_key:
                logger.error(f"HTNPlanner: Error generating cache key: {e_cache_key}")
        # --- End Cache Key Calculation ---

        # --- 1. Check for completed LLM sketch first if one was pending for this problem ---
        if cache_key_for_this_problem and cache_key_for_this_problem in self._pending_llm_sketches:
            logger.info(f"HTNPlanner: Found pending LLM sketch for problem key '{cache_key_for_this_problem}'. Processing...")
            llm_sketch_result_ops = await self._process_completed_llm_sketch(cache_key_for_this_problem)
            if llm_sketch_result_ops: 
                logger.info("HTNPlanner: LLM sketch processed. Attempting to use this sketch directly for the current plan.")
                final_action_plan_from_sketch = self._format_plan_result(llm_sketch_result_ops)
                if final_action_plan_from_sketch:
                    logger.info(f"Successfully formatted plan from LLM sketch (len: {len(final_action_plan_from_sketch)}). Using it.")
                    if self._cache and cache_key_for_this_problem:
                        try:
                            await self._cache.put(cache_key_for_this_problem, final_action_plan_from_sketch, ttl_override=self._plan_cache_ttl)
                            logger.info(f"Stored LLM sketch plan in cache. Key: {cache_key_for_this_problem}")
                        except Exception as e_cache_llm:
                            logger.error(f"Error storing LLM sketch plan in cache: {e_cache_llm}")
                    return final_action_plan_from_sketch
                else:
                    logger.error("HTNPlanner: Failed to format the processed LLM sketch. Proceeding to standard planning.")
            elif cache_key_for_this_problem in self._pending_llm_sketches: 
                logger.info(f"HTNPlanner: LLM sketch for problem key '{cache_key_for_this_problem}' is still in progress. No plan this cycle.")
                return None 

        # --- 2. Try Plan Cache (existing logic) ---
        if self._cache and cache_key_for_this_problem:
            cached_plan_data = await self._cache.get(cache_key_for_this_problem)
            if cached_plan_data: 
                if isinstance(cached_plan_data, list) and all(isinstance(action, dict) and "type" in action for action in cached_plan_data):
                    logger.info(f"Cache hit for goal '{goal.description}'. Returning cached plan (len: {len(cached_plan_data)}).")
                    return cached_plan_data
        # --- End Plan Cache ---

        effective_max_depth = self.max_depth 
        if self._controller and hasattr(self._controller, 'config') and isinstance(self._controller.config, dict): # type: ignore
            live_agent_config = self._controller.config 
            performance_config_from_live_agent = live_agent_config.get("performance", {})
            if isinstance(performance_config_from_live_agent, dict):
                config_max_depth = performance_config_from_live_agent.get("max_planning_depth", self.max_depth) 
                if isinstance(config_max_depth, int) and config_max_depth > 0:
                    effective_max_depth = config_max_depth 
        
        current_cs_level_for_plan = None
        _ConsciousState_planner = globals().get('ConsciousState') 

        if self._controller and hasattr(self._controller, 'consciousness_level') and _ConsciousState_planner:
            controller_cs_obj = getattr(self._controller, 'consciousness_level')
            if isinstance(controller_cs_obj, _ConsciousState_planner): # type: ignore
                current_cs_level_for_plan = controller_cs_obj
        
        if current_cs_level_for_plan: 
            _ConsciousState_plan = globals().get('ConsciousState')
            if _ConsciousState_plan and not hasattr(_ConsciousState_plan, '_is_dummy'): # type: ignore
                htn_planner_live_config = self._controller.config.get("htn_planner", {}) if self._controller and hasattr(self._controller, 'config') else {}
                min_depth_low_cs = htn_planner_live_config.get("min_planning_depth_on_low_cs", 1)
                max_depth_low_cs = htn_planner_live_config.get("max_planning_depth_on_low_cs", 2)
                if current_cs_level_for_plan.value <= _ConsciousState_plan.PRE_CONSCIOUS.value: #type: ignore
                    effective_max_depth = min(max_depth_low_cs, effective_max_depth)
                    effective_max_depth = max(min_depth_low_cs, effective_max_depth)

    
        current_full_cognitive_context_for_planner = {} 
        if self._controller and hasattr(self._controller, '_oscar_get_cognitive_state'):
            get_cog_state_method = getattr(self._controller, '_oscar_get_cognitive_state')
            if asyncio.iscoroutinefunction(get_cog_state_method):
                current_full_cognitive_context_for_planner = await get_cog_state_method(skip_component_statuses=True)


        logger.info(f"Planning for task: {initial_task}, MaxDepth: {effective_max_depth}")
        plan_result_from_decompose: Optional[PlanResultType] = None
        
        # --- Simplified iterative deepening for snippet clarity ---
        for current_depth_limit_iter in range(1, effective_max_depth + 1):
            logger.debug(f"HTN ID: Trying decomposition with depth_limit: {current_depth_limit_iter}")
            plan_result_from_decompose = await self._decompose(
                task=initial_task, state=current_state, depth=0, depth_limit=current_depth_limit_iter,
                full_cognitive_context_for_pwm=current_full_cognitive_context_for_planner
            )
            if plan_result_from_decompose:
                logger.info(f"HTN ID: Plan found at depth_limit {current_depth_limit_iter}.")
                break 
        # --- End simplified iterative deepening ---

        final_action_plan: Optional[List[Dict[str, Any]]] = None
        if plan_result_from_decompose:
            final_action_plan = self._format_plan_result(plan_result_from_decompose)


        # --- 3. If HTN planning failed, and task is complex, try LLM sketching ---
        if final_action_plan is None:
            task_name_str_for_llm_check = self._get_task_name_from_tasktype(initial_task)
            if task_name_str_for_llm_check not in self.operators: # Only sketch for complex tasks
                if cache_key_for_this_problem: # Ensure we have a key
                    logger.info(f"HTN Planning failed for '{initial_task}'. Initiating LLM sketch (Key: {cache_key_for_this_problem}).")
                    sketch_status = await self._llm_assisted_plan_sketch(goal, initial_task, current_state, cache_key_for_this_problem)
                    if sketch_status == "sketching_initiated" or sketch_status == "sketching_already_pending":
                        logger.info(f"HTNPlanner: LLM sketch is now pending for '{initial_task}'. No plan this cycle.")
                        return None # Signal that planning is deferred to LLM sketch
                else:
                    logger.warning("HTNPlanner: Cannot initiate LLM sketch as cache_key_for_this_problem could not be generated.")
            else:
                logger.debug(f"HTNPlanner: Task '{task_name_str_for_llm_check}' is a primitive operator. Standard HTN planning failed, no LLM sketch triggered for it.")

        if final_action_plan and self._cache and cache_key_for_this_problem:
            await self._cache.put(cache_key_for_this_problem, final_action_plan, ttl_override=self._plan_cache_ttl)
            logger.info(f"Stored successful plan (len: {len(final_action_plan)}) in cache. Key: {cache_key_for_this_problem}")

        if final_action_plan:
            return final_action_plan
        else:
            goal_desc_log_final = goal.description if hasattr(goal,'description') else 'Unknown Goal'
            logger.warning(f"HTN Planning failed for goal: {goal_desc_log_final} (task: {initial_task}). No plan returned.") # type: ignore
            return None
        
    def _format_plan_result(self, plan_ops_with_bindings: PlanResultType) -> Optional[List[Dict[str, Any]]]:
        action_plan_formatted: List[Dict[str, Any]] = []
        if not plan_ops_with_bindings: return None
        
        for operator_name, bindings_dict in plan_ops_with_bindings:
            op_def = self.operators.get(operator_name)
            if not op_def:
                logger.error(f"Formatting plan: Unknown operator '{operator_name}'. Invalidating plan.")
                return None
            
            params_for_action: Dict[str, Any] = {}
            for p_name in op_def.parameters:
                if p_name not in bindings_dict:
                    logger.error(f"Formatting plan: Operator '{operator_name}' missing binding for parameter '{p_name}'. Invalidating plan.")
                    return None
                params_for_action[p_name] = bindings_dict[p_name]
            
            action_plan_formatted.append({"type": operator_name, "params": params_for_action})
        return action_plan_formatted


    def _goal_to_task(self, goal: 'Goal') -> Optional[TaskType]: # type: ignore
        """Maps a Goal object description to an initial TaskType for planning."""
        if not self._GoalClass or (hasattr(self._GoalClass, '_is_dummy') and self._GoalClass._is_dummy()): # type: ignore
            logger.error("HTNPlanner: Goal class definition is missing or a dummy. Cannot map goal to task."); return None
        if not isinstance(goal, self._GoalClass): # type: ignore
            logger.error(f"HTNPlanner: Invalid object passed as goal (type: {type(goal)}). Expected {self._GoalClass}."); return None
        if not hasattr(goal, 'description') or not isinstance(goal.description, str):
            logger.error(f"HTNPlanner: Goal object missing 'description' string attribute."); return None

        desc = goal.description.strip().lower() 
        logger.debug(f"HTNPlanner: Mapping goal description to task: '{desc}'")

        if desc == "observe and learn from the environment": 
            return "task_observe_and_learn"
        
        if desc.startswith("analyze and break loop"):
            return "task_analyze_and_break_loop"
        if desc == "report status":
            return "task_report_status"
        if desc == "explain goal":
            return "task_explain_current_goal"

        match_read = re.match(r"(?:read\s+file|get\s+content\s+of)\s*:\s*(.+)", desc, re.IGNORECASE)
        if match_read:
            path = match_read.group(1).strip()
            logger.debug(f"HTNPlanner: Mapped to task_read_file with path: '{path}'")
            return ("task_read_file", path) if path else None

        match_explore = re.match(r"(?:explore|list|ls|dir)\s*(?:directory|files(?:\s+in)?)?\s*:\s*(.+)", desc, re.IGNORECASE)
        if match_explore:
            path = match_explore.group(1).strip()
            logger.debug(f"HTNPlanner: Mapped to task_explore_directory with path: '{path}'")
            return ("task_explore_directory", path) if path else None

        if desc in ["explore", "list", "explore directory", "list files"]:
            logger.debug(f"HTNPlanner: Mapped to task_explore_directory with default path '.'")
            return ("task_explore_directory", ".")
        
        # Add this new block for task_has_method
        match_task_has_method = re.match(r"task_has_method\s*:\s*(.+)", desc, re.IGNORECASE)
        if match_task_has_method:
            param_val = match_task_has_method.group(1).strip()
            logger.debug(f"HTNPlanner: Mapped to task_has_method with param: '{param_val}'")
            return ("task_has_method", param_val) if param_val else None
        
        # Add this new block for task_needs_sketch
        match_task_needs_sketch = re.match(r"task_needs_sketch\s*:\s*(.+)", desc, re.IGNORECASE)
        if match_task_needs_sketch:
            param_val_sketch = match_task_needs_sketch.group(1).strip()
            logger.debug(f"HTNPlanner: Mapped to task_needs_sketch with param: '{param_val_sketch}'")
            return ("task_needs_sketch", param_val_sketch) if param_val_sketch else None

        # Add this new block for task_llm_will_fail
        match_task_llm_fail = re.match(r"task_llm_will_fail\s*:\s*(.+)", desc, re.IGNORECASE)
        if match_task_llm_fail:
            param_val_fail = match_task_llm_fail.group(1).strip()
            logger.debug(f"HTNPlanner: Mapped to task_llm_will_fail with param: '{param_val_fail}'")
            return ("task_llm_will_fail", param_val_fail) if param_val_fail else None
        
        # --- NEW: Match for "task_direct_response_content" description ---
        match_direct_response = re.match(r"task_direct_response_content\s*:\s*(.+)", desc, re.IGNORECASE | re.DOTALL)
        if match_direct_response:
            response_text_to_send = match_direct_response.group(1).strip()
            if response_text_to_send:
                logger.debug(f"HTNPlanner: Mapped to task_direct_response_content with text: '{response_text_to_send[:50]}...'")
                return ("task_direct_response_content", response_text_to_send)
        # --- END NEW ---

        # General "respond to user : " match
        match_respond = re.match(r"respond\s+to\s+user\s*:\s*(.+)", desc, re.IGNORECASE)
        if match_respond:
            user_text = match_respond.group(1).strip()
            logger.debug(f"HTNPlanner: Mapped to task_respond_simple with text: '{user_text[:50]}...'")
            return ("task_respond_simple", user_text) if user_text else None
        
        # Regex for write: match "write file : path content : content_text"
        match_write = re.match(
            r"(?:write|create|save)\s+(?:file|to)?\s*:?\s*([^:]+?)\s+(?:content|with|text)\s*:?\s*(.+)", 
            desc, re.IGNORECASE | re.DOTALL
        )
        if match_write:
            path_param = match_write.group(1).strip().strip("'\"")
            content_param = match_write.group(2).strip().strip("'\"")
            if path_param and content_param is not None: # content can be empty string
                logger.debug(f"HTNPlanner: Mapped to WRITE_FILE task with path: '{path_param}', content_len: {len(content_param)}")
                return ("WRITE_FILE", path_param, content_param) 
        
        # Try to match LLM query
        match_llm_query = re.match(r"(?:llm query|ask llm|query llm)\s*:\s*(.+)", desc, re.IGNORECASE)
        if match_llm_query:
            query_text = match_llm_query.group(1).strip()
            if query_text:
                logger.debug(f"HTNPlanner: Mapped to task_call_llm_direct_query with prompt: '{query_text[:50]}...'")
                return ("task_call_llm_direct_query", query_text) 
        
        logger.warning(f"HTNPlanner: No specific task mapping found for goal description: '{goal.description}'. Cannot plan.") # type: ignore
        return None


    async def _decompose(self, task: TaskType, state: Set['Predicate'], depth: int, depth_limit: int,
                         full_cognitive_context_for_pwm: Dict[str, Any]) -> Optional[PlanResultType]: # type: ignore
        
        logger.info(f"HTNPlanner._decompose DEBUG ENTRY: Received full_cognitive_context_for_pwm with keys: {list(full_cognitive_context_for_pwm.keys())}")
        task_name = self._get_task_name_from_tasktype(task)
        logger.debug(f"{'  ' * depth}Decomposing task: {task} (Depth: {depth}, Limit: {depth_limit})")

        if depth >= depth_limit: 
            logger.debug(f"{'  ' * depth}Depth limit {depth_limit} hit for task: {task} at depth {depth}. Backtracking.")
            return None

        if task_name in self.operators:
            operator = self.operators[task_name]; bindings = {}
            task_args = task[1:] if isinstance(task, tuple) else ()
            if len(task_args) == len(operator.parameters): bindings = {p: v for p, v in zip(operator.parameters, task_args)}
            elif task_args or operator.parameters: logger.warning(f"Arg count mismatch op {task_name}. Task:{task_args}, Op:{operator.parameters}"); return None
            if operator.is_applicable(state, bindings):
                logger.debug(f"{'  ' * depth}Applicable primitive: {task_name} w/ {bindings}")
                return [(task_name, bindings)]
            else:
                logger.debug(f"{'  ' * depth}Op '{task_name}' preconds not met w/ {bindings}.")
                return None

        if task_name in self.methods:
            applicable_methods_with_heuristic_info: List[Tuple[Method, Dict[str, Any], float, float, int]] = []
            
            for method_candidate in self.methods[task_name]:
                bindings = method_candidate.get_parameter_bindings(task)
                if bindings is not None and method_candidate.is_applicable(state, bindings):
                    duration_cost, success_prob, subtask_count = await self._estimate_method_properties(
                        method_candidate, bindings, state, full_cognitive_context_for_pwm
                    )
                    applicable_methods_with_heuristic_info.append(
                        (method_candidate, bindings, duration_cost, success_prob, subtask_count)
                    )

            if not applicable_methods_with_heuristic_info:
                logger.debug(f"{'  ' * depth}No applicable methods found for task '{task_name}'.")
                return None

            processed_methods_for_sorting: List[Tuple[Method, Dict[str, Any], float]] = []
            _ConsciousState_planner_decompose = globals().get('ConsciousState') 

            for method_info_tuple in applicable_methods_with_heuristic_info:
                method_cand, bind_dict, dur_cost, succ_prob, st_count = method_info_tuple
                
                safe_succ_prob = max(succ_prob, 1e-6) 
                combined_heuristic_value = dur_cost / safe_succ_prob

                current_cs_level_obj = None 
                if _ConsciousState_planner_decompose and full_cognitive_context_for_pwm:
                    cs_name_from_ctx = full_cognitive_context_for_pwm.get("consciousness_level") # From _oscar_get_cognitive_state
                    if cs_name_from_ctx and isinstance(cs_name_from_ctx, str):
                        try:
                            current_cs_level_obj = _ConsciousState_planner_decompose[cs_name_from_ctx.upper()] 
                        except KeyError:
                             logger.warning(f"Could not map CS name '{cs_name_from_ctx}' to ConsciousState enum member in _decompose.")
                
                if current_cs_level_obj: 
                    cs_val_received = current_cs_level_obj.value
                    pre_cs_val_enum = _ConsciousState_planner_decompose.PRE_CONSCIOUS.value
                    
                    if cs_val_received <= pre_cs_val_enum:
                        htn_planner_live_config_decompose = {}
                        if self._controller and hasattr(self._controller, 'config') and isinstance(self._controller.config, dict):
                            htn_planner_live_config_decompose = self._controller.config.get("htn_planner", {}) 
                        
                        simplicity_penalty_factor_plan = float(htn_planner_live_config_decompose.get(
                            "low_cs_simplicity_penalty_factor", 
                            self._config.get("low_cs_simplicity_penalty_factor", 0.5) 
                        ))
                        
                        penalty_amount = st_count * simplicity_penalty_factor_plan * (dur_cost + 1) 
                        original_h_val_before_penalty = combined_heuristic_value 
                        combined_heuristic_value += penalty_amount
                        
                        if penalty_amount > 0:
                            logger.debug(
                                f"{'  ' * depth}Method '{method_cand.name}' combined_heuristic adjusted for low CS ({current_cs_level_obj.name}). "
                                f"Original H: {original_h_val_before_penalty:.2f}, "
                                f"Penalty (using subtask_count {st_count}): {penalty_amount:.2f}, New Combined H: {combined_heuristic_value:.2f}"
                            )
                processed_methods_for_sorting.append((method_cand, bind_dict, combined_heuristic_value))
            
            processed_methods_for_sorting.sort(key=lambda x: x[2])
            
            logger.debug(
                f"{'  ' * depth}Found {len(processed_methods_for_sorting)} applicable method(s) for task '{task_name}'. "
                f"Sorted by new heuristic: "
                f"{[(m.name, round(h_val, 2)) for m, _, h_val in processed_methods_for_sorting]}"
            )

            for method, bindings, _heuristic_score_val in processed_methods_for_sorting: 
                logger.debug(f"{'  ' * depth}Trying method: {method.name} (Combined Heuristic: {_heuristic_score_val:.2f}) w/ {bindings}")
                plan_fragment: PlanResultType = []
                current_plan_state = state.copy() 
                possible = True
                
                for subtask_template in method.subtasks:
                    concrete_subtask = method.bind_subtask(subtask_template, bindings)
                    if concrete_subtask is None:
                        logger.error(f"Subtask binding failed in {method.name} for {subtask_template}")
                        possible = False; break
                    sub_plan = await self._decompose(concrete_subtask, current_plan_state, depth + 1, depth_limit, full_cognitive_context_for_pwm) 
                    if sub_plan:
                        plan_fragment.extend(sub_plan)
                        temp_state_after_sub_plan = current_plan_state 
                        for op_name_eff, op_bindings_eff in sub_plan: 
                             if op_name_eff in self.operators:
                                  applied_state_effect = self.operators[op_name_eff].apply(temp_state_after_sub_plan, op_bindings_eff)
                                  if applied_state_effect is None: possible = False; break
                                  temp_state_after_sub_plan = applied_state_effect
                             else: possible = False; break 
                        if not possible: break
                        current_plan_state = temp_state_after_sub_plan 
                    else: possible = False; break
                
                if possible:
                    logger.debug(f"{'  ' * depth}Method '{method.name}' successfully decomposed task '{task_name}'.")
                    if hasattr(method, 'metadata') and isinstance(method.metadata, dict):
                        method.metadata["usage_count"] = method.metadata.get("usage_count", 0) + 1
                        method.metadata["last_used_ts"] = time.time()
                        logger.debug(f"Updated usage stats for method '{method.name}': Usage={method.metadata['usage_count']}, LastUsed={method.metadata['last_used_ts']:.0f}")
                    
                    return plan_fragment 
                else: 
                    logger.debug(f"{'  ' * depth}Method '{method.name}' failed to decompose. Trying next.")

            logger.debug(f"{'  ' * depth}No successful method found for task: {task} within depth limit {depth_limit}")
            return None
        else:
            logger.warning(f"{'  ' * depth}Task name '{task_name}' is unknown (not an operator or complex task).")
            return None

    async def _llm_assisted_plan_sketch(self,
                                  goal: 'Goal', # type: ignore
                                  task_to_solve: TaskType,
                                  current_state: Set['Predicate'], # type: ignore
                                  cache_key_for_sketch: str # Add cache_key parameter
                                 ) -> str: # Returns a status string
        """
        Schedules an LLM call to generate a plan sketch.
        Stores the future and context in _pending_llm_sketches.
        Returns "sketching_initiated" or "sketching_already_pending" or "sketching_failed_to_schedule".
        """
        if not self._controller or not hasattr(self._controller, 'model_name') or \
           not hasattr(self._controller, '_asyncio_loop') or not hasattr(self._controller, 'schedule_offline_task') or \
           not call_ollama:
            logger.error("HTNPlanner._llm_assisted_plan_sketch: Controller, its attributes, or call_ollama not available.")
            return "sketching_failed_to_schedule"

        if cache_key_for_sketch in self._pending_llm_sketches:
            logger.info(f"HTNPlanner: LLM sketch for problem key '{cache_key_for_sketch}' already pending.")
            return "sketching_already_pending"

        # --- Construct LLM prompt (same logic as before) ---
        _Goal_llm_sketch = globals().get('Goal')
        _Predicate_llm_sketch = globals().get('Predicate')
        if not _Goal_llm_sketch or not _Predicate_llm_sketch:
            logger.error("HTNPlanner._llm_assisted_plan_sketch: Goal or Predicate type not available for prompt.")
            return "sketching_failed_to_schedule"

        goal_desc = goal.description if hasattr(goal, 'description') else "Unknown goal"
        success_criteria_text_list = []
        if hasattr(goal, 'success_criteria') and isinstance(goal.success_criteria, set):
            for p_sc in goal.success_criteria:
                if isinstance(p_sc, _Predicate_llm_sketch) and all(hasattr(p_sc, attr) for attr in ['name', 'args', 'value']):
                    success_criteria_text_list.append(f"{p_sc.name}{str(p_sc.args)}={p_sc.value}") # type: ignore
        success_criteria_text = "; ".join(success_criteria_text_list) if success_criteria_text_list else "Not specified."
        
        current_state_summary_list = []
        if isinstance(current_state, set):
            count = 0
            sorted_current_state_for_prompt = sorted(list(current_state), key=lambda p_item: (getattr(p_item,'name',''), str(getattr(p_item,'args',''))))

            for p_state in sorted_current_state_for_prompt:
                if isinstance(p_state, _Predicate_llm_sketch) and all(hasattr(p_state, attr) for attr in ['name', 'args', 'value']):
                    current_state_summary_list.append(f"{p_state.name}{str(p_state.args)}={p_state.value}") # type: ignore
                    count += 1
                    if count >= 15: break
        current_state_text_summary = "; ".join(current_state_summary_list) if current_state_summary_list else "No specific predicates."

        available_operators_text = "\n".join([
            f"- {op_name}({', '.join(op_def.parameters if hasattr(op_def, 'parameters') else [])}) # Cost: {getattr(op_def, 'estimated_cost', 'N/A')}"
            for op_name, op_def in self.operators.items()
        ])
        
        system_prompt = ( 
            "You are an AI HTN planning assistant. Your goal is to decompose a high-level task into a sequence of "
            "available primitive operators. Provide the plan as a JSON list of lists, where each inner list "
            "is an operator name followed by its string parameter values. For example: "
            "[[\"OPERATOR_NAME1\", \"param1_val\"], [\"OPERATOR_NAME2\"]]. "
            "Instantiate parameters with concrete string values based on the goal and current state. "
            "If an operator takes no parameters, the inner list should only contain the operator name, e.g., [\"OPERATOR_NAME_NO_PARAMS\"]. "
            "Focus on creating a valid and logical sequence using ONLY the provided 'Available Primitive Operators'. "
            "Do not invent new operators. Ensure parameter counts match. "
            "If you cannot form a plan, return JSON null. "
            "After the JSON block, on new lines, you can optionally add 'Precondition Hints:' followed by text if this were to become a reusable method."
        )
        task_name_for_prompt = self._get_task_name_from_tasktype(task_to_solve)
        task_params_for_prompt_list = []
        if isinstance(task_to_solve, tuple) and len(task_to_solve) > 1:
            for i, arg_val in enumerate(task_to_solve[1:]):
                task_params_for_prompt_list.append(f"parameter ?param{i} = '{arg_val}'")
        task_params_str_for_prompt = "; ".join(task_params_for_prompt_list)
        if task_params_str_for_prompt:
            task_params_str_for_prompt = f" (with parameters: {task_params_str_for_prompt})"

        user_prompt = (
            f"High-Level Task to Solve: {task_name_for_prompt}{task_params_str_for_prompt}\n"
            f"(Original Goal Description: '{goal_desc}')\n"
            f"Goal Success Criteria to Achieve: {success_criteria_text}\n"
            f"Key Current State Predicates (summary): {current_state_text_summary}\n\n"
            f"Available Primitive Operators (use only these):\n{available_operators_text}\n\n"
            f"IMPORTANT: For QUERY_KB, provide exactly THREE parameters: name (string), args (list of strings/numbers), and value (true/false).\n"
            f"Suggest a sequence of operators as a JSON list of lists to achieve the goal. "
            f"When an operator in your plan needs a parameter that corresponds to one of the parameters from the 'High-Level Task to Solve' (e.g., ?param0), "
            f"YOU MUST USE THE VALUE PROVIDED FOR THAT PARAMETER (e.g., '{task_to_solve[1] if isinstance(task_to_solve, tuple) and len(task_to_solve) > 1 else 'example_param_value'}'). "
            f"For other parameters, use sensible concrete string values based on the goal and current state, or common paths like '.' for current directory if appropriate. "
            f"Format: [[\"OPERATOR_NAME1\", \"param1_val_as_string\"], [\"OPERATOR_NAME2\"]]. "
            f"Do not include comments inside the JSON. "
            f"If the plan is impossible with the given operators, return JSON null."
        )
        # --- End LLM Prompt Construction ---

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        # Get LLM config from self._config (which should have htn_planner section)
        llm_sketch_timeout = float(self._config.get("llm_sketch_timeout_s", 60.0))
        llm_sketch_temp = float(self._config.get("llm_sketch_temperature", 0.2))

        logger.info(f"HTN_LLM_SKETCH: Scheduling LLM plan sketch for task '{task_name_for_prompt}' (Key: {cache_key_for_sketch}).")

        # Schedule the call_ollama coroutine
        llm_future = self._controller.schedule_offline_task( # type: ignore
            call_ollama, # The coroutine function
            self._controller.model_name, # type: ignore
            messages,
            llm_sketch_temp,
            self._controller._asyncio_loop, # type: ignore
            timeout=llm_sketch_timeout
        )

        self._pending_llm_sketches[cache_key_for_sketch] = {
            "future": llm_future,
            "goal": goal, # Store for context when processing result
            "task_to_solve": task_to_solve,
            "initial_state": current_state,
            "timestamp_scheduled": time.time()
        }
        return "sketching_initiated"

    async def _process_completed_llm_sketch(self, cache_key_for_sketch: str) -> Optional[PlanResultType]:
        """
        Checks a specific pending LLM sketch. If completed, parses the result,
        attempts method abstraction, and returns the PlanResultType (sketch) or None.
        Removes the task from pending if processed.
        """
        if cache_key_for_sketch not in self._pending_llm_sketches:
            return None # Not pending or already processed

        task_info = self._pending_llm_sketches[cache_key_for_sketch]
        future = task_info["future"]

        if not future.done():
            logger.debug(f"HTN_LLM_SKETCH_PROC: Sketch for key '{cache_key_for_sketch}' still pending.")
            return None # Not yet complete

        logger.info(f"HTN_LLM_SKETCH_PROC: Processing completed LLM sketch for key '{cache_key_for_sketch}'.")
        # Remove from pending once we start processing it
        del self._pending_llm_sketches[cache_key_for_sketch]

        llm_plan_sketch_ops: Optional[PlanResultType] = None
        try:
            response_str, llm_error = await future # Get result (already awaited by secondary loop)

            if llm_error or not response_str:
                logger.error(f"HTN_LLM_SKETCH_PROC: LLM sketch task failed or empty. Error: {llm_error}")
            elif not response_str.strip() or response_str.strip().lower() == "null":
                logger.info("HTN_LLM_SKETCH_PROC: LLM sketch task returned null/empty response.")
            else:
                json_sketch_str = None
                match_json_block = re.search(r"```json\s*([\s\S]+?)\s*```", response_str, re.DOTALL)
                if match_json_block:
                    json_sketch_str = match_json_block.group(1).strip()
                else:
                    first_bracket = response_str.find('[') # Check for list first
                    last_bracket = response_str.rfind(']')
                    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                        potential_json_candidate = response_str[first_bracket : last_bracket + 1]
                        try:
                            parsed_candidate = json.loads(potential_json_candidate)
                            if isinstance(parsed_candidate, list):
                                json_sketch_str = json.dumps(parsed_candidate)
                        except json.JSONDecodeError: pass # Ignore if not valid list
                    if not json_sketch_str and response_str.strip().lower() == "null":
                         json_sketch_str = "null"


                if not json_sketch_str or json_sketch_str.lower() == "null":
                    logger.info(f"HTN_LLM_SKETCH_PROC: LLM sketch response contained no valid plan JSON (null or not found).")
                else:
                    try:
                        llm_plan_list_of_lists = json.loads(json_sketch_str)
                        if not isinstance(llm_plan_list_of_lists, list):
                             logger.error(f"HTN_LLM_SKETCH_PROC: Parsed JSON sketch is not a list. Type: {type(llm_plan_list_of_lists)}")
                        else:
                            validated_ops: PlanResultType = []
                            valid_sketch = True
                            for i, op_entry in enumerate(llm_plan_list_of_lists):
                                if not isinstance(op_entry, list) or not op_entry: valid_sketch = False; break
                                op_name_llm = op_entry[0] # First element is name
                                op_args_llm_raw = op_entry[1:] # Rest are args

                                if not isinstance(op_name_llm, str) or op_name_llm not in self.operators: valid_sketch = False; break
                                op_def = self.operators[op_name_llm]
                                expected_param_count = len(op_def.parameters)
                                
                                op_args_llm_str = [str(arg) for arg in op_args_llm_raw]
                                
                                if len(op_args_llm_str) != expected_param_count: 
                                    # Allow empty list of args if operator expects 0 and LLM gives [""]
                                    if not (expected_param_count == 0 and len(op_args_llm_str) == 1 and op_args_llm_str[0] == ""):
                                        valid_sketch = False; break
                                    else: # Correcting empty arg list if LLM gave [""] for no-param op
                                        op_args_llm_str = []


                                bindings_dict = {p_name: arg_val for p_name, arg_val in zip(op_def.parameters, op_args_llm_str)}
                                validated_ops.append((op_name_llm, bindings_dict))

                            if valid_sketch and validated_ops:
                                llm_plan_sketch_ops = validated_ops
                                logger.info(f"HTN_LLM_SKETCH_PROC: Successfully parsed and validated LLM plan sketch with {len(llm_plan_sketch_ops)} ops.")
                            elif not validated_ops and valid_sketch: # Empty valid list
                                 logger.info("HTN_LLM_SKETCH_PROC: LLM returned an empty plan list [].")
                            else: # Invalid sketch
                                 logger.error(f"HTN_LLM_SKETCH_PROC: LLM plan sketch failed validation. Sketch content: {str(llm_plan_list_of_lists)[:200]}")


                    except json.JSONDecodeError as e:
                        logger.error(f"HTN_LLM_SKETCH_PROC: Error decoding LLM JSON sketch: {e}. Sketch string: '{json_sketch_str}'")

        except asyncio.CancelledError:
            logger.warning("HTN_LLM_SKETCH_PROC: LLM sketch task was cancelled.")
        except Exception as e_res:
            logger.error(f"HTN_LLM_SKETCH_PROC: Exception retrieving result for LLM sketch task: {e_res}", exc_info=True)

        # If a sketch was obtained, try to abstract it into a method
        if llm_plan_sketch_ops:
            goal_ctx = task_info["goal"]
            task_to_solve_ctx = task_info["task_to_solve"]
            initial_state_ctx = task_info["initial_state"]
            task_name_str_abstract = self._get_task_name_from_tasktype(task_to_solve_ctx)

            new_method = self._abstract_plan_to_method(task_to_solve_ctx, llm_plan_sketch_ops, initial_state_ctx)
            if new_method:
                self.methods.setdefault(task_name_str_abstract, []).append(new_method)
                logger.info(f"HTN_LLM_SKETCH_PROC: Learned new method '{new_method.name}' via LLM sketch for task '{task_name_str_abstract}'.")
                await self._persist_learned_methods() 
            return llm_plan_sketch_ops 
        
        return None 
        
    def _abstract_plan_to_method(self, 
                                 task: TaskType, 
                                 plan_operators_with_bindings: PlanResultType, 
                                 initial_state: Set['Predicate'] # type: ignore
                                ) -> Optional['Method']: # type: ignore
        _Method_abstract = globals().get('Method')
        _Predicate_abstract = globals().get('Predicate')
        if not _Method_abstract or not _Predicate_abstract:
            logger.error("HTNPlanner._abstract_plan_to_method: Method or Predicate class not available.")
            return None
        if not plan_operators_with_bindings: 
            logger.warning("HTNPlanner._abstract_plan_to_method: Received empty plan_operators_with_bindings. Cannot abstract.")
            return None

        task_name_str = self._get_task_name_from_tasktype(task)
        task_args_values = task[1:] if isinstance(task, tuple) else ()
        method_param_names = [f"?param{i}" for i in range(len(task_args_values))]
        method_task_signature = (task_name_str,) + tuple(method_param_names)
        initial_method_bindings: Dict[str, Any] = {
            var_name[1:]: val for var_name, val in zip(method_param_names, task_args_values)
        }
        method_subtasks: List[TaskType] = []
        
        for op_idx, (op_name, op_concrete_bindings_from_llm) in enumerate(plan_operators_with_bindings):
            op_def = self.operators.get(op_name)
            if not op_def:
                logger.error(f"Method Abstraction: Unknown operator '{op_name}' found in plan sketch. Cannot abstract.")
                return None
            subtask_args_template: List[Any] = []
            for formal_op_param_name in op_def.parameters:
                concrete_val_from_llm = op_concrete_bindings_from_llm.get(formal_op_param_name)
                if concrete_val_from_llm is None:
                    logger.error(f"Method Abstraction: Operator '{op_name}' in sketch missing required parameter '{formal_op_param_name}'. Cannot abstract.")
                    return None
                matched_method_param_placeholder = None
                for method_p_name_stripped, original_task_arg_concrete_val in initial_method_bindings.items():
                    if original_task_arg_concrete_val == concrete_val_from_llm:
                        matched_method_param_placeholder = f"?{method_p_name_stripped}"
                        break
                if matched_method_param_placeholder:
                    subtask_args_template.append(matched_method_param_placeholder)
                else:
                    subtask_args_template.append(concrete_val_from_llm)
            method_subtasks.append((op_name,) + tuple(subtask_args_template))

        method_preconditions: Set[Predicate] = set() # type: ignore
        if plan_operators_with_bindings:
            first_op_name, first_op_concrete_bindings = plan_operators_with_bindings[0]
            first_op_def = self.operators.get(first_op_name)

            if first_op_def:
                logger.debug(f"Method Abstraction: Generating preconditions from first operator '{first_op_name}' which has templates: {[(p.name, p.args) for p in first_op_def.preconditions]}")
                for precond_template in first_op_def.preconditions:
                    bound_first_op_precond = first_op_def._bind_predicate(precond_template, first_op_concrete_bindings)
                    
                    if bound_first_op_precond and hasattr(bound_first_op_precond, 'args'): 
                        generalized_args_list = []
                        is_relevant_to_method_params = False
                        
                        for arg_val_in_concrete_precond in bound_first_op_precond.args: # type: ignore
                            matched_method_p_name = None
                            for method_p_placeholder_name_stripped, original_task_arg_concrete_val in initial_method_bindings.items():
                                if arg_val_in_concrete_precond == original_task_arg_concrete_val:
                                    matched_method_p_name = f"?{method_p_placeholder_name_stripped}"
                                    break
                            
                            if matched_method_p_name:
                                generalized_args_list.append(matched_method_p_name)
                                is_relevant_to_method_params = True
                            else:
                                generalized_args_list.append(arg_val_in_concrete_precond)
                        
                        template_has_variables = any(isinstance(a, str) and a.startswith('?') for a in precond_template.args) # type: ignore

                        if is_relevant_to_method_params or not method_param_names or not template_has_variables:
                            final_generalized_precond = _Predicate_abstract( # type: ignore
                                name=bound_first_op_precond.name, # type: ignore
                                args=tuple(generalized_args_list),
                                value=bound_first_op_precond.value, # type: ignore
                                timestamp=0 
                            )
                            method_preconditions.add(final_generalized_precond)
                            logger.debug(f"Method Abstraction: Added generalized precondition: {final_generalized_precond.name}{final_generalized_precond.args}={final_generalized_precond.value}")
                        else:
                            logger.debug(f"Method Abstraction: Skipped precondition {bound_first_op_precond.name}{bound_first_op_precond.args} as not directly relevant to method params and method has params.")
            else:
                logger.warning(f"Method Abstraction: First operator '{first_op_name}' from sketch not found in self.operators.")
        
        if task_name_str == "task_read_file" and method_param_names:
            path_param_for_precond = method_param_names[0] 
            default_read_precond = _Predicate_abstract( 
                name="isFile", args=(path_param_for_precond,), value=True, timestamp=0
            ) # type: ignore
            method_preconditions.add(default_read_precond)
            logger.debug(f"Method Abstraction: Added default 'isFile({path_param_for_precond})' precondition for task_read_file.")

        if not method_subtasks: 
            logger.warning("Method Abstraction: No subtasks generated for the new method. Aborting method creation.")
            return None

        new_method_name = f"learned_{task_name_str}_{int(time.time())}"
        new_method_instance = _Method_abstract( # type: ignore
            name=new_method_name,
            task_signature=method_task_signature,
            preconditions=method_preconditions,
            subtasks=method_subtasks
        )
        
        # Initialize all required metadata fields
        new_method_instance.metadata = { # type: ignore
            "confidence": 0.7, 
            "success_rate": 0.0, 
            "usage_count": 0,    
            "total_recorded_successes": 0, 
            "learned_via": "llm_sketch",
            "creation_timestamp": time.time(),
            "last_used_ts": None, 
            "last_successful_use_ts": None 
        }
        
        logger.info(f"Method Abstraction: Successfully created new method '{new_method_instance.name}' "
                    f"for task signature {method_task_signature} with {len(method_preconditions)} preconditions "
                    f"and {len(method_subtasks)} subtasks.")
        logger.debug(f"  New method preconditions: {[(p.name, p.args, p.value) for p in new_method_instance.preconditions]}") # type: ignore
        logger.debug(f"  New method subtasks: {new_method_instance.subtasks}") # type: ignore

        return new_method_instance
        
    async def _persist_learned_methods(self) -> None:
        """
        Persists the current HTN library (operators and methods, especially learned ones) 
        to a JSON file. (MDP C.3.1)
        """
        if not self._learned_methods_path:
            logger.debug("HTNPlanner: Learned methods path not set. Skipping persistence.")
            return

        logger.info(f"HTNPlanner: Persisting HTN library to {self._learned_methods_path}...")
        
        serializable_operators = {}
        for op_name, op_obj in self.operators.items():
            op_dict = {
                "name": op_obj.name,
                "parameters": op_obj.parameters,
                "preconditions": [self._predicate_to_dict(p) for p in op_obj.preconditions if self._predicate_to_dict(p)],
                "effects": [self._predicate_to_dict(p) for p in op_obj.effects if self._predicate_to_dict(p)],
                "estimated_cost": op_obj.estimated_cost
            }
            serializable_operators[op_name] = op_dict

        serializable_methods = {}
        for task_name, method_list in self.methods.items():
            serializable_method_list = []
            for method_obj in method_list:
                method_dict = {
                    "name": method_obj.name,
                    "task_signature": list(method_obj.task_signature), 
                    "preconditions": [self._predicate_to_dict(p) for p in method_obj.preconditions if self._predicate_to_dict(p)],
                    "subtasks": method_obj.subtasks, 
                    "heuristic_score": method_obj.heuristic_score, 
                    "metadata": getattr(method_obj, 'metadata', {}) 
                }
                serializable_method_list.append(method_dict)
            serializable_methods[task_name] = serializable_method_list
        
        library_data = {
            "operators": serializable_operators,
            "methods": serializable_methods,
            "timestamp": time.time()
        }

        try:
            self._learned_methods_path.parent.mkdir(parents=True, exist_ok=True)
            temp_file_path = self._learned_methods_path.with_suffix(".tmp")
            with open(temp_file_path, 'w', encoding='utf-8') as f:
                json.dump(library_data, f, indent=2)
            temp_file_path.replace(self._learned_methods_path)
            logger.info(f"HTNPlanner: Library persisted successfully to {self._learned_methods_path}.")
        except Exception as e:
            logger.error(f"HTNPlanner: Failed to persist library: {e}", exc_info=True)
            if 'temp_file_path' in locals() and temp_file_path.exists(): # type: ignore
                try: temp_file_path.unlink() # type: ignore
                except OSError: pass

    async def _load_learned_methods(self) -> None:
        """
        Loads HTN library (operators and methods) from the JSON file, merging with
        or replacing existing hardcoded ones. (MDP C.3.1)
        """
        if not self._learned_methods_path or not self._learned_methods_path.exists():
            logger.debug("HTNPlanner: Learned methods file not found or path not set. Using only hardcoded library.")
            return
        
        logger.info(f"HTNPlanner: Loading HTN library from {self._learned_methods_path}...")
        _Operator_load = globals().get('Operator')
        _Method_load = globals().get('Method')

        if not _Operator_load or not _Method_load:
            logger.error("HTNPlanner: Operator or Method class not available for loading library. Aborting load.")
            return

        try:
            with open(self._learned_methods_path, 'r', encoding='utf-8') as f:
                library_data = json.load(f)

            loaded_ops_count = 0
            if "operators" in library_data and isinstance(library_data["operators"], dict):
                for op_name, op_dict in library_data["operators"].items():
                    try:
                        preconds = {self._dict_to_predicate(p_d) for p_d in op_dict.get("preconditions", []) if self._dict_to_predicate(p_d)}
                        effects = {self._dict_to_predicate(p_d) for p_d in op_dict.get("effects", []) if self._dict_to_predicate(p_d)}
                        
                        new_op = _Operator_load( # type: ignore
                            name=op_dict["name"],
                            parameters=op_dict.get("parameters", []),
                            preconditions=preconds, # type: ignore
                            effects=effects, # type: ignore
                            estimated_cost=op_dict.get("estimated_cost", 1.0)
                        )
                        self.operators[op_name] = new_op # Overwrites if name clashes
                        loaded_ops_count +=1
                    except Exception as e_op_load:
                        logger.error(f"Error loading operator '{op_name}' from library: {e_op_load}. Data: {op_dict}")
            
            loaded_methods_count = 0
            if "methods" in library_data and isinstance(library_data["methods"], dict):
                for task_name, method_list_data in library_data["methods"].items():
                    if not isinstance(method_list_data, list): continue
                    
                    current_methods_for_task = self.methods.setdefault(task_name, [])
                    
                    for method_dict in method_list_data:
                        try:
                            method_name_to_load = method_dict.get("name")
                            if any(m.name == method_name_to_load for m in current_methods_for_task):
                                logger.debug(f"Method '{method_name_to_load}' for task '{task_name}' already exists. Skipping duplicate from file.")
                                continue

                            preconds_m = {self._dict_to_predicate(p_d) for p_d in method_dict.get("preconditions", []) if self._dict_to_predicate(p_d)}
                            task_sig_list = method_dict.get("task_signature", [])
                            task_sig_tuple = tuple(task_sig_list) if isinstance(task_sig_list, list) else ()

                            loaded_subtasks_raw = method_dict.get("subtasks", [])
                            reconstructed_subtasks: List[TaskType] = []
                            if isinstance(loaded_subtasks_raw, list):
                                for sub_entry in loaded_subtasks_raw:
                                    if isinstance(sub_entry, list): 
                                        reconstructed_subtasks.append(tuple(sub_entry))
                                    elif isinstance(sub_entry, str): 
                                        reconstructed_subtasks.append(sub_entry)
                                    else:
                                        logger.warning(f"Unsupported subtask entry type during load: {type(sub_entry)} in method '{method_dict.get('name')}'. Skipping subtask.")
                            
                            new_m = _Method_load( # type: ignore
                                name=method_name_to_load,
                                task_signature=task_sig_tuple, # type: ignore
                                preconditions=preconds_m, # type: ignore
                                subtasks=reconstructed_subtasks, 
                                heuristic_score=method_dict.get("heuristic_score"), 
                            )
                            if "metadata" in method_dict and isinstance(method_dict["metadata"], dict):
                                new_m.metadata = method_dict["metadata"] # type: ignore
                            else: 
                                new_m.metadata = {} # type: ignore
                            
                            if hasattr(new_m, '__post_init__') and callable(getattr(new_m, '__post_init__')):
                                new_m.__post_init__()


                            current_methods_for_task.append(new_m) 
                            loaded_methods_count += 1
                        except Exception as e_m_load:
                             logger.error(f"Error loading method '{method_dict.get('name')}' for task '{task_name}': {e_m_load}. Data: {method_dict}")
            
            logger.info(f"HTNPlanner: Loaded {loaded_ops_count} operators and {loaded_methods_count} methods from {self._learned_methods_path}.")

        except json.JSONDecodeError as e:
            logger.error(f"HTNPlanner: Failed to decode JSON from library file {self._learned_methods_path}: {e}")
        except Exception as e:
            logger.error(f"HTNPlanner: Error loading HTN library: {e}", exc_info=True)


    async def update_method_performance_stats(self, 
                                            task_name: str, 
                                            method_name: str, 
                                            plan_succeeded: bool) -> None:
        """
        Updates the success_rate and confidence of a specific method based on plan outcome.
        Called by AgentController. (MDP C.3.3)
        """
        if task_name in self.methods:
            method_found_for_update = False
            for method_obj in self.methods[task_name]:
                if method_obj.name == method_name:
                    method_found_for_update = True
                    if hasattr(method_obj, 'metadata') and isinstance(method_obj.metadata, dict):
                        
                        usage_count_current_attempt_included = method_obj.metadata.get("usage_count", 0)
                        if usage_count_current_attempt_included == 0:
                            logger.warning(f"update_method_performance_stats for '{method_name}': usage_count is 0. Assuming this is the first use and it was pre-incremented.")
                        
                        total_recorded_successes = method_obj.metadata.setdefault("total_recorded_successes", 0)
                        
                        if plan_succeeded:
                            total_recorded_successes += 1
                            method_obj.metadata["last_successful_use_ts"] = time.time()
                        
                        method_obj.metadata["total_recorded_successes"] = total_recorded_successes
                        
                        new_success_rate = 0.0
                        if usage_count_current_attempt_included > 0: 
                            new_success_rate = total_recorded_successes / usage_count_current_attempt_included
                        
                        method_obj.metadata["success_rate"] = round(new_success_rate, 3)

                        if method_obj.metadata.get("learned_via") == "llm_sketch":
                            base_confidence = method_obj.metadata.setdefault("confidence", 0.7)
                            new_confidence_val = base_confidence * 0.7 + new_success_rate * 0.3
                            if usage_count_current_attempt_included < 5:
                                if plan_succeeded and new_success_rate > 0:
                                    new_confidence_val = max(new_confidence_val, 0.65 + (0.1 * total_recorded_successes))
                                elif not plan_succeeded:
                                    new_confidence_val *= 0.8 
                            method_obj.metadata["confidence"] = round(max(0.1, min(1.0, new_confidence_val)), 3)
                        
                        logger.info(
                            f"Method '{method_name}' stats updated. Outcome: {'SUCCESS' if plan_succeeded else 'FAILURE'}. "
                            f"New SuccessRate: {method_obj.metadata['success_rate']:.2f} ({total_recorded_successes}/{usage_count_current_attempt_included}), " 
                            f"New Confidence: {method_obj.metadata.get('confidence', 'N/A')}, "
                            f"Usage: {usage_count_current_attempt_included}"
                        )
                        
                        if method_obj.metadata.get("learned_via") == "llm_sketch":
                            await self._persist_learned_methods()
                        break 
            if not method_found_for_update:
                logger.warning(f"update_method_performance_stats: Method '{method_name}' not found for task '{task_name}'.")
        else:
            logger.warning(f"update_method_performance_stats: Task '{task_name}' not found in methods.")

    async def _prune_learned_methods(self, current_timestamp: Optional[float] = None) -> None:
        """
        Periodically reviews and prunes learned methods from the library.
        (MDP C.3.3)
        """
        if current_timestamp is None:
            current_timestamp = time.time()
            
        logger.info("HTNPlanner: Starting pruning process for learned methods...")
        pruned_count = 0
        
        pruning_config = self._config.get("htn_planner_pruning", {}) 
        min_usage_for_pruning = int(pruning_config.get("min_usage_for_pruning", 10))
        low_success_rate_threshold = float(pruning_config.get("low_success_rate_threshold", 0.2))
        low_confidence_threshold = float(pruning_config.get("low_confidence_threshold", 0.3))
        max_age_unused_days = int(pruning_config.get("max_age_unused_days", 30))
        max_age_unused_seconds = max_age_unused_days * 24 * 60 * 60

        for task_name in list(self.methods.keys()):
            methods_to_keep_for_task = []
            changed_for_this_task = False
            for method_obj in self.methods[task_name]:
                if not (hasattr(method_obj, 'metadata') and isinstance(method_obj.metadata, dict)):
                    methods_to_keep_for_task.append(method_obj)
                    continue
                
                if method_obj.metadata.get("learned_via") != "llm_sketch":
                    methods_to_keep_for_task.append(method_obj) 
                    continue

                usage = method_obj.metadata.get("usage_count", 0)
                success_rate = method_obj.metadata.get("success_rate", 0.0)
                confidence = method_obj.metadata.get("confidence", 0.7)
                last_used = method_obj.metadata.get("last_used_ts") 

                should_prune = False
                prune_reason = ""

                if usage >= min_usage_for_pruning:
                    if success_rate < low_success_rate_threshold and confidence < low_confidence_threshold:
                        should_prune = True
                        prune_reason = f"low success_rate ({success_rate:.2f}) and low confidence ({confidence:.2f}) after {usage} uses."
                
                if not should_prune and last_used and isinstance(last_used, (int, float)) and \
                   (current_timestamp - last_used > max_age_unused_seconds):
                    if confidence < 0.8: 
                        should_prune = True
                        prune_reason = f"unused for over {max_age_unused_days} days (confidence: {confidence:.2f}, last_used: {last_used})."
                
                if should_prune:
                    logger.info(f"Pruning learned method '{method_obj.name}' for task '{task_name}'. Reason: {prune_reason}")
                    pruned_count += 1
                    changed_for_this_task = True
                else:
                    methods_to_keep_for_task.append(method_obj)
            
            if changed_for_this_task: 
                if methods_to_keep_for_task:
                    self.methods[task_name] = methods_to_keep_for_task
                else: 
                    del self.methods[task_name]
                    logger.info(f"Removed task entry '{task_name}' from methods as all its learned methods were pruned.")


        if pruned_count > 0:
            logger.info(f"HTNPlanner: Pruned {pruned_count} learned methods.")
            await self._persist_learned_methods() 
        else:
            logger.info("HTNPlanner: No learned methods met pruning criteria in this review.")


    # --- CognitiveComponent Methods ---
    async def process(self,*args,**kwargs) -> Optional[Dict[str, Any]]: return None
    async def reset(self): logger.info("HTNPlanner reset. Library remains.")
    async def get_status(self) -> Dict[str, Any]: return { "component": "HTNPlanner", "status": "operational", "max_depth": self.max_depth, "operators_count": len(self.operators), "methods_count": sum(len(m_list) for m_list in self.methods.values()) }
    async def shutdown(self): logger.info("HTNPlanner shutting down.")

# --- END OF cognitive_modules/htn_planner.py (Corrected ImportError Fallback) ---