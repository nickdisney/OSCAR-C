# tests/unit/test_htn_planner.py

import pytest
import asyncio
import time
from typing import Dict, Any, List, Set, Optional, Tuple, Union # Added Optional, Tuple, Union
from unittest.mock import MagicMock, patch, AsyncMock
import pytest_asyncio
import logging
from pathlib import Path
import re # Added re for test_decompose_sorts_by_new_heuristic

# Attempt to import from the project structure
try:
    from consciousness_experiment.cognitive_modules.htn_planner import HTNPlanner, Operator, Method, TaskType, PlanResultType # Key HTN imports
    from consciousness_experiment.models.datatypes import Goal, Predicate # Core datatypes
    from consciousness_experiment.protocols import CognitiveComponent, Planner # Protocols
    from consciousness_experiment.cognitive_modules.cognitive_cache import CognitiveCache # If used by HTNPlanner
    from consciousness_experiment.models.enums import ConsciousState # If used by HTNPlanner
    
    # Import call_ollama if HTNPlanner directly uses it (it will for sketching)
    from consciousness_experiment.external_comms import call_ollama
    from consciousness_experiment.cognitive_modules.predictive_world_model import PredictiveWorldModel 


    HTN_MODELS_AVAILABLE = True
    logger_test_htn = logging.getLogger(__name__) # Logger for this test file
except ImportError as e:
    HTN_MODELS_AVAILABLE = False
    logging.basicConfig(level=logging.INFO) # Basic logging if imports fail early
    logger_test_htn = logging.getLogger(__name__)
    logger_test_htn.error(f"Critical imports for test_htn_planner.py failed: {e}. Tests will likely be skipped or fail.")
    
    # Minimal fallbacks for the test structure to be parseable if imports fail
    class CognitiveComponent: pass #type: ignore
    class Planner(CognitiveComponent): pass # type: ignore
    class HTNPlanner(Planner): # Basic mock for structure # type: ignore
        async def initialize(self, config, controller): 
            self.operators = {}
            self.methods = {}
            self._controller = controller # Store controller for LLM calls
            self._config = config.get("htn_planner", {})
            self._planning_session_pwm_cache = {} # Ensure this exists in fallback
            self._pending_llm_sketches = {} # Add for fallback
            return True
        def _get_task_name_from_tasktype(self, task): return "dummy_task"
        async def _llm_assisted_plan_sketch(self,g,t,s,c): return "sketching_failed_to_schedule" # Updated fallback
        def _abstract_plan_to_method(self,t,p,s): return None
        async def _persist_learned_methods(self): pass
        async def _estimate_method_properties(self, m, b, s, full_cognitive_context_for_pwm): return 1.0, 0.5, 1 # duration, prob, count
        async def _process_completed_llm_sketch(self, cache_key): return None # Add for fallback


    class Operator: 
        def __init__(self, name, parameters=None, preconditions=None, effects=None, estimated_cost=1.0):
            self.name = name; self.parameters = parameters or []; 
            self.preconditions = preconditions or set(); self.effects = effects or set()
            self.estimated_cost = estimated_cost
        def _bind_predicate(self, p, b): return p # Dummy
        def is_applicable(self, s, b): return True # Dummy
        def apply(self, s, b): return s # Dummy

    class Method: 
        def __init__(self, name, task_signature=None, preconditions=None, subtasks=None, heuristic_score=None, metadata=None):
            self.name=name; self.task_signature=task_signature or ();
            self.preconditions = preconditions or set(); self.subtasks = subtasks or []
            self.metadata = metadata or {}; self.heuristic_score = heuristic_score # Ensure heuristic_score is set
        def _bind_predicate(self, p, b): return p
        def is_applicable(self,s,b): return True
        def get_parameter_bindings(self,t): return {}
        def bind_subtask(self,st,b): return st


    class Goal: 
        def __init__(self, description, priority=1.0, id=None, success_criteria=None):
            self.description = description; self.priority=priority; self.id=id or "dummy_goal_id"
            self.success_criteria = success_criteria or set()

    class Predicate:
        def __init__(self, name, args, value=True, timestamp=None):
            self.name=name; self.args=args; self.value=value; self.timestamp=timestamp or time.time()
        def __hash__(self): return hash((self.name, self.args, self.value))
        def __eq__(self, other): 
            return isinstance(other, Predicate) and self.name == other.name and self.args == other.args and self.value == other.value

    class ValueCategory(str): pass #type: ignore
    class ValueJudgment: pass
    class CognitiveCache: pass #type: ignore
    
    class MockEnum: # Simple Mock for Enum if ConsciousState fails to import properly
        def __init__(self, name, value):
            self.name = name
            self.value = value
        def __eq__(self, other):
            if isinstance(other, MockEnum):
                return self.name == other.name and self.value == other.value
            return False
        def __hash__(self):
            return hash((self.name, self.value))
            
    class ConsciousState(MockEnum): # type: ignore
        PRE_CONSCIOUS = MockEnum("PRE_CONSCIOUS", 1)
        CONSCIOUS = MockEnum("CONSCIOUS", 2)
        META_CONSCIOUS = MockEnum("META_CONSCIOUS", 3)
        UNCONSCIOUS = MockEnum("UNCONSCIOUS", 0)
    
    class PredictiveWorldModel(CognitiveComponent): pass 
    globals()['PredictiveWorldModel'] = PredictiveWorldModel 

    TaskType = Union[str, Tuple[str, ...]]
    PlanResultType = List[Tuple[str, Dict[str, Any]]]
    
    async def call_ollama(*args, **kwargs): return None, "call_ollama_dummy_fallback"
    globals()['call_ollama'] = call_ollama

# ... (after existing imports)
# --- NEW IMPORTS FOR ASYNC LLM SKETCH TESTS ---
try:
    from .htn_test_mocks import ( # Relative import for test files
        MockAgentControllerForHTN,
        MOCK_LLM_SKETCH_SUCCESS_READ_FILE,
        MOCK_LLM_SKETCH_MULTI_STEP_WRITE_THEN_READ,
        MOCK_LLM_SKETCH_FAILURE_NULL_RESPONSE,
        MOCK_LLM_SKETCH_FAILURE_INVALID_JSON,
        MOCK_LLM_SKETCH_FAILURE_UNKNOWN_OPERATOR,
        MOCK_LLM_SKETCH_FAILURE_PARAM_MISMATCH,
        MOCK_LLM_SKETCH_API_ERROR,
        MOCK_LLM_SKETCH_SUCCESS_NO_PARAMS_OP,
        MOCK_LLM_SKETCH_SUCCESS_QUERY_KB_CORRECT_PARAMS,
        Goal as MockGoal, # Rename to avoid conflict if Real Goal is also imported
        Predicate as MockPredicate # Rename
    )
    HTN_MOCKS_AVAILABLE = True
except ImportError as e_mock:
    HTN_MOCKS_AVAILABLE = False
    logger_test_htn.error(f"Could not import HTN test mocks: {e_mock}. LLM sketch tests may be skipped.")
# --- END NEW IMPORTS ---


# --- Fixture for HTNPlanner instance with some operators ---
@pytest_asyncio.fixture
async def htn_planner_instance_with_ops():
    if not HTN_MODELS_AVAILABLE:
        pytest.skip("Skipping HTNPlanner tests as core models/components could not be imported.")

    mock_controller = MagicMock()
    mock_controller.model_name = "test_llm_for_htn"
    mock_controller._asyncio_loop = asyncio.get_running_loop()
    mock_controller.config = { 
        "htn_planner": {
            "max_planning_depth": 5, 
            "llm_sketch_timeout_s": 10.0, 
            "llm_sketch_temperature": 0.1
        },
        "performance": { 
             "max_planning_depth": 5 
        },
        "cognitive_cache": {"default_ttl": 1.0}, 
        "agent_data_paths": {"htn_library_path": "dummy_htn_lib.json"} 
    }
    
    mock_cache = MagicMock(spec=CognitiveCache)
    async def mock_cache_get(key): return None
    async def mock_cache_put(key, value, ttl_override=None): pass
    mock_cache.get = mock_cache_get
    mock_cache.put = mock_cache_put
    mock_controller.cache = mock_cache
    
    planner = HTNPlanner()
    await planner.initialize(mock_controller.config, mock_controller) 

    _Predicate_fixture = globals().get('Predicate', Predicate) 
    
    planner.operators["READ_FILE"] = Operator(
        name="READ_FILE", parameters=["path"],
        preconditions={_Predicate_fixture("isFile", ("?path",), True)},
        effects={_Predicate_fixture("readFileContent", ("?path",), True)},
        estimated_cost=1.0
    )
    planner.operators["THINKING"] = Operator(
        name="THINKING", parameters=["content"],
        effects={_Predicate_fixture("thoughtAbout", ("?content",), True)},
        estimated_cost=0.1
    )
    planner.operators["WRITE_FILE"] = Operator(
        name="WRITE_FILE", parameters=["path", "content_var"],
        effects={_Predicate_fixture("fileWritten", ("?path",), True)},
        estimated_cost=1.0
    )
    return planner

@pytest_asyncio.fixture
async def htn_planner_for_llm_sketch_tests(tmp_path: Path):
    """
    Provides an HTNPlanner instance configured with MockAgentControllerForHTN
    and an asyncio.Queue for injecting mock LLM responses.
    Includes basic operators for sketch validation.
    """
    if not HTN_MODELS_AVAILABLE or not HTN_MOCKS_AVAILABLE:
        pytest.skip("Skipping HTN LLM sketch tests: core/mock models not available.")

    mock_llm_response_queue = asyncio.Queue()
    mock_controller = MockAgentControllerForHTN(mock_llm_response_queue)
    
    # Configure paths for learned method persistence
    test_lib_path_llm = tmp_path / "htn_lib_llm_sketch_test.json"
    mock_controller.agent_root_path = tmp_path # So relative path works
    mock_controller.config["agent_data_paths"] = {"htn_library_path": str(test_lib_path_llm.name)}


    planner = HTNPlanner()
    # Ensure the mock_controller has a cache attribute, even if basic
    if not hasattr(mock_controller, 'cache'):
        mock_controller.cache = MagicMock(spec=CognitiveCache) # type: ignore
        async def mock_cache_get_llm(key): return None
        async def mock_cache_put_llm(key, value, ttl_override=None): pass
        mock_controller.cache.get = mock_cache_get_llm # type: ignore
        mock_controller.cache.put = mock_cache_put_llm # type: ignore
        
    await planner.initialize(mock_controller.config, mock_controller)

    # Define some operators that the LLM sketches might use
    # Use MockPredicate from htn_test_mocks to avoid conflicts if real one is complex
    _PredicateSketchTest = globals().get('MockPredicate', Predicate) 

    planner.operators["READ_FILE"] = Operator(
        name="READ_FILE", parameters=["path"],
        preconditions={_PredicateSketchTest("isFile", ("?path",), True)},
        effects={_PredicateSketchTest("readFileContent", ("?path",), True)},
        estimated_cost=1.0
    )
    planner.operators["WRITE_FILE"] = Operator(
        name="WRITE_FILE", parameters=["path", "content_var"],
        effects={_PredicateSketchTest("fileWritten", ("?path",), True)},
        estimated_cost=1.0
    )
    planner.operators["THINKING"] = Operator(
        name="THINKING", parameters=["content"],
        effects={_PredicateSketchTest("thoughtAbout", ("?content",), True)},
        estimated_cost=0.1
    )
    planner.operators["OBSERVE_SYSTEM"] = Operator( # For no-param op test
        name="OBSERVE_SYSTEM", parameters=[],
        effects={_PredicateSketchTest("systemObserved", (), True)},
        estimated_cost=0.2
    )
    planner.operators["QUERY_KB"] = Operator( # For QUERY_KB param test
        name="QUERY_KB", parameters=["name", "args", "value"],
        effects={_PredicateSketchTest("kbQueried", ("?name", "?args"), True)},
        estimated_cost=0.3
    )


    # Return planner and the queue for tests to inject responses
    return planner, mock_llm_response_queue

@pytest.mark.asyncio
async def test_abstract_plan_to_method_basic(htn_planner_instance_with_ops: HTNPlanner):
    _Predicate_test = globals().get('Predicate', Predicate)
    _Method_test = globals().get('Method', Method) 
    _TaskType_test = globals().get('TaskType', TaskType) 
    _PlanResultType_test = globals().get('PlanResultType', PlanResultType) 

    planner = htn_planner_instance_with_ops
    
    task_to_learn: _TaskType_test = ("task_process_file_content", "data.txt", "analysis_prefix") # type: ignore
    
    plan_sketch: _PlanResultType_test = [ # type: ignore
        ("READ_FILE", {"path": "data.txt"}), 
        ("THINKING", {"content": "analysis_prefix: Content of data.txt was interesting."}) 
    ]
    
    initial_state: Set[_Predicate_test] = { # type: ignore
        _Predicate_test("isFile", ("data.txt",), True) # type: ignore
    }

    new_method = planner._abstract_plan_to_method(task_to_learn, plan_sketch, initial_state)

    assert new_method is not None
    if HTN_MODELS_AVAILABLE: assert isinstance(new_method, Method) 

    assert new_method.name.startswith("learned_task_process_file_content_")
    assert new_method.task_signature == ("task_process_file_content", "?param0", "?param1")
    
    assert len(new_method.subtasks) == 2
    assert new_method.subtasks[0] == ("READ_FILE", "?param0")
    assert new_method.subtasks[1] == ("THINKING", "analysis_prefix: Content of data.txt was interesting.")

    assert len(new_method.preconditions) == 1
    precond = list(new_method.preconditions)[0]
    assert getattr(precond, 'name', None) == "isFile"
    assert getattr(precond, 'args', None) == ("?param0",)
    assert getattr(precond, 'value', None) is True

    assert "learned_via" in new_method.metadata
    assert new_method.metadata["learned_via"] == "llm_sketch"
    assert new_method.metadata["confidence"] == 0.7


@pytest.mark.asyncio
async def test_persist_and_load_learned_methods(htn_planner_instance_with_ops: HTNPlanner, tmp_path: Path):
    planner = htn_planner_instance_with_ops
    
    _Method_persist = globals().get('Method', Method)
    _Predicate_persist = globals().get('Predicate', Predicate)

    learned_method = _Method_persist( # type: ignore
        name="learned_test_method_123",
        task_signature=("task_dummy_learn", "?obj"),
        preconditions={_Predicate_persist("isKnown", ("?obj",), True)}, # type: ignore
        subtasks=[("THINKING", "I learned this for ?obj")],
        metadata={"learned_via": "test_case", "confidence": 0.95}
    )
    planner.methods.setdefault("task_dummy_learn", []).append(learned_method)
    
    test_lib_path = tmp_path / "test_htn_lib.json"
    planner._learned_methods_path = test_lib_path

    await planner._persist_learned_methods()
    assert test_lib_path.exists()
    
    mock_controller_new = MagicMock()
    mock_controller_new.config = planner._controller.config # type: ignore
    mock_controller_new.config["agent_data_paths"]["htn_library_path"] = str(test_lib_path.relative_to(tmp_path)) 
    mock_controller_new.agent_root_path = tmp_path 
    mock_cache_new = MagicMock(spec=CognitiveCache)
    async def mock_cache_get_new(key): return None
    async def mock_cache_put_new(key, value, ttl_override=None): pass
    mock_cache_new.get = mock_cache_get_new
    mock_cache_new.put = mock_cache_put_new
    mock_controller_new.cache = mock_cache_new


    new_planner = HTNPlanner()
    await new_planner.initialize(mock_controller_new.config, mock_controller_new) 

    assert "task_dummy_learn" in new_planner.methods
    assert len(new_planner.methods["task_dummy_learn"]) >= 1
    found_loaded_method = False
    for m in new_planner.methods["task_dummy_learn"]:
        if m.name == "learned_test_method_123":
            found_loaded_method = True
            assert m.task_signature == ("task_dummy_learn", "?obj")
            assert len(m.preconditions) == 1
            loaded_precond = list(m.preconditions)[0]
            assert getattr(loaded_precond, 'name', None) == "isKnown" 
            assert getattr(loaded_precond, 'args', None) == ("?obj",)
            assert getattr(m, 'metadata', {}).get("confidence") == 0.95
            break
    assert found_loaded_method, "Learned method was not found after loading."

    assert "THINKING" in new_planner.operators
    assert new_planner.operators["THINKING"].name == "THINKING"
    
@pytest_asyncio.fixture
async def htn_planner_with_mock_pwm(htn_planner_instance_with_ops: HTNPlanner):
    planner = htn_planner_instance_with_ops
    
    if not planner._controller:
        planner._controller = MagicMock()
    if not hasattr(planner._controller, 'config') or not planner._controller.config: # type: ignore
        planner._controller.config = { # type: ignore
             "htn_planner": {"llm_sketch_temperature": 0.2, "llm_sketch_timeout_s": 10.0}, 
             "performance": {"max_planning_depth": 5 }, 
             "cognitive_cache": {"default_ttl": 1.0}, 
             "agent_data_paths": {"htn_library_path": "dummy_htn_lib.json"},
             "predictive_world_model": {} 
        }
    if not hasattr(planner._controller, '_get_active_goal_type'):
        mock_get_active_goal_type = MagicMock(return_value="test_goal_type")
        planner._controller._get_active_goal_type = mock_get_active_goal_type # type: ignore

    mock_pwm_instance = MagicMock(spec=PredictiveWorldModel) 

    async def actual_mock_pwm_process_logic(input_state):
        action_to_predict = input_state.get("predict_request", {}).get("action_to_execute", {})
        action_type_to_predict = action_to_predict.get("type")
        
        logger_test_htn.debug(f"Mock PWM process called for action_type: {action_type_to_predict}, params: {action_to_predict.get('params')}")

        if action_type_to_predict == "OP_HIGH_SUCCESS":
            return {"prediction": {"all_probabilities": {"success": 0.9, "failure": 0.1}, "predicted_outcome": "success", "confidence": 0.9}}
        elif action_type_to_predict == "OP_MED_SUCCESS":
            return {"prediction": {"all_probabilities": {"success": 0.6, "failure": 0.4}, "predicted_outcome": "success", "confidence": 0.6}}
        elif action_type_to_predict == "OP_LOW_SUCCESS":
            return {"prediction": {"all_probabilities": {"success": 0.2, "failure": 0.8}, "predicted_outcome": "failure", "confidence": 0.8}}
        elif action_type_to_predict == "OP_DEFAULT_SUCCESS": 
            return {"prediction": {"all_probabilities": {"success": 0.5, "failure": 0.5}, "predicted_outcome": "success", "confidence": 0.5}}
        else: 
            logger_test_htn.warning(f"Mock PWM: Unhandled action_type '{action_type_to_predict}' in mock_pwm_process. Returning default 0.5 success.")
            return {"prediction": {"all_probabilities": {"success": 0.5, "failure": 0.5}, "predicted_outcome": "success", "confidence": 0.5}}

    mock_pwm_instance.process = AsyncMock(side_effect=actual_mock_pwm_process_logic)
    
    planner._controller.predictive_world_model = mock_pwm_instance # type: ignore
    
    _Predicate_fixture_pwm = globals().get('Predicate', Predicate) 
    
    if "OP_HIGH_SUCCESS" not in planner.operators:
        planner.operators["OP_HIGH_SUCCESS"] = Operator(name="OP_HIGH_SUCCESS", estimated_cost=1.0)
    if "OP_MED_SUCCESS" not in planner.operators:
        planner.operators["OP_MED_SUCCESS"] = Operator(name="OP_MED_SUCCESS", estimated_cost=1.0)
    if "OP_LOW_SUCCESS" not in planner.operators:
        planner.operators["OP_LOW_SUCCESS"] = Operator(name="OP_LOW_SUCCESS", estimated_cost=1.0)
    if "OP_DEFAULT_SUCCESS" not in planner.operators: 
        planner.operators["OP_DEFAULT_SUCCESS"] = Operator(name="OP_DEFAULT_SUCCESS", estimated_cost=2.0)
    
    return planner


@pytest.mark.asyncio
async def test_estimate_method_properties_basic(htn_planner_with_mock_pwm: HTNPlanner):
    planner = htn_planner_with_mock_pwm
    _Method_est_prop = globals().get('Method', Method)
    
    method = _Method_est_prop( # type: ignore
        name="test_method_basic_props",
        task_signature=("complex_task_props",),
        subtasks=["OP_HIGH_SUCCESS", "OP_MED_SUCCESS"]
    )
    
    if hasattr(planner, '_planning_session_pwm_cache'):
        planner._planning_session_pwm_cache.clear()

    minimal_cognitive_context = {
        "consciousness_level": "CONSCIOUS", 
        "current_cs_level_name": "CONSCIOUS",
        "active_goal_type": "test_basic_props_goal",
        "php_levels": {"pain": 0, "happiness": 5, "purpose": 5},
        "drives": {"curiosity": 0.5}
    }

    duration, success_prob, subtask_count = await planner._estimate_method_properties(
        method, 
        bindings={}, 
        current_state_for_heuristic=set(), 
        full_cognitive_context_for_pwm=minimal_cognitive_context 
    )
    
    assert duration == pytest.approx(1.0 + 1.0 + 0.01 * (1 + 0)) 
    assert success_prob == pytest.approx(0.9 * 0.6)
    assert subtask_count == 2
    
    assert planner._controller.predictive_world_model.process.call_count >= 2 # type: ignore
    assert len(planner._planning_session_pwm_cache) == 2


@pytest.mark.asyncio
async def test_estimate_method_properties_with_complex_subtask(htn_planner_with_mock_pwm: HTNPlanner):
    planner = htn_planner_with_mock_pwm
    _Method_est_prop_complex = globals().get('Method', Method) 

    method_with_complex = _Method_est_prop_complex( # type: ignore
        name="method_main_complex_sub",
        task_signature=("task_top_level",), 
        subtasks=["OP_HIGH_SUCCESS", "sub_complex_task_A"] 
    )
    
    if "sub_complex_task_A" not in planner.methods: 
        planner.methods["sub_complex_task_A"] = [
            _Method_est_prop_complex(name="dummy_method_for_sub_A", task_signature=("sub_complex_task_A",), subtasks=["OP_DEFAULT_SUCCESS"]) # type: ignore
        ]
    
    if hasattr(planner, '_planning_session_pwm_cache'):
        planner._planning_session_pwm_cache.clear()

    minimal_cognitive_context_complex = {
        "consciousness_level": "CONSCIOUS",
        "current_cs_level_name": "CONSCIOUS",
        "active_goal_type": "test_complex_props_goal",
        "php_levels": {"pain": 0, "happiness": 5, "purpose": 5},
        "drives": {"curiosity": 0.5}
    }

    duration, success_prob, subtask_count = await planner._estimate_method_properties(
        method_with_complex, 
        bindings={}, 
        current_state_for_heuristic=set(), 
        full_cognitive_context_for_pwm=minimal_cognitive_context_complex 
    )
    
    assert duration == pytest.approx(2.51)
    assert success_prob == pytest.approx(0.9 * 0.75)
    assert subtask_count == 2


@pytest.mark.asyncio
async def test_decompose_sorts_by_new_heuristic(htn_planner_with_mock_pwm: HTNPlanner):
    planner = htn_planner_with_mock_pwm
    _Method_decompose_heuristic = globals().get('Method', Method)
    
    method_A = _Method_decompose_heuristic(name="method_A_low_succ", task_signature=("goal_task",), subtasks=["OP_LOW_SUCCESS"]) # type: ignore
    method_B = _Method_decompose_heuristic(name="method_B_med_succ", task_signature=("goal_task",), subtasks=["OP_MED_SUCCESS"]) # type: ignore
    method_C = _Method_decompose_heuristic(name="method_C_high_succ_long", task_signature=("goal_task",), subtasks=["OP_DEFAULT_SUCCESS", "OP_HIGH_SUCCESS"]) # type: ignore

    planner.methods["goal_task"] = [method_A, method_B, method_C] 
    
    _ConsciousState_test = globals().get('ConsciousState', ConsciousState)
    mock_cs_normal_name = _ConsciousState_test.CONSCIOUS.name if hasattr(_ConsciousState_test.CONSCIOUS, 'name') else str(_ConsciousState_test.CONSCIOUS) # type: ignore

    cognitive_context_normal_cs = {
        "consciousness_level": mock_cs_normal_name, 
        "current_cs_level_name": mock_cs_normal_name, 
        "active_goal_type": "test_heuristic_sort_goal",
        "php_levels": {"pain": 0, "happiness": 5, "purpose": 5},
        "drives": {"curiosity": 0.5}
    }

    if hasattr(planner, '_planning_session_pwm_cache'): planner._planning_session_pwm_cache.clear()

    async def mock_estimate_props_for_sort_test(method_obj, bindings, state_h, full_ctx_pwm):
        if method_obj.name == "method_A_low_succ":
            return 1.0, 0.2, 1  
        elif method_obj.name == "method_B_med_succ":
            return 1.0, 0.6, 1
        elif method_obj.name == "method_C_high_succ_long":
            return 3.0, 0.45, 2 
        return 100.0, 0.1, 1 

    with patch.object(planner, '_estimate_method_properties', side_effect=mock_estimate_props_for_sort_test), \
         patch.object(logging.getLogger("consciousness_experiment.cognitive_modules.htn_planner"), 'debug') as mock_logger_debug:

        await planner._decompose(
            task=("goal_task",), 
            state=set(), 
            depth=0, 
            depth_limit=1, 
            full_cognitive_context_for_pwm=cognitive_context_normal_cs
        )

        found_sorted_log = False
        expected_method_order_in_log = ["method_B_med_succ", "method_A_low_succ", "method_C_high_succ_long"]
        
        for call_args in mock_logger_debug.call_args_list:
            log_message = call_args[0][0] 
            if "Sorted by new heuristic:" in log_message: 
                found_sorted_log = True
                logger_test_htn.info(f"Found sorting log: {log_message}")
                
                method_names_in_log = re.findall(r"\('([^']+)',", log_message)
                assert method_names_in_log == expected_method_order_in_log, \
                    f"Method sort order incorrect. Expected {expected_method_order_in_log}, got {method_names_in_log}"
                break
        assert found_sorted_log, "Log message showing sorted methods by new heuristic was not found."


@pytest.mark.asyncio
async def test_decompose_low_cs_penalty(htn_planner_with_mock_pwm: HTNPlanner):
    planner = htn_planner_with_mock_pwm
    _Method_cs_penalty = globals().get('Method', Method)
    _ConsciousState_cs_penalty = globals().get('ConsciousState', ConsciousState)

    method_simple = _Method_cs_penalty(name="method_simple_cs", task_signature=("cs_penalty_task",), subtasks=["OP_HIGH_SUCCESS"]) # type: ignore
    method_complex = _Method_cs_penalty(name="method_complex_cs", task_signature=("cs_penalty_task",), subtasks=["OP_HIGH_SUCCESS", "OP_MED_SUCCESS", "OP_DEFAULT_SUCCESS"]) # type: ignore
    
    planner.methods["cs_penalty_task"] = [method_simple, method_complex]
    
    mock_cs_low = _ConsciousState_cs_penalty.PRE_CONSCIOUS if _ConsciousState_cs_penalty != Any else "PRE_CONSCIOUS_FALLBACK" # type: ignore
    
    if planner._controller and hasattr(planner._controller, 'config'):
        planner._controller.config.setdefault("htn_planner", {})["low_cs_simplicity_penalty_factor"] = 1.0 # type: ignore
    else: 
        pytest.fail("Controller or config not available in planner for CS penalty test.")

    if hasattr(planner, '_planning_session_pwm_cache'): planner._planning_session_pwm_cache.clear()

    cognitive_context_low_cs = {
        "consciousness_level": mock_cs_low.name if hasattr(mock_cs_low, 'name') else str(mock_cs_low),
        "current_cs_level_name": mock_cs_low.name if hasattr(mock_cs_low, 'name') else str(mock_cs_low),
        "active_goal_type": "test_low_cs_penalty_goal",
        "php_levels": {"pain": 0, "happiness": 5, "purpose": 5},
        "drives": {"curiosity": 0.5}
    }

    with patch.object(logging.getLogger("consciousness_experiment.cognitive_modules.htn_planner"), 'debug') as mock_logger_debug_cs:
        await planner._decompose(
            task=("cs_penalty_task",), 
            state=set(), 
            depth=0, 
            depth_limit=1, 
            full_cognitive_context_for_pwm=cognitive_context_low_cs 
        )

        found_cs_penalty_log_for_simple = False
        found_cs_penalty_log_for_complex = False
        sorted_order_in_cs_log = []

        for call_args in mock_logger_debug_cs.call_args_list:
            log_msg = call_args[0][0]
            if "heuristic adjusted for low CS" in log_msg and "method_simple_cs" in log_msg:
                found_cs_penalty_log_for_simple = True
            if "heuristic adjusted for low CS" in log_msg and "method_complex_cs" in log_msg:
                found_cs_penalty_log_for_complex = True
            if "Sorted by new heuristic:" in log_msg: 
                 sorted_order_in_cs_log = re.findall(r"\('([^']+)',", log_msg)


        assert found_cs_penalty_log_for_simple, "Penalty log for simple method not found."
        assert found_cs_penalty_log_for_complex, "Penalty log for complex method not found."
        
        assert sorted_order_in_cs_log and sorted_order_in_cs_log[0] == "method_simple_cs", \
            f"Expected simple method to be preferred under low CS. Order: {sorted_order_in_cs_log}"
        

@pytest.mark.asyncio
async def test_update_method_performance_stats_success(htn_planner_instance_with_ops: HTNPlanner):
    planner = htn_planner_instance_with_ops
    _Method_stats = globals().get('Method', Method)
    
    task_name = "test_task_for_stats"
    method_name = "learned_method_stats_1"
    
    learned_method = _Method_stats( # type: ignore
        name=method_name,
        task_signature=(task_name, "?arg1"),
        subtasks=[("THINKING", "test")],
        metadata={"learned_via": "llm_sketch", "confidence": 0.7, "usage_count": 0, "success_rate": 0.0}
    )
    planner.methods[task_name] = [learned_method]
    
    learned_method.metadata["usage_count"] = 1 
    learned_method.metadata["last_used_ts"] = time.time() - 10 

    await planner.update_method_performance_stats(task_name, method_name, plan_succeeded=True)
    
    assert learned_method.metadata["success_rate"] == pytest.approx(1.0) 
    assert learned_method.metadata["confidence"] > 0.7 
    assert learned_method.metadata["last_successful_use_ts"] is not None

    learned_method.metadata["usage_count"] = 2
    learned_method.metadata["last_used_ts"] = time.time() - 5
    prev_confidence = learned_method.metadata["confidence"]
    last_successful_ts = learned_method.metadata["last_successful_use_ts"]

    await planner.update_method_performance_stats(task_name, method_name, plan_succeeded=False)
    
    assert learned_method.metadata["success_rate"] == pytest.approx(0.5) 
    assert learned_method.metadata["confidence"] < prev_confidence 
    assert learned_method.metadata["last_successful_use_ts"] == last_successful_ts 


@pytest.mark.asyncio
async def test_update_method_performance_stats_early_failures_confidence(htn_planner_instance_with_ops: HTNPlanner):
    planner = htn_planner_instance_with_ops
    _Method_stats_conf = globals().get('Method', Method)
    
    task_name = "test_task_conf"
    method_name = "learned_method_conf_1"
    method = _Method_stats_conf( # type: ignore
        name=method_name, task_signature=(task_name,), subtasks=[("THINKING", "t")],
        metadata={"learned_via": "llm_sketch", "confidence": 0.7, "usage_count": 0, "success_rate": 0.0}
    )
    planner.methods[task_name] = [method]

    method.metadata["usage_count"] = 1
    await planner.update_method_performance_stats(task_name, method_name, False)
    assert method.metadata["success_rate"] == 0.0
    assert method.metadata["confidence"] == pytest.approx(0.7 * 0.7 * 0.8, abs=1e-3)


@pytest.mark.asyncio
async def test_prune_learned_methods_low_success_and_confidence(htn_planner_instance_with_ops: HTNPlanner):
    planner = htn_planner_instance_with_ops
    _Method_prune = globals().get('Method', Method)
    
    task_name = "task_to_prune"
    method_to_prune = _Method_prune( # type: ignore
        name="method_bad_learned", task_signature=(task_name,), subtasks=[],
        metadata={
            "learned_via": "llm_sketch", "usage_count": 15, 
            "success_rate": 0.1, "confidence": 0.2, "last_used_ts": time.time() - 1000
        }
    )
    method_to_keep = _Method_prune( # type: ignore
        name="method_good_learned", task_signature=(task_name,), subtasks=[],
        metadata={
            "learned_via": "llm_sketch", "usage_count": 15,
            "success_rate": 0.8, "confidence": 0.9, "last_used_ts": time.time() - 1000
        }
    )
    method_hardcoded = _Method_prune(name="method_hardcoded", task_signature=(task_name,), subtasks=[]) # type: ignore
    
    planner.methods[task_name] = [method_to_prune, method_to_keep, method_hardcoded]
    
    planner._config["htn_planner_pruning"] = {
        "min_usage_for_pruning": 10,
        "low_success_rate_threshold": 0.25, 
        "low_confidence_threshold": 0.25,   
        "max_age_unused_days": 30
    }
    
    with patch.object(planner, '_persist_learned_methods', new_callable=AsyncMock) as mock_persist:
        await planner._prune_learned_methods()

    assert method_to_prune.name not in [m.name for m in planner.methods.get(task_name, [])]
    assert method_to_keep.name in [m.name for m in planner.methods.get(task_name, [])]
    assert method_hardcoded.name in [m.name for m in planner.methods.get(task_name, [])]
    assert mock_persist.call_count == 1


@pytest.mark.asyncio
async def test_prune_learned_methods_old_unused(htn_planner_instance_with_ops: HTNPlanner):
    planner = htn_planner_instance_with_ops
    _Method_prune_old = globals().get('Method', Method)

    task_name = "task_old"
    thirty_five_days_ago = time.time() - (35 * 24 * 60 * 60)
    method_old_low_conf = _Method_prune_old( # type: ignore
        name="method_old_low_conf", task_signature=(task_name,), subtasks=[],
        metadata={
            "learned_via": "llm_sketch", "usage_count": 5, 
            "success_rate": 0.6, "confidence": 0.4, "last_used_ts": thirty_five_days_ago
        }
    )
    method_old_high_conf = _Method_prune_old( # type: ignore
        name="method_old_high_conf", task_signature=(task_name,), subtasks=[],
        metadata={
            "learned_via": "llm_sketch", "usage_count": 5,
            "success_rate": 0.9, "confidence": 0.85, "last_used_ts": thirty_five_days_ago
        }
    ) 
    method_recent = _Method_prune_old( # type: ignore
        name="method_recent_low_conf", task_signature=(task_name,), subtasks=[],
        metadata={
            "learned_via": "llm_sketch", "usage_count": 2,
            "success_rate": 0.1, "confidence": 0.3, "last_used_ts": time.time() - 1000 
        }
    )

    planner.methods[task_name] = [method_old_low_conf, method_old_high_conf, method_recent]
    
    planner._config["htn_planner_pruning"] = {
        "min_usage_for_pruning": 10, 
        "low_success_rate_threshold": 0.2,
        "low_confidence_threshold": 0.25,
        "max_age_unused_days": 30 
    }

    with patch.object(planner, '_persist_learned_methods', new_callable=AsyncMock) as mock_persist:
        await planner._prune_learned_methods()
        
    current_methods_for_task = [m.name for m in planner.methods.get(task_name, [])]
    assert method_old_low_conf.name not in current_methods_for_task, "Old, low confidence method should be pruned"
    assert method_old_high_conf.name in current_methods_for_task, "Old, but high confidence method should be kept"
    assert method_recent.name in current_methods_for_task, "Recent method should be kept"
    assert mock_persist.call_count == 1


@pytest.mark.asyncio
async def test_prune_learned_methods_task_entry_removed_if_all_pruned(htn_planner_instance_with_ops: HTNPlanner):
    planner = htn_planner_instance_with_ops
    _Method_prune_all = globals().get('Method', Method)

    task_name_prune_all = "task_all_prune"
    method_to_be_pruned_1 = _Method_prune_all( # type: ignore
        name="prune_me_1", task_signature=(task_name_prune_all,), subtasks=[],
        metadata={"learned_via": "llm_sketch", "usage_count": 20, "success_rate": 0.1, "confidence": 0.1}
    )
    method_to_be_pruned_2 = _Method_prune_all( # type: ignore
        name="prune_me_2", task_signature=(task_name_prune_all,), subtasks=[],
        metadata={"learned_via": "llm_sketch", "usage_count": 20, "success_rate": 0.05, "confidence": 0.15}
    )
    planner.methods[task_name_prune_all] = [method_to_be_pruned_1, method_to_be_pruned_2]
    
    planner._config["htn_planner_pruning"] = {
        "min_usage_for_pruning": 10, "low_success_rate_threshold": 0.2,
        "low_confidence_threshold": 0.2, "max_age_unused_days": 30
    }
    with patch.object(planner, '_persist_learned_methods', new_callable=AsyncMock): 
        await planner._prune_learned_methods()
    
    assert task_name_prune_all not in planner.methods, "Task entry should be removed if all its learned methods are pruned."


@pytest.mark.asyncio
async def test_estimate_method_properties_uses_full_cognitive_context_for_pwm(htn_planner_with_mock_pwm: HTNPlanner):
    planner = htn_planner_with_mock_pwm
    _Method_test_ctx = globals().get('Method', Method)
    
    test_method = _Method_test_ctx( # type: ignore
        name="method_for_context_test",
        task_signature=("task_ctx_test",),
        subtasks=["OP_HIGH_SUCCESS"] 
    )

    rich_cognitive_context = {
        "consciousness_level": "CONSCIOUS",
        "current_cs_level_name": "CONSCIOUS",
        "current_goal_type": "test_goal_type_rich_ctx", 
        "php_levels": {
            "pain": 1.5,
            "happiness": 6.5,
            "purpose": 7.5
        },
        "drives": { 
            "drive_curiosity": 0.8, 
            "drive_satisfaction": 0.4,
            "drive_competence": 0.6
        },
    }
    
    planner._controller.predictive_world_model.process.reset_mock() # type: ignore
    if hasattr(planner, '_planning_session_pwm_cache'):
        planner._planning_session_pwm_cache.clear()

    await planner._estimate_method_properties(
        method=test_method,
        bindings={},
        current_state_for_heuristic=set(),
        full_cognitive_context_for_pwm=rich_cognitive_context 
    )

    assert planner._controller.predictive_world_model.process.call_count > 0, "Mock PWM process was not called" # type: ignore
    
    call_args_list = planner._controller.predictive_world_model.process.call_args_list # type: ignore
    assert len(call_args_list) == 1, f"Expected 1 call to PWM, got {len(call_args_list)}"
    
    input_state_to_pwm = call_args_list[0][0][0] 
    assert "predict_request" in input_state_to_pwm
    predict_request = input_state_to_pwm["predict_request"]
    
    assert "context" in predict_request, "PWM predict_request missing 'context' key"
    pwm_context_received = predict_request["context"]
    
    logger_test_htn.debug(f"Context received by mock PWM: {pwm_context_received}")

    assert pwm_context_received.get("current_cs_level_name") == "CONSCIOUS"
    assert pwm_context_received.get("active_goal_type") == "test_goal_type_rich_ctx"
    
    assert "php_levels" in pwm_context_received, "php_levels missing in context to PWM"
    assert isinstance(pwm_context_received["php_levels"], dict)
    assert pwm_context_received["php_levels"].get("pain") == 1.5
    assert pwm_context_received["php_levels"].get("happiness") == 6.5
    assert pwm_context_received["php_levels"].get("purpose") == 7.5
    
    assert "drives" in pwm_context_received, "drives missing in context to PWM"
    assert isinstance(pwm_context_received["drives"], dict)
    
    assert pwm_context_received["drives"].get("drive_curiosity") == 0.8
    assert pwm_context_received["drives"].get("drive_satisfaction") == 0.4
    assert pwm_context_received["drives"].get("drive_competence") == 0.6

@pytest.mark.asyncio
async def test_plan_uses_existing_method_no_llm_sketch(htn_planner_for_llm_sketch_tests: Tuple[HTNPlanner, asyncio.Queue]):
    planner, mock_llm_q = htn_planner_for_llm_sketch_tests
    _GoalSketchTest = globals().get('MockGoal', Goal)
    _MethodSketchTest = globals().get('Method', Method) # If Method is complex

    # Define an existing method for a task
    existing_method = _MethodSketchTest( # type: ignore
        name="method_for_existing_task",
        task_signature=("task_has_method", "?param"),
        subtasks=[("READ_FILE", "?param")]
    )
    planner.methods["task_has_method"] = [existing_method]

    goal = _GoalSketchTest(description="task_has_method : existing_file.txt") # type: ignore
    initial_state = {Predicate("isFile", ("existing_file.txt",), True)} # type: ignore

    # Mock _llm_assisted_plan_sketch to ensure it's NOT called
    with patch.object(planner, '_llm_assisted_plan_sketch', new_callable=AsyncMock) as mock_llm_sketch_call:
        mock_llm_sketch_call.return_value = "sketching_not_needed_should_not_be_called" # Should not return this

        plan = await planner.plan(goal, initial_state)

    assert plan is not None
    assert len(plan) == 1
    assert plan[0]["type"] == "READ_FILE"
    assert plan[0]["params"]["path"] == "existing_file.txt"
    mock_llm_sketch_call.assert_not_called() # Crucial: ensure LLM sketch was not attempted
    assert mock_llm_q.empty(), "LLM response queue should be empty if sketch not called"

@pytest.mark.asyncio
async def test_plan_llm_sketch_success_and_method_learning(htn_planner_for_llm_sketch_tests: Tuple[HTNPlanner, asyncio.Queue], tmp_path: Path):
    planner, mock_llm_q = htn_planner_for_llm_sketch_tests
    _GoalSketchTest = globals().get('MockGoal', Goal)
    _PredicateSketchTest = globals().get('MockPredicate', Predicate)

    if planner._learned_methods_path is None:
        planner._learned_methods_path = tmp_path / "htn_lib_s2.json"
        planner._controller.config["agent_data_paths"]["htn_library_path"] = str(planner._learned_methods_path.name) 
        planner._controller.agent_root_path = tmp_path 

    goal_desc_for_sketch = "task_needs_sketch : data/file_to_read.txt"
    goal = _GoalSketchTest(description=goal_desc_for_sketch) 
    initial_state = {_PredicateSketchTest("isFile", ("data/file_to_read.txt",), True)} 

    await mock_llm_q.put( (MOCK_LLM_SKETCH_SUCCESS_READ_FILE["response_str"], MOCK_LLM_SKETCH_SUCCESS_READ_FILE["error"]) )
    
    # --- Capture the future from schedule_offline_task ---
    scheduled_future: Optional[asyncio.Future] = None
    original_schedule_task = planner._controller.schedule_offline_task

    def schedule_task_wrapper(*args, **kwargs):
        nonlocal scheduled_future
        # Call the original and store the future
        f = original_schedule_task(*args, **kwargs)
        scheduled_future = f
        return f

    with patch.object(planner, '_persist_learned_methods', new_callable=AsyncMock) as mock_persist, \
         patch.object(planner._controller, 'schedule_offline_task', side_effect=schedule_task_wrapper) as mock_schedule_call:
        
        plan_attempt1 = await planner.plan(goal, initial_state)

    assert plan_attempt1 is None, "Plan should be None when LLM sketch is initiated and not yet complete."
    
    # --- NEW: Wait for the mock LLM response to be processed ---
    assert mock_schedule_call.called, "schedule_offline_task was not called by planner.plan"
    assert scheduled_future is not None, "Future was not captured"
    if scheduled_future: # Should always be true if mock_schedule_call.called
        await planner._controller.wait_for_mock_completion(scheduled_future)
    # --- END NEW ---

    assert mock_llm_q.empty(), "LLM response queue should have been consumed after mock completion."
    
    # --- Second planning attempt: LLM sketch result processed, method learned, plan generated ---
    with patch.object(planner, '_persist_learned_methods', new_callable=AsyncMock) as mock_persist_attempt2:
        plan_attempt2 = await planner.plan(goal, initial_state)
    
    assert plan_attempt2 is not None, "Plan should be generated in the second attempt using the learned method."
    assert len(plan_attempt2) == 1
    assert plan_attempt2[0]["type"] == "READ_FILE"
    assert plan_attempt2[0]["params"]["path"] == "data/file_to_read.txt"
    
    assert "task_needs_sketch" in planner.methods
    assert len(planner.methods["task_needs_sketch"]) > 0
    learned_method = planner.methods["task_needs_sketch"][0] 
    assert learned_method.name.startswith("learned_task_needs_sketch_")
    assert learned_method.task_signature == ("task_needs_sketch", "?param0")
    assert ("READ_FILE", "?param0") in learned_method.subtasks

    mock_persist_attempt2.assert_called_once()

@pytest.mark.asyncio
async def test_plan_llm_sketch_failure_null_response(htn_planner_for_llm_sketch_tests: Tuple[HTNPlanner, asyncio.Queue]):
    planner, mock_llm_q = htn_planner_for_llm_sketch_tests
    _GoalSketchTest = globals().get('MockGoal', Goal)

    goal = _GoalSketchTest(description="task_llm_will_fail : some_param") 
    initial_state = set()

    await mock_llm_q.put( (MOCK_LLM_SKETCH_FAILURE_NULL_RESPONSE["response_str"], MOCK_LLM_SKETCH_FAILURE_NULL_RESPONSE["error"]) )
    
    # --- Capture the future from schedule_offline_task ---
    scheduled_future_null_resp: Optional[asyncio.Future] = None
    original_schedule_task_null_resp = planner._controller.schedule_offline_task

    def schedule_task_wrapper_null_resp(*args, **kwargs):
        nonlocal scheduled_future_null_resp
        f = original_schedule_task_null_resp(*args, **kwargs)
        scheduled_future_null_resp = f
        return f

    with patch.object(planner, '_persist_learned_methods', new_callable=AsyncMock) as mock_persist, \
         patch.object(planner._controller, 'schedule_offline_task', side_effect=schedule_task_wrapper_null_resp) as mock_schedule_call_null_resp:
        
        plan_attempt1 = await planner.plan(goal, initial_state)
    
    assert plan_attempt1 is None, "Plan should be None when LLM sketch is initiated."

    # --- NEW: Wait for the mock LLM response to be processed ---
    assert mock_schedule_call_null_resp.called, "schedule_offline_task was not called for null response test"
    assert scheduled_future_null_resp is not None, "Future was not captured for null response test"
    if scheduled_future_null_resp:
        await planner._controller.wait_for_mock_completion(scheduled_future_null_resp)
    # --- END NEW ---
    
    assert mock_llm_q.empty(), "LLM response queue should have been consumed after mock completion for null response."
    
    with patch.object(planner, '_persist_learned_methods', new_callable=AsyncMock) as mock_persist_attempt2:
        plan_attempt2 = await planner.plan(goal, initial_state)

    assert plan_attempt2 is None, "Plan should still be None if LLM sketch returned null and no method was learned."
    assert "task_llm_will_fail" not in planner.methods 
    
    mock_persist.assert_not_called() 
    mock_persist_attempt2.assert_not_called()

@pytest.mark.asyncio
async def test_scenario3b_llm_sketch_returns_error(htn_planner_for_llm_sketch_tests: Tuple[HTNPlanner, asyncio.Queue]):
    planner_instance, mock_llm_q_3b = htn_planner_for_llm_sketch_tests # Unpack
    _Goal_s3_err = globals().get('MockGoal', Goal)
    _Predicate_s3_err = globals().get('MockPredicate', Predicate)

    goal_desc_s3_err = "perform task with error from llm"
    initial_task_s3_err: TaskType = ("task_error_from_llm",) # This task name is used for patching _goal_to_task

    # --- Capture the future for this test ---
    scheduled_future_s3b: Optional[asyncio.Future] = None
    original_schedule_task_s3b = planner_instance._controller.schedule_offline_task
    def schedule_task_wrapper_s3b(*args, **kwargs):
        nonlocal scheduled_future_s3b
        f = original_schedule_task_s3b(*args, **kwargs)
        scheduled_future_s3b = f
        return f
    
    with patch.object(planner_instance, '_goal_to_task', return_value=initial_task_s3_err), \
         patch.object(planner_instance._controller, 'schedule_offline_task', side_effect=schedule_task_wrapper_s3b) as mock_schedule_call_s3b:
        test_goal_s3_err = _Goal_s3_err(description=goal_desc_s3_err, id="goal_s3_err") 
        current_state_s3_err: Set[_Predicate_s3_err] = set() 

        logger_test_htn.info("SCENARIO 3b (ERROR): First plan attempt (expect LLM sketch -> error)")
        await mock_llm_q_3b.put( (MOCK_LLM_SKETCH_API_ERROR["response_str"], MOCK_LLM_SKETCH_API_ERROR["error"]) )
        plan_attempt1 = await planner_instance.plan(test_goal_s3_err, current_state_s3_err)
        assert plan_attempt1 is None

        # --- NEW: Wait for mock completion ---
        assert mock_schedule_call_s3b.called
        assert scheduled_future_s3b is not None
        if scheduled_future_s3b:
            await planner_instance._controller.wait_for_mock_completion(scheduled_future_s3b)
        assert mock_llm_q_3b.empty(), "LLM queue should be empty after mock completion (scenario 3b)"
        # --- END NEW ---

        logger_test_htn.info("SCENARIO 3b (ERROR): Simulating LLM (error) response received, second plan attempt (expect no plan)")
        plan_attempt2 = await planner_instance.plan(test_goal_s3_err, current_state_s3_err)
        assert plan_attempt2 is None, "Expected no plan if LLM sketch returns an error"
        assert "task_error_from_llm" not in planner_instance.methods or \
               not any(m.name.startswith("learned_task_error_from_llm_")
                       for m in planner_instance.methods.get("task_error_from_llm", [])), \
               "No method should be learned if LLM sketch has an error."

@pytest.mark.asyncio
async def test_scenario3c_llm_sketch_returns_invalid_json(htn_planner_for_llm_sketch_tests: Tuple[HTNPlanner, asyncio.Queue]):
    planner_instance, mock_llm_q_3c = htn_planner_for_llm_sketch_tests # Unpack
    _Goal_s3_inv = globals().get('MockGoal', Goal)
    _Predicate_s3_inv = globals().get('MockPredicate', Predicate)

    goal_desc_s3_inv = "perform task with invalid json from llm"
    initial_task_s3_inv: TaskType = ("task_invalid_json_sketch",)

    # --- Capture the future from schedule_offline_task ---
    scheduled_future_s3c: Optional[asyncio.Future] = None
    original_schedule_task_s3c = planner_instance._controller.schedule_offline_task

    def schedule_task_wrapper_s3c(*args, **kwargs):
        nonlocal scheduled_future_s3c
        f = original_schedule_task_s3c(*args, **kwargs)
        scheduled_future_s3c = f
        return f

    with patch.object(planner_instance, '_goal_to_task', return_value=initial_task_s3_inv), \
         patch.object(planner_instance._controller, 'schedule_offline_task', side_effect=schedule_task_wrapper_s3c) as mock_schedule_call_s3c:

        test_goal_s3_inv = _Goal_s3_inv(description=goal_desc_s3_inv, id="goal_s3_inv")
        current_state_s3_inv: Set[_Predicate_s3_inv] = set()

        logger_test_htn.info("SCENARIO 3c (INVALID_JSON): First plan attempt (expect LLM sketch -> invalid JSON)")
        await mock_llm_q_3c.put( (MOCK_LLM_SKETCH_FAILURE_INVALID_JSON["response_str"], MOCK_LLM_SKETCH_FAILURE_INVALID_JSON["error"]) )
        plan_attempt1 = await planner_instance.plan(test_goal_s3_inv, current_state_s3_inv)
        assert plan_attempt1 is None

        # --- NEW: Wait for mock completion ---
        assert mock_schedule_call_s3c.called
        assert scheduled_future_s3c is not None
        if scheduled_future_s3c:
            await planner_instance._controller.wait_for_mock_completion(scheduled_future_s3c)
        assert mock_llm_q_3c.empty(), "LLM queue should be empty after mock completion (scenario 3c)"
        # --- END NEW ---

        logger_test_htn.info("SCENARIO 3c (INVALID_JSON): Simulating LLM (invalid JSON) response received, second plan attempt (expect no plan)")
        plan_attempt2 = await planner_instance.plan(test_goal_s3_inv, current_state_s3_inv)
        assert plan_attempt2 is None, "Expected no plan if LLM sketch returns invalid JSON"
        assert "task_invalid_json_sketch" not in planner_instance.methods or \
               not any(m.name.startswith("learned_task_invalid_json_sketch_")
                       for m in planner_instance.methods.get("task_invalid_json_sketch", [])), \
               "No method should be learned if LLM sketch is invalid JSON."
# --- END OF FILE tests/unit/test_htn_planner.py ---