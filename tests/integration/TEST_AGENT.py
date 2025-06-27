#---START OF FILE/TEST_AGENT.py

import pytest
import asyncio
import time
import os
import sys # Added for other parts of the file if they use it
from pathlib import Path
import queue # Import queue for mocking
from unittest.mock import patch, MagicMock # Use for mocking
import toml # For writing dummy config
import logging
import pytest_asyncio

# Conditionally import resource
RESOURCE_AVAILABLE = False
try:
    import resource
    if hasattr(resource, 'getrlimit') and hasattr(resource, 'RLIMIT_NOFILE'):
         RESOURCE_AVAILABLE = True
except ImportError:
    pass

# Attempt to import using absolute package path
PACKAGE_NAME = "consciousness_experiment"
CONTROLLER_AVAILABLE = False
AgentController = None
MODELS_AVAILABLE = False
# Define component classes as None initially
KnowledgeBase, CognitiveCache, PerformanceOptimizer, HTNPlanner = None, None, None, None
AttentionController, GlobalWorkspaceManager, ExperienceStream = None, None, None
ConsciousnessLevelAssessor, MetaCognitiveMonitor, LoopDetector = None, None, None
ErrorRecoverySystem, PredictiveWorldModel, DynamicSelfModel = None, None, None
EmergentMotivationSystem, NarrativeConstructor, ValueSystem = None, None, None
# Define data types and enums as None initially
Predicate, Goal, PhenomenalState, PainSource, ValueJudgment = None, None, None, None, None
create_goal_from_descriptor = None
AgentState, ConsciousState, GoalStatus, RecoveryMode, ValueCategory = None, None, None, None, None
CognitiveComponent = None # Base protocol
# Component init order from AgentController
COMPONENT_INIT_ORDER = [ # Use the name from AgentController if it's imported, otherwise define
    "knowledge_base", "cache", "performance_optimizer", "htn_planner",
    "attention_controller", "global_workspace", "experience_stream",
    "consciousness_assessor", "meta_cognition", "loop_detector",
    "predictive_world_model", "dynamic_self_model", "emergent_motivation_system",
    "narrative_constructor", 
    "value_system",         
    "error_recovery",
]
# For USER_GOAL_PRIORITY, etc. - these are module-level in AgentController
DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC = "Observe and learn from the environment"
DEFAULT_OBSERVE_GOAL_PRIORITY = 1.0
USER_GOAL_PRIORITY = 5.0


try:
    from consciousness_experiment.agent_controller import AgentController, component_classes, COMPONENT_INIT_ORDER as AGENT_CONTROLLER_COMPONENT_INIT_ORDER, DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC, DEFAULT_OBSERVE_GOAL_PRIORITY, USER_GOAL_PRIORITY
    # Overwrite global COMPONENT_INIT_ORDER if successfully imported from agent_controller
    COMPONENT_INIT_ORDER = AGENT_CONTROLLER_COMPONENT_INIT_ORDER

    from consciousness_experiment.models.datatypes import Predicate, Goal, PhenomenalState, PainSource, ValueJudgment, create_goal_from_descriptor
    from consciousness_experiment.models.enums import AgentState, ConsciousState, GoalStatus, RecoveryMode, ValueCategory
    from consciousness_experiment.protocols import CognitiveComponent

    # Import all component classes referenced in agent_controller.component_classes
    from consciousness_experiment.cognitive_modules.knowledge_base import KnowledgeBase
    from consciousness_experiment.cognitive_modules.cognitive_cache import CognitiveCache
    from consciousness_experiment.cognitive_modules.performance_optimizer import PerformanceOptimizer
    from consciousness_experiment.cognitive_modules.htn_planner import HTNPlanner
    from consciousness_experiment.cognitive_modules.attention_controller import AttentionController
    from consciousness_experiment.cognitive_modules.global_workspace_manager import GlobalWorkspaceManager
    from consciousness_experiment.cognitive_modules.experience_stream import ExperienceStream
    from consciousness_experiment.cognitive_modules.consciousness_level_assessor import ConsciousnessLevelAssessor
    from consciousness_experiment.cognitive_modules.meta_cognitive_monitor import MetaCognitiveMonitor
    from consciousness_experiment.cognitive_modules.loop_detector import LoopDetector
    from consciousness_experiment.cognitive_modules.error_recovery import ErrorRecoverySystem
    from consciousness_experiment.cognitive_modules.predictive_world_model import PredictiveWorldModel
    from consciousness_experiment.cognitive_modules.dynamic_self_model import DynamicSelfModel
    from consciousness_experiment.cognitive_modules.emergent_motivation_system import EmergentMotivationSystem
    from consciousness_experiment.cognitive_modules.narrative_constructor import NarrativeConstructor
    from consciousness_experiment.cognitive_modules.value_system import ValueSystem


    CONTROLLER_AVAILABLE = True
    MODELS_AVAILABLE = True
    # print(f"\nDEBUG (test_cognitive_cycle): Successfully imported AgentController and components via package path '{PACKAGE_NAME}'.") # Too verbose for every test

except ImportError as e:
    # print(f"\nDEBUG (test_cognitive_cycle): Failed to import AgentController or dependencies via package path '{PACKAGE_NAME}': {e}") # Too verbose
    # Fallback definitions for type hints if imports fail
    class MockProtocol: pass
    CognitiveComponent = MockProtocol # type: ignore
    AgentController = type('DummyAgentController', (object,), {}) # type: ignore
    component_classes = {} # Fallback
    # Fallback for module-level constants if AgentController import fails
    DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC = "Observe and learn from the environment (fallback)"
    DEFAULT_OBSERVE_GOAL_PRIORITY = 1.0
    USER_GOAL_PRIORITY = 5.0


# Logger for the test file
logger = logging.getLogger(__name__) # For test file's own logging


@pytest_asyncio.fixture(scope="function")
async def test_agent(tmp_path: Path):
    """Provides a fresh AgentController instance for each test, with enhanced debugging."""
    if not CONTROLLER_AVAILABLE:
        pytest.fail("FIXTURE_ABORT: AgentController class not available. Imports at top of test file failed.")

    timestamp = int(time.time() * 1000)
    logger.info(f"FIXTURE_SETUP: test_agent starting for tmp_path: {tmp_path}")

    # --- Standardized Config Override ---
    # This should match the structure your AgentController and components expect
    # and include keys used by the PHP system and other new features.
    config_override = {
        "agent_data_paths": {
            "pid_directory": "run", "kb_db_path": f"data/test_kb_{timestamp}.db", # Unique DB per test
            "narrative_log_path": f"data/test_narr_{timestamp}.jsonl",
            "self_model_path": f"data/test_self_{timestamp}.json",
            "predictive_model_path": f"data/test_pred_{timestamp}.json",
            "performance_adjustments_path": f"data/test_perf_adj_{timestamp}.json",
        },
        "agent": {
            "pid_file_name": f"test_agent_{timestamp}.pid", "ui_meter_update_interval_s": 0.001,
            "goal_re_evaluation_interval_cycles": 3, "default_goal_cooldown_cycles": 2,
            "min_curiosity_for_observe": 0.1, "max_consecutive_planning_failures": 2,
            "max_execution_failures_per_goal": 2,
        },
        "knowledge_base": {"default_context_retrieval_count": 3},
        "dynamic_self_model": {
            "learning_rate": 0.1, "max_invalid_paths": 10,
            "learning_events_history_size": 5, "reflection_interval_cycles": 3
        },
        "narrative_constructor": {
            "max_length": 5, "valence_change_threshold": 0.1, "intensity_threshold": 0.6,
            "save_interval_s": 10, "drive_change_threshold": 0.1, "significance_threshold": 0.05,
            "llm_model_name": "test_narrative_model", "temperature": 0.6, "timeout_s": 0.02, # Shortened timeout for tests
            "pain_change_threshold_sig": 0.3, "happiness_change_threshold_sig": 0.3,
            "purpose_change_threshold_sig": 0.3,
        },
        "filesystem": {"allow_file_write": True, "allow_overwrite": True, "max_list_items": 5, "max_read_chars": 500},
        "performance": {"target_cycle_time": 0.001, "max_planning_depth": 3, "profiler_history_size":5},
        "global_workspace": { "capacity": 5, "broadcast_threshold": 0.05, "min_items_if_any_attended": 1 },
        "attention_controller": {
            "max_candidates": 10, "recency_weight": 0.3, "hint_weight": 0.4, "goal_relevance_weight":0.3,
            "softmax_temperature": 0.5, "novelty_window_size": 3, "novelty_bonus_weight": 0.1,
            "surprise_bonus_weight": 0.2, "pain_attention_distraction_factor": 0.2, "pain_rumination_threshold_cycles": 3,
        },
        "htn_planner": {
            "plan_cache_ttl_s": 0.05, "min_planning_depth_on_low_cs": 1,
            "max_planning_depth_on_low_cs": 2, "low_cs_simplicity_penalty_factor": 0.5,
        },
        "cognitive_cache": {"default_ttl": 0.05},
        "consciousness_assessor": {
            "meta_conscious_threshold": 0.75, "conscious_threshold": 0.50, "pre_conscious_threshold": 0.25,
            "unconscious_threshold": 0.1, "diff_weight_sources": 0.4, "diff_weight_lexical": 0.6,
            "int_weight_shared_concepts": 0.5, "phi_contrib_diff": 0.5, "phi_contrib_int": 0.5,
            "global_workspace_capacity_for_norm": 5,
        },
        "loop_detection": {
            "window_size": 3, "max_consecutive_actions": 2, "frequency_threshold": 0.67,
            "ignore_thinking_actions": True
        },
        "error_recovery": {"max_error_history": 3, "frequency_window": 2, "frequency_threshold": 1},
        "telemetry": {"enabled": False, "host": "localhost", "port": 8765},
        "predictive_world_model": {"initial_version": 0, "learning_rate": 0.1, "memory_length": 5, "save_interval_versions": 2},
        "emergent_motivation_system": {
            "detailed_evaluation_interval_cycles": 1, "ems_cs_history_maxlen": 3,
            "ems_low_cs_persistence_threshold": 2, "ems_low_cs_curiosity_boost_factor": 0.1,
            "drives": { # Ensure drives sub-table is present
                "curiosity": {"gain_prediction_error": 0.1, "gain_from_high_pain_for_distraction": 0.05, "value":0.5, "decay":0.02, "gain_discovery": 0.08, "loss_repetition": 0.04, "threshold_high_pain_for_curiosity":7.0, "threshold_low_purpose_for_curiosity":2.0},
                "satisfaction": {"loss_from_pain_factor": 0.15, "gain_from_happiness_factor": 0.08, "value":0.5, "decay":0.03, "gain_success_rate":0.1, "loss_failure_rate":0.15, "gain_goal_achieved":0.3, "loss_goal_failed":0.25},
                "competence": {"gain_from_low_purpose_for_efficacy": 0.1, "value":0.5, "decay":0.01, "gain_capability_increase":0.1, "loss_limitation_increase":0.1, "gain_success_rate":0.05, "loss_failure_rate":0.07, "threshold_low_purpose_for_competence":2.5}
            }
        },
        "experience_stream": {"intensity_factor": 1.0, "valence_decay": 0.1, "custom_stopwords": ["testword"]},
        "meta_cognition": {
            "stagnation_threshold_s": 0.1, "failure_rate_threshold": 0.6, "low_consciousness_threshold_s": 0.1,
            "history_size": 3, "reflection_trigger_frequency_cycles": 2
        },
        "performance_optimizer": {"history_size": 3, "auto_apply_adjustments": True},
        "llm_settings": {
            "default_timeout_s": 0.02, "action_selection_temperature": 0.7, # Shortened timeout
            "max_recent_turns_for_prompt": 2, "intent_mapping_temperature": 0.3
        },
        "value_system": {
            "plan_rejection_value_threshold": -0.3, "action_safety_veto_threshold": -0.7,
            "value_weights": {"safety": 2.5, "affective_balance": 1.5, "goal_achievement": 1.8 }
        },
        "internal_states": {
            "baseline_pain_age_factor": 0.00005, "max_baseline_pain_from_age": 1.5,
            "acute_pain_goal_fail_priority_scale_factor": 0.4, "pain_event_max_initial_intensity": 2.5,
            "default_pain_event_decay_rate_per_cycle": 0.01, "pain_event_min_intensity_to_retain": 0.02,
            "happiness_from_goal_priority_scale_factor": 0.25, "happiness_decay_to_baseline_factor": 0.04,
            "happiness_baseline_target": 5.0, "pain_impact_on_happiness_scale_factor": 0.25,
            "purpose_from_capability_gain_factor": 0.06, "purpose_from_high_priority_goal_factor": 0.25,
            "purpose_decay_rate_per_cycle": 0.0003,
            "complex_goal_priority_threshold": (USER_GOAL_PRIORITY if USER_GOAL_PRIORITY else 5.0) - 1.0,
            "max_pain_shutdown_threshold": 9.5, "min_purpose_shutdown_threshold": 0.8,
        }
    }
    # --- End Standardized Config Override ---

    # Create necessary data directories within tmp_path
    for key in ["kb_db_path", "narrative_log_path", "self_model_path", "predictive_model_path", "performance_adjustments_path"]:
        path_val = config_override["agent_data_paths"].get(key)
        if path_val: (tmp_path / Path(path_val).parent).mkdir(parents=True, exist_ok=True)
    pid_dir_val = config_override["agent_data_paths"].get("pid_directory")
    if pid_dir_val: (tmp_path / Path(pid_dir_val)).mkdir(parents=True, exist_ok=True)

    mock_ui_queue = MagicMock(spec=queue.Queue)
    agent: AgentController # For type hinting
    initialized_components_names: List[str] = [] # Store names of successfully initialized components

    dummy_config_file = tmp_path / f"test_config_{timestamp}.toml"
    try:
        with open(dummy_config_file, "w") as f:
            toml.dump(config_override, f)
        logger.info(f"FIXTURE_SETUP: Successfully wrote dummy config to {dummy_config_file}")
    except Exception as e:
        logger.error(f"FIXTURE_ERROR: Failed to write dummy config file: {e}", exc_info=True)
        pytest.fail(f"Fixture: Failed to write dummy config for AgentController: {e}")

    # This is where AgentController and its internal components are instantiated.
    # Its __init__ also instantiates the helper managers.
    agent = AgentController(ui_queue=mock_ui_queue, model_name="test_integration_model", config_path=str(dummy_config_file))
    # Override agent_root_path to use tmp_path for this test instance
    agent.agent_root_path = tmp_path
    # Force the config_override to be used by the agent instance after its __init__
    # This ensures tests use the specific test config, not one potentially loaded from a default path by __init__
    agent.config = config_override
    logger.info(f"FIXTURE_SETUP: AgentController instance created. Root path: {agent.agent_root_path}")


    # Set up asyncio loop for the agent
    try:
        agent_loop_for_init = asyncio.get_running_loop()
    except RuntimeError:
        agent_loop_for_init = asyncio.new_event_loop()
        asyncio.set_event_loop(agent_loop_for_init)
    agent._asyncio_loop = agent_loop_for_init


    # This section ensures that the agent instance used by the tests
    # has actual component instances and type references, not mocks from fallbacks
    # in case of import errors at the top of agent_controller.py itself.
    if CONTROLLER_AVAILABLE and MODELS_AVAILABLE: # Check if actual classes were imported
        logger.info("FIXTURE_DEBUG: Forcing real component instances AND type references into AgentController...")
        # Iterate through the component_classes map from agent_controller.py (or this file's fallback)
        for name_key_in_map, RealComponentClass_from_map in component_classes.items():
            if RealComponentClass_from_map and not (hasattr(RealComponentClass_from_map, '_is_dummy') and RealComponentClass_from_map._is_dummy()): # type: ignore
                try:
                    # Only reinstantiate if it's in COMPONENT_INIT_ORDER (expected by controller)
                    # or if agent.components doesn't have it / has a dummy
                    if name_key_in_map in COMPONENT_INIT_ORDER or \
                       name_key_in_map not in agent.components or \
                       (hasattr(agent.components[name_key_in_map], '_is_dummy') and agent.components[name_key_in_map]._is_dummy()): # type: ignore

                        instance = RealComponentClass_from_map()
                        agent.components[name_key_in_map] = instance
                        # Also set as direct attribute if agent_controller.py's __init__ does this
                        if hasattr(agent, name_key_in_map):
                            setattr(agent, name_key_in_map, instance)
                        logger.debug(f"Fixture: Replaced/Ensured agent.components['{name_key_in_map}'] with real {RealComponentClass_from_map}")
                except Exception as e_inst_real:
                    logger.error(f"Fixture: Error instantiating REAL component {name_key_in_map} ({RealComponentClass_from_map}): {e_inst_real}")
                    pytest.fail(f"Fixture: Failed to instantiate REAL component {name_key_in_map}: {e_inst_real}")

        type_references_to_force = {
            "_Predicate": Predicate, "_Goal": Goal, "_PhenomenalState": PhenomenalState,
            "_PainSource": PainSource, "_ValueJudgment": ValueJudgment, "_AgentState": AgentState,
            "_ConsciousState": ConsciousState, "_GoalStatus": GoalStatus, "_RecoveryMode": RecoveryMode,
            "_ValueCategory": ValueCategory, "_create_goal_from_descriptor": create_goal_from_descriptor,
            "_CognitiveComponentBase": CognitiveComponent
        }
        for attr_name, RealType in type_references_to_force.items():
            if RealType and not (hasattr(RealType, '_is_dummy') and RealType._is_dummy()): # type: ignore
                setattr(agent, attr_name, RealType)
                logger.debug(f"Fixture: Set agent.{attr_name} to real type {RealType}.")
            else:
                logger.warning(f"Fixture: Real type for {attr_name} not available (is None or Mock). Agent may use mock/fallback if its internal imports failed.")
        
        # Sanity check KB instance type
        if "knowledge_base" in agent.components:
            actual_kb_instance = agent.components["knowledge_base"]
            ExpectedKBClass = component_classes.get("knowledge_base") # Get from the map used for instantiation
            if ExpectedKBClass and not isinstance(actual_kb_instance, ExpectedKBClass):
                 logger.error(f"Fixture ERROR: agent.components['knowledge_base'] is type {type(actual_kb_instance)}, expected {ExpectedKBClass}")
                 pytest.fail("KnowledgeBase component type mismatch after forcing real instances.")
            else:
                 logger.debug(f"Fixture: agent.components['knowledge_base'] type {type(actual_kb_instance)} matches expected.")

    else: # Should have been caught by module-level skip if CONTROLLER_AVAILABLE is False
        pytest.fail("FIXTURE_ABORT: CONTROLLER_AVAILABLE or MODELS_AVAILABLE is False but fixture execution continued.")


    # Initialize components on the agent instance
    init_success_fixture = True
    logger.info("FIXTURE_DEBUG: Starting component initialization loop within fixture...")
    for name in COMPONENT_INIT_ORDER: # Use the agent_controller's COMPONENT_INIT_ORDER
        if not init_success_fixture: break
        if name in agent.components:
            component = agent.components[name]
            logger.info(f"FIXTURE_DEBUG: Attempting to initialize component: '{name}' of type {type(component)}")
            try:
                init_method = getattr(component, 'initialize', None)
                if init_method:
                     if asyncio.iscoroutinefunction(init_method):
                         success_flag = await init_method(agent.config, agent)
                     else:
                         logger.warning(f"FIXTURE_DEBUG: Component '{name}' initialize method is not async. Calling synchronously.")
                         success_flag = component.initialize(agent.config, agent) # type: ignore
                     
                     if not success_flag:
                         logger.error(f"FIXTURE_ERROR: Component '{name}' .initialize() returned False! ABORTING FIXTURE.")
                         init_success_fixture = False; break
                     else:
                         initialized_components_names.append(name)
                         logger.info(f"FIXTURE_DEBUG: Component '{name}' .initialize() returned True.")
                else:
                     logger.warning(f"FIXTURE_DEBUG: Component '{name}' has no initialize method. Assuming success by default for fixture.")
                     initialized_components_names.append(name) # Still add to list for shutdown
            except Exception as e_init_fixture_loop:
                 logger.error(f"FIXTURE_EXCEPTION: Exception during '{name}' .initialize(): {e_init_fixture_loop}", exc_info=True)
                 init_success_fixture = False; break
        elif name in component_classes: # Check if it *should* have been in agent.components
             logger.error(f"FIXTURE_ERROR: Component '{name}' in COMPONENT_INIT_ORDER but NOT FOUND in agent.components. Expected class: {component_classes.get(name)}. ABORTING FIXTURE.")
             init_success_fixture = False; break
        else: # Should not happen if COMPONENT_INIT_ORDER is aligned with component_classes
            logger.error(f"FIXTURE_ERROR: Component '{name}' in COMPONENT_INIT_ORDER but NOT DEFINED in test's component_classes map. ABORTING FIXTURE.")
            init_success_fixture = False; break

    if not init_success_fixture:
        if agent: # Agent instance exists
            # Attempt to call agent's own shutdown components method if available
            # The attribute name changed in AgentController to _shutdown_components_by_name_list
            shutdown_method_on_agent = getattr(agent, '_shutdown_components_by_name_list', getattr(agent, '_shutdown_components', None))
            if shutdown_method_on_agent and asyncio.iscoroutinefunction(shutdown_method_on_agent):
                logger.info("FIXTURE_CLEANUP: Calling agent's shutdown for partially initialized components due to init failure.")
                await shutdown_method_on_agent(initialized_components_names)
            else:
                logger.warning("FIXTURE_CLEANUP: Agent or its shutdown method not available/async for failed init cleanup.")
        pytest.fail("Agent component initialization failed in test fixture. Check FIXTURE_ERROR logs for details.")

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ START OF ADDED SECTION TO ENSURE HELPER MANAGERS +++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logger.info("FIXTURE_DEBUG: Ensuring helper managers are instantiated on agent (post-component init)...")
    try:
        from consciousness_experiment.agent_helpers.goal_manager import GoalLifecycleManager
        if not hasattr(agent, 'goal_manager') or not isinstance(agent.goal_manager, GoalLifecycleManager):
            agent.goal_manager = GoalLifecycleManager(agent) # Pass agent reference
            logger.info("Fixture: Instantiated agent.goal_manager.")
    except ImportError:
        logger.error("Fixture: Failed to import GoalLifecycleManager for explicit instantiation.")
        pytest.fail("Fixture: Could not import GoalLifecycleManager.")
    except Exception as e_gm_fix:
        logger.error(f"Fixture: Error ensuring GoalLifecycleManager: {e_gm_fix}")
        pytest.fail(f"Fixture: Error with GoalLifecycleManager: {e_gm_fix}")

    try:
        from consciousness_experiment.agent_helpers.action_executor import ActionExecutor
        if not hasattr(agent, 'action_executor') or not isinstance(agent.action_executor, ActionExecutor):
            agent.action_executor = ActionExecutor(agent) # Pass agent reference
            logger.info("Fixture: Instantiated agent.action_executor.")
    except ImportError:
        logger.error("Fixture: Failed to import ActionExecutor for explicit instantiation.")
        pytest.fail("Fixture: Could not import ActionExecutor.")
    except Exception as e_ae_fix:
        logger.error(f"Fixture: Error ensuring ActionExecutor: {e_ae_fix}")
        pytest.fail(f"Fixture: Error with ActionExecutor: {e_ae_fix}")

    try:
        from consciousness_experiment.agent_helpers.internal_state_manager import InternalStateUpkeepManager
        if not hasattr(agent, 'internal_state_upkeep_manager') or not isinstance(agent.internal_state_upkeep_manager, InternalStateUpkeepManager):
            agent.internal_state_upkeep_manager = InternalStateUpkeepManager(agent) # Pass agent reference
            logger.info("Fixture: Instantiated agent.internal_state_upkeep_manager.")
    except ImportError:
        logger.error("Fixture: Failed to import InternalStateUpkeepManager for explicit instantiation.")
        pytest.fail("Fixture: Could not import InternalStateUpkeepManager.")
    except Exception as e_ism_fix:
        logger.error(f"Fixture: Error ensuring InternalStateUpkeepManager: {e_ism_fix}")
        pytest.fail(f"Fixture: Error with InternalStateUpkeepManager: {e_ism_fix}")
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++ END OF ADDED SECTION +++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Final assertions for fixture validity (with specific fail messages)
    if agent is None: pytest.fail("FIXTURE_FAIL: AgentController instance is None after creation.")
    if not (hasattr(agent, 'DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC') and agent.DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC == DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC):
        pytest.fail(f"FIXTURE_FAIL: agent.DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC assertion failed. Agent has: {getattr(agent, 'DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC', 'MISSING')}")
    if not (hasattr(agent, 'USER_GOAL_PRIORITY') and agent.USER_GOAL_PRIORITY == USER_GOAL_PRIORITY):
        pytest.fail(f"FIXTURE_FAIL: agent.USER_GOAL_PRIORITY assertion failed. Agent has: {getattr(agent, 'USER_GOAL_PRIORITY', 'MISSING')}")
    if not (hasattr(agent, 'DEFAULT_OBSERVE_GOAL_PRIORITY') and agent.DEFAULT_OBSERVE_GOAL_PRIORITY == DEFAULT_OBSERVE_GOAL_PRIORITY):
        pytest.fail(f"FIXTURE_FAIL: agent.DEFAULT_OBSERVE_GOAL_PRIORITY assertion failed. Agent has: {getattr(agent, 'DEFAULT_OBSERVE_GOAL_PRIORITY', 'MISSING')}")
    if not (hasattr(agent, 'config') and isinstance(agent.config, dict)):
        pytest.fail(f"FIXTURE_FAIL: agent.config assertion failed. Type is {type(getattr(agent, 'config', None))}")
    if not hasattr(agent, 'agent_root_path'):
        pytest.fail("FIXTURE_FAIL: agent.agent_root_path assertion failed.")
    if not (hasattr(agent, 'components') and isinstance(agent.components, dict)):
        pytest.fail(f"FIXTURE_FAIL: agent.components assertion failed. Type is {type(getattr(agent, 'components', None))}")
    
    # Check for helper managers after explicit instantiation block
    if not hasattr(agent, 'goal_manager'):
        pytest.fail("FIXTURE_FAIL: Agent missing 'goal_manager' attribute after explicit setup.")
    if not hasattr(agent, 'action_executor'):
        pytest.fail("FIXTURE_FAIL: Agent missing 'action_executor' attribute after explicit setup.")
    if not hasattr(agent, 'internal_state_upkeep_manager'): # This was the original failing assertion
        pytest.fail("FIXTURE_FAIL: Agent missing 'internal_state_upkeep_manager' attribute after explicit setup.")

    logger.info(f"FIXTURE_SETUP: test_agent setup complete for {tmp_path}. Agent ready.")
    yield agent # Provide the initialized agent to the test

    # --- Teardown ---
    logger.info(f"FIXTURE_TEARDOWN: test_agent for {tmp_path} starting teardown...")
    if agent and hasattr(agent, '_is_running_flag') and agent._is_running_flag.is_set():
        logger.warning(f"FIXTURE_TEARDOWN: Agent '{agent.model_name}' still running. Attempting stop.")
        try:
            # Ensure stop is awaited if it's async (it should be)
            stop_method_agent = getattr(agent, 'stop', None)
            if stop_method_agent:
                if asyncio.iscoroutinefunction(stop_method_agent): await stop_method_agent()
                else: stop_method_agent() # Should be async
            # Give a very brief moment for tasks to cancel if stop() schedules cancellation
            await asyncio.sleep(0.01)
        except Exception as e_stop:
            logger.error(f"FIXTURE_TEARDOWN: Error stopping agent during teardown: {e_stop}", exc_info=True)

    # Explicitly call the more specific shutdown method if available
    if agent and hasattr(agent, '_shutdown_components_by_name_list'):
        logger.info("FIXTURE_TEARDOWN: Calling _shutdown_components_by_name_list.")
        await agent._shutdown_components_by_name_list(initialized_components_names)
    elif agent and hasattr(agent, '_shutdown_components'): # Fallback to older name
        logger.info("FIXTURE_TEARDOWN: Calling _shutdown_components (fallback).")
        await agent._shutdown_components(initialized_components_names) # type: ignore
    else:
        logger.info("FIXTURE_TEARDOWN: Agent or its shutdown method not found.")

    if agent and hasattr(agent, '_cleanup_pid_file') and callable(agent._cleanup_pid_file):
        agent._cleanup_pid_file() # Explicitly call if exists

    # Attempt to clean up any database files created if KB was initialized
    if "knowledge_base" in initialized_components_names and agent and hasattr(agent, 'knowledge_base'):
        kb_instance_fixture = getattr(agent, 'knowledge_base')
        if kb_instance_fixture and hasattr(kb_instance_fixture, 'db_path') and kb_instance_fixture.db_path:
            db_file_to_remove = Path(kb_instance_fixture.db_path)
            if db_file_to_remove.exists():
                try:
                    # Close connection if KB holds it open
                    if hasattr(kb_instance_fixture, '_connection') and kb_instance_fixture._connection:
                        try: kb_instance_fixture._connection.close()
                        except Exception: pass
                    db_file_to_remove.unlink()
                    logger.info(f"FIXTURE_TEARDOWN: Removed test KB DB: {db_file_to_remove}")
                except OSError as e_db_del:
                    logger.warning(f"FIXTURE_TEARDOWN: Could not remove test KB DB {db_file_to_remove}: {e_db_del}")

    # Clean up other temp files created by the fixture using agent_root_path
    paths_to_check_from_config = [
        config_override["agent_data_paths"]["narrative_log_path"],
        config_override["agent_data_paths"]["self_model_path"],
        config_override["agent_data_paths"]["predictive_model_path"],
        config_override["agent_data_paths"]["performance_adjustments_path"],
        # dummy_config_file itself is already in tmp_path handled by pytest
    ]
    for rel_path_str in paths_to_check_from_config:
        abs_path_to_clean = tmp_path / Path(rel_path_str)
        if abs_path_to_clean.exists():
            try: abs_path_to_clean.unlink(); logger.info(f"FIXTURE_TEARDOWN: Cleaned up {abs_path_to_clean}")
            except OSError: pass # Ignore if cannot delete

    # Clean PID file if it was created by the test setup explicitly
    pid_dir_val_cleanup = config_override["agent_data_paths"].get("pid_directory")
    pid_file_name_cleanup = config_override["agent"].get("pid_file_name")
    if pid_dir_val_cleanup and pid_file_name_cleanup:
        pid_file_abs_cleanup = tmp_path / Path(pid_dir_val_cleanup) / pid_file_name_cleanup
        if pid_file_abs_cleanup.exists():
            try: pid_file_abs_cleanup.unlink(); logger.info(f"FIXTURE_TEARDOWN: Cleaned up PID file {pid_file_abs_cleanup}")
            except OSError: pass
            
    if dummy_config_file.exists():
        try: dummy_config_file.unlink(); logger.info(f"FIXTURE_TEARDOWN: Cleaned up dummy config {dummy_config_file}")
        except OSError: pass


    logger.info(f"FIXTURE_TEARDOWN: test_agent for {tmp_path} teardown complete.")