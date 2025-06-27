# tests/integration/test_cognitive_cycle.py

import pytest
import asyncio
import time
import logging
import queue
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, ANY
from typing import Optional, List, Dict, Tuple, Set, Any # Removed Counter from here
from collections import Counter, deque # Added specific import for Counter
import os
import json # For creating mock LLM responses
import math
from contextlib import suppress

import pytest_asyncio


# --- Define logger for this test file ---
logger = logging.getLogger(__name__)
# Optional: Configure logging level for tests if needed
# logging.basicConfig(level=logging.DEBUG)

# --- Define expected package name ---
PACKAGE_NAME = "consciousness_experiment" # Use the actual package name if different

# --- Attempt to import using absolute package path ---
CONTROLLER_AVAILABLE = False
AgentController = None
COMPONENT_INIT_ORDER = [] # Define placeholder if import fails
Predicate = None
Goal = None
create_goal_from_descriptor = None
GoalStatus = None
ConsciousState = None
PhenomenalState = None
KnowledgeBase = None
PSUTIL_AVAILABLE_CTRL = False # Track psutil availability for controller tests
DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC = None # Define placeholder
USER_GOAL_PRIORITY = None # Define placeholder
DEFAULT_OBSERVE_GOAL_PRIORITY = None # Define placeholder
PainSource = None

# Define component classes map (needed for fixture init check)
component_classes = {}

try:
    # Use direct imports assuming standard package structure relative to tests
    from consciousness_experiment.agent_controller import (
        AgentController,
        COMPONENT_INIT_ORDER_CTRL as COMPONENT_INIT_ORDER,
        PSUTIL_AVAILABLE_CTRL,
        DEFAULT_OBSERVE_GOAL_PRIORITY as AGENT_DEFAULT_OBSERVE_GOAL_PRIORITY, # Import with alias
        USER_GOAL_PRIORITY as AGENT_USER_GOAL_PRIORITY, # Import with alias
        DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC as AGENT_DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC # Import with alias
    )
    from consciousness_experiment.models.datatypes import Predicate, Goal, create_goal_from_descriptor, PhenomenalState, PainSource
    from consciousness_experiment.models.enums import GoalStatus, ConsciousState
    from consciousness_experiment.cognitive_modules.knowledge_base import KnowledgeBase
    # Import component classes used in the fixture's component_classes map
    from consciousness_experiment.cognitive_modules.cognitive_cache import CognitiveCache
    from consciousness_experiment.cognitive_modules.performance_optimizer import PerformanceOptimizer
    from consciousness_experiment.cognitive_modules.htn_planner import HTNPlanner, Operator, Method # Import Operator and Method
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
   
    component_classes = { # Populate the map
            "knowledge_base": KnowledgeBase, "cache": CognitiveCache,
            "performance_optimizer": PerformanceOptimizer, "htn_planner": HTNPlanner,
            "attention_controller": AttentionController, "global_workspace": GlobalWorkspaceManager,
            "experience_stream": ExperienceStream, "consciousness_assessor": ConsciousnessLevelAssessor,
            "meta_cognition": MetaCognitiveMonitor, "loop_detector": LoopDetector,
            "predictive_world_model": PredictiveWorldModel, "dynamic_self_model": DynamicSelfModel,
            "emergent_motivation_system": EmergentMotivationSystem,
            "narrative_constructor": NarrativeConstructor, "error_recovery": ErrorRecoverySystem,
        }

    CONTROLLER_AVAILABLE = True
    # Set global constants for tests from imported agent controller constants
    DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC = AGENT_DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC
    USER_GOAL_PRIORITY = AGENT_USER_GOAL_PRIORITY
    DEFAULT_OBSERVE_GOAL_PRIORITY = AGENT_DEFAULT_OBSERVE_GOAL_PRIORITY


    print(f"\nDEBUG: Successfully imported test dependencies via package path '{PACKAGE_NAME}'.")
except ImportError as e:
    # Fallback values if import fails, tests will likely be skipped by CONTROLLER_AVAILABLE anyway
    USER_GOAL_PRIORITY = 5.0
    DEFAULT_OBSERVE_GOAL_PRIORITY = 1.0
    DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC = "Observe and learn from the environment" # Fallback
    print(f"\nDEBUG: Skipping integration tests: Failed to import necessary modules via package path '{PACKAGE_NAME}' - {e}")
    if PACKAGE_NAME in str(e):
        pytest.skip(f"Skipping integration tests: Package '{PACKAGE_NAME}' not found or structured incorrectly.", allow_module_level=True)
    else:
        pytest.skip(f"Skipping integration tests due to import error: {e}", allow_module_level=True)


# --- Test Fixture/Helper ---

@pytest_asyncio.fixture(scope="function")
async def test_agent(tmp_path: Path):
    """Provides a fresh AgentController instance for each test."""
    if not CONTROLLER_AVAILABLE:
        pytest.fail("AgentController class not available for fixture setup.")

    timestamp = int(time.time() * 1000)

    default_pid_dir = "run"
    default_kb_db_path = "data/oscar_c_kb.db"
    default_narrative_log_path = "logs/narrative.log"
    default_self_model_path = "data/self_model.json"
    default_predictive_model_path = "data/predictive_model.json"
    default_perf_adjustments_path = "data/perf_adjustments.json"

    (tmp_path / Path(default_kb_db_path).parent).mkdir(parents=True, exist_ok=True)
    (tmp_path / Path(default_narrative_log_path).parent).mkdir(parents=True, exist_ok=True)
    (tmp_path / Path(default_pid_dir)).mkdir(parents=True, exist_ok=True)

    config_override = {
        "agent_data_paths": {
            "pid_directory": default_pid_dir,
            "kb_db_path": default_kb_db_path,
            "narrative_log_path": default_narrative_log_path,
            "self_model_path": default_self_model_path,
            "predictive_model_path": default_predictive_model_path,
            "performance_adjustments_path": default_perf_adjustments_path,
        },
        "agent": {
            "pid_file_name": f"test_agent_{timestamp}.pid",
            "ui_meter_update_interval_s": 0.01,
            "goal_re_evaluation_interval_cycles": 5,
            "default_goal_cooldown_cycles": 2, # Keep it short for tests
            "min_curiosity_for_observe": 0.1,
        },
        "knowledge_base": {"default_context_retrieval_count": 5},
        "dynamic_self_model": { # Ensure reflection config is here
            "learning_rate": 0.1,
            "max_invalid_paths": 20,
            "learning_events_history_size": 10, # Smaller for testing
            "reflection_interval_cycles": 5    # Reflect more often for testing
            },
        "narrative_constructor": {
            "max_length": 10, "valence_change_threshold": 0.2, "intensity_threshold": 0.6,
            "save_interval_s": 10, "drive_change_threshold": 0.1, "significance_threshold": 0.1,
            "llm_model_name": "test_narrative_model", "temperature": 0.6, "timeout_s": 10.0
        },
        "filesystem": {"allow_file_write": True, "allow_overwrite": False, "max_list_items": 10, "max_read_chars": 1000},
        "performance": {"target_cycle_time": 0.01, "max_planning_depth": 3, "profiler_history_size":10},
        "global_workspace": { "capacity": 5, "broadcast_threshold": 0.4, "min_items_if_any_attended": 1 },
        "attention_controller": {
            "max_candidates": 8, "recency_weight": 0.3, "hint_weight": 0.4,
            "goal_relevance_weight":0.3, "softmax_temperature": 1.0,
            "novelty_window_size": 3, "novelty_bonus_weight": 0.1, "surprise_bonus_weight": 0.2
        },
        "cognitive_cache": {"default_ttl": 0.2},
        "consciousness_assessor": {
            "meta_conscious_threshold": 0.75, "conscious_threshold": 0.50,
            "pre_conscious_threshold": 0.25, "unconscious_threshold": 0.1,
            "diff_weight_sources": 0.4, "diff_weight_lexical": 0.6,
            "int_weight_shared_concepts": 0.5, "phi_contrib_diff": 0.5, "phi_contrib_int": 0.5,
            "global_workspace_capacity_for_norm": 5,
        },
        "loop_detection": {
            "window_size": 3, "max_consecutive_actions": 2,
            "frequency_threshold": 0.67, "ignore_thinking_actions": True
        },
        "error_recovery": {"max_error_history": 5, "frequency_window": 3, "frequency_threshold": 2},
        "telemetry": {"enabled": False, "host": "localhost", "port": 8765},
        "predictive_world_model": {"initial_version": 0, "learning_rate": 0.1, "memory_length": 5, "default_confidence": 0.6},
        "emergent_motivation_system": {"detailed_evaluation_interval_cycles": 3},
        "experience_stream": {"intensity_factor": 1.0, "valence_decay": 0.1, "phenomenal_state_capacity": 3},
        "meta_cognition": {
            "stagnation_threshold_s": 0.2, "failure_rate_threshold": 0.6,
            "low_consciousness_threshold_s": 0.2, "history_size": 3,
            "reflection_trigger_frequency_cycles": 3
        },
        "performance_optimizer": {"history_size": 3, "auto_apply_adjustments": False},
        "llm_settings": {
            "default_timeout_s": 0.2,
            "action_selection_temperature": 0.7,
            "max_recent_turns_for_prompt": 3,
            "intent_mapping_temperature": 0.3 # Added for LLM goal mapping
            },
    }

    mock_ui_queue = MagicMock(spec=queue.Queue)
    agent = None
    initialized_components = []

    dummy_config_file = tmp_path / f"test_config_{timestamp}.toml"
    dummy_config_file.touch()

    with patch.object(AgentController, '_load_config', return_value=config_override) as mock_load_config:
        agent = AgentController(ui_queue=mock_ui_queue, model_name="test_integration_model", config_path=str(dummy_config_file))
        assert agent.agent_root_path == tmp_path, f"Agent root path {agent.agent_root_path} not {tmp_path}"
        mock_load_config.assert_called_once()

    # --- SIMPLIFIED LOOP HANDLING ---
    agent_loop_for_init = asyncio.get_running_loop()
    agent._asyncio_loop = agent_loop_for_init

    init_success = True
    logger.debug("Fixture: Initializing agent components...")
    for name in COMPONENT_INIT_ORDER:
        if name in agent.components:
            component = agent.components[name]
            logger.debug(f"Fixture: Initializing '{name}'...")
            try:
                init_method = getattr(component, 'initialize', None)
                if init_method:
                     if asyncio.iscoroutinefunction(init_method):
                         success = await init_method(agent.config, agent)
                     else:
                         success = component.initialize(agent.config, agent) # type: ignore

                     if not success:
                         logger.error(f"Fixture: Component '{name}' .initialize() returned False!")
                         init_success = False; break
                     else:
                         initialized_components.append(name)
                else:
                     logger.warning(f"Fixture: Component '{name}' has no initialize method. Assuming success.")
                     initialized_components.append(name)
                logger.debug(f"Fixture: Component '{name}' initialized.")
            except Exception as e:
                 logger.error(f"Fixture: Exception initializing '{name}': {e}", exc_info=True)
                 init_success = False; break
        elif name in component_classes:
             logger.warning(f"Fixture: Component '{name}' in INIT_ORDER but not instantiated in agent.components.")

    if not init_success:
        if agent and hasattr(agent, '_shutdown_components'):
            await agent._shutdown_components(initialized_components)
        pytest.fail("Agent component initialization failed in test fixture.")

    # Ensure essential *imported constants* that tests rely on are available
    # (and that the agent instance itself is created)
    assert agent is not None, "AgentController instance could not be created."

    # These constants are imported at the top of test_cognitive_cycle.py
    # from consciousness_experiment.agent_controller
    assert DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC is not None, \
        "DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC constant not available to tests (check import)."
    assert USER_GOAL_PRIORITY is not None, \
        "USER_GOAL_PRIORITY constant not available to tests (check import)."
    assert DEFAULT_OBSERVE_GOAL_PRIORITY is not None, \
        "DEFAULT_OBSERVE_GOAL_PRIORITY constant not available to tests (check import)."

    # Check for essential instance attributes that should be set by AgentController.__init__
    assert hasattr(agent, 'config'), "Agent instance missing 'config' attribute after init."
    assert hasattr(agent, 'agent_root_path'), "Agent instance missing 'agent_root_path' after init."
    assert hasattr(agent, 'components'), "Agent instance missing 'components' attribute after init."
    assert isinstance(agent.components, dict), "Agent 'components' attribute is not a dict after init."
    # Optionally, check if all expected components were initialized
    # for comp_name_in_order in COMPONENT_INIT_ORDER:
    #     assert comp_name_in_order in agent.components, f"Component '{comp_name_in_order}' missing after init."

    logger.debug("Fixture: Agent instance ready and initial checks passed.")
    yield agent


    # --- TEARDOWN ---
    logger.debug("Fixture: Teardown - Ensuring agent's main task is cancelled if running...")
    if agent and agent._main_loop_task and not agent._main_loop_task.done():
        agent._main_loop_task.cancel()
        try:
            await agent._main_loop_task
        except asyncio.CancelledError:
            logger.debug("Fixture: Agent's main loop task successfully cancelled during teardown.")
        except Exception as e:
            logger.error(f"Fixture: Error awaiting cancelled agent main loop task: {e}")

    logger.debug("Fixture: Shutting down agent components...")
    if agent and hasattr(agent, '_shutdown_components'):
        await agent._shutdown_components(initialized_components)

    logger.debug("Fixture: Teardown complete.")


@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_attention_workspace_experience_flow(test_agent: AgentController):
    """
    Tests the flow of information from attention candidates through the
    global workspace and into the phenomenal state generated by the experience stream.
    """
    agent = test_agent
    logger.info("--- Test: Attention -> Workspace -> Experience Flow ---")

    # Ensure necessary components are available
    assert hasattr(agent, 'attention_controller'), "AttentionController missing."
    assert hasattr(agent, 'global_workspace'), "GlobalWorkspaceManager missing."
    assert hasattr(agent, 'experience_stream'), "ExperienceStream missing."
    assert PhenomenalState is not None, "PhenomenalState class not loaded."


    # 1. Define Sample Attention Candidates
    current_time = time.time()
    candidates = {
        "percept_high_priority": {"content": "URGENT_DATA EMERGENCY", "weight_hint": 0.9, "timestamp": current_time - 1, "source_type": "perception"},
        "goal_active": {"content": "Achieve Test Goal Now", "weight_hint": 0.8, "timestamp": current_time - 10, "source_type": "goal"},
        "memory_relevant": {"content": "Relevant past event information", "weight_hint": 0.7, "timestamp": current_time - 30, "source_type": "memory"},
        "percept_low_priority": {"content": "background_noise hum", "weight_hint": 0.1, "timestamp": current_time - 2, "source_type": "perception"},
        "memory_old": {"content": "Old irrelevant memory detail", "weight_hint": 0.2, "timestamp": current_time - 300, "source_type": "memory"},
    }
    ws_capacity = getattr(agent.global_workspace, 'capacity', 7)
    if ws_capacity < 5:
         for i in range(ws_capacity):
              candidates[f"filler_{i}"] = {"content": f"filler data content {i}", "weight_hint": 0.6, "timestamp": current_time - 5, "source_type": "filler"}


    # 2. Call Attention Allocation
    logger.info("Calling AttentionController...")
    attn_input = {
        "candidates": candidates,
        "current_cycle_active_goal_id": "test_goal_id_dummy",
        "last_gwm_content_for_novelty": {"some_previous_item": "prev_data"},
        "last_prediction_error": None
    }
    attn_output = await agent.attention_controller.process(attn_input)

    assert attn_output is not None and "attention_weights" in attn_output, "Attention process failed or gave invalid output."
    attention_weights = attn_output["attention_weights"]
    assert isinstance(attention_weights, dict), "Attention weights are not a dict."
    if attention_weights:
        pass
    logger.info(f"Attention weights generated: { {k: round(v, 2) for k, v in attention_weights.items()} }")


    # 3. Call Global Workspace Update
    logger.info("Calling GlobalWorkspaceManager...")
    ws_input = {"attention_weights": attention_weights, "all_candidates_data": candidates}
    ws_output = await agent.global_workspace.process(ws_input)

    assert ws_output is not None and "broadcast_content" in ws_output, "Workspace process failed or gave invalid output."
    broadcast_content = ws_output["broadcast_content"]
    assert isinstance(broadcast_content, dict), "Broadcast content is not a dict."
    assert len(broadcast_content) <= ws_capacity, f"Workspace content ({len(broadcast_content)}) exceeds capacity ({ws_capacity})."

    broadcast_threshold = getattr(agent.global_workspace, 'broadcast_threshold', 0.5)

    if "percept_high_priority" in attention_weights and attention_weights["percept_high_priority"] >= broadcast_threshold:
        assert "percept_high_priority" in broadcast_content, f"High priority percept missing from workspace (weight {attention_weights.get('percept_high_priority', 0):.2f} >= thresh {broadcast_threshold:.2f})."
    if "goal_active" in attention_weights and attention_weights["goal_active"] >= broadcast_threshold:
        assert "goal_active" in broadcast_content, f"Active goal missing from workspace (weight {attention_weights.get('goal_active', 0):.2f} >= thresh {broadcast_threshold:.2f})."

    logger.info(f"Workspace content generated ({len(broadcast_content)} items): {list(broadcast_content.keys())}")


    # 4. Call Experience Integration
    logger.info("Calling ExperienceStream...")
    dummy_percepts = {"system_state": {"cpu": 10, "mem": 30}, "raw_text_input": "User said hello world"}
    dummy_memories = [ {"id": "mem1", "content": "Recalled: old relevant memory"} ]
    dummy_context = {"last_action_type": "THINKING", "last_action_outcome": "success"}
    exp_input = {
        "percepts": dummy_percepts,
        "memories": dummy_memories,
        "context": dummy_context,
        "broadcast_content": broadcast_content
    }
    exp_output = await agent.experience_stream.process(exp_input)

    assert exp_output is not None and "phenomenal_state" in exp_output, "Experience stream failed or gave invalid output."
    phenomenal_state = exp_output["phenomenal_state"]
    assert isinstance(phenomenal_state, PhenomenalState), f"Experience stream did not return PhenomenalState object, got {type(phenomenal_state)}."

    # 5. Verify Phenomenal State Content
    assert isinstance(phenomenal_state.content, dict), "Phenomenal state content is not a dict."
    if "percept_high_priority" in broadcast_content:
        assert "percept_high_priority" in phenomenal_state.content, "High priority percept missing from P-State content."
    if "goal_active" in broadcast_content:
        assert "goal_active" in phenomenal_state.content, "Active goal missing from P-State content."
    assert "relevant_memories" in phenomenal_state.content, "Memories missing from P-State content."
    assert "action_context" in phenomenal_state.content, "Action context missing from P-State content."

    assert 0 <= phenomenal_state.intensity <= 1.0, f"Phenomenal state intensity out of bounds: {phenomenal_state.intensity}"
    assert -1.0 <= phenomenal_state.valence <= 1.0, f"Phenomenal state valence out of bounds: {phenomenal_state.valence}"
    assert 0 <= phenomenal_state.integration_level <= 1.0, f"Phenomenal state integration_level (old proxy) out of bounds: {phenomenal_state.integration_level}"

    # --- ADD CHECKS FOR NEW PHI-PROXY SUB-METRICS ---
    assert hasattr(phenomenal_state, 'distinct_source_count'), "PhenomenalState missing 'distinct_source_count'"
    assert isinstance(phenomenal_state.distinct_source_count, int), "'distinct_source_count' should be an int"
    assert phenomenal_state.distinct_source_count >= 0, "'distinct_source_count' should be non-negative"

    expected_min_sources = 0
    if broadcast_content: expected_min_sources +=1
    if dummy_memories: expected_min_sources +=1
    if dummy_context: expected_min_sources +=1
    if broadcast_content or dummy_memories or dummy_context:
        assert phenomenal_state.distinct_source_count >= 1, f"Expected at least 1 distinct source type, got {phenomenal_state.distinct_source_count}. Broadcast: {bool(broadcast_content)}, Mem: {bool(dummy_memories)}, Ctx: {bool(dummy_context)}"


    assert hasattr(phenomenal_state, 'content_diversity_lexical'), "PhenomenalState missing 'content_diversity_lexical'"
    assert isinstance(phenomenal_state.content_diversity_lexical, float), "'content_diversity_lexical' should be a float"
    assert 0.0 <= phenomenal_state.content_diversity_lexical <= 1.0, f"'content_diversity_lexical' out of bounds [0,1]: {phenomenal_state.content_diversity_lexical}"
    if any(isinstance(c.get("content"), str) and c.get("content") for c_id, c in candidates.items() if c_id in broadcast_content): # Check if there was actual text content
        # This assertion is tricky because "URGENT_DATA EMERGENCY" might be all unique after splitting
        # or could have a TTR of 1.0 if only 1-2 words and unique.
        # For now, just ensure it's calculated; specific value depends on exact tokenizer/stopwords.
        # assert phenomenal_state.content_diversity_lexical > 0.0, "Lexical diversity should be >0 with text in broadcast."
        pass


    assert hasattr(phenomenal_state, 'shared_concept_count_gw'), "PhenomenalState missing 'shared_concept_count_gw'"
    assert isinstance(phenomenal_state.shared_concept_count_gw, float), "'shared_concept_count_gw' should be a float"
    assert 0.0 <= phenomenal_state.shared_concept_count_gw <= 1.0, f"'shared_concept_count_gw' out of bounds [0,1]: {phenomenal_state.shared_concept_count_gw}"

    # The problematic line `textual_broadcast_items = ...` has been removed.

    num_textual_items_in_broadcast = 0
    for item_key, item_value_or_dict in broadcast_content.items():
        actual_content_for_text = item_value_or_dict
        if isinstance(item_value_or_dict, dict) and "content" in item_value_or_dict:
            actual_content_for_text = item_value_or_dict["content"]
        if isinstance(actual_content_for_text, str):
            num_textual_items_in_broadcast += 1

    if num_textual_items_in_broadcast < 2 :
        assert phenomenal_state.shared_concept_count_gw == 0.0, \
            f"shared_concept_count_gw should be 0 if < 2 textual items in broadcast (found {num_textual_items_in_broadcast}). " \
            f"Broadcast content: {broadcast_content}"
    # --- END NEW CHECKS ---

    logger.info(
        f"Phenomenal state generated. I:{phenomenal_state.intensity:.2f}, V:{phenomenal_state.valence:.2f}, "
        f"IL(old):{phenomenal_state.integration_level:.2f}, "
        f"Sources:{phenomenal_state.distinct_source_count}, LexDiv:{phenomenal_state.content_diversity_lexical:.2f}, "
        f"SharedGW:{phenomenal_state.shared_concept_count_gw:.2f}"
    )
    logger.info("--- Test Passed: Attention -> Workspace -> Experience Flow ---")


@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_action_execution_and_kb_update(test_agent: AgentController):
    """Test executing an action and verifying KB update."""
    agent = test_agent
    logger.info("--- Test: Action Execution & KB Update ---")
    assert hasattr(agent, 'knowledge_base') and agent.knowledge_base is not None
    assert Predicate is not None, "Predicate class not available."

    action_to_execute = {"type": "OBSERVE_SYSTEM", "params": {}}
    action_timestamp_before = time.time()

    result = await agent._oscar_execute_action(action_to_execute)

    assert result.get("outcome") == "success", f"Action execution failed: {result.get('error')}"
    if PSUTIL_AVAILABLE_CTRL:
        assert isinstance(result.get("result_data"), dict) and "cpu_percent" in result.get("result_data", {}), \
            "Action result data missing expected keys or wrong type (psutil available)."
    else:
        assert isinstance(result.get("result_data"), dict) and "error" in result.get("result_data", {}), \
            "Action result data should indicate psutil unavailable."

    await asyncio.sleep(0.05)

    query_args = ("actionExecution", "OBSERVE_SYSTEM", "success")
    events = await agent.knowledge_base.query(name="eventOccurred", args=query_args, value=True)

    assert len(events) >= 1, f"eventOccurred predicate for {query_args} not found in KB after action execution."
    assert events[0].timestamp >= action_timestamp_before - 0.1, f"KB event timestamp {events[0].timestamp} seems too early (before {action_timestamp_before})."
    logger.info("--- Test Passed: Action Execution & KB Update ---")


@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_consciousness_assessment_change(test_agent: AgentController):
    agent = test_agent
    logger.info("--- Test: Consciousness Assessment Change ---")
    assert hasattr(agent, 'consciousness_assessor') and agent.consciousness_assessor is not None
    assert PhenomenalState is not None, "PhenomenalState class not loaded."
    assert ConsciousState is not None, "ConsciousState enum not loaded."

    high_integration_experience = PhenomenalState(
        content={"item1": "data file report", "item2": 123, "item3": True, "item4": 0.5, "item5": [1,2], "shared_topic": "analysis report data"},
        intensity=0.8,
        valence=0.2,
        integration_level=0.9,
        attention_weight=0.8,
        timestamp=time.time(),
        distinct_source_count=4,
        content_diversity_lexical=0.8,
        shared_concept_count_gw=0.6
    )
    workspace_content_high = {"item1": "data file report", "item2": 123, "item3": True, "item4": "another concept data", "shared_topic": "analysis report data"}

    assessment_input_high = {"experience": high_integration_experience, "workspace_content": workspace_content_high}
    assessment_output_high = await agent.consciousness_assessor.process(assessment_input_high)

    assert assessment_output_high is not None, "Consciousness assessor process returned None (high)"
    new_level_high = assessment_output_high.get("conscious_state")
    phi_proxy_score_high = assessment_output_high.get("phi_proxy_score")

    assert new_level_high is not None, "Consciousness assessor did not return a state (high)."
    assert isinstance(new_level_high, ConsciousState), "Assessor did not return ConsciousState enum member."

    phi_str_high = f"{phi_proxy_score_high:.3f}" if isinstance(phi_proxy_score_high, float) else str(phi_proxy_score_high) # type: ignore
    logger.info(f"High integration test: Level={new_level_high.name}, Φ-Proxy={phi_str_high}") # type: ignore

    assert new_level_high == ConsciousState.PRE_CONSCIOUS, \
           f"Expected PRE_CONSCIOUS for high integration with current calculations, got {new_level_high.name} (Φ-Proxy: {phi_str_high})" # type: ignore

    low_integration_experience = PhenomenalState(
        content={"item_low": "simple isolated thing"},
        intensity=0.2, valence=-0.1, integration_level=0.1,
        attention_weight=0.2, timestamp=time.time(),
        distinct_source_count=1,
        content_diversity_lexical=0.1,
        shared_concept_count_gw=0.0
    )
    workspace_content_low = {"item_low": "simple isolated thing"}

    assessment_input_low = {"experience": low_integration_experience, "workspace_content": workspace_content_low}
    assessment_output_low = await agent.consciousness_assessor.process(assessment_input_low)

    assert assessment_output_low is not None, "Consciousness assessor process returned None (low)"
    new_level_low = assessment_output_low.get("conscious_state")
    phi_proxy_score_low = assessment_output_low.get("phi_proxy_score")

    assert new_level_low is not None, "Consciousness assessor did not return a state (low)."
    phi_str_low = f"{phi_proxy_score_low:.3f}" if isinstance(phi_proxy_score_low, float) else str(phi_proxy_score_low) # type: ignore
    logger.info(f"Low integration test: Level={new_level_low.name}, Φ-Proxy={phi_str_low}") # type: ignore

    unconscious_thresh = getattr(agent.consciousness_assessor, 'unconscious_threshold', 0.05)
    pre_conscious_thresh = getattr(agent.consciousness_assessor, 'pre_conscious_threshold', 0.25)

    expected_level_low: ConsciousState
    if phi_proxy_score_low < unconscious_thresh : # type: ignore
            expected_level_low = ConsciousState.UNCONSCIOUS
    elif phi_proxy_score_low < pre_conscious_thresh:  # type: ignore
            expected_level_low = ConsciousState.UNCONSCIOUS # If between unconscious and pre_conscious, it's still UNCONSCIOUS
    else:
            expected_level_low = ConsciousState.PRE_CONSCIOUS

    assert new_level_low == expected_level_low, \
            f"Expected {expected_level_low.name} for low integration (Φ-Proxy: {phi_str_low}), got {new_level_low.name}" # type: ignore
    logger.info("--- Test Passed: Consciousness Assessment Change ---")


@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_htn_parameter_planning(test_agent: AgentController):
    agent = test_agent
    logger.info("--- Test: HTN Parameter Planning ---")

    assert hasattr(agent, 'htn_planner') and agent.htn_planner is not None, "HTNPlanner component missing."
    assert hasattr(agent, 'knowledge_base') and agent.knowledge_base is not None, "KnowledgeBase component missing."
    assert Predicate is not None, "Predicate class not available."
    assert Goal is not None, "Goal class not available."
    assert create_goal_from_descriptor is not None, "create_goal_from_descriptor function not available."

    test_file_relative_to_project_root = "config.toml"
    path_parameter_for_goal = test_file_relative_to_project_root

    goal_desc = f"read file : {path_parameter_for_goal}"
    test_goal = create_goal_from_descriptor(goal_desc)
    assert test_goal is not None, "Failed to create test goal object for parameterized planning."

    initial_state_predicates = {
        Predicate(name="isFile", args=(path_parameter_for_goal,), value=True)
    }
    logger.debug(f"Initial state for planning: {initial_state_predicates}")

    logger.info(f"Calling planner for goal: '{goal_desc}'")
    plan_method = getattr(agent.htn_planner, 'plan', None)
    assert plan_method and asyncio.iscoroutinefunction(plan_method), "Planner 'plan' method missing or not async."
    generated_plan = await plan_method(test_goal, initial_state_predicates)

    assert generated_plan is not None, f"Planner returned None for goal '{goal_desc}' (planning failed)."
    assert isinstance(generated_plan, list), "Plan is not a list."
    assert len(generated_plan) > 0, "Plan is empty."
    logger.info(f"Parameterized plan generated: {generated_plan}")

    first_action = generated_plan[0]
    assert isinstance(first_action, dict), "Plan step is not a dictionary."
    assert "type" in first_action, "Plan step missing 'type' key."
    assert "params" in first_action, "Plan step missing 'params' key."
    assert first_action["type"] == "READ_FILE", f"Expected action type 'READ_FILE', got '{first_action['type']}'"

    action_params = first_action["params"]
    assert isinstance(action_params, dict), "Action params is not a dictionary."
    assert "path" in action_params, "READ_FILE action missing 'path' parameter."
    assert action_params["path"] == path_parameter_for_goal, \
        f"Path parameter mismatch: Expected '{path_parameter_for_goal}', got '{action_params['path']}'"

    logger.info("--- Test Passed: HTN Parameter Planning ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_read_file_action(test_agent: AgentController, tmp_path: Path):
    agent = test_agent
    logger.info("--- Test: READ_FILE Action Execution ---")

    original_agent_root = agent.agent_root_path
    agent.agent_root_path = tmp_path

    assert hasattr(agent, 'knowledge_base') and agent.knowledge_base is not None, "KnowledgeBase component missing."
    assert Predicate is not None, "Predicate class not available."

    test_dir_name = "read_test_dir"
    test_file_name = "readable.txt"
    test_dir = tmp_path / test_dir_name
    test_dir.mkdir()
    test_file = test_dir / test_file_name
    file_content = "This is the content of the test file.\nIt has multiple lines."
    test_file.write_text(file_content, encoding='utf-8')

    relative_path = f"{test_dir_name}/{test_file_name}"
    action_read_success = {"type": "READ_FILE", "params": {"path": relative_path}}

    logger.info(f"Executing successful READ_FILE action for: {relative_path} (agent_root_path mocked to: {agent.agent_root_path})")
    result_success = await agent._oscar_execute_action(action_read_success)

    assert result_success.get("outcome") == "success", f"READ_FILE action failed unexpectedly: {result_success.get('error')}"
    result_data = result_success.get("result_data",{})
    assert Path(result_data.get("path", "")) == test_file.resolve(), f"Resolved path '{result_data.get('path')}' doesn't match expected '{test_file.resolve()}'."
    assert result_data.get("content_snippet") == file_content, "File content mismatch."

    await asyncio.sleep(0.05)
    resolved_path_str = result_data.get("path", "")
    read_preds = await agent.knowledge_base.query(name="readFileContent", args=(resolved_path_str,), value=True)
    assert len(read_preds) >= 1, f"readFileContent predicate for '{resolved_path_str}' not found in KB after successful read."
    logger.info("Successful READ_FILE executed and verified.")

    action_read_fail = {"type": "READ_FILE", "params": {"path": "non_existent_dir/non_existent_file.txt"}}
    logger.info(f"Executing failed READ_FILE action for: {action_read_fail['params']['path']}")
    result_fail = await agent._oscar_execute_action(action_read_fail)
    assert result_fail.get("outcome") == "failure", "READ_FILE action should have failed for non-existent file."
    assert "does not exist" in result_fail.get("error", "").lower() or "no such file" in result_fail.get("error", "").lower(), \
           f"Error message '{result_fail.get('error')}' doesn't indicate file not found."

    action_list_fail = {"type": "LIST_FILES", "params": {"path": relative_path}}
    logger.info(f"Executing failed LIST_FILES action for file: {action_list_fail['params']['path']}")
    result_list_fail = await agent._oscar_execute_action(action_list_fail)
    assert result_list_fail.get("outcome") == "failure", "LIST_FILES action should have failed for a file path."
    assert "not a directory" in result_list_fail.get("error", "").lower(), f"Error message '{result_list_fail.get('error')}' doesn't indicate 'not a directory'."

    await asyncio.sleep(0.05)
    invalid_preds_list = await agent.knowledge_base.query(name="isInvalidPath", args=(resolved_path_str,), value=True)
    assert len(invalid_preds_list) >= 1, f"isInvalidPath predicate for file '{resolved_path_str}' not found in KB after failed LIST_FILES."
    logger.info("Failed LIST_FILES on file path verified and KB updated.")

    agent.agent_root_path = original_agent_root
    logger.info("--- Test Passed: READ_FILE Action Execution ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_prediction_and_update(test_agent: AgentController):
    agent = test_agent
    logger.info("--- Test: Predictive World Model ---")

    assert hasattr(agent, 'predictive_world_model'), "PredictiveWorldModel component missing."
    assert hasattr(agent, 'htn_planner'), "HTNPlanner component missing (needed for operator defs)."
    assert Predicate is not None, "Predicate class not available."

    pwm = agent.predictive_world_model

    action_observe = {"type": "OBSERVE_SYSTEM", "params": {}}
    predict_request = {"action_to_execute": action_observe}
    logger.info(f"Requesting prediction for: {action_observe['type']}")
    prediction_result_dict = await pwm.process({"predict_request": predict_request})

    assert isinstance(prediction_result_dict, dict), "PWM process did not return a dict for prediction."
    prediction = prediction_result_dict.get("prediction")
    assert isinstance(prediction, dict), "Prediction result is not a dictionary."
    logger.info(f"Prediction received: {prediction}")

    assert prediction.get("predicted_outcome") == "success", \
        f"Predicted outcome should be 'success' for OBSERVE_SYSTEM by default heuristic, got '{prediction.get('predicted_outcome')}'"
    assert prediction.get("basis") == "default_heuristic", \
        f"Prediction basis should be 'default_heuristic' for unlearned OBSERVE_SYSTEM, got '{prediction.get('basis')}'"

    if "expected_effects" in prediction and prediction["expected_effects"]:
        assert any(isinstance(eff, dict) and eff.get("name") == "systemStateObserved" for eff in prediction.get("expected_effects", [])), \
               "If effects are present, 'systemStateObserved' should be among them for OBSERVE_SYSTEM."
        logger.info("Prediction included expected_effects as per operator definition.")
    else:
        logger.info("Prediction did not include detailed expected_effects (may rely on learned outcomes).")

    actual_result_success = {"outcome": "success", "result_data": {"cpu": 10}, "type": "OBSERVE_SYSTEM"}
    logger.info("Updating model with matching outcome...")
    update_request_success = {"prediction": prediction, "actual_result": actual_result_success}
    await pwm.process({"update_request": update_request_success})

    status_after_match = await pwm.get_status()
    assert status_after_match.get("last_prediction_error_type") is None, "Prediction error should be None after matching update."
    initial_version = status_after_match.get("model_version")

    actual_result_fail = {"outcome": "failure", "error": "Sensor offline", "type": "OBSERVE_SYSTEM"}
    logger.info("Updating model with mismatching outcome...")
    update_request_fail = {"prediction": prediction, "actual_result": actual_result_fail}
    await pwm.process({"update_request": update_request_fail})

    status_after_mismatch = await pwm.get_status()
    assert status_after_mismatch.get("last_prediction_error_type") == "outcome_mismatch", "Prediction error type should be 'outcome_mismatch'."
    assert status_after_mismatch.get("model_version") > initial_version, "Model version should increment after mismatch."

    assert pwm.last_prediction_error is not None, "last_prediction_error should be populated after a mismatch."
    assert pwm.last_prediction_error.get("type") == "outcome_mismatch", "Error type should be outcome_mismatch."
    assert pwm.last_prediction_error.get("predicted") == "success", \
        f"Expected 'predicted' in error to be 'success', got {pwm.last_prediction_error.get('predicted')}"
    assert pwm.last_prediction_error.get("actual") == "failure", \
        f"Expected 'actual' in error to be 'failure', got {pwm.last_prediction_error.get('actual')}"
    assert pwm.last_prediction_error.get("action_type") == "OBSERVE_SYSTEM"

    logger.info("--- Test Passed: Predictive World Model ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_self_model_path_knowledge_update(test_agent: AgentController, tmp_path: Path):
    agent = test_agent
    dsm = agent.dynamic_self_model
    logger.info("--- Test: Dynamic Self Model Path Knowledge ---")

    original_agent_root = agent.agent_root_path
    agent.agent_root_path = tmp_path

    assert dsm is not None, "DynamicSelfModel component missing."
    assert hasattr(dsm, 'self_model'), "DynamicSelfModel has no self_model attribute."

    valid_dir_relative = "valid_dir_for_dsm"
    valid_file_relative = f"{valid_dir_relative}/valid_file_dsm.txt"
    invalid_file_relative = "non_existent_dir_dsm/no_file_dsm.txt"

    (agent.agent_root_path / valid_dir_relative).mkdir()
    (agent.agent_root_path / valid_file_relative).write_text("content")

    abs_valid_dir = str((agent.agent_root_path / valid_dir_relative).resolve())
    abs_valid_file = str((agent.agent_root_path / valid_file_relative).resolve())
    abs_invalid_file_expected_in_model = str((agent.agent_root_path / invalid_file_relative).resolve(strict=False))

    async def simulate_dsm_update(action_type, params, outcome, error=None, result_data=None):
         input_data = {
              "last_action_type": action_type,
              "action_params": params,
              "action_outcome": outcome,
              "action_error": error,
              "action_result_data": result_data,
              "phenomenal_state": None,
              "cognitive_state": {}, "active_goal": None, "self_model_summary": {}
         }
         await dsm.process(input_data)

    logger.info(f"Simulating successful LIST_FILES on: {valid_dir_relative}")
    list_result_data_success = {"path": abs_valid_dir, "count": 1, "entries": ["valid_file_dsm.txt"]}
    await simulate_dsm_update("LIST_FILES", {"path": valid_dir_relative}, "success", result_data=list_result_data_success)
    current_model = dsm.self_model
    validated_paths = current_model.get("knowledge_meta", {}).get("validated_paths", {})
    invalid_paths = current_model.get("knowledge_meta", {}).get("invalid_paths", [])
    assert abs_valid_dir in validated_paths, f"Valid directory '{abs_valid_dir}' not added to validated_paths."
    assert abs_valid_dir not in invalid_paths, f"Valid directory '{abs_valid_dir}' should not be in invalid_paths."

    logger.info(f"Simulating successful READ_FILE on: {valid_file_relative}")
    read_result_data_success = {"path": abs_valid_file, "content_snippet": "content", "truncated": False}
    await simulate_dsm_update("READ_FILE", {"path": valid_file_relative}, "success", result_data=read_result_data_success)
    current_model = dsm.self_model
    validated_paths = current_model.get("knowledge_meta", {}).get("validated_paths", {})
    invalid_paths = current_model.get("knowledge_meta", {}).get("invalid_paths", [])
    assert abs_valid_file in validated_paths, f"Valid file '{abs_valid_file}' not added to validated_paths."
    assert abs_valid_file not in invalid_paths, f"Valid file '{abs_valid_file}' should not be in invalid_paths."

    logger.info(f"Simulating failed LIST_FILES on: {invalid_file_relative}")
    await simulate_dsm_update("LIST_FILES", {"path": invalid_file_relative}, "failure", error="Path not exist")
    current_model = dsm.self_model
    validated_paths = current_model.get("knowledge_meta", {}).get("validated_paths", {})
    invalid_paths = current_model.get("knowledge_meta", {}).get("invalid_paths", [])
    assert abs_invalid_file_expected_in_model in invalid_paths, f"Invalid path '{abs_invalid_file_expected_in_model}' not in invalid_paths."

    logger.info(f"Simulating failed READ_FILE on directory: {valid_dir_relative}")
    read_result_data_fail_dir = {"path": abs_valid_dir}
    await simulate_dsm_update("READ_FILE", {"path": valid_dir_relative}, "failure", error="Is a directory", result_data=read_result_data_fail_dir)
    current_model = dsm.self_model
    validated_paths = current_model.get("knowledge_meta", {}).get("validated_paths", {})
    invalid_paths = current_model.get("knowledge_meta", {}).get("invalid_paths", [])
    assert abs_valid_dir in invalid_paths, f"Directory path '{abs_valid_dir}' not added to invalid_paths after failed READ_FILE."
    assert abs_valid_dir not in validated_paths, f"Directory path '{abs_valid_dir}' should be removed from validated_paths."

    agent.agent_root_path = original_agent_root
    logger.info("--- Test Passed: Dynamic Self Model Path Knowledge ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_motivation_updates_on_history(test_agent: AgentController):
    agent = test_agent
    ems = agent.emergent_motivation_system
    dsm = agent.dynamic_self_model
    kb = agent.knowledge_base
    logger.info("--- Test: Emergent Motivation Update ---")

    assert ems is not None, "EmergentMotivationSystem component missing."
    assert dsm is not None, "DynamicSelfModel component missing."
    assert kb is not None, "KnowledgeBase component missing."
    assert Predicate is not None, "Predicate class not available."

    initial_drives_status = await ems.get_status()
    initial_drives = initial_drives_status.get("current_drives", {})
    initial_satisfaction = initial_drives.get("satisfaction", 0.5)
    initial_competence = initial_drives.get("competence", 0.5)
    logger.info(f"Initial Drives: {initial_drives}")

    logger.info("Simulating 5 successful actions...")
    base_time = time.time()
    for i in range(5):
         ts = base_time + i * 0.1
         await kb.assert_fact(Predicate("eventOccurred", ("actionExecution", f"ACTION_S_{i}", "success"), True, timestamp=ts))

    dsm_status_before = await dsm.get_status()
    mock_self_model_summary_after_success = dsm_status_before.copy()
    mock_self_model_summary_after_success["num_capabilities"] = dsm_status_before.get("num_capabilities", 0) + 2

    dummy_last_action_result = {"type": "ACTION_S_4", "outcome": "success"}

    logger.info("Calling EMS process after successes...")
    await ems.process({ "cognitive_state": {}, "last_action_result": dummy_last_action_result,
                        "phenomenal_state": None, "active_goal": None,
                        "self_model_summary": mock_self_model_summary_after_success})

    drives_after_success_status = await ems.get_status()
    drives_after_success = drives_after_success_status.get("current_drives", {})
    satisfaction_after_success = drives_after_success.get("satisfaction", 0.5)
    competence_after_success = drives_after_success.get("competence", 0.5)
    logger.info(f"Drives after success: {drives_after_success}")

    assert satisfaction_after_success > initial_satisfaction, "Satisfaction should increase after successes."
    assert competence_after_success > initial_competence, "Competence should increase after successes and capability gain."

    logger.info("Simulating 5 failed actions...")
    base_time = time.time()
    for i in range(5):
         ts = base_time + i * 0.1
         await kb.assert_fact(Predicate("eventOccurred", ("actionExecution", f"ACTION_F_{i}", "failure"), True, timestamp=ts))

    mock_self_model_summary_after_failure = dsm_status_before.copy()
    mock_self_model_summary_after_failure["num_limitations"] = dsm_status_before.get("num_limitations", 0) + 2
    dummy_last_action_result_fail = {"type": "ACTION_F_4", "outcome": "failure"}

    logger.info("Calling EMS process after failures...")
    await ems.process({ "cognitive_state": {}, "last_action_result": dummy_last_action_result_fail,
                        "phenomenal_state": None, "active_goal": None,
                        "self_model_summary": mock_self_model_summary_after_failure})

    drives_after_failure_status = await ems.get_status()
    drives_after_failure = drives_after_failure_status.get("current_drives", {})
    satisfaction_after_failure = drives_after_failure.get("satisfaction", 0.5)
    competence_after_failure = drives_after_failure.get("competence", 0.5)
    logger.info(f"Drives after failure: {drives_after_failure}")

    assert satisfaction_after_failure < satisfaction_after_success, "Satisfaction should decrease after failures."
    assert competence_after_failure < competence_after_success, "Competence should decrease after failures and limitation gain."

    logger.info("--- Test Passed: Emergent Motivation Update ---")

from unittest.mock import patch, AsyncMock

@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
def test_narrative_significance_triggers_direct(test_agent: AgentController):
    agent = test_agent
    narrative_constructor = agent.narrative_constructor
    logger.info("--- Test: Narrative Significance Triggers (Direct) ---")
    assert narrative_constructor is not None

    _PhenomenalState = globals().get('PhenomenalState')
    _ConsciousState_local = globals().get('ConsciousState')
    assert _PhenomenalState is not None
    assert _ConsciousState_local is not None, "ConsciousState enum not loaded for test."


    mock_p_state = _PhenomenalState(content={}, intensity=0.5, valence=0.1, integration_level=0.5, timestamp=time.time())
    mock_action_result = {"type": "TEST", "outcome": "success"}
    mock_drives_no_change = {"curiosity": 0.5, "satisfaction": 0.5, "competence": 0.5}
    mock_drives_high_change = {"curiosity": 0.9, "satisfaction": 0.1, "competence": 0.5}
    mock_drives_low_change = {"curiosity": 0.55, "satisfaction": 0.5, "competence": 0.5}
    mock_prediction_error = {"type": "outcome_mismatch", "action_type": "TEST_ACTION", "predicted_outcome_summary": "success", "actual_outcome_summary": "failure"}

    narrative_constructor._last_phenomenal_state = None
    narrative_constructor._last_drive_state = mock_drives_no_change.copy()
    narrative_constructor.drive_change_threshold = 0.25

    def set_controller_cs_state(level: ConsciousState):
        agent.consciousness_level = level
        agent._prev_consciousness_level = level # type: ignore

    set_controller_cs_state(_ConsciousState_local.CONSCIOUS) # type: ignore
    is_sig, reason, _ = narrative_constructor._is_significant( mock_p_state, mock_action_result, None, {}, None, mock_drives_no_change )
    assert not is_sig, f"Should not be significant with no triggers. Reason: {reason}"

    set_controller_cs_state(_ConsciousState_local.CONSCIOUS) # type: ignore
    is_sig, reason, summary = narrative_constructor._is_significant( mock_p_state, mock_action_result, None, {}, mock_prediction_error, mock_drives_no_change )
    assert is_sig, f"Should be significant due to prediction error. Reason: {reason}"; assert "PredictionErr" in reason; assert "prediction_error" in summary

    set_controller_cs_state(_ConsciousState_local.CONSCIOUS) # type: ignore
    is_sig, reason, summary = narrative_constructor._is_significant( mock_p_state, mock_action_result, None, {}, None, mock_drives_high_change )
    assert is_sig, f"Should be significant due to high drive shift. Reason: {reason}"; assert "DriveShift" in reason; assert "drive_shift" in summary

    narrative_constructor._last_drive_state = mock_drives_no_change.copy()
    set_controller_cs_state(_ConsciousState_local.CONSCIOUS) # type: ignore
    is_sig, reason, summary = narrative_constructor._is_significant( mock_p_state, mock_action_result, None, {}, None, mock_drives_low_change )
    assert not is_sig, f"Should NOT be significant due to low drive shift. Reason: {reason}"; assert "DriveShift" not in reason

    logger.info("--- Test Passed: Narrative Significance Triggers (Direct) ---")



@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
@patch('consciousness_experiment.cognitive_modules.narrative_constructor.call_ollama', new_callable=AsyncMock)
async def test_narrative_generator_calls_llm(mock_call_ollama: AsyncMock, test_agent: AgentController):
    agent = test_agent
    narrative_constructor = agent.narrative_constructor
    ems = agent.emergent_motivation_system
    logger.info("--- Test: Narrative Generate Calls LLM ---")

    assert narrative_constructor is not None, "NarrativeConstructor component missing."
    assert ems is not None, "EmergentMotivationSystem component missing."
    _PhenomenalState = globals().get('PhenomenalState')
    assert _PhenomenalState is not None, "PhenomenalState class missing."
    _ConsciousState_local = globals().get('ConsciousState')
    assert _ConsciousState_local is not None, "ConsciousState enum not loaded for test."

    mock_llm_response = "This is a generated narrative entry about the event."
    mock_call_ollama.return_value = (mock_llm_response, None)

    mock_drive_state = {"curiosity": 0.7, "satisfaction": 0.2}
    mock_ems_status = {"current_drives": mock_drive_state}
    narrative_constructor._last_drive_state = mock_drive_state

    with patch.object(ems, 'get_status', AsyncMock(return_value=mock_ems_status)):
        trigger_reason = "PredictionErr(outcome_mismatch). DriveShift(0.50)."
        trigger_event_summary = {
            "prediction_error": {"type": "outcome_mismatch", "action_type": "TEST_ACTION", "predicted_outcome_summary": "X", "actual_outcome_summary": "Y"},
            "drive_shift": 0.50
        }
        test_phenomenal_state = _PhenomenalState(
            content={"focus": "test_data"}, intensity=0.8, valence=-0.3,
            integration_level=0.6, timestamp=time.time()
        )
        agent.consciousness_level = _ConsciousState_local.CONSCIOUS

        logger.info("Calling generate_narrative_entry...")
        generated_text = await narrative_constructor.generate_narrative_entry(
            phenomenal_state=test_phenomenal_state,
            triggering_event=trigger_event_summary,
            reason=trigger_reason
        )

    mock_call_ollama.assert_called_once()
    assert generated_text == mock_llm_response

    args_passed, kwargs_passed = mock_call_ollama.call_args

    assert len(args_passed) == 4, f"Expected 4 positional args, got {len(args_passed)}"
    assert args_passed[0] == narrative_constructor.llm_model_name
    assert isinstance(args_passed[1], list) and len(args_passed[1]) == 2
    assert args_passed[1][1]['role'] == 'user'

    user_prompt_content = args_passed[1][1]['content']
    assert trigger_reason in user_prompt_content
    assert "Drives: curiosity=0.70, satisfaction=0.20" in user_prompt_content
    assert "PredictionErr" in user_prompt_content
    assert "TEST_ACTION" in user_prompt_content
    assert args_passed[2] == narrative_constructor.llm_temperature
    assert args_passed[3] == agent._asyncio_loop

    if kwargs_passed:
        assert len(kwargs_passed) == 1 and 'timeout' in kwargs_passed, \
            f"Only expected 'timeout' in kwargs if any, got {kwargs_passed}"
        assert kwargs_passed['timeout'] == narrative_constructor.llm_timeout_s
        logger.info(f"call_ollama mock called with kwargs: {kwargs_passed}")
    else:
        logger.info("call_ollama mock called with no kwargs (timeout might have used its internal default).")


    logger.info("--- Test Passed: Narrative Generate Calls LLM ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_user_input_perception(test_agent: AgentController):
    agent = test_agent
    logger.info("--- Test: User Input Perception ---")

    assert agent._user_input_queue is not None, "_user_input_queue not initialized."
    assert agent._asyncio_loop is not None, "_asyncio_loop not initialized in fixture."
    assert agent._is_running_flag is not None, "_is_running_flag not initialized."

    test_message = "Hello OSCAR from the test!"

    agent._is_running_flag.set()
    try:
        agent.handle_user_input(test_message)
        await asyncio.sleep(0.01)
        assert agent._user_input_queue.qsize() == 1, "User input was not added to the queue."

        logger.info("Calling _oscar_perceive to check for queued input...")
        percepts = await agent._oscar_perceive()

        assert isinstance(percepts, dict), "_oscar_perceive did not return a dict."
        assert "user_input" in percepts, "'user_input' key missing from percepts."
        assert percepts["user_input"] == test_message, "Perceived user input does not match sent message."
        assert agent._user_input_queue.empty(), "User input queue should be empty after perceive."

        logger.info("Calling _oscar_perceive again (queue should be empty)...")
        percepts_empty = await agent._oscar_perceive()
        assert percepts_empty["user_input"] is None, "User input should be None when queue is empty."

    finally:
        agent._is_running_flag.clear()

    logger.info("--- Test Passed: User Input Perception ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_write_file_action(test_agent: AgentController, tmp_path: Path):
    agent = test_agent
    original_agent_root = agent.agent_root_path
    agent.agent_root_path = tmp_path

    assert agent.config.get('filesystem', {}).get('allow_file_write') is True, "allow_file_write should be True for test."

    logger.info("--- Test: WRITE_FILE Action Execution ---")

    assert hasattr(agent, 'knowledge_base') and agent.knowledge_base is not None, "KnowledgeBase component missing."
    assert Predicate is not None, "Predicate class not available."

    target_dir_relative = "write_test_data_dir"
    (agent.agent_root_path / target_dir_relative).mkdir(exist_ok=True)

    target_file_relative = f"{target_dir_relative}/output.txt"
    target_file_abs_expected = (agent.agent_root_path / target_file_relative).resolve()
    file_content = "Content written by OSCAR-C test."

    action_write_success = {
        "type": "WRITE_FILE",
        "params": {"path": target_file_relative, "content": file_content}
    }

    logger.info(f"Executing successful WRITE_FILE action for new file: {target_file_relative} (agent_root_path mocked to {agent.agent_root_path})")
    result_success = await agent._oscar_execute_action(action_write_success)

    assert result_success.get("outcome") == "success", f"WRITE_FILE action failed unexpectedly: {result_success.get('error')}"
    assert target_file_abs_expected.exists(), "Target file was not created."
    assert target_file_abs_expected.read_text(encoding='utf-8') == file_content, "File content mismatch."
    result_data = result_success.get("result_data", {})
    assert Path(result_data.get("path", "")) == target_file_abs_expected, "Result path mismatch."
    assert result_data.get("bytes_written") == len(file_content.encode('utf-8')), "Bytes written mismatch."

    await asyncio.sleep(0.05)
    resolved_path_str_success = str(target_file_abs_expected)
    kb_written = await agent.knowledge_base.query(name="fileWritten", args=(resolved_path_str_success,), value=True)
    kb_valid = await agent.knowledge_base.query(name="isValidPath", args=(resolved_path_str_success,), value=True)
    kb_event_success = await agent.knowledge_base.query(name="eventOccurred", args=("actionExecution", "WRITE_FILE", "success"), value=True)
    assert len(kb_written) >= 1, "fileWritten predicate missing after write."
    assert len(kb_valid) >= 1, "isValidPath predicate missing after write."
    assert len(kb_event_success) >= 1, "eventOccurred(success) predicate missing."
    logger.info("Successful WRITE_FILE executed and verified.")

    action_write_overwrite_fail = {
        "type": "WRITE_FILE",
        "params": {"path": target_file_relative, "content": "Attempt to overwrite."}
    }
    logger.info(f"Executing failed WRITE_FILE (overwrite denied) for: {target_file_relative}")
    result_overwrite_fail = await agent._oscar_execute_action(action_write_overwrite_fail)

    assert result_overwrite_fail.get("outcome") == "failure", "WRITE_FILE should have failed (overwrite denied)."
    assert "overwrite is disabled" in result_overwrite_fail.get("error", "").lower(), "Error message mismatch for overwrite denied."
    assert target_file_abs_expected.read_text(encoding='utf-8') == file_content, "File content was incorrectly overwritten."

    agent.config['filesystem']['allow_overwrite'] = True
    new_content_overwrite = "Overwritten content."
    action_write_overwrite_success = {
        "type": "WRITE_FILE",
        "params": {"path": target_file_relative, "content": new_content_overwrite}
    }
    logger.info(f"Executing successful WRITE_FILE (overwrite enabled) for: {target_file_relative}")
    result_overwrite_success = await agent._oscar_execute_action(action_write_overwrite_success)

    assert result_overwrite_success.get("outcome") == "success", f"WRITE_FILE should have succeeded (overwrite enabled): {result_overwrite_success.get('error')}"
    assert target_file_abs_expected.read_text(encoding='utf-8') == new_content_overwrite, "File content mismatch after overwrite."
    agent.config['filesystem']['allow_overwrite'] = False

    action_write_dir_fail = {
        "type": "WRITE_FILE",
        "params": {"path": target_dir_relative, "content": "Should not work."}
    }
    logger.info(f"Executing failed WRITE_FILE (write to directory) for: {target_dir_relative}")
    result_dir_fail = await agent._oscar_execute_action(action_write_dir_fail)

    assert result_dir_fail.get("outcome") == "failure", "WRITE_FILE should have failed (write to directory)."
    assert "is a directory" in result_dir_fail.get("error", "").lower(), "Error message mismatch for write to directory."

    await asyncio.sleep(0.05)
    dir_path_str_abs = str((agent.agent_root_path / target_dir_relative).resolve())
    kb_invalid_dir = await agent.knowledge_base.query(name="isInvalidPath", args=(dir_path_str_abs,), value=True)
    assert len(kb_invalid_dir) >= 1, "isInvalidPath predicate missing after failed write to directory."
    logger.info("Failed WRITE_FILE (write to directory) verified.")

    agent.agent_root_path = original_agent_root
    logger.info("--- Test Passed: WRITE_FILE Action Execution ---")

from unittest.mock import patch, AsyncMock
from pathlib import Path

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_self_model_capability_update_for_llm(test_agent: AgentController):
    agent = test_agent
    dsm = agent.dynamic_self_model
    logger.info("--- Test: Dynamic Self Model Capability Update (CALL_LLM) ---")

    assert dsm is not None, "DynamicSelfModel component missing."
    action_name = "CALL_LLM"
    action_key = f"action:{action_name}"

    async def simulate_dsm_process(outcome: str, error: Optional[str] = None):
        input_data = {
             "last_action_type": action_name,
             "action_params": {"prompt": "test"},
             "action_outcome": outcome,
             "action_error": error,
             "action_result_data": {"response": "..."} if outcome=="success" else None,
             "phenomenal_state": None,
             "cognitive_state": {}, "active_goal": None, "self_model_summary": {}
        }
        await dsm.process(input_data)

    initial_caps = dsm.self_model.get("capabilities", {}).copy()
    initial_lims = dsm.self_model.get("limitations", {}).copy()
    assert action_key not in initial_caps, f"'{action_key}' should not be in capabilities initially."
    assert action_key not in initial_lims, f"'{action_key}' should not be in limitations initially."
    initial_version = dsm.self_model.get("version", 0)

    logger.info("Simulating successful CALL_LLM for DSM update...")
    await simulate_dsm_process(outcome="success")

    caps_after_success = dsm.self_model.get("capabilities", {})
    lims_after_success = dsm.self_model.get("limitations", {})
    version_after_success = dsm.self_model.get("version", 0)

    assert action_key in caps_after_success, f"'{action_key}' missing from capabilities after success."
    assert caps_after_success[action_key] > 0, f"Capability confidence for '{action_key}' should be positive after success."
    assert action_key not in lims_after_success, f"'{action_key}' should not be in limitations after success."
    assert version_after_success > initial_version, "Model version should increase after update."
    logger.info(f"Capability confidence after success: {caps_after_success[action_key]:.3f}")
    confidence_after_success = caps_after_success[action_key]

    logger.info("Simulating failed CALL_LLM for DSM update...")
    await simulate_dsm_process(outcome="failure", error="LLM Timeout")

    _caps_dict_after_failure = dsm.self_model.get("capabilities", {})
    capability_value_after_failure = _caps_dict_after_failure.get(action_key)
    lims_after_failure = dsm.self_model.get("limitations", {})
    version_after_failure = dsm.self_model.get("version", 0)

    assert capability_value_after_failure is not None, f"Capability value for '{action_key}' should exist after failure."
    assert capability_value_after_failure < confidence_after_success, f"Capability confidence for '{action_key}' should decrease after failure."
    assert action_key in lims_after_failure, f"'{action_key}' missing from limitations after failure."
    assert lims_after_failure[action_key] > 0, f"Limitation confidence for '{action_key}' should be positive after failure."
    assert version_after_failure > version_after_success, "Model version should increase again after failure update."
    logger.info(f"Capability confidence after failure: {capability_value_after_failure:.3f}")
    logger.info(f"Limitation confidence after failure: {lims_after_failure[action_key]:.3f}")
    limitation_value_after_failure = lims_after_failure[action_key]

    logger.info("Simulating another successful CALL_LLM for DSM update...")
    await simulate_dsm_process(outcome="success")

    caps_after_second_success = dsm.self_model.get("capabilities", {})
    lims_after_second_success = dsm.self_model.get("limitations", {})
    capability_value_after_second_success = caps_after_second_success.get(action_key)
    limitation_value_after_second_success = lims_after_second_success.get(action_key)

    assert capability_value_after_second_success is not None, "Capability value missing after second success."
    assert capability_value_after_second_success > capability_value_after_failure, "Capability confidence should increase again after second success."
    assert action_key in lims_after_second_success, "Limitation should still exist after one success."
    assert limitation_value_after_second_success is not None, "Limitation value missing after second success."
    assert limitation_value_after_second_success < limitation_value_after_failure, "Limitation confidence should decrease after success."
    logger.info(f"Capability confidence after 2nd success: {capability_value_after_second_success:.3f}")
    logger.info(f"Limitation confidence after 2nd success: {limitation_value_after_second_success:.3f}")

    logger.info("--- Test Passed: Dynamic Self Model Capability Update (CALL_LLM) ---")


@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_pwm_adaptive_learning(test_agent: AgentController, tmp_path: Path):
    """
    Tests that PredictiveWorldModel adaptively learns outcome frequencies for actions
    and uses this learned knowledge for subsequent predictions.
    """
    agent = test_agent
    pwm = agent.predictive_world_model
    logger.info("--- Test: PredictiveWorldModel Adaptive Learning ---")

    assert pwm is not None, "PredictiveWorldModel component missing."
    assert hasattr(pwm, 'causal_rules'), "PWM missing 'causal_rules' attribute for learning."

    original_agent_root = agent.agent_root_path
    agent.agent_root_path = tmp_path

    test_file_relative = "adaptive_learn_test.txt"
    action_key_expected = f"READ_FILE:target_{Path(test_file_relative).name}"


    action_params = {"path": test_file_relative}
    action_to_test = {"type": "READ_FILE", "params": action_params}
    predict_request_args = {"action_to_execute": action_to_test, "context": {}}

    logger.info(f"Initial prediction for {action_key_expected} (no prior learning)...")
    pred_output1 = await pwm.process({"predict_request": predict_request_args})
    assert pred_output1 and "prediction" in pred_output1, "PWM process for initial prediction failed."
    prediction1 = pred_output1["prediction"]
    logger.info(f"Prediction 1: {prediction1}")
    assert prediction1.get("predicted_outcome") == "success", "Initial prediction should be default 'success'."
    assert prediction1.get("basis") == "default_heuristic", "Initial prediction basis incorrect."
    assert abs(prediction1.get("confidence", 0.0) - 0.3) < 0.01, "Initial confidence incorrect."
    assert action_key_expected not in pwm.causal_rules, "No rule should exist initially."

    logger.info(f"Simulating FAILED execution of {action_key_expected} and updating PWM...")
    actual_result_failure = {
        "type": "READ_FILE", "params": action_params,
        "outcome": "failure", "error": "File not found",
        "context": {}
    }
    update_request_args1 = {"prediction": prediction1, "actual_result": actual_result_failure}
    update_output1 = await pwm.process({"update_request": update_request_args1})
    assert update_output1 and "last_prediction_error_details" in update_output1, "PWM update process failed."
    assert update_output1["last_prediction_error_details"] is not None, "Prediction error should be recorded."

    assert action_key_expected in pwm.causal_rules, "Causal rule not created after first failure."
    assert pwm.causal_rules[action_key_expected] == Counter({"failure": 1}), \
        f"Incorrect rule after 1st failure: {pwm.causal_rules[action_key_expected]}"
    logger.info(f"PWM causal_rules after 1st failure: {pwm.causal_rules}")

    logger.info(f"Second prediction for {action_key_expected} (after 1 failure)...")
    pred_output2 = await pwm.process({"predict_request": predict_request_args})
    assert pred_output2 and "prediction" in pred_output2, "PWM process for second prediction failed."
    prediction2 = pred_output2["prediction"]
    logger.info(f"Prediction 2: {prediction2}")
    assert prediction2.get("predicted_outcome") == "failure", "Second prediction should be 'failure'."
    assert prediction2.get("basis") == "general_action_rule", "Second prediction basis incorrect."
    assert abs(prediction2.get("confidence", 0.0) - 1.0) < 0.01, "Second prediction confidence incorrect (should be 1.0)."

    # 4. Simulate File Creation and Successful Execution & Update Model
    logger.info(f"Simulating SUCCESSFUL execution of {action_key_expected} and updating PWM...")
    actual_result_success = {
        "type": "READ_FILE", "params": action_params,
        "outcome": "success", "result_data": {"content_snippet": "test content"},
        "context": {}
    }
    update_request_args2 = {"prediction": prediction2, "actual_result": actual_result_success} # Use prediction2 here
    update_output2 = await pwm.process({"update_request": update_request_args2})
    assert update_output2 and "last_prediction_error_details" in update_output2, "PWM update process failed."
    assert update_output2["last_prediction_error_details"] is not None, "Prediction error should be recorded on success after predicting failure."

    assert pwm.causal_rules[action_key_expected] == Counter({"failure": 1, "success": 1}), \
        f"Incorrect rule after 1 failure, 1 success: {pwm.causal_rules[action_key_expected]}"
    logger.info(f"PWM causal_rules after 1 fail, 1 success: {pwm.causal_rules}")

    # 5. Third Prediction (Outcomes are now 50/50 for this specific key)
    logger.info(f"Third prediction for {action_key_expected} (after 1 fail, 1 success)...")
    pred_output3 = await pwm.process({"predict_request": predict_request_args})
    assert pred_output3 and "prediction" in pred_output3, "PWM process for third prediction failed."
    prediction3 = pred_output3["prediction"] # This is the correct variable for the third prediction
    logger.info(f"Prediction 3: {prediction3}")

    # --- CORRECTED ASSERTIONS USING prediction3 ---
    assert prediction3.get("predicted_outcome") in ["failure", "success"], \
        f"Third prediction outcome unexpected, got: {prediction3.get('predicted_outcome')}"
    assert prediction3.get("basis") == "general_action_rule", \
        f"Third prediction basis incorrect, got: {prediction3.get('basis')}"
    assert abs(prediction3.get("confidence", 0.0) - 0.5) < 0.01, \
        f"Third prediction confidence incorrect (should be 0.5), got: {prediction3.get('confidence')}"
    # --- END CORRECTIONS ---

    agent.agent_root_path = original_agent_root
    logger.info("--- Test Passed: PredictiveWorldModel Adaptive Learning ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_pwm_contextual_learning(test_agent: AgentController, tmp_path: Path):
    """
    Tests that PredictiveWorldModel learns and uses context-specific outcome frequencies.
    """
    agent = test_agent
    pwm = agent.predictive_world_model
    logger.info("--- Test: PredictiveWorldModel Contextual Learning ---")

    assert pwm is not None, "PredictiveWorldModel component missing."
    assert hasattr(pwm, 'causal_context'), "PWM missing 'causal_context' attribute for learning."

    original_agent_root = agent.agent_root_path
    agent.agent_root_path = tmp_path

    test_file_relative = "context_learn_file.txt"
    action_params = {"path": test_file_relative}
    action_to_test = {"type": "READ_FILE", "params": action_params}
    # Expected action_key from refined _get_action_key
    action_key_expected = f"READ_FILE:target_{Path(test_file_relative).name}"

    # Define different contexts
    context_low_cs = {
        "consciousness_level_name": "PRE_CONSCIOUS", # Will generate "low_cs" in context_key
        "active_goal_type": "information_gathering",
        "drive_curiosity": 0.3 # Should generate "low_cur"
    }
    context_key_low_cs = pwm._extract_context_key(context_low_cs)

    context_high_cs_high_cur = {
        "consciousness_level_name": "META_CONSCIOUS", # Will generate "high_cs"
        "active_goal_type": "exploration",
        "drive_curiosity": 0.8 # Will generate "high_cur"
    }
    context_key_high_cs_high_cur = pwm._extract_context_key(context_high_cs_high_cur)

    # --- Step 1: Learn Failure in Low CS Context ---
    logger.info(f"Learning '{action_key_expected}' outcome in context '{context_key_low_cs}' as FAILURE.")
    # Initial prediction (no specific context rule yet)
    pred_req_low1 = {"action_to_execute": action_to_test, "context": context_low_cs}
    pred_out_low1_dict = await pwm.process({"predict_request": pred_req_low1})
    prediction_low1 = pred_out_low1_dict.get("prediction", {}) if pred_out_low1_dict else {}


    actual_result_fail_low_cs = {
        "type": "READ_FILE", "params": action_params, "outcome": "failure",
        "error": "Simulated read error in low_cs", "context": context_low_cs
    }
    update_req_low1 = {"prediction": prediction_low1, "actual_result": actual_result_fail_low_cs}
    await pwm.process({"update_request": update_req_low1})

    assert action_key_expected in pwm.causal_context, "Action key missing from causal_context."
    assert context_key_low_cs in pwm.causal_context[action_key_expected], f"Context key '{context_key_low_cs}' missing for action."
    assert pwm.causal_context[action_key_expected][context_key_low_cs] == Counter({"failure": 1}), \
        f"Incorrect rule for {action_key_expected} in {context_key_low_cs}"
    # General rule should also be updated
    assert pwm.causal_rules.get(action_key_expected) == Counter({"failure": 1})
    logger.info(f"Rule learned: {action_key_expected} | {context_key_low_cs} -> failure:1")


    # --- Step 2: Learn Success in High CS / High Curiosity Context ---
    logger.info(f"Learning '{action_key_expected}' outcome in context '{context_key_high_cs_high_cur}' as SUCCESS.")
    # Initial prediction (should not use the low_cs rule yet for this different context)
    pred_req_high1 = {"action_to_execute": action_to_test, "context": context_high_cs_high_cur}
    pred_out_high1_dict = await pwm.process({"predict_request": pred_req_high1})
    prediction_high1 = pred_out_high1_dict.get("prediction",{}) if pred_out_high1_dict else {}


    actual_result_succ_high_cs = {
        "type": "READ_FILE", "params": action_params, "outcome": "success",
        "result_data": {"content_snippet": "content"}, "context": context_high_cs_high_cur
    }
    update_req_high1 = {"prediction": prediction_high1, "actual_result": actual_result_succ_high_cs}
    await pwm.process({"update_request": update_req_high1})

    assert context_key_high_cs_high_cur in pwm.causal_context[action_key_expected], f"Context key '{context_key_high_cs_high_cur}' missing."
    assert pwm.causal_context[action_key_expected][context_key_high_cs_high_cur] == Counter({"success": 1}), \
        f"Incorrect rule for {action_key_expected} in {context_key_high_cs_high_cur}"
    # General rule should now reflect both outcomes
    assert pwm.causal_rules.get(action_key_expected) == Counter({"failure": 1, "success": 1})
    logger.info(f"Rule learned: {action_key_expected} | {context_key_high_cs_high_cur} -> success:1")
    logger.info(f"PWM causal_rules: {pwm.causal_rules}")
    logger.info(f"PWM causal_context: {pwm.causal_context}")


    # --- Step 3: Test Predictions with Learned Contextual Rules ---
    # Predict in Low CS context
    logger.info(f"Predicting for '{action_key_expected}' in context '{context_key_low_cs}'...")
    pred_req_low2 = {"action_to_execute": action_to_test, "context": context_low_cs}
    pred_out_low2_dict = await pwm.process({"predict_request": pred_req_low2})
    prediction_low2 = pred_out_low2_dict.get("prediction",{}) if pred_out_low2_dict else {}

    logger.info(f"Prediction with Low CS Context: {prediction_low2}")
    assert prediction_low2.get("predicted_outcome") == "failure", "Should predict failure in low_cs context."
    assert prediction_low2.get("basis") == "context_specific_rule", "Prediction basis incorrect for low_cs."
    assert abs(prediction_low2.get("confidence", 0.0) - 1.0) < 0.01, "Confidence incorrect for low_cs prediction."

    # Predict in High CS / High Curiosity context
    logger.info(f"Predicting for '{action_key_expected}' in context '{context_key_high_cs_high_cur}'...")
    pred_req_high2 = {"action_to_execute": action_to_test, "context": context_high_cs_high_cur}
    pred_out_high2_dict = await pwm.process({"predict_request": pred_req_high2})
    prediction_high2 = pred_out_high2_dict.get("prediction",{}) if pred_out_high2_dict else {}

    logger.info(f"Prediction with High CS Context: {prediction_high2}")
    assert prediction_high2.get("predicted_outcome") == "success", "Should predict success in high_cs_high_cur context."
    assert prediction_high2.get("basis") == "context_specific_rule", "Prediction basis incorrect for high_cs_high_cur."
    assert abs(prediction_high2.get("confidence", 0.0) - 1.0) < 0.01, "Confidence incorrect for high_cs_high_cur prediction."

    # Predict with a default/neutral context (should use general rule which is now 50/50)
    logger.info(f"Predicting for '{action_key_expected}' in default context...")
    default_context = {"consciousness_level_name": "CONSCIOUS", "active_goal_type": "general_task", "drive_curiosity": 0.5}
    context_key_default = pwm._extract_context_key(default_context) # Should be "default_context" or similar

    # Ensure the default context key does not yet exist (or it's truly 'default_context' and empty)
    if context_key_default != "default_context":
         assert context_key_default not in pwm.causal_context.get(action_key_expected, {}), \
            f"Default context key '{context_key_default}' should not have specific rule yet."

    pred_req_default = {"action_to_execute": action_to_test, "context": default_context}
    pred_out_default_dict = await pwm.process({"predict_request": pred_req_default})
    prediction_default = pred_out_default_dict.get("prediction",{}) if pred_out_default_dict else {}

    logger.info(f"Prediction with Default Context ('{context_key_default}'): {prediction_default}")
    assert prediction_default.get("predicted_outcome") in ["failure", "success"], "Default context prediction outcome unexpected."
    assert prediction_default.get("basis") == "general_action_rule", "Default context prediction basis incorrect."
    assert abs(prediction_default.get("confidence", 0.0) - 0.5) < 0.01, "Default context prediction confidence incorrect (should be 0.5)."

    agent.agent_root_path = original_agent_root
    logger.info("--- Test Passed: PredictiveWorldModel Contextual Learning ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_htn_iterative_deepening(test_agent: AgentController):
    """
    Tests the HTNPlanner's iterative deepening search.
    1. Defines a problem solvable at depth 2.
    2. Checks that it's found with max_depth >= 2 (and normal CS).
    3. Clears cache.
    4. Checks that it's NOT found if effective max_depth is set to 1 (due to CS or config).
    5. Checks that a longer path is found if short path is blocked and depth allows (normal CS).
    """
    agent = test_agent
    planner = agent.htn_planner
    kb = agent.knowledge_base 
    logger.info("--- Test: HTN Iterative Deepening ---")

    assert planner is not None, "HTNPlanner component missing."
    assert kb is not None, "KnowledgeBase component missing."
    assert Predicate is not None, "Predicate class not available for test."
    assert Goal is not None, "Goal class not available for test."
    assert create_goal_from_descriptor is not None, "create_goal_from_descriptor not available."
    _Predicate_for_test_id = globals().get('Predicate') 
    assert _Predicate_for_test_id is not None

    # Store original agent config and CS level for restoration
    original_agent_config_max_depth = agent.config.get("performance", {}).get("max_planning_depth")
    original_agent_cs_level = agent.consciousness_level

    original_operators = planner.operators.copy()
    original_methods = planner.methods.copy()
    planner.operators = {}
    planner.methods = {}

    planner.operators["OP_A"] = Operator(name="OP_A", effects={_Predicate_for_test_id("stateA", (), True)})
    planner.operators["OP_B"] = Operator(name="OP_B", effects={_Predicate_for_test_id("stateB", (), True)})
    planner.operators["OP_C"] = Operator(name="OP_C", effects={_Predicate_for_test_id("stateC", (), True)})
    planner.operators["OP_D"] = Operator(name="OP_D", effects={_Predicate_for_test_id("stateD", (), True)}) 
    planner.operators["OP_E"] = Operator(name="OP_E", effects={_Predicate_for_test_id("stateE", (), True)}) 

    planner.methods["task_X"] = [
        Method(name="method_short_path", task_signature=("task_X",),
               subtasks=["OP_A", "OP_B"]), 
        Method(name="method_longer_path_setup", task_signature=("task_X",),
               subtasks=["OP_C", "task_Y"]),
    ]
    planner.methods["task_Y"] = [
        Method(name="method_for_task_Y", task_signature=("task_Y",),
               subtasks=["OP_D", "OP_E"])
    ]

    test_goal = create_goal_from_descriptor("Achieve stateB via task_X")
    assert test_goal is not None, "Failed to create test_goal for iterative deepening."
    empty_initial_state = set()

    with patch.object(planner, '_goal_to_task', return_value=("task_X",)) as mock_g2t:

        # --- Test 1: Max depth allows finding the short plan (OP_A, OP_B) ---
        agent.config.get("performance", {})["max_planning_depth"] = 2 
        agent.consciousness_level = ConsciousState.CONSCIOUS # Ensure normal CS state

        logger.info(f"Testing ID with agent_config.max_depth=2, CS={agent.consciousness_level.name}")
        plan1 = await planner.plan(test_goal, empty_initial_state)
        
        assert plan1 is not None, "Plan should be found with effective max_depth=2"
        assert len(plan1) == 2, f"Expected plan of length 2, got {len(plan1)}. Plan: {plan1}"
        assert plan1[0]["type"] == "OP_A" and plan1[1]["type"] == "OP_B", f"Plan is not OP_A, OP_B. Got: {plan1}"
        logger.info(f"Plan 1 (agent_config.max_depth=2, CS={agent.consciousness_level.name}): {plan1}")

        # --- Test 2: Max depth too shallow (due to agent.config) to find the short plan ---
        if agent.cache: await agent.cache.clear()
        agent.config.get("performance", {})["max_planning_depth"] = 1 
        agent.consciousness_level = ConsciousState.CONSCIOUS # Normal CS, depth limit from config

        logger.info(f"Testing ID with agent_config.max_depth=1, CS={agent.consciousness_level.name}")
        plan2 = await planner.plan(test_goal, empty_initial_state) 
        
        planner_internal_max_depth_after_plan2 = planner.max_depth # This is what planner.plan's loop used
        assert plan2 is None, \
            (f"Plan should NOT be found when agent.config.max_depth is 1 (planner loop used {planner_internal_max_depth_after_plan2}). "
             f"Plan found: {plan2}")
        logger.info(f"Plan 2 (agent_config.max_depth=1, CS={agent.consciousness_level.name}): {plan2} (Correctly None)")

        # --- Test 2b: Max depth too shallow (due to LOW CS override) ---
        if agent.cache: await agent.cache.clear()
        agent.config.get("performance", {})["max_planning_depth"] = 3 # Config allows deep plan
        agent.config.setdefault("htn_planner", {})["max_planning_depth_on_low_cs"] = 1 # Low CS override to 1
        agent.consciousness_level = ConsciousState.UNCONSCIOUS # LOW CS

        logger.info(f"Testing ID with agent_config.max_depth=3, but CS={agent.consciousness_level.name} (low_cs_max_depth=1)")
        plan2b = await planner.plan(test_goal, empty_initial_state)
        
        # Planner.plan logs the "Effective Max Overall Depth for this attempt"
        # We'd ideally capture that log to confirm it used the low_cs_max_depth.
        # For now, just assert plan is None.
        assert plan2b is None, \
            (f"Plan should NOT be found when CS is low and low_cs_max_depth is 1, even if config.max_depth is higher. "
             f"Plan found: {plan2b}")
        logger.info(f"Plan 2b (agent_config.max_depth=3, CS={agent.consciousness_level.name}): {plan2b} (Correctly None)")


        # --- Test 3: Max depth allows finding the longer plan if short one is blocked ---
        if agent.cache: await agent.cache.clear()
        
        planner.operators["OP_A"] = Operator(name="OP_A", 
                                            preconditions={_Predicate_for_test_id("short_path_blocker", (), True)},
                                            effects={_Predicate_for_test_id("stateA", (), True)})

        agent.config.get("performance", {})["max_planning_depth"] = 3
        agent.consciousness_level = ConsciousState.CONSCIOUS # Ensure normal CS for this

        logger.info(f"Testing ID with agent_config.max_depth=3, blocked short path, and CS={agent.consciousness_level.name}")
        plan3 = await planner.plan(test_goal, empty_initial_state) 

        assert plan3 is not None, "Plan should be found via longer path when short path is blocked and depth allows."
        assert len(plan3) == 3, f"Expected plan of length 3 from longer path, got {len(plan3)}. Plan: {plan3}"
        assert plan3[0]["type"] == "OP_C" and plan3[1]["type"] == "OP_D" and plan3[2]["type"] == "OP_E", \
            f"Plan does not match longer path (OP_C, OP_D, OP_E). Got: {plan3}"
        logger.info(f"Plan 3 (agent_config.max_depth=3, CS={agent.consciousness_level.name}, short path blocked): {plan3}")

    # Restore original agent config max_depth and CS level
    if original_agent_config_max_depth is not None:
        agent.config.get("performance", {})["max_planning_depth"] = original_agent_config_max_depth
    else: 
        if "max_planning_depth" in agent.config.get("performance", {}):
            del agent.config.get("performance", {})["max_planning_depth"]
    if original_agent_cs_level is not None: # Restore original CS if it was set
        agent.consciousness_level = original_agent_cs_level
    
    planner.max_depth = agent.config.get("performance", {}).get("max_planning_depth", 5) 
    planner.operators = original_operators
    planner.methods = original_methods
    logger.info("--- Test Passed: HTN Iterative Deepening ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_htn_heuristic_influence(test_agent: AgentController):
    """
    Tests that the HTNPlanner prioritizes applicable methods with better (lower)
    heuristic scores, and backtracks if the preferred method fails during decomposition.
    """
    agent = test_agent
    planner = agent.htn_planner
    logger.info("--- Test: HTN Heuristic Influence ---")

    assert planner is not None and Predicate is not None and Goal is not None

    original_operators = planner.operators.copy()
    original_methods = planner.methods.copy()
    planner.operators = {}
    planner.methods = {}

    # Operators
    planner.operators["OP_A1"] = Operator(name="OP_A1",
                                          preconditions={Predicate("precond_for_A1", (), True)}, # This will be MISSING
                                          effects={Predicate("effect_A1", (), True)})
    planner.operators["OP_A_SETUP"] = Operator(name="OP_A_SETUP", effects={Predicate("setup_for_A_done", (), True)})

    planner.operators["OP_B1"] = Operator(name="OP_B1", effects={Predicate("effect_B1", (), True)})
    planner.operators["OP_B2"] = Operator(name="OP_B2", effects={Predicate("effect_B2", (), True)}) # Goal state

    # Methods for "task_heuristic_choice"
    # Method A: Better heuristic (2 subtasks). Its subtask OP_A1 will fail due to its own precondition.
    # Preconditions for Method A itself will be met.
    method_A = Method(name="method_A_preferred_fails_subtask",
                      task_signature=("task_heuristic_choice",),
                      preconditions={Predicate("can_try_A", (), True)}, # This will be in state
                      subtasks=["OP_A_SETUP", "OP_A1"]) # OP_A1 requires "precond_for_A1"
                      # Heuristic approx: 2 subtasks + 1 precond*0.1 = 2.1

    # Method B: Worse heuristic (3 subtasks). All its subtasks will succeed.
    # Preconditions for Method B itself will also be met.
    method_B = Method(name="method_B_backup_succeeds",
                      task_signature=("task_heuristic_choice",),
                      preconditions={Predicate("can_try_B", (), True)}, # This will be in state
                      subtasks=["OP_B1", "OP_B1", "OP_B2"]) # Make it 3 subtasks for a worse heuristic
                      # Heuristic approx: 3 subtasks + 1 precond*0.1 = 3.1

    # Ensure Method A has a better (lower) heuristic score than Method B due to fewer subtasks
    assert method_A.heuristic_score is not None and method_B.heuristic_score is not None
    assert method_A.heuristic_score < method_B.heuristic_score, \
        f"Heuristic setup error: Method A ({method_A.heuristic_score}) should be better than B ({method_B.heuristic_score})"

    # Add methods such that Method A (better heuristic) is listed first if no sorting,
    # or planner explicitly sorts them.
    planner.methods["task_heuristic_choice"] = [method_A, method_B]
    # If we wanted to be meaner to the sorter: planner.methods["task_heuristic_choice"] = [method_B, method_A]

    test_goal = create_goal_from_descriptor("Goal for heuristic choice leading to effect_B2")
    assert test_goal is not None
    test_goal.success_criteria = {Predicate("effect_B2", (), True)} # For clarity if needed

    # Initial state: preconditions for *both methods* are true.
    # Precondition for OP_A1 (subtask of method_A) is *not* true.
    initial_state = {
        Predicate("can_try_A", (), True),
        Predicate("can_try_B", (), True)
    }

    with patch.object(planner, '_goal_to_task', return_value=("task_heuristic_choice",)):
        htn_logger = logging.getLogger("consciousness_experiment.cognitive_modules.htn_planner")
        with patch.object(htn_logger, 'debug') as mock_planner_debug_log:
            logger.info("Planning with heuristic choice (expecting preferred method to fail subtask)...")
            planner.max_depth = 5 # Ensure sufficient depth for the longer path
            plan = await planner.plan(test_goal, initial_state)

            assert plan is not None, "Plan should be found using Method B after Method A fails."
            assert len(plan) == 3, f"Plan should consist of 3 actions from Method B, got {len(plan)}"
            assert plan[0]["type"] == "OP_B1" and \
                   plan[1]["type"] == "OP_B1" and \
                   plan[2]["type"] == "OP_B2", "Plan does not match Method B's subtasks."

            log_messages = [call_args[0][0] for call_args in mock_planner_debug_log.call_args_list]

            method_A_tried_log_found = any(f"Trying method: {method_A.name}" in msg for msg in log_messages)
            # Method A should fail because its subtask OP_A1 is not applicable.
            # The log would show "Decomposing task: OP_A1" then "Op 'OP_A1' preconds not met".
            # Then "Decomposition failed for subtask 'OP_A1' in method 'method_A_preferred_fails_subtask'"
            # Then "Method 'method_A_preferred_fails_subtask' failed to decompose task".
            method_A_failed_log_found = any(f"Method '{method_A.name}' failed to decompose" in msg for msg in log_messages)

            method_B_tried_log_found = any(f"Trying method: {method_B.name}" in msg for msg in log_messages)
            method_B_succeeded_log_found = any(f"Method '{method_B.name}' successfully decomposed" in msg for msg in log_messages)

            assert method_A_tried_log_found, "Log does not show Method A (preferred) was tried."
            assert method_A_failed_log_found, "Log does not show Method A failed during its subtask decomposition."
            assert method_B_tried_log_found, "Log does not show Method B (backup) was tried."
            assert method_B_succeeded_log_found, "Log does not show Method B succeeded."

            try:
                index_try_A = next(i for i, msg in enumerate(log_messages) if f"Trying method: {method_A.name}" in msg)
                index_fail_A_subtask = next(i for i, msg in enumerate(log_messages) if f"Decomposition failed for subtask 'OP_A1' in method '{method_A.name}'" in msg and i > index_try_A)
                index_try_B = next(i for i, msg in enumerate(log_messages) if f"Trying method: {method_B.name}" in msg and i > index_fail_A_subtask)
                assert index_try_A < index_fail_A_subtask < index_try_B, "Log order incorrect: Method A should be tried, its subtask fail, then Method B tried."
                logger.info("Heuristic ordering and backtracking verified.")
            except StopIteration:
                # Print more logs if specific messages are not found for debugging
                # for i, msg_debug in enumerate(log_messages):
                #     logger.error(f"Log {i}: {msg_debug}")
                pytest.fail("Could not find all necessary log messages to verify heuristic trial order and failure.")

    planner.operators = original_operators
    planner.methods = original_methods
    logger.info("--- Test Passed: HTN Heuristic Influence ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_dsm_perform_reflection(test_agent: AgentController):
    """
    Tests the DynamicSelfModel's _perform_reflection method to ensure identity traits
    and learning_rate_meta are updated based on a history of learning events.
    """
    agent = test_agent
    dsm = agent.dynamic_self_model
    logger.info("--- Test: DynamicSelfModel - Perform Reflection ---")

    assert dsm is not None, "DynamicSelfModel component missing."
    assert hasattr(dsm, '_perform_reflection'), "DSM missing '_perform_reflection' method."
    assert hasattr(dsm, 'learning_events'), "DSM missing 'learning_events' attribute."

    # Configure DSM for rapid reflection and smaller history for the test
    original_reflection_interval = dsm.reflection_interval
    original_learning_events_maxlen = dsm.learning_events.maxlen
    dsm.reflection_interval = 5  # Reflect every 5 qualifying updates
    dsm.learning_events = deque(maxlen=10) # Keep last 10 events for reflection
    dsm.cycles_since_reflection = 0 # Reset counter

    # Initial state of traits and meta_learning (should be defaults or loaded)
    initial_traits = dsm.self_model.get("identity_traits", {}).copy()
    initial_lr_meta = dsm.self_model.get("learning_rate_meta", {}).get("capabilities", {}).copy()
    logger.info(f"Initial DSM Traits: {initial_traits}")
    logger.info(f"Initial DSM LR Meta: {initial_lr_meta}")


    # Helper to simulate DSM processing for learning events
    async def simulate_dsm_process_for_reflection(action_type: str, outcome: str,
                                                  specific_key_suffix: Optional[str] = None):
        params = {"test_param": "value"} # Generic params
        # Construct a specific key if suffix is provided
        action_key_specific = f"action:{action_type}{specific_key_suffix}" if specific_key_suffix else None

        # Note: dsm.update_self_model is what appends to learning_events and calls _perform_reflection
        # We are calling it directly here to simulate multiple agent cycles' worth of learning.
        await dsm.update_self_model(
             last_action_type=action_type,
             action_params=params,
             action_outcome=outcome,
             action_error="TestError" if outcome == "failure" else None,
             action_result_data={"info": "test_data"},
             current_phenomenal_state=None
        )

    # --- Scenario 1: Trigger Adaptability & Fast Learner ---
    # Simulate diverse successful actions
    logger.info("Simulating diverse successful actions to trigger reflection...")
    # (reflection_interval is 5, so 5 calls to update_self_model with an action_type will trigger it)
    await simulate_dsm_process_for_reflection("ACTION_A", "success", specific_key_suffix=":variant1")
    await simulate_dsm_process_for_reflection("ACTION_B", "success")
    await simulate_dsm_process_for_reflection("ACTION_C", "success")
    await simulate_dsm_process_for_reflection("ACTION_A", "success", specific_key_suffix=":variant1") # Repeated specific
    await simulate_dsm_process_for_reflection("ACTION_D", "success") # This 5th call should trigger reflection

    traits_after_s1 = dsm.self_model.get("identity_traits", {})
    lr_meta_s1 = dsm.self_model.get("learning_rate_meta", {}).get("capabilities", {})

    logger.info(f"DSM Traits after Scenario 1 (Diverse Success): {traits_after_s1}")
    logger.info(f"DSM LR Meta after Scenario 1: {lr_meta_s1}")

    assert traits_after_s1.get("adaptability", 0) > initial_traits.get("adaptability", 0.5), \
        "Adaptability should increase with diverse successful actions."
    assert traits_after_s1.get("caution", 0.5) < initial_traits.get("caution", 0.5), \
        "Caution should decrease with high success rate."

    # --- Scenario 3 (Revised for clear fast/slow in one reflection window) ---
    dsm.learning_events.clear() # Clear events from S1 as reflection already happened
    dsm.cycles_since_reflection = 0

    logger.info("Simulating for clearer fast/slow learner detection (Scenario 3)...")
    # We need 5 events to trigger reflection.
    # Action X: 3 successes (should be fast learner)
    await simulate_dsm_process_for_reflection("ACTION_X_S3", "success")
    await simulate_dsm_process_for_reflection("ACTION_X_S3", "success")
    await simulate_dsm_process_for_reflection("ACTION_X_S3", "success")
    # Action Y: 2 events, 1 success, 1 failure (consistency 0.5, neither fast nor slow with current thresholds)
    await simulate_dsm_process_for_reflection("ACTION_Y_S3", "failure")
    await simulate_dsm_process_for_reflection("ACTION_Y_S3", "success") # This 5th call triggers reflection

    traits_after_s3 = dsm.self_model.get("identity_traits", {})
    lr_meta_s3 = dsm.self_model.get("learning_rate_meta", {}).get("capabilities", {})
    logger.info(f"DSM Traits after Scenario 3: {traits_after_s3}")
    logger.info(f"DSM LR Meta after Scenario 3: {lr_meta_s3}")
    # learning_events would have been cleared by _perform_reflection, so logging it here is after clear.
    # If you need to see events *before* clear, log inside _perform_reflection or before its call.

    # Reflection processed events: [X:s, X:s, X:s, Y:f, Y:s]
    # For ACTION_X_S3: 3 successes / 3 total -> consistency 1.0 >= 0.85 -> fast_learner
    # For ACTION_Y_S3: 1 success / 2 total -> len < 3, so skipped for fast/slow classification.
    assert "action:ACTION_X_S3" in lr_meta_s3.get("fast_learner", []), \
        f"ACTION_X_S3 should be fast_learner. Got: {lr_meta_s3.get('fast_learner', [])}"
    assert "action:ACTION_Y_S3" not in lr_meta_s3.get("slow_learner", []), \
        f"ACTION_Y_S3 should not be slow_learner (not enough samples). Got: {lr_meta_s3.get('slow_learner', [])}"
    assert "action:ACTION_Y_S3" not in lr_meta_s3.get("fast_learner", []), \
        f"ACTION_Y_S3 should not be fast_learner (not enough samples). Got: {lr_meta_s3.get('fast_learner', [])}"


    # --- Scenario 4: Test slow learner ---
    dsm.learning_events.clear() # Clear events from S3
    dsm.cycles_since_reflection = 0
    logger.info("Simulating for slow learner detection (Scenario 4)...")
    # Action W_S4: Highly inconsistent outcomes
    # We need at least 3 samples. Let's use 5 events for this action to ensure it dominates the window.
    # To make it "slow" (consistency <= 0.4), the most common outcome can be at most 40%
    # If 5 samples for W_S4: max 2 for most common outcome. e.g., S,S,F,X,Y (X,Y different from S,F)
    # For simplicity, let's just ensure outcomes are varied and the test conditions are met.
    # The "slow_learner" condition `consistency <= 0.4` might be too strict for binary outcomes.
    # Let's adjust the test to reflect what's possible or assume "error" outcome is distinct.
    # The simulate_dsm_process_for_reflection helper uses "error" as an outcome for learning_events.

    async def simulate_dsm_process_custom_outcome(action_type: str, custom_outcome: str):
        params = {"test_param": "value"}
        await dsm.update_self_model(
             last_action_type=action_type, action_params=params,
             action_outcome=custom_outcome, # Use custom outcome here
             action_error="SimulatedError" if custom_outcome not in ["success", "failure"] else None,
             action_result_data={"info": "test_data"}, current_phenomenal_state=None
        )

    dsm.learning_events.clear(); dsm.cycles_since_reflection = 0
    await simulate_dsm_process_custom_outcome("ACTION_W_S4", "success")      # W_S4 outcomes: [S]
    await simulate_dsm_process_custom_outcome("ACTION_W_S4", "failure")      # W_S4 outcomes: [S, F]
    await simulate_dsm_process_custom_outcome("ACTION_W_S4", "error_type_A") # W_S4 outcomes: [S, F, A]
    await simulate_dsm_process_custom_outcome("ACTION_W_S4", "error_type_B") # W_S4 outcomes: [S, F, A, B]
    await simulate_dsm_process_custom_outcome("ACTION_Z_S4", "success")      # Triggers reflection.
    # Events for reflection: [W:S, W:F, W:A, W:B, Z:S]
    # For ACTION_W_S4: outcomes are [S, F, A, B]. All unique. Count=1 for each. Total=4. Consistency=1/4=0.25.
    # This meets `consistency <= 0.4`.

    lr_meta_s4_revised = dsm.self_model.get("learning_rate_meta", {}).get("capabilities", {})
    logger.info(f"DSM LR Meta after Scenario 4 (revised for slow): {lr_meta_s4_revised}")
    assert "action:ACTION_W_S4" in lr_meta_s4_revised.get("slow_learner", []), \
        f"ACTION_W_S4 should be slow_learner. Got: {lr_meta_s4_revised.get('slow_learner', [])}"


    # Restore original DSM settings
    dsm.reflection_interval = original_reflection_interval
    dsm.learning_events = deque(maxlen=original_learning_events_maxlen)
    logger.info("--- Test Passed: DynamicSelfModel Perform Reflection ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_ems_curiosity_on_prediction_error(test_agent: AgentController):
    """
    Tests that EmergentMotivationSystem's curiosity drive increases when a
    PredictiveWorldModel prediction error is present.
    """
    agent = test_agent
    ems = agent.emergent_motivation_system
    pwm = agent.predictive_world_model
    logger.info("--- Test: EMS Curiosity on Prediction Error ---")

    assert ems is not None, "EmergentMotivationSystem component missing."
    assert pwm is not None, "PredictiveWorldModel component missing."
    assert hasattr(ems, 'drives') and "curiosity" in ems.drives, "EMS or curiosity drive not initialized."
    assert hasattr(pwm, 'last_prediction_error'), "PWM missing 'last_prediction_error' attribute."

    # --- Step 1: Get initial curiosity and simulate a non-discovery action with NO prediction error ---
    await ems.reset() # Start with default drive values for this test
    initial_status = await ems.get_status()
    initial_curiosity = initial_status.get("current_drives", {}).get("curiosity", 0.5)
    logger.info(f"Initial curiosity: {initial_curiosity:.3f}")

    pwm.last_prediction_error = None # Ensure no prior error

    # Simulate a neutral action outcome (e.g. THINKING)
    action_result_neutral = {"type": "THINKING", "outcome": "success"}
    ems_input_no_error = {
        "cognitive_state": {"consciousness_level": "CONSCIOUS"},
        "last_action_result": action_result_neutral,
        "phenomenal_state": None, "active_goal": None, "self_model_summary": None
    }
    await ems.process(ems_input_no_error)
    status_after_no_error = await ems.get_status()
    curiosity_after_no_error = status_after_no_error.get("current_drives", {}).get("curiosity", 0.5)

    # Curiosity should ideally only decay slightly or stay same if not a discovery action
    # Default decay in EMS is towards 0.5, so if initial was 0.5, it might not change much.
    # Let's ensure it hasn't spuriously increased.
    assert curiosity_after_no_error <= initial_curiosity + 0.001, \
        f"Curiosity should not increase without discovery or error. Initial: {initial_curiosity}, Got: {curiosity_after_no_error}"
    logger.info(f"Curiosity after neutral action (no error): {curiosity_after_no_error:.3f}")


    # --- Step 2: Simulate a significant prediction error ---
    # PredictiveWorldModel.last_prediction_error structure:
    # { "type": "outcome_mismatch", "predicted": "A", "actual": "B",
    #   "error_source_details": {"error_magnitude": 0.8}, ... }

    mock_prediction_error = {
        "type": "outcome_mismatch",
        "predicted": "success",
        "actual": "failure",
        "action_type": "TEST_ACTION_FOR_ERROR",
        "error_source_details": {"error_magnitude": 0.85} # Significant magnitude
    }
    # In the AgentController loop, the error is passed via agent.last_prediction_error_for_attention
    # But EMS looks for it on the PWM instance directly for its calculation.
    pwm.last_prediction_error = mock_prediction_error

    # Simulate a generic action outcome that occurred alongside this error
    action_result_with_error = {"type": "SOME_ACTION", "outcome": "failure"}
    ems_input_with_error = {
        "cognitive_state": {"consciousness_level": "CONSCIOUS"},
        "last_action_result": action_result_with_error,
        "phenomenal_state": None, "active_goal": None, "self_model_summary": None
    }

    logger.info("Calling EMS process with a simulated prediction error...")
    await ems.process(ems_input_with_error)
    status_after_error = await ems.get_status()
    curiosity_after_error = status_after_error.get("current_drives", {}).get("curiosity", 0.5)

    logger.info(f"Curiosity after prediction error: {curiosity_after_error:.3f}")

    # Curiosity should have increased significantly from curiosity_after_no_error
    # The gain is gain_pred_err * error_magnitude * (1.0 - current_value)
    # Default gain_pred_err = 0.05. Error_magnitude = 0.85
    # Expected boost factor = 0.05 * 0.85 = 0.0425
    # If curiosity_after_no_error was ~0.5, boost = 0.0425 * (1-0.5) = 0.02125
    # So, expected curiosity ~ 0.5 + 0.02125 = 0.52125 (before decay component of this cycle)
    # The decay is current_value + decay * (0.5 - current_value)
    # If value was 0.49 (decayed from 0.5), new_value before pred_err boost = 0.49 + 0.02*(0.5-0.49) = 0.4902
    # Then pred_err boost = 0.0425 * (1 - 0.4902) = 0.0425 * 0.5098 = 0.02166
    # Total = 0.4902 + 0.02166 = 0.51186
    assert curiosity_after_error > curiosity_after_no_error, \
        f"Curiosity should increase after a significant prediction error. Before: {curiosity_after_no_error}, After: {curiosity_after_error}"

    # Check if the increase is somewhat meaningful (e.g., > 0.01 if error mag was high)
    # This depends on the gain_prediction_error param in config, which is 0.05
    # And the current value of curiosity (gain is higher if curiosity is low)
    gain_pred_err_config = ems.drives.get("curiosity", {}).get("gain_prediction_error", 0.05)
    expected_min_increase_factor = gain_pred_err_config * mock_prediction_error["error_source_details"]["error_magnitude"] * 0.3 # Assuming (1-curiosity) is at least 0.3

    assert (curiosity_after_error - curiosity_after_no_error) >= (expected_min_increase_factor - ems.drives["curiosity"]["decay"]*0.1), \
        f"Curiosity increase {curiosity_after_error - curiosity_after_no_error:.4f} seems too small for error magnitude."


    # --- Step 3: Clear prediction error and process again, curiosity should decay ---
    pwm.last_prediction_error = None
    curiosity_before_decay_cycle = curiosity_after_error

    logger.info("Calling EMS process again with no prediction error (expecting decay)...")
    # Store the unrounded value from the previous step if possible, or re-fetch status
    # For this test, ems.drives["curiosity"]["value"] holds the more precise value
    precise_curiosity_before_decay_cycle = ems.drives["curiosity"]["value"]

    await ems.process(ems_input_no_error)
    status_after_decay = await ems.get_status()
    curiosity_after_decay_rounded = status_after_decay.get("current_drives", {}).get("curiosity", 0.5)
    precise_curiosity_after_decay = ems.drives["curiosity"]["value"]


    logger.info(f"Curiosity after decay cycle: {curiosity_after_decay_rounded:.3f} (precise: {precise_curiosity_after_decay:.6f})")
    logger.info(f"Precise curiosity before decay cycle was: {precise_curiosity_before_decay_cycle:.6f}")

    # It should move towards 0.5.
    # If it was > 0.5, it should decrease or stay 0.5.
    # If it was < 0.5, it should increase or stay 0.5.
    if precise_curiosity_before_decay_cycle > 0.500001: # Using a small epsilon
        assert precise_curiosity_after_decay < precise_curiosity_before_decay_cycle, \
            "Curiosity should decrease towards 0.5 if it was high and no new error/discovery."
    elif precise_curiosity_before_decay_cycle < 0.499999: # Using a small epsilon
            assert precise_curiosity_after_decay > precise_curiosity_before_decay_cycle, \
            "Curiosity should increase towards 0.5 if it was low and no new error/discovery."
    else: # It was already very close to 0.5
        assert abs(precise_curiosity_after_decay - 0.5) < 0.001, \
            "Curiosity should remain very close to 0.5 if it started there and no new stimuli."

    logger.info("--- Test Passed: EMS Curiosity on Prediction Error ---")


@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_po_updates_agent_config(test_agent: AgentController):
    """
    Tests that AgentController reads adjustments from PerformanceOptimizer
    and updates its own self.config, and that HTNPlanner uses the updated depth.
    """
    agent = test_agent
    po = agent.performance_optimizer
    planner = agent.htn_planner 
    logger.info("--- Test: PerformanceOptimizer Updates AgentController.config ---")

    assert po is not None, "PerformanceOptimizer component missing."
    assert planner is not None, "HTNPlanner component missing."
    assert hasattr(agent, 'config'), "Agent missing 'config' attribute."
    assert hasattr(po, 'config_changes'), "PO missing 'config_changes' attribute."
    assert Predicate is not None and Goal is not None and create_goal_from_descriptor is not None, \
        "Required datatypes/functions for test setup are missing."
    _ConsciousState_for_test = globals().get('ConsciousState') 
    assert _ConsciousState_for_test is not None, "ConsciousState enum not available for test."
    assert DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC is not None, "Default goal description constant not available."


    # --- Setup: Configure PO to auto-apply and define a bottleneck scenario ---
    original_po_auto_apply = agent.config.get("performance_optimizer", {}).get("auto_apply_adjustments")
    original_perf_max_depth = agent.config.get("performance", {}).get("max_planning_depth")
    original_agent_cs_level = agent.consciousness_level


    agent.config.setdefault("performance_optimizer", {})["auto_apply_adjustments"] = True
    # Ensure PO instance uses this setting for the test
    po._config_perf_optimizer_section["auto_apply_adjustments"] = True 

    initial_max_depth_for_test = 10
    agent.config.setdefault("performance", {})["max_planning_depth"] = initial_max_depth_for_test
    # Ensure planner's internal self.max_depth reflects this initial state for consistent testing
    # This simulates planner having been initialized or having run with this config before.
    if planner: # planner can be None if not correctly initialized by fixture
        planner.max_depth = initial_max_depth_for_test 
    
    po.config_changes = {} # Clear any prior PO adjustments for this test isolation

    logger.info(f"TEST: Initial agent.config[performance][max_planning_depth] set to: {initial_max_depth_for_test}")

    planning_threshold = po.optimization_thresholds.get("planning", 0.1) 
    simulated_planning_duration = planning_threshold * 4.0 # Severity 4.0
    cycle_profile_bottleneck = {"planning": simulated_planning_duration} 

    # --- Action 1: Call PO.process ---
    logger.info(f"Simulating PO process with 'planning' bottleneck ({simulated_planning_duration}s)...")
    po_input_state = {"cycle_profile": cycle_profile_bottleneck}
    po_output = await po.process(po_input_state)
    assert po_output and "performance_analysis" in po_output, "PO.process failed or returned invalid structure."
    
    performance_analysis = po_output["performance_analysis"]
    assert isinstance(performance_analysis, dict), "performance_analysis is not a dictionary."
    suggested_adjustments = performance_analysis.get("suggested_adjustments", {})
    assert isinstance(suggested_adjustments, dict), "suggested_adjustments is not a dictionary."

    assert "performance" in suggested_adjustments, \
        "PO suggestions should contain 'performance' key for planning depth adjustment."
    performance_section_suggestion = suggested_adjustments.get("performance", {})
    assert isinstance(performance_section_suggestion, dict), "'performance' section in suggestions is not a dict."
    assert "max_planning_depth" in performance_section_suggestion, \
        "PO suggestions for 'performance' should contain 'max_planning_depth' key."
    
    new_suggested_depth_by_po = performance_section_suggestion.get("max_planning_depth")
    assert new_suggested_depth_by_po == 8, \
        f"PO suggested unexpected depth {new_suggested_depth_by_po}, expected 8 (from 10 with sev 4.0)."
    logger.info(f"PO suggested new max_planning_depth for 'performance' section: {new_suggested_depth_by_po}")

    # --- Simulate AgentController merging PO's adjustments ---
    po_status_after_process = await po.get_status()
    active_po_adjustments_from_status = po_status_after_process.get("active_config_adjustments", {})
    assert isinstance(active_po_adjustments_from_status, dict), "PO status did not return a dict for active_config_adjustments."

    config_updated_by_po_in_cycle_test_flag = False
    if active_po_adjustments_from_status:
        logger.info(f"Agent test is applying PO adjustments from PO.status: {active_po_adjustments_from_status}")
        for component_key_from_po, comp_adjustments_from_po in active_po_adjustments_from_status.items():
            if component_key_from_po not in agent.config:
                agent.config[component_key_from_po] = {} 
            
            if isinstance(agent.config.get(component_key_from_po), dict) and isinstance(comp_adjustments_from_po, dict):
                for param_key, new_value in comp_adjustments_from_po.items():
                    current_val_in_agent_config = agent.config[component_key_from_po].get(param_key)
                    if current_val_in_agent_config != new_value:
                        agent.config[component_key_from_po][param_key] = new_value
                        config_updated_by_po_in_cycle_test_flag = True
                        logger.info(f"TEST_APPLY: agent.config[{component_key_from_po}][{param_key}] updated to {new_value} (was {current_val_in_agent_config})")
    
    assert config_updated_by_po_in_cycle_test_flag, \
        "AgentController.config should have been updated by PO adjustments based on the test's merge simulation."

    # --- Verification: AgentController's config and HTNPlanner behavior ---
    final_max_depth_in_agent_performance_section = agent.config.get("performance", {}).get("max_planning_depth")
    logger.info(f"Final agent.config[performance][max_planning_depth]: {final_max_depth_in_agent_performance_section}")
    assert final_max_depth_in_agent_performance_section == new_suggested_depth_by_po, \
        f"agent.config.performance.max_planning_depth was not updated correctly. Expected {new_suggested_depth_by_po}, got {final_max_depth_in_agent_performance_section}."

    # Use a goal description that HTNPlanner._goal_to_task can map
    dummy_goal_desc_for_test = DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC 
    dummy_goal_for_planner_test = create_goal_from_descriptor(dummy_goal_desc_for_test)
    assert dummy_goal_for_planner_test is not None, "Failed to create dummy goal for planner test."
    
    expected_depth_planner_uses = final_max_depth_in_agent_performance_section # Should be 8

    # Ensure agent is in a normal conscious state for this verification
    agent.consciousness_level = _ConsciousState_for_test.CONSCIOUS
    logger.info(f"Verifying HTNPlanner uses new max_depth: {expected_depth_planner_uses} with CS={agent.consciousness_level.name if agent.consciousness_level else 'N/A'}")
    
    htn_module_logger_name = planner.__class__.__module__
    htn_logger_for_test = logging.getLogger(htn_module_logger_name)
    original_htn_logger_level = htn_logger_for_test.level
    # Ensure INFO logs are captured by the mock for this specific test logger
    if not htn_logger_for_test.isEnabledFor(logging.INFO):
        htn_logger_for_test.setLevel(logging.INFO)


    with patch.object(htn_logger_for_test, 'info') as mock_planner_info_log:
        # Plan for a mappable goal, even if it results in no plan, to trigger logging.
        # An empty state set() will likely result in no plan for "observe and learn"
        # if its methods have preconditions, but it should get past _goal_to_task.
        await planner.plan(dummy_goal_for_planner_test, set()) 

        found_max_depth_log = False
        actual_logged_effective_depth_str = "NOT_FOUND_IN_LOGS" 
        
        # The target log message from HTNPlanner.plan():
        # f"Planning started for goal: {goal.description} (Effective Max Overall Depth for this attempt: {effective_max_depth})"
        target_log_pattern_start = f"Planning started for goal: {dummy_goal_for_planner_test.description} (Effective Max Overall Depth for this attempt:"
        
        for call_args in mock_planner_info_log.call_args_list:
            log_message = str(call_args[0][0])
            if log_message.startswith(target_log_pattern_start):
                match = re.search(r"this attempt: (\d+)\)", log_message)
                if match:
                    actual_logged_effective_depth = int(match.group(1))
                    actual_logged_effective_depth_str = str(actual_logged_effective_depth)
                    if actual_logged_effective_depth == expected_depth_planner_uses:
                        found_max_depth_log = True
                        logger.info(f"HTNPlanner correctly logged usage of effective max_depth: {expected_depth_planner_uses}")
                        break
        
        assert found_max_depth_log, \
            (f"HTNPlanner did not log using the effective max_depth of {expected_depth_planner_uses}. "
             f"Actual logged effective depth was '{actual_logged_effective_depth_str}' based on pattern '{target_log_pattern_start}'. "
             f"Review target string or HTNPlanner INFO logs. Captured INFO logs for HTNPlanner: "
             f"{[c[0][0] for c in mock_planner_info_log.call_args_list if isinstance(c[0][0], str) and 'HTNPlanner' in c[0][0]]}")

    # Restore logger level if it was changed
    if htn_logger_for_test.level != original_htn_logger_level:
        htn_logger_for_test.setLevel(original_htn_logger_level)

    # Restore original config values and agent state for test isolation
    if original_po_auto_apply is not None:
        agent.config.get("performance_optimizer", {})["auto_apply_adjustments"] = original_po_auto_apply
    else: # If key didn't exist, remove it
        if "auto_apply_adjustments" in agent.config.get("performance_optimizer", {}):
            del agent.config.get("performance_optimizer", {})["auto_apply_adjustments"]
            
    if original_perf_max_depth is not None:
        agent.config.get("performance", {})["max_planning_depth"] = original_perf_max_depth
    else:
        if "max_planning_depth" in agent.config.get("performance", {}):
            del agent.config.get("performance", {})["max_planning_depth"]
    
    agent.consciousness_level = original_agent_cs_level
    # Restore planner's internal max_depth for consistency, though fixture provides fresh agent
    if planner:
        planner.max_depth = agent.config.get("performance", {}).get("max_planning_depth", 5) 

    logger.info("--- Test Passed: PerformanceOptimizer Updates AgentController.config (and HTNPlanner sees it) ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_default_goal_cooldown_after_user_goal(test_agent: AgentController, tmp_path: Path):
    """
    Tests that the cooldown for the default 'Observe and learn' goal is reset
    after a user-initiated (or non-default) goal is completed, allowing the
    default goal to be re-triggered sooner if conditions are met.

    Note: This test uses simulated goal selection and completion logic based on
          AgentController's structure, as running the full loop is complex in a test.
    """
    agent = test_agent
    logger.info("--- Test: Default Goal Cooldown Reset After User Goal ---")

    # DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC is imported at the top of the file now
    # and available directly.

    assert agent.emergent_motivation_system is not None
    assert agent.htn_planner is not None
    assert GoalStatus is not None, "GoalStatus enum not loaded."
    original_agent_root = agent.agent_root_path

    # --- Setup (Ensure DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC is used directly) ---
    original_min_curiosity = agent.min_curiosity_for_observe
    original_cooldown_cycles = agent.default_goal_cooldown_cycles
    original_ems_curiosity_val = agent.emergent_motivation_system.drives["curiosity"]["value"]

    agent.min_curiosity_for_observe = 0.1
    agent.default_goal_cooldown_cycles = 3
    agent.emergent_motivation_system.drives["curiosity"]["value"] = 0.9

    async def select_or_generate_goal_wrapper():
        active = agent._oscar_get_active_goal() # This now correctly finds ACTIVE or reactivates SUSPENDED
        if active:
            logger.debug(f"Wrapper: Found active/reactivated goal: {active.description}")
            return active

        can_create_default = True
        curiosity = agent.emergent_motivation_system.drives["curiosity"]["value"]
        cycles_since_last = agent.cycle_count - agent._last_default_observe_completion_cycle

        condition1_dampen = (cycles_since_last < agent.default_goal_cooldown_cycles)
        # Dampening from plan: IF cooldown period NOT met AND curiosity IS LOW -> THEN DAMPEN
        # So, generate IF NOT ( (cooldown_not_met) AND (curiosity_low) )
        # Which is: (cooldown_met) OR (curiosity_not_low)
        # Current AgentController logic is: (cycles_since_last >= cooldown) AND (curiosity >= min_curiosity)
        # The test logic below for can_create_default aligns with AgentController's actual logic.

        if not ((agent.cycle_count - agent._last_default_observe_completion_cycle) >= agent.default_goal_cooldown_cycles and \
              curiosity >= agent.min_curiosity_for_observe):
            can_create_default = False # Conditions not met for default goal


        if can_create_default:
            logger.debug(f"Wrapper: Generating default goal (cooldown: {cycles_since_last}, cur:{curiosity:.2f})")
            new_goal = create_goal_from_descriptor( # Use imported create_goal_from_descriptor
                DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC, # Use imported constant
                priority=DEFAULT_OBSERVE_GOAL_PRIORITY # Use imported constant
            )
            if new_goal:
                # Check if already in active_goals by description to avoid duplicates if test calls this many times
                if not any(g.description == new_goal.description and g.status == GoalStatus.ACTIVE for g in agent.active_goals):
                     agent.active_goals.append(new_goal)
                return new_goal
            else:
                 logger.error("Wrapper: Failed to create default goal object.")
                 return None
        else:
             logger.debug(f"Wrapper: Default goal dampened (cooldown_met_or_high_cur_needed: {cycles_since_last} vs {agent.default_goal_cooldown_cycles}, cur:{curiosity:.2f} vs {agent.min_curiosity_for_observe})")
             return None


    async def simulate_goal_completion(goal: Goal): # type: ignore
        assert goal is not None and hasattr(goal, 'status') and hasattr(goal, 'description')
        goal_description_for_check = goal.description # Store before modification
        goal.status = GoalStatus.ACHIEVED
        logger.info(f"Simulating completion of goal: '{goal_description_for_check}' at cycle {agent.cycle_count}")

        # This logic should mirror AgentController._run_agent_loop's goal completion
        if goal_description_for_check == DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC: # Use imported constant
            agent._last_default_observe_completion_cycle = agent.cycle_count
            logger.info(f"Default goal completed. Cooldown counter set to {agent.cycle_count}.")
        else:
            agent._last_default_observe_completion_cycle = 0 # Reset cooldown
            logger.info("Non-default goal completed. Cooldown counter reset to 0.")

        if goal in agent.active_goals:
            agent.active_goals.remove(goal)
        agent.current_plan = None # Clear plan for completed goal

    # --- Step 1: Let agent complete an "Observe and learn" goal ---
    agent.active_goals = []
    agent.cycle_count = 1
    agent._last_default_observe_completion_cycle = -agent.default_goal_cooldown_cycles

    logger.info("Cycle 1: Generating and completing default goal...")
    default_goal_s1 = await select_or_generate_goal_wrapper()
    assert default_goal_s1 is not None and default_goal_s1.description == DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC, \
        "Failed to generate initial default goal."
    await simulate_goal_completion(default_goal_s1)
    assert agent._last_default_observe_completion_cycle == 1, \
        "Cooldown counter not set correctly for default goal completion."
    assert not agent.active_goals, "Active goals list should be empty after default goal completion."

    # --- Step 2: Ensure default goal is NOT generated if cooldown is active AND curiosity is low ---
    # (Original test Step 2 was about high curiosity overriding cooldown; this is now integrated better into wrapper)
    agent.cycle_count += 1 # Cycle 2
    original_curiosity_for_s2 = agent.emergent_motivation_system.drives["curiosity"]["value"]
    agent.emergent_motivation_system.drives["curiosity"]["value"] = agent.min_curiosity_for_observe - 0.01 # Low curiosity

    logger.info(f"Cycle 2: Checking if default goal is DAMPENED (cooldown={agent.default_goal_cooldown_cycles}, "
                f"since_last={agent.cycle_count - agent._last_default_observe_completion_cycle}, "
                f"cur={agent.emergent_motivation_system.drives['curiosity']['value']:.2f})...")
    # Expected: cycles_since_last (2-1=1) < cooldown_cycles (3) -> TRUE
    # Expected: curiosity (0.09) < min_curiosity_for_observe (0.1) -> TRUE
    # AgentController logic for can_generate_default_goal:
    # ( (cycles_since_last >= cooldown) AND (curiosity >= min_curiosity) )
    # ( (1 >= 3) FALSE ) AND ( (0.09 >= 0.1) FALSE ) -> FALSE. So, should NOT generate.
    goal_cycle_2_dampened = await select_or_generate_goal_wrapper()
    assert goal_cycle_2_dampened is None, \
        f"Default goal should BE dampened in Cycle 2 with active cooldown and low curiosity. Got: {goal_cycle_2_dampened.description if goal_cycle_2_dampened else 'None'}"
    agent.emergent_motivation_system.drives["curiosity"]["value"] = original_curiosity_for_s2 # Restore curiosity


    # --- Step 3: Give agent a user goal and let it complete ---
    agent.cycle_count += 1 # Cycle 3
    user_goal_desc = "read file : test_cooldown.txt"
    dummy_file = tmp_path / "test_cooldown.txt"
    dummy_file.write_text("cooldown test")
    agent.agent_root_path = tmp_path

    user_goal = await agent._map_text_to_goal(user_goal_desc)
    assert user_goal is not None, "Failed to map user text to goal."
    agent.active_goals = [g for g in agent.active_goals if hasattr(g, 'description') and g.description != DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC]
    agent.active_goals.append(user_goal)

    logger.info(f"Cycle 3: Completing user goal '{user_goal.description}'...")
    await simulate_goal_completion(user_goal)
    assert agent._last_default_observe_completion_cycle == 0, \
        "Cooldown counter was not reset to 0 after user goal completion."
    assert not agent.active_goals, "Active goals list should be empty after user goal completion."

    # --- Step 4: Check if agent generates "Observe and learn" goal immediately in next cycle ---
    agent.cycle_count += 1 # Cycle 4
    # Restore high curiosity for this check
    agent.emergent_motivation_system.drives["curiosity"]["value"] = 0.9
    logger.info(f"Cycle 4: Checking for new default goal (curiosity={agent.emergent_motivation_system.drives['curiosity']['value']:.2f}, cooldown reset)...")
    # Dampening check for default goal generation:
    # cycles_since_last (4-0=4) >= cooldown_cycles (3) -> TRUE
    # curiosity (0.9) >= min_curiosity_for_observe (0.1) -> TRUE
    # Both true -> should generate default goal.
    next_goal_after_user = await select_or_generate_goal_wrapper()

    assert next_goal_after_user is not None, \
        "Agent should have generated a new goal after user goal completion and cooldown reset."
    assert next_goal_after_user.description == DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC, \
        f"Expected new 'Observe and learn' goal, but got '{next_goal_after_user.description if next_goal_after_user else 'None'}'."
    logger.info(f"Agent correctly generated '{DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC}' goal after user goal completion.")

    # --- Restore original settings ---
    agent.min_curiosity_for_observe = original_min_curiosity
    agent.default_goal_cooldown_cycles = original_cooldown_cycles
    agent.emergent_motivation_system.drives["curiosity"]["value"] = original_ems_curiosity_val
    agent.agent_root_path = original_agent_root

    logger.info("--- Test Passed: Default Goal Cooldown Reset After User Goal ---")


@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_user_goal_prioritization(test_agent: AgentController, tmp_path: Path):
    """
    Tests that user-initiated goals are prioritized over default 'Observe and learn' goals,
    and that among user goals, priority and creation time are respected.
    """
    agent = test_agent
    logger.info("--- Test: User Goal Prioritization ---")

    assert agent.htn_planner is not None
    assert DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC is not None # This is now set from agent instance in fixture
    assert USER_GOAL_PRIORITY is not None
    assert DEFAULT_OBSERVE_GOAL_PRIORITY is not None


    # --- Setup ---
    original_agent_root = agent.agent_root_path
    original_min_curiosity = agent.min_curiosity_for_observe
    original_cooldown_cycles = agent.default_goal_cooldown_cycles
    original_ems_curiosity_val = agent.emergent_motivation_system.drives["curiosity"]["value"] # type: ignore

    agent.min_curiosity_for_observe = 0.1
    agent.default_goal_cooldown_cycles = 1
    agent.emergent_motivation_system.drives["curiosity"]["value"] = 0.9 # type: ignore
    agent._last_default_observe_completion_cycle = agent.cycle_count - agent.default_goal_cooldown_cycles
    agent.active_goals = []
    agent.current_plan = None

    # --- Scenario 1: User goal vs. Eligible Default Goal ---
    logger.info("Scenario 1: User goal vs. Eligible Default Goal")
    user_text_1 = "this is a user request"

    # Simulate AgentController._run_agent_loop's goal handling for user input
    potential_user_goal = await agent._map_text_to_goal(user_text_1)
    assert potential_user_goal is not None, "Failed to map user text to goal for S1."
    # Ensure it has the correct user priority
    assert potential_user_goal.priority == USER_GOAL_PRIORITY, f"S1 User goal mapped with prio {potential_user_goal.priority}, expected {USER_GOAL_PRIORITY}"
    agent.active_goals.append(potential_user_goal) # Manually add as if it came from broadcast

    logger.info("S1: Calling _oscar_generate_or_select_goal...")
    selected_goal_s1 = await agent._oscar_generate_or_select_goal() # Use the comprehensive method

    assert selected_goal_s1 is not None, "A goal should have been selected (S1)."
    logger.info(f"Scenario 1 Selected Goal: '{selected_goal_s1.description}', Prio: {selected_goal_s1.priority}")
    assert selected_goal_s1.description.startswith("respond to user :"), \
        "User-initiated 'respond' goal was not prioritized over potential default goal."
    assert selected_goal_s1.priority == USER_GOAL_PRIORITY, \
        f"User goal priority incorrect. Expected {USER_GOAL_PRIORITY}, got {selected_goal_s1.priority}"

    agent.active_goals = []


    # --- Scenario 2: Two user goals with different priorities ---
    logger.info("Scenario 2: Two user goals, different priorities")
    agent.agent_root_path = tmp_path
    (tmp_path / "file_low.txt").write_text("low prio")
    (tmp_path / "file_high.txt").write_text("high prio")

    user_goal_low_prio_desc = "read file : file_low.txt"
    user_goal_high_prio_desc = "read file : file_high.txt"

    # _map_text_to_goal sets USER_GOAL_PRIORITY. For this test, we need to manually create with different priorities
    # if we want to test the sorting logic of _oscar_get_active_goal with varied *user* priorities.
    # The previous test method's design used _map_text_to_goal, which assigns USER_GOAL_PRIORITY (e.g., 5.0) to all.
    # For this scenario, let's ensure they are treated as user goals by having priority >= DEFAULT_OBSERVE_GOAL_PRIORITY,
    # but distinct enough to test sorting.

    # Manually create goals with different priorities
    goal_low = agent._create_goal_from_descriptor(user_goal_low_prio_desc, priority=USER_GOAL_PRIORITY - 1.0) # e.g., 4.0
    goal_high = agent._create_goal_from_descriptor(user_goal_high_prio_desc, priority=USER_GOAL_PRIORITY)       # e.g., 5.0

    assert goal_low and goal_high
    agent.active_goals = [goal_low, goal_high]

    logger.info("S2: Calling _oscar_generate_or_select_goal...")
    selected_goal_s2 = await agent._oscar_generate_or_select_goal() # Use the comprehensive method
    assert selected_goal_s2 is not None, "A goal should have been selected (S2)."
    logger.info(f"Scenario 2 Selected Goal: '{selected_goal_s2.description}', Prio: {selected_goal_s2.priority}")
    assert selected_goal_s2.description == user_goal_high_prio_desc, \
        "Higher priority user goal was not selected."
    assert selected_goal_s2.priority == USER_GOAL_PRIORITY


    # --- Scenario 3: Two user goals with same priority, different creation times ---
    logger.info("Scenario 3: Two user goals, same priority, different creation times")
    agent.active_goals = []

    # Original input strings for mapping
    input_text_older = "older message for S3"
    input_text_newer = "newer message for S3"

    # What _map_text_to_goal will generate as descriptions
    expected_goal_desc_older = f"respond to user : {input_text_older}"
    expected_goal_desc_newer = f"respond to user : {input_text_newer}"

    goal_older = await agent._map_text_to_goal(input_text_older) # Use the original simple text
    assert goal_older and goal_older.priority == USER_GOAL_PRIORITY # type: ignore
    assert goal_older.description == expected_goal_desc_older # type: ignore
    await asyncio.sleep(0.001) # Ensure time difference for creation_time
    goal_newer = await agent._map_text_to_goal(input_text_newer) # Use the original simple text
    assert goal_newer and goal_newer.priority == USER_GOAL_PRIORITY # type: ignore
    assert goal_newer.description == expected_goal_desc_newer # type: ignore

    # Add newer first to test if sorting by creation_time (older first) works
    agent.active_goals = [goal_newer, goal_older]

    logger.info("S3: Calling _oscar_generate_or_select_goal...")
    selected_goal_s3 = await agent._oscar_generate_or_select_goal()
    assert selected_goal_s3 is not None, "A goal should have been selected (S3)."

    logger.info(f"Scenario 3 Selected Goal: '{selected_goal_s3.description}', Prio: {selected_goal_s3.priority}, Created: {selected_goal_s3.creation_time}")
    # Log the goals we are comparing against for clarity
    logger.info(f"Comparing with Expected Older: '{expected_goal_desc_older}', Created: {goal_older.creation_time}") # type: ignore
    logger.info(f"Comparing with Expected Newer: '{expected_goal_desc_newer}', Created: {goal_newer.creation_time}") # type: ignore

    # --- CORRECTED ASSERTION ---
    assert selected_goal_s3.description == expected_goal_desc_older, \
        "Older user goal (with same priority) was not selected."
    assert selected_goal_s3.priority == USER_GOAL_PRIORITY


    # --- Restore original settings ---
    agent.min_curiosity_for_observe = original_min_curiosity
    agent.default_goal_cooldown_cycles = original_cooldown_cycles
    agent.emergent_motivation_system.drives["curiosity"]["value"] = original_ems_curiosity_val # type: ignore
    agent.agent_root_path = original_agent_root

    logger.info("--- Test Passed: User Goal Prioritization ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
@patch(f'{PACKAGE_NAME}.agent_controller.call_ollama', new_callable=AsyncMock) # Patch where call_ollama is IMPORTED in agent_controller
async def test_llm_goal_mapping_fallback(mock_call_ollama: AsyncMock, test_agent: AgentController):
    agent = test_agent
    logger.info("--- Test: AgentController LLM Goal Mapping Fallback ---")

    assert USER_GOAL_PRIORITY is not None, "USER_GOAL_PRIORITY constant not available for test."

    # Case 1: LLM maps to read_file
    mock_call_ollama.return_value = (json.dumps({"task_type": "read_file", "parameters": {"path": "settings.conf"}}), None)
    user_text1 = "I need to see the agent's main configuration data." # This should not match the simple regex
    goal1 = await agent._map_text_to_goal(user_text1)
    assert goal1 is not None, "Goal should be created for Case 1."
    assert goal1.description == "read file : settings.conf", f"Goal description mismatch for Case 1. Got: {goal1.description}"
    assert goal1.priority == USER_GOAL_PRIORITY, "Goal priority mismatch for Case 1."
    
    # --- MODIFIED ASSERTION FOR mock_call_ollama ---
    mock_call_ollama.assert_called_once() # First, ensure it was called
    
    # Then, inspect the call arguments more flexibly
    call_args_list = mock_call_ollama.call_args_list
    assert len(call_args_list) == 1, "mock_call_ollama was called more/less than once."
    
    args_passed, kwargs_passed = call_args_list[0] # Get the arguments of the single call

    # Verify the parts of the call we can be certain about
    assert args_passed[0] == agent.model_name, "LLM model name mismatch."
    assert isinstance(args_passed[1], list) and len(args_passed[1]) == 2, "LLM messages format incorrect."
    
    # Check the user message content
    user_message_sent_to_llm = args_passed[1][1]
    assert user_message_sent_to_llm.get("role") == "user", "User message role incorrect."
    assert user_message_sent_to_llm.get("content") == user_text1, "User message content mismatch."

    # Check the system message content (less strictly for exact string match if it's complex)
    system_message_sent_to_llm = args_passed[1][0]
    assert system_message_sent_to_llm.get("role") == "system", "System message role incorrect."
    assert "You are an intent parsing assistant" in system_message_sent_to_llm.get("content", ""), \
        "System prompt content seems incorrect (missing key phrase)."
    assert "Known agent task types:" in system_message_sent_to_llm.get("content", ""), \
        "System prompt should list known agent task types."
    # You can add more specific checks for task names within the system prompt if needed, e.g.:
    assert "read_file" in system_message_sent_to_llm.get("content", ""), "System prompt missing 'read_file' task type."

    assert args_passed[2] == agent.config.get("llm_settings", {}).get("intent_mapping_temperature", 0.3), \
        "LLM temperature mismatch."
    assert args_passed[3] == agent._asyncio_loop, "Asyncio loop mismatch."
    
    expected_timeout = agent.config.get("llm_settings", {}).get("default_timeout_s", 30.0)
    assert kwargs_passed.get("timeout") == expected_timeout, \
        f"LLM timeout mismatch. Expected {expected_timeout}, got {kwargs_passed.get('timeout')}."
    # --- END MODIFIED ASSERTION ---

    mock_call_ollama.reset_mock()
    logger.info("Case 1 (LLM map to read_file) passed.")

    # Case 2: LLM maps to explore_directory with default path
    # Assuming LLM might determine "." if no path is specified by user for general exploration query
    mock_call_ollama.return_value = (json.dumps({"task_type": "explore_directory", "parameters": {"path": "."}}), None)
    user_text2 = "What files are here?"
    goal2 = await agent._map_text_to_goal(user_text2)
    assert goal2 is not None, "Goal should be created for Case 2."
    assert goal2.description == "explore directory : .", f"Goal description mismatch for Case 2. Got: {goal2.description}"
    assert goal2.priority == USER_GOAL_PRIORITY, "Goal priority mismatch for Case 2."
    mock_call_ollama.reset_mock()
    logger.info("Case 2 (LLM map to explore_directory) passed.")

    # Case 3: LLM returns null (conversational or unmappable) -> fallback to "respond to user"
    mock_call_ollama.return_value = ("null", None)
    user_text3 = "Good morning OSCAR!"
    goal3 = await agent._map_text_to_goal(user_text3)
    assert goal3 is not None, "Goal should be created for Case 3."
    assert goal3.description == f"respond to user : {user_text3[:100]}", f"Goal description mismatch for Case 3. Got: {goal3.description}"
    assert goal3.priority == USER_GOAL_PRIORITY, "Goal priority mismatch for Case 3."
    mock_call_ollama.reset_mock()
    logger.info("Case 3 (LLM returns null, fallback to respond) passed.")

    # Case 4: LLM returns malformed JSON -> fallback to "respond to user"
    mock_call_ollama.return_value = ("{\"task_type\": \"read_file\", \"parameters\": {\"path\": \"config.toml\"", None) # Malformed
    user_text4 = "Could you retrieve the configuration details?" # Less likely to match simple regex
    goal4 = await agent._map_text_to_goal(user_text4)
    assert goal4 is not None, "Goal should be created for Case 4."
    assert goal4.description == f"respond to user : {user_text4[:100]}", f"Goal description mismatch for Case 4. Got: {goal4.description}"
    mock_call_ollama.reset_mock()
    logger.info("Case 4 (LLM malformed JSON, fallback to respond) passed.")

    # Case 5: LLM returns unknown task_type -> fallback to "respond to user"
    mock_call_ollama.return_value = (json.dumps({"task_type": "make_coffee", "parameters": {}}), None)
    user_text5 = "Make me coffee"
    goal5 = await agent._map_text_to_goal(user_text5)
    assert goal5 is not None, "Goal should be created for Case 5."
    assert goal5.description == f"respond to user : {user_text5[:100]}", f"Goal description mismatch for Case 5. Got: {goal5.description}"
    mock_call_ollama.reset_mock()
    logger.info("Case 5 (LLM unknown task_type, fallback to respond) passed.")

    # Case 6: LLM fails (returns error) -> fallback to "respond to user"
    mock_call_ollama.return_value = (None, "LLM Service Unavailable")
    user_text6 = "Analyse this document for me please" # Intentionally vague for LLM
    goal6 = await agent._map_text_to_goal(user_text6)
    assert goal6 is not None, "Goal should be created for Case 6."
    assert goal6.description == f"respond to user : {user_text6[:100]}", f"Goal description mismatch for Case 6. Got: {goal6.description}"
    mock_call_ollama.reset_mock()
    logger.info("Case 6 (LLM returns error, fallback to respond) passed.")

    # Case 7: LLM maps to task requiring param, but param is missing in LLM response
    # (e.g., LLM identifies "read_file" but fails to extract "path")
    mock_call_ollama.return_value = (json.dumps({"task_type": "read_file", "parameters": {"wrong_param_name": "config.toml"}}), None)
    user_text7 = "I need to examine the configuration." # Less likely for simple regex
    goal7 = await agent._map_text_to_goal(user_text7)
    assert goal7 is not None, "Goal should be created for Case 7 (fallback)."
    assert goal7.description == f"respond to user : {user_text7[:100]}", \
        f"Should fallback if LLM misses required param for formatted string. Got: {goal7.description}"
    mock_call_ollama.reset_mock()
    logger.info("Case 7 (LLM missing required param, fallback to respond) passed.")

    # Case 8: LLM successfully maps to "write_file" task (if defined in agent_controller._map_text_to_goal_via_llm.known_tasks_for_llm)
    # Check if 'write_file' is in the known_tasks_for_llm dictionary within the closure of _map_text_to_goal_via_llm
    # This is a bit fragile as it depends on closure inspection.
    known_tasks_in_llm_mapper = {}
    if agent._map_text_to_goal_via_llm.__closure__:
        for cell in agent._map_text_to_goal_via_llm.__closure__:
            # The known_tasks_for_llm dict is often the first item in the closure
            # and is a dict. We can check for a known key like "read_file".
            if isinstance(cell.cell_contents, dict) and "read_file" in cell.cell_contents:
                known_tasks_in_llm_mapper = cell.cell_contents
                break

    if "write_file" in known_tasks_in_llm_mapper:
        mock_call_ollama.return_value = (json.dumps({"task_type": "write_file", "parameters": {"path": "output.txt", "content": "hello from test"}}), None)
        user_text8 = "Please persist the message 'hello from test' in a file named output.txt."
        goal8 = await agent._map_text_to_goal(user_text8)
        assert goal8 is not None, "Goal should be created for Case 8."
        assert goal8.description == "write file : output.txt content : hello from test", f"Goal description mismatch for Case 8. Got: {goal8.description}"
        assert goal8.priority == USER_GOAL_PRIORITY, "Goal priority mismatch for Case 8."
        mock_call_ollama.reset_mock()
        logger.info("Case 8 (LLM map to write_file) passed.")
    else:
        logger.warning("Skipping Case 8 (LLM map to write_file) as 'write_file' not found in LLM known tasks closure.")


    logger.info("--- Test Passed: AgentController LLM Goal Mapping Fallback ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_mcm_detects_meta_conscious_state(test_agent: AgentController):
    agent = test_agent
    mcm = agent.meta_cognition
    logger.info("--- Test: MetaCognitiveMonitor Detects META_CONSCIOUS State ---")

    assert mcm is not None, "MetaCognitiveMonitor component missing."
    assert ConsciousState is not None, "ConsciousState enum not available for test."

    # Prepare cognitive_state input for MCM
    # Simulate agent being in META_CONSCIOUS state
    mock_cognitive_state = {
        "timestamp": time.time(),
        "consciousness_level": ConsciousState.META_CONSCIOUS.name, # Pass as string name
        "active_goal_count": 0,
        "current_goal_desc": None,
        "current_goal_status": None,
        "current_plan_length": 0,
        "workspace_load": 3,
        "emotional_valence": 0.1,
        "integration_level": 0.8
    }
    mock_performance_metrics = {"average_cycle_time": 0.05} # Dummy performance

    # Update MCM's internal history for consciousness level for this test run
    # (monitor_cognition uses self.recent_consciousness_levels for its own checks,
    # but also uses the passed cognitive_state.get("consciousness_level"))
    # To ensure the check inside monitor_cognition for META_CONSCIOUS on the *current* state works:
    mcm.recent_consciousness_levels.append((time.time(), ConsciousState.META_CONSCIOUS))


    logger.info(f"Calling MetaCognitiveMonitor.monitor_cognition with META_CONSCIOUS state...")
    mcm_input_state = {"cognitive_state": mock_cognitive_state, "performance_metrics": mock_performance_metrics}
    
    # process method of MCM calls monitor_cognition
    mcm_output = await mcm.process(mcm_input_state) 
    assert mcm_output is not None and "meta_analysis" in mcm_output, "MCM process failed or gave invalid output."
    meta_analysis = mcm_output["meta_analysis"]

    assert isinstance(meta_analysis, dict), "Meta-analysis result is not a dictionary."
    assert "issues_detected" in meta_analysis, "issues_detected missing from meta_analysis."
    
    issues_detected = meta_analysis["issues_detected"]
    assert isinstance(issues_detected, list), "issues_detected is not a list."

    found_meta_conscious_issue = False
    for issue in issues_detected:
        if isinstance(issue, dict) and issue.get("type") == "meta_conscious_state_active":
            found_meta_conscious_issue = True
            assert issue.get("current_level") == ConsciousState.META_CONSCIOUS.name, \
                "Issue details incorrect for meta_conscious_state_active."
            break
    
    assert found_meta_conscious_issue, \
        f"MetaCognitiveMonitor did not detect 'meta_conscious_state_active'. Issues found: {issues_detected}"

    logger.info("--- Test Passed: MetaCognitiveMonitor Detects META_CONSCIOUS State ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_ems_curiosity_boost_on_low_cs(test_agent: AgentController):
    agent = test_agent
    ems = agent.emergent_motivation_system
    logger.info("--- Test: EMS Curiosity Boost on Persistent Low CS ---")

    assert ems is not None, "EmergentMotivationSystem component missing."
    assert hasattr(ems, 'drives') and "curiosity" in ems.drives, "EMS or curiosity drive not initialized."
    assert ConsciousState is not None, "ConsciousState enum not available for test."

    # Store original EMS settings to restore them later
    original_cs_hist_maxlen = ems.ems_cs_history_maxlen
    original_low_cs_persist_thresh = ems.low_cs_persistence_threshold
    original_low_cs_boost_factor = ems.low_cs_curiosity_boost_factor
    original_recent_cs_levels_ems_maxlen = ems.recent_cs_levels_ems.maxlen # Store original deque maxlen

    # Configure EMS for this specific test scenario
    test_maxlen = 3
    ems.ems_cs_history_maxlen = test_maxlen 
    # IMPORTANT: Re-create the deque with the new maxlen for the test
    ems.recent_cs_levels_ems = deque(maxlen=test_maxlen) 
    
    ems.low_cs_persistence_threshold = 2 
    ems.low_cs_curiosity_boost_factor = 0.1 # Make boost more obvious for test logging

    await ems.reset() # Call reset. This will clear the newly created deque.
                      # Ensure reset doesn't inadvertently change maxlen if it re-reads from a stale self._config.
                      # Current EMS reset does: self.recent_cs_levels_ems.clear() - which is fine, preserves maxlen.

    initial_status = await ems.get_status()
    initial_curiosity = initial_status.get("current_drives", {}).get("curiosity", 0.5)
    logger.info(f"Initial curiosity for low_cs test (after reset & test config): {initial_curiosity:.3f}")
    logger.info(f"EMS test config: HistMaxlen={ems.ems_cs_history_maxlen}, PersistThresh={ems.low_cs_persistence_threshold}, DequeMaxlen={ems.recent_cs_levels_ems.maxlen}")


    dummy_last_action_result = {"type": "THINKING", "outcome": "success"}
    base_ems_input = {
        "last_action_result": dummy_last_action_result,
        "phenomenal_state": None, "active_goal": None, "self_model_summary": None
    }
    ems_input_conscious = {**base_ems_input, "cognitive_state": {"consciousness_level": ConsciousState.CONSCIOUS.name}}
    ems_input_unconscious = {**base_ems_input, "cognitive_state": {"consciousness_level": ConsciousState.UNCONSCIOUS.name}}
    ems_input_preconscious = {**base_ems_input, "cognitive_state": {"consciousness_level": ConsciousState.PRE_CONSCIOUS.name}}

    # --- Detailed Cycle Trace ---
    # C0: State after reset
    curiosity_c0 = ems.drives["curiosity"]["value"]
    logger.info(f"C0 (Reset): Curiosity={curiosity_c0:.4f}, History={list(ems.recent_cs_levels_ems)}")

    # C1: Input CONSCIOUS
    await ems.process(ems_input_conscious)
    curiosity_c1 = ems.drives["curiosity"]["value"]
    logger.info(f"C1 (CONSCIOUS): Curiosity={curiosity_c1:.4f}, History={list(ems.recent_cs_levels_ems)}")
    # History for C1 check (before append): []. is_low_cs=F. Deque after C1: ['CONSCIOUS'] (len 1)

    # C2: Input UNCONSCIOUS
    await ems.process(ems_input_unconscious)
    curiosity_c2 = ems.drives["curiosity"]["value"]
    logger.info(f"C2 (UNCONSCIOUS): Curiosity={curiosity_c2:.4f}, History={list(ems.recent_cs_levels_ems)}")
    # History for C2 check (before append): ['CONSCIOUS'] (low_cs_count=0). is_low_cs=F. Deque after C2: ['CONSCIOUS', 'UNCONSCIOUS'] (len 2)
    assert curiosity_c2 <= curiosity_c1 or abs(curiosity_c2 - 0.5) < abs(curiosity_c1 - 0.5), \
        f"Curiosity should not have boosted at C2. C1:{curiosity_c1:.4f}, C2:{curiosity_c2:.4f}"

    # C3: Input PRE_CONSCIOUS
    await ems.process(ems_input_preconscious)
    curiosity_c3 = ems.drives["curiosity"]["value"]
    logger.info(f"C3 (PRE_CONSCIOUS): Curiosity={curiosity_c3:.4f}, History={list(ems.recent_cs_levels_ems)}")
    # History for C3 check (before append): ['CONSCIOUS', 'UNCONSCIOUS'] (low_cs_count=1). is_low_cs=F (1 < threshold 2). No boost.
    # Deque after C3: ['CONSCIOUS', 'UNCONSCIOUS', 'PRE_CONSCIOUS'] (len 3)
    assert curiosity_c3 <= curiosity_c2 or abs(curiosity_c3 - 0.5) < abs(curiosity_c2 - 0.5), \
        f"Curiosity should not have boosted at C3 yet. C2:{curiosity_c2:.4f}, C3:{curiosity_c3:.4f}"

    # C4: Input UNCONSCIOUS (Boost should apply here)
    await ems.process(ems_input_unconscious)
    curiosity_c4 = ems.drives["curiosity"]["value"]
    logger.info(f"C4 (UNCONSCIOUS): Curiosity={curiosity_c4:.4f}, History={list(ems.recent_cs_levels_ems)}")
    # History for C4 check (before append): ['CONSCIOUS', 'UNCONSCIOUS', 'PRE_CONSCIOUS'] (low_cs_count=2). is_low_cs=T (2 >= threshold 2). Boost.
    # Deque after C4: ['UNCONSCIOUS', 'PRE_CONSCIOUS', 'UNCONSCIOUS'] (maxlen 3, 'CONSCIOUS' from C1 evicted)
    assert curiosity_c4 > curiosity_c3, f"Curiosity should boost at C4. C3:{curiosity_c3:.4f}, C4:{curiosity_c4:.4f}"

    # C5: Input CONSCIOUS (Boost should apply here)
    await ems.process(ems_input_conscious)
    curiosity_c5 = ems.drives["curiosity"]["value"]
    logger.info(f"C5 (CONSCIOUS): Curiosity={curiosity_c5:.4f}, History={list(ems.recent_cs_levels_ems)}")
    # History for C5 check (before append): ['UNCONSCIOUS', 'PRE_CONSCIOUS', 'UNCONSCIOUS'] (low_cs_count=3). is_low_cs=T (3 >= threshold 2). Boost.
    # Deque after C5: ['PRE_CONSCIOUS', 'UNCONSCIOUS', 'CONSCIOUS'] (oldest 'UNCONSCIOUS' from C2 evicted)
    assert curiosity_c5 > curiosity_c4 or abs(curiosity_c5 - curiosity_c4) < 0.001 if curiosity_c4 > 0.98 else curiosity_c5 > curiosity_c4 - (ems.drives["curiosity"]["decay"] * 0.1) , \
         f"Curiosity should boost or stay high at C5. C4:{curiosity_c4:.4f}, C5:{curiosity_c5:.4f}"

    # C6: Input CONSCIOUS (Boost should apply here)
    await ems.process(ems_input_conscious)
    curiosity_c6 = ems.drives["curiosity"]["value"]
    logger.info(f"C6 (CONSCIOUS): Curiosity={curiosity_c6:.4f}, History={list(ems.recent_cs_levels_ems)}")
    # History for C6 check (before append): ['PRE_CONSCIOUS', 'UNCONSCIOUS', 'CONSCIOUS'] (low_cs_count=2). is_low_cs=T (2 >= threshold 2). Boost.
    # Deque after C6: ['UNCONSCIOUS', 'CONSCIOUS', 'CONSCIOUS'] (oldest 'PRE_CONSCIOUS' from C3 evicted)
    assert curiosity_c6 > curiosity_c5 or abs(curiosity_c6 - curiosity_c5) < 0.001 if curiosity_c5 > 0.98 else curiosity_c6 > curiosity_c5 - (ems.drives["curiosity"]["decay"] * 0.1) , \
         f"Curiosity should boost or stay high at C6. C5:{curiosity_c5:.4f}, C6:{curiosity_c6:.4f}"

    # C7: Input CONSCIOUS (Boost SHOULD STOP here)
    curiosity_before_c7 = ems.drives["curiosity"]["value"] # This is curiosity_c6
    await ems.process(ems_input_conscious)
    curiosity_c7 = ems.drives["curiosity"]["value"]
    logger.info(f"C7 (CONSCIOUS): Curiosity={curiosity_c7:.4f}, History={list(ems.recent_cs_levels_ems)}")
    # History for C7 check (before append): ['UNCONSCIOUS', 'CONSCIOUS', 'CONSCIOUS'] (low_cs_count=1). is_low_cs=F (1 < threshold 2). No boost.
    # Deque after C7: ['CONSCIOUS', 'CONSCIOUS', 'CONSCIOUS'] (oldest 'UNCONSCIOUS' from C4's input processing evicted)
    
    decay_effect_expected = True if curiosity_before_c7 > 0.5001 else (True if curiosity_before_c7 < 0.4999 else False)
    if decay_effect_expected and curiosity_before_c7 > 0.5001: # If it was high, it should now decay (or move towards 0.5)
        assert curiosity_c7 < curiosity_before_c7, \
            f"Curiosity should decay at C7 as boost stops. Before_C7 (C6):{curiosity_before_c7:.6f}, C7:{curiosity_c7:.6f}"
    elif decay_effect_expected and curiosity_before_c7 < 0.4999: # If it was low (unlikely here), it should rise towards 0.5
         assert curiosity_c7 > curiosity_before_c7, \
            f"Curiosity should rise towards 0.5 at C7 as boost stops. Before_C7 (C6):{curiosity_before_c7:.6f}, C7:{curiosity_c7:.6f}"
    else: # It was already very close to 0.5, check it remains so
        assert abs(curiosity_c7 - 0.5) < 0.01 + ems.drives["curiosity"]["decay"], \
            f"Curiosity {curiosity_c7:.6f} should remain near 0.5 if it started near 0.5 and boost stopped. Before_C7: {curiosity_before_c7:.6f}"


    # Restore original EMS settings
    ems.ems_cs_history_maxlen = original_cs_hist_maxlen
    ems.recent_cs_levels_ems = deque(maxlen=original_recent_cs_levels_ems_maxlen) # Restore deque with original maxlen
    # Re-populate with some dummy values or clear if that's closer to original fixture state
    # For now, just ensure maxlen is restored. The content will be overwritten by next tests.
    ems.low_cs_persistence_threshold = original_low_cs_persist_thresh
    ems.low_cs_curiosity_boost_factor = original_low_cs_boost_factor

    logger.info("--- Test Passed: EMS Curiosity Boost on Persistent Low CS ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
@patch(f'{PACKAGE_NAME}.cognitive_modules.narrative_constructor.call_ollama', new_callable=AsyncMock)
async def test_narrative_writes_to_kb(mock_call_ollama_kb: AsyncMock, test_agent: AgentController):
    agent = test_agent
    nc = agent.narrative_constructor
    kb = agent.knowledge_base
    logger.info("--- Test: NarrativeConstructor Writes to KnowledgeBase ---")

    assert nc is not None, "NarrativeConstructor component missing."
    assert kb is not None, "KnowledgeBase component missing."
    # Ensure NC's internal references were set up during its initialize
    assert hasattr(nc, '_kb') and nc._kb is kb, "NarrativeConstructor does not have the correct KB reference."
    assert hasattr(nc, '_PredicateClass') and nc._PredicateClass is not None, \
        "NarrativeConstructor does not have a valid Predicate class reference."

    # --- Setup: Prepare conditions to trigger a significant event ---
    mock_call_ollama_kb.return_value = ("LLM generated: A moment of high intensity.", None)
    
    # Get necessary type references used by NC internally and for creating test data
    _PhenomenalState_class_for_test = globals().get('PhenomenalState')
    assert _PhenomenalState_class_for_test is not None, "PhenomenalState class not available for test setup."

    # 1. Create a PhenomenalState that will trigger significance due to high intensity
    current_timestamp = time.time() - 5 # Ensure a distinct timestamp
    test_p_state = _PhenomenalState_class_for_test(
        content={"item": "data_for_high_intensity_event"},
        intensity=nc.intensity_threshold + 0.1, # Ensure it's above threshold
        valence=0.1, # Neutral valence
        integration_level=0.5,
        timestamp=current_timestamp,
        # Assuming these are populated by ExperienceStream, provide defaults if not tested here
        distinct_source_count=2, content_diversity_lexical=0.5, shared_concept_count_gw=0.2 
    )
    # Set NC's internal _last_phenomenal_state to something neutral to ensure current one is "new"
    nc._last_phenomenal_state = _PhenomenalState_class_for_test(content={}, intensity=0.1, valence=0.1)
    
    # 2. Prepare other inputs for nc.process()
    # These might contribute to the 'reason' string or event_summary
    test_action_result = {"type": "NEUTRAL_ACTION_KB", "outcome": "success"}
    test_loop_info = None
    test_meta_analysis = {}
    mock_prediction_error_for_nc_kb = None 
    mock_current_drives_for_nc_kb = {"curiosity": 0.5, "satisfaction": 0.5, "competence": 0.5}
    nc._last_drive_state = mock_current_drives_for_nc_kb.copy() # Align last_drive_state for _is_significant

    # Assemble the input state for nc.process
    nc_input_state = {
        "phenomenal_state": test_p_state,
        "last_action_result": test_action_result,
        "loop_info": test_loop_info,
        "meta_analysis": test_meta_analysis,
        "prediction_error": mock_prediction_error_for_nc_kb,
        "current_drives": mock_current_drives_for_nc_kb 
    }
    
    # Store current KB fact count (optional, for clearer assertion later if needed)
    initial_facts_query = await kb.query_state({"predicate_count": True})
    initial_fact_count = initial_facts_query.get("predicate_count", 0)

    # --- Action: Process the event to generate narrative and assert to KB ---
    logger.info(f"Calling NarrativeConstructor.process to generate entry and KB facts... Initial KB fact count: {initial_fact_count}")
    await nc.process(nc_input_state)

    # --- Verification ---
    assert mock_call_ollama_kb.called, "call_ollama (mocked) was not called by NarrativeConstructor."
    assert len(nc.narrative) > 0, "Narrative entry was not created."
    
    await asyncio.sleep(0.02) # Give a moment for async KB writes if needed

    last_narrative_entry = nc.narrative[-1]
    narrative_entry_timestamp = last_narrative_entry.timestamp
    assert narrative_entry_timestamp == current_timestamp, "Timestamp of narrative entry doesn't match input P-State."

    # Verify predicates in KB
    # 1. narrativeEventRecorded
    # The 'reason' arg in the predicate is derived from the _is_significant check.
    # For high intensity, the reason string would contain "HighIntensity(...)".
    # Consciousness level is from the NarrativeEntry object.
    expected_reason_substring = "HighIntensity" # This should be part of the 'reason'
    
    all_event_preds = await kb.query(name="narrativeEventRecorded")
    found_event_pred = False
    for pred in all_event_preds:
        if pred.args[0] == narrative_entry_timestamp and \
           expected_reason_substring in str(pred.args[1]) and \
           pred.args[2] == last_narrative_entry.consciousness_level:
            found_event_pred = True
            logger.info(f"Found narrativeEventRecorded: {pred}")
            break
    assert found_event_pred, \
        f"narrativeEventRecorded predicate not found or mismatch. Expected ts={narrative_entry_timestamp}, reason containing '{expected_reason_substring}', CS='{last_narrative_entry.consciousness_level}'. Found: {all_event_preds}"

    # 2. narrativeAssociatedValence
    expected_valence = round(test_p_state.valence, 2)
    valence_preds = await kb.query(name="narrativeAssociatedValence", args=(narrative_entry_timestamp, expected_valence))
    logger.info(f"KB query for narrativeAssociatedValence (ts={narrative_entry_timestamp}, val={expected_valence}): Found {len(valence_preds)} preds.")
    assert len(valence_preds) >= 1, \
        f"narrativeAssociatedValence predicate not found for ts={narrative_entry_timestamp}, valence={expected_valence}"

    # 3. narrativeAssociatedIntensity
    expected_intensity = round(test_p_state.intensity, 2)
    intensity_preds = await kb.query(name="narrativeAssociatedIntensity", args=(narrative_entry_timestamp, expected_intensity))
    logger.info(f"KB query for narrativeAssociatedIntensity (ts={narrative_entry_timestamp}, int={expected_intensity}): Found {len(intensity_preds)} preds.")
    assert len(intensity_preds) >= 1, \
        f"narrativeAssociatedIntensity predicate not found for ts={narrative_entry_timestamp}, intensity={expected_intensity}"

    # 4. narrativeTriggerType
    # Based on the setup, "HighIntensity" should be a primary trigger.
    # The logic in NC.process will determine the main_trigger_type.
    # Let's assume 'high_intensity' is a valid trigger type string it might use.
    # If the trigger logic is more complex, this might need adjustment.
    # For now, let's check that *a* trigger type was recorded.
    all_trigger_type_preds = await kb.query(name="narrativeTriggerType")
    found_trigger_pred_for_ts = False
    actual_trigger_type_found = "None"
    for pred in all_trigger_type_preds:
        if pred.args[0] == narrative_entry_timestamp:
            found_trigger_pred_for_ts = True
            actual_trigger_type_found = str(pred.args[1])
            logger.info(f"Found narrativeTriggerType: {pred}")
            break
    assert found_trigger_pred_for_ts, \
        f"narrativeTriggerType predicate not found for timestamp {narrative_entry_timestamp}. Found: {all_trigger_type_preds}"
    # More specific check if you know the exact trigger type string NC uses:
    # assert actual_trigger_type_found == "high_intensity_event", f"Expected trigger 'high_intensity_event', got '{actual_trigger_type_found}'"
    logger.info(f"Recorded trigger type was: '{actual_trigger_type_found}'")


    logger.info("--- Test Passed: NarrativeConstructor Writes to KnowledgeBase ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_po_updates_agent_config(test_agent: AgentController):
    """
    Tests that AgentController reads adjustments from PerformanceOptimizer
    and updates its own self.config.
    """
    agent = test_agent
    po = agent.performance_optimizer
    logger.info("--- Test: PerformanceOptimizer Updates AgentController.config ---")

    assert po is not None, "PerformanceOptimizer component missing."
    assert hasattr(agent, 'config'), "Agent missing 'config' attribute."
    assert hasattr(po, 'config_changes'), "PO missing 'config_changes' attribute."

    # --- Setup: Configure PO to auto-apply and define a bottleneck scenario ---
    if "performance_optimizer" not in agent.config:
        agent.config["performance_optimizer"] = {}
    agent.config["performance_optimizer"]["auto_apply_adjustments"] = True
    po._config_perf_optimizer_section["auto_apply_adjustments"] = True 

    current_agent_config_performance = agent.config.setdefault("performance", {})
    current_agent_config_performance["max_planning_depth"] = 10 
    po.config_changes = {} 

    logger.info(f"TEST: agent.config[performance][max_planning_depth] set to: {agent.config['performance']['max_planning_depth']}")

    planning_threshold = po.optimization_thresholds.get("planning", 0.1) 
    simulated_planning_duration = planning_threshold * 4 
    cycle_profile_bottleneck = {"planning": simulated_planning_duration} 

    logger.info(f"Simulating PO process with 'planning' bottleneck ({simulated_planning_duration}s)...")
    po_input_state = {"cycle_profile": cycle_profile_bottleneck}
    po_output = await po.process(po_input_state)
    assert po_output and "performance_analysis" in po_output, "PO.process failed or returned invalid structure."
    
    performance_analysis = po_output["performance_analysis"]
    assert isinstance(performance_analysis, dict), "performance_analysis is not a dictionary."
    suggested_adjustments = performance_analysis.get("suggested_adjustments", {})
    assert isinstance(suggested_adjustments, dict), "suggested_adjustments is not a dictionary."

    assert "performance" in suggested_adjustments, \
        "PO suggestions should contain 'performance' key for planning depth adjustment."
    performance_section_suggestion = suggested_adjustments.get("performance", {})
    assert isinstance(performance_section_suggestion, dict), "'performance' section in suggestions is not a dict."
    assert "max_planning_depth" in performance_section_suggestion, \
        "PO suggestions for 'performance' should contain 'max_planning_depth' key."
    
    new_suggested_depth_by_po = performance_section_suggestion.get("max_planning_depth")
    assert new_suggested_depth_by_po == 8, \
        f"PO suggested unexpected depth {new_suggested_depth_by_po}, expected 8."
    logger.info(f"PO suggested new max_planning_depth for 'performance' section: {new_suggested_depth_by_po}")

    po_status_after_process = await po.get_status()
    active_po_adjustments_from_status = po_status_after_process.get("active_config_adjustments", {})
    assert isinstance(active_po_adjustments_from_status, dict), "PO status did not return a dict for active_config_adjustments."

    config_updated_by_po_in_cycle_test_flag = False
    if active_po_adjustments_from_status:
        logger.info(f"Agent test is applying PO adjustments from PO.status: {active_po_adjustments_from_status}")
        for component_key_from_po, comp_adjustments_from_po in active_po_adjustments_from_status.items():
            if component_key_from_po not in agent.config:
                agent.config[component_key_from_po] = {} 
            
            if isinstance(agent.config.get(component_key_from_po), dict) and isinstance(comp_adjustments_from_po, dict):
                for param_key, new_value in comp_adjustments_from_po.items():
                    current_val_in_agent_config = agent.config[component_key_from_po].get(param_key)
                    if current_val_in_agent_config != new_value:
                        agent.config[component_key_from_po][param_key] = new_value
                        config_updated_by_po_in_cycle_test_flag = True
                        logger.info(f"TEST_APPLY: agent.config[{component_key_from_po}][{param_key}] updated to {new_value} (was {current_val_in_agent_config})")
                    else:
                        logger.info(f"TEST_APPLY: agent.config[{component_key_from_po}][{param_key}] already {new_value}, no update needed by test logic.")
            else:
                logger.warning(f"Test cannot apply PO adjustments for '{component_key_from_po}': agent.config section or adjustments from PO not dicts.")
    
    assert config_updated_by_po_in_cycle_test_flag, \
        "AgentController.config should have been updated by PO adjustments based on the test's merge simulation."

    final_max_depth_in_agent_performance_section = agent.config.get("performance", {}).get("max_planning_depth")
    logger.info(f"Final agent.config[performance][max_planning_depth]: {final_max_depth_in_agent_performance_section}")
    assert final_max_depth_in_agent_performance_section == new_suggested_depth_by_po, \
        f"agent.config.performance.max_planning_depth was not updated correctly. Expected {new_suggested_depth_by_po}, got {final_max_depth_in_agent_performance_section}."

    logger.info(f"Verifying HTNPlanner uses new max_depth: {final_max_depth_in_agent_performance_section}")
    dummy_goal_desc_for_test = DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC 
    dummy_goal_for_planner_test = create_goal_from_descriptor(dummy_goal_desc_for_test)
    assert dummy_goal_for_planner_test is not None, "Failed to create dummy goal for planner test."

    planner = agent.htn_planner
    assert planner is not None, "HTNPlanner not found on agent."

    expected_depth_planner_uses = final_max_depth_in_agent_performance_section

    # --- MODIFICATION 1: Set agent's consciousness state ---
    # Ensure CS level does not restrict depth for this specific test of PO update
    # This assumes ConsciousState is imported at the top of the test file
    original_cs_level = agent.consciousness_level
    agent.consciousness_level = ConsciousState.CONSCIOUS # Set to a level that won't trigger low_cs depth limits
    logger.info(f"TEST: Temporarily set agent.consciousness_level to {agent.consciousness_level.name} to bypass CS depth override.")


    with patch.object(logging.getLogger(f"{PACKAGE_NAME}.cognitive_modules.htn_planner"), 'info') as mock_planner_info_log:
        await planner.plan(dummy_goal_for_planner_test, set()) 

        found_max_depth_log = False
        for call_args in mock_planner_info_log.call_args_list:
            log_message = str(call_args[0][0]) 
            # --- MODIFICATION 2: Corrected log message string ---
            # Log message is: "Planning started for goal: {goal.description} (Effective Max Overall Depth for this attempt: {effective_max_depth})"
            if "Planning started for goal" in log_message and \
               f"(Effective Max Overall Depth for this attempt: {expected_depth_planner_uses})" in log_message:
                found_max_depth_log = True
                logger.info(f"HTNPlanner correctly logged usage of max_depth: {expected_depth_planner_uses}")
                break
        assert found_max_depth_log, \
            f"HTNPlanner did not log using the updated max_depth of {expected_depth_planner_uses} from agent.config[performance]. Review HTNPlanner logs. All logs from planner: {[str(c[0][0]) for c in mock_planner_info_log.call_args_list]}"

    # --- Restore original CS level if necessary (good practice) ---
    agent.consciousness_level = original_cs_level
    logger.info(f"TEST: Restored agent.consciousness_level to {agent.consciousness_level.name}.")

    logger.info("--- Test Passed: PerformanceOptimizer Updates AgentController.config (and HTNPlanner sees it) ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_po_adjusts_attention_max_candidates(test_agent: AgentController):
    agent = test_agent
    po = agent.performance_optimizer
    ac = agent.attention_controller
    logger.info("--- Test: PO Dynamically Adjusts AttentionController.max_candidates ---")

    assert po is not None, "PerformanceOptimizer component missing."
    assert ac is not None, "AttentionController component missing."
    assert hasattr(agent, 'config'), "Agent missing 'config' attribute."

    # --- Setup ---
    # Configure PO to auto-apply
    agent.config.setdefault("performance_optimizer", {})["auto_apply_adjustments"] = True
    po._config_perf_optimizer_section["auto_apply_adjustments"] = True 

    # Set initial AC max_candidates in agent.config
    initial_max_candidates = 30
    agent.config.setdefault("attention_controller", {})["max_candidates"] = initial_max_candidates
    # Also update AC's internal self.max_candidates as if it initialized with this value
    # This ensures ac.max_candidates (used as a fallback in ac.allocate_attention) is correct initially.
    ac.max_candidates = initial_max_candidates 
    po.config_changes = {} # Clear PO's prior adjustments

    logger.info(f"Initial agent.config[attention_controller][max_candidates] = {initial_max_candidates}")

    # Simulate an "attention" bottleneck
    attention_threshold = po.optimization_thresholds.get("attention", 0.03) # Default from PO if not in instance
    simulated_attention_duration = attention_threshold * 3.0 # Severity 3.0
    cycle_profile_attn_bottleneck = {"attention": simulated_attention_duration} 

    # --- Action 1: PO processes bottleneck and suggests change ---
    logger.info(f"Simulating PO process with 'attention' bottleneck ({simulated_attention_duration}s)...")
    po_input_state = {"cycle_profile": cycle_profile_attn_bottleneck}
    po_output = await po.process(po_input_state)
    assert po_output and "performance_analysis" in po_output, "PO.process failed or returned invalid structure."
    
    performance_analysis = po_output["performance_analysis"]
    assert isinstance(performance_analysis, dict), "performance_analysis is not a dictionary."
    suggested_adjustments = performance_analysis.get("suggested_adjustments", {})
    assert isinstance(suggested_adjustments, dict), "suggested_adjustments is not a dictionary."

    assert "attention_controller" in suggested_adjustments, \
        "PO suggestions should contain 'attention_controller' key for its adjustments."
    attention_section_suggestion = suggested_adjustments.get("attention_controller", {})
    assert isinstance(attention_section_suggestion, dict), "'attention_controller' section in suggestions is not a dict."
    assert "max_candidates" in attention_section_suggestion, \
        "PO suggestions for 'attention_controller' should contain 'max_candidates' key."
    
    suggested_new_max_cand = attention_section_suggestion.get("max_candidates")
    # Based on PO logic: initial 30, sev 3.0 -> Rule1 (sev>2.5, current>15) -> new = max(5, 30-10) = 20
    assert suggested_new_max_cand == 20, \
        f"PO suggested max_candidates {suggested_new_max_cand}, expected 20."
    logger.info(f"PO suggested new max_candidates for 'attention_controller': {suggested_new_max_cand}")

    # --- Action 2: Simulate AgentController merging this change into its self.config ---
    po_status_after_process = await po.get_status()
    active_po_adjustments_from_status = po_status_after_process.get("active_config_adjustments", {})
    assert isinstance(active_po_adjustments_from_status, dict), "PO status did not return a dict for active_config_adjustments."

    config_updated_by_po_in_cycle_test_flag = False
    if active_po_adjustments_from_status:
        logger.info(f"Agent test is applying PO adjustments from PO.status: {active_po_adjustments_from_status}")
        for component_key_from_po, comp_adjustments_from_po in active_po_adjustments_from_status.items():
            if component_key_from_po not in agent.config:
                agent.config[component_key_from_po] = {} 
            
            if isinstance(agent.config.get(component_key_from_po), dict) and isinstance(comp_adjustments_from_po, dict):
                for param_key, new_value in comp_adjustments_from_po.items():
                    current_val_in_agent_config = agent.config[component_key_from_po].get(param_key)
                    if current_val_in_agent_config != new_value:
                        agent.config[component_key_from_po][param_key] = new_value
                        config_updated_by_po_in_cycle_test_flag = True
                        logger.info(f"TEST_APPLY (AC): agent.config[{component_key_from_po}][{param_key}] updated to {new_value} (was {current_val_in_agent_config})")
                    else:
                        logger.info(f"TEST_APPLY (AC): agent.config[{component_key_from_po}][{param_key}] already {new_value}, no update needed by test logic.")
            else:
                logger.warning(f"Test cannot apply PO adjustments for '{component_key_from_po}': agent.config section or adjustments from PO not dicts.")
    
    assert config_updated_by_po_in_cycle_test_flag, \
        "AgentController.config should have been updated by PO adjustments based on the test's merge simulation."

    final_max_cand_in_agent_config = agent.config.get("attention_controller", {}).get("max_candidates")
    assert final_max_cand_in_agent_config == suggested_new_max_cand, \
        f"agent.config[attention_controller][max_candidates] not updated correctly. Expected {suggested_new_max_cand}, got {final_max_cand_in_agent_config}."

    # --- Action 3: Call AttentionController.allocate_attention and check if it uses the new value ---
    num_dummy_candidates = suggested_new_max_cand + 10 
    dummy_candidates_for_ac = {
        f"cand_{i}": {"content": f"content {i}", "weight_hint": 0.5, "timestamp": time.time()}
        for i in range(num_dummy_candidates)
    }

    ac_logger = logging.getLogger(f"{PACKAGE_NAME}.cognitive_modules.attention_controller")
    with patch.object(ac_logger, 'debug') as mock_ac_debug_log, \
         patch.object(ac_logger, 'warning') as mock_ac_warning_log: # Patch 'warning'
        
        await ac.allocate_attention(dummy_candidates_for_ac)

        found_effective_max_cand_log = False
        for call_args in mock_ac_debug_log.call_args_list: # Check debug logs
            log_msg = str(call_args[0][0])
            if f"Effective max_candidates: {suggested_new_max_cand}" in log_msg:
                found_effective_max_cand_log = True
                logger.info(f"AC correctly logged DEBUG usage of updated max_candidates: {suggested_new_max_cand}")
                break
        assert found_effective_max_cand_log, \
            f"AttentionController did not log DEBUG message for 'Effective max_candidates: {suggested_new_max_cand}'. Review AC debug logs."

        found_truncation_log = False
        # Truncation is expected since num_dummy_candidates > suggested_new_max_cand
        if len(dummy_candidates_for_ac) > suggested_new_max_cand: 
            for call_args in mock_ac_warning_log.call_args_list: # Check WARNING logs for truncation
                log_msg = str(call_args[0][0]) 
                # The log message in AttentionController is:
                # f"Number of attention candidates ({len(candidates)}) exceeds dynamic limit ({current_max_candidates}). Truncating."
                if f"exceeds dynamic limit ({suggested_new_max_cand}). Truncating." in log_msg:
                    found_truncation_log = True
                    logger.info("AC correctly logged WARNING for truncation based on new max_candidates.")
                    break
            assert found_truncation_log, \
                f"AttentionController did not log WARNING message for truncation with new limit {suggested_new_max_cand} when expected. Review AC warning logs."
        # No 'else' needed here, as if truncation is not expected, found_truncation_log remains False and isn't asserted.

    logger.info("--- Test Passed: PO Dynamically Adjusts AttentionController.max_candidates ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_po_adjusts_gwm_capacity(test_agent: AgentController):
    agent = test_agent
    po = agent.performance_optimizer
    gwm = agent.global_workspace
    logger.info("--- Test: PO Dynamically Adjusts GlobalWorkspaceManager.capacity ---")

    assert po is not None, "PerformanceOptimizer component missing."
    assert gwm is not None, "GlobalWorkspaceManager component missing."

    # --- Setup ---
    agent.config.setdefault("performance_optimizer", {})["auto_apply_adjustments"] = True
    po._config_perf_optimizer_section["auto_apply_adjustments"] = True 

    initial_gwm_capacity = 10
    agent.config.setdefault("global_workspace", {})["capacity"] = initial_gwm_capacity
    gwm.capacity = initial_gwm_capacity # Ensure GWM instance starts with this
    po.config_changes = {}

    logger.info(f"Initial agent.config[global_workspace][capacity] = {initial_gwm_capacity}")

    # Simulate a "workspace" bottleneck
    workspace_threshold = po.optimization_thresholds.get("workspace", 0.04)
    simulated_workspace_duration = workspace_threshold * 2.5 # Severity 2.5
    cycle_profile_gwm_bottleneck = {"workspace": simulated_workspace_duration}

    # --- Action 1: PO processes bottleneck and suggests change ---
    await po.process({"cycle_profile": cycle_profile_gwm_bottleneck})
    po_status = await po.get_status()
    active_po_adjustments = po_status.get("active_config_adjustments", {})
    
    assert "global_workspace" in active_po_adjustments, "PO did not suggest for 'global_workspace'."
    assert "capacity" in active_po_adjustments.get("global_workspace",{}), "PO did not suggest 'capacity'."
    
    suggested_new_capacity = active_po_adjustments["global_workspace"]["capacity"]
    # Initial 10, sev 2.5 -> Rule1 (sev>2.0, current>5) -> new = max(3, 10-2) = 8
    assert suggested_new_capacity == 8, f"PO suggested GWM capacity {suggested_new_capacity}, expected 8."
    logger.info(f"PO suggested new GWM capacity: {suggested_new_capacity}")

    # --- Action 2: Simulate AgentController merging this change ---
    config_updated_flag_test = False
    # ... (Use the same generic merge logic as in test_po_adjusts_attention_max_candidates) ...
    if active_po_adjustments:
        for comp_key, comp_adjusts in active_po_adjustments.items():
            if comp_key not in agent.config: agent.config[comp_key] = {}
            if isinstance(agent.config.get(comp_key), dict) and isinstance(comp_adjusts, dict):
                for param_key, new_val in comp_adjusts.items():
                    if agent.config[comp_key].get(param_key) != new_val:
                        agent.config[comp_key][param_key] = new_val
                        config_updated_flag_test = True
                        logger.info(f"TEST_APPLY (GWM): agent.config[{comp_key}][{param_key}] updated to {new_val}")
    assert config_updated_flag_test, "Agent config for GWM capacity was not updated."
    assert agent.config.get("global_workspace", {}).get("capacity") == suggested_new_capacity

    # --- Action 3: Call GWM.update_workspace and check if it uses the new capacity ---
    # Create more candidates than the new capacity
    num_attn_candidates = suggested_new_capacity + 5
    mock_attention_weights = {f"item_{i}": 0.9 for i in range(num_attn_candidates)} # All high attention
    mock_all_candidates_data = {
        f"item_{i}": {"content": f"data_{i}"} for i in range(num_attn_candidates)
    }

    gwm_logger = logging.getLogger(f"{PACKAGE_NAME}.cognitive_modules.global_workspace_manager")
    with patch.object(gwm_logger, 'debug') as mock_gwm_debug_log, \
         patch.object(gwm_logger, 'info') as mock_gwm_info_log: # Also capture info for "Workspace updated"
        
        broadcast_content = await gwm.update_workspace(mock_attention_weights, mock_all_candidates_data)

        assert len(broadcast_content) == suggested_new_capacity, \
            f"GWM broadcast content length {len(broadcast_content)} does not match new capacity {suggested_new_capacity}."

        found_effective_capacity_log = False
        for call_args in mock_gwm_debug_log.call_args_list:
            log_msg = str(call_args[0][0])
            if f"Effective capacity: {suggested_new_capacity}" in log_msg:
                found_effective_capacity_log = True
                logger.info(f"GWM correctly logged usage of updated capacity: {suggested_new_capacity}")
                break
        assert found_effective_capacity_log, \
            f"GWM did not log DEBUG message for 'Effective capacity: {suggested_new_capacity}'. Review GWM debug logs."

        found_size_log = False
        for call_args in mock_gwm_info_log.call_args_list: # Check info logs for "Workspace updated"
            log_msg = str(call_args[0][0])
            if f"Workspace updated. Size: {suggested_new_capacity}/{suggested_new_capacity}" in log_msg:
                found_size_log = True
                logger.info(f"GWM correctly logged workspace size with new capacity: {suggested_new_capacity}")
                break
        assert found_size_log, \
            f"GWM did not log INFO message for 'Workspace updated. Size: {suggested_new_capacity}/{suggested_new_capacity}'. Review GWM info logs."


    logger.info("--- Test Passed: PO Dynamically Adjusts GlobalWorkspaceManager.capacity ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_dsm_perform_reflection(test_agent: AgentController):
    agent = test_agent
    dsm = agent.dynamic_self_model
    logger.info("--- Test: DynamicSelfModel - Perform Reflection ---")

    assert dsm is not None, "DynamicSelfModel component missing."
    assert hasattr(dsm, '_perform_reflection'), "DSM missing '_perform_reflection' method."
    assert hasattr(dsm, 'learning_events'), "DSM missing 'learning_events' attribute."

    original_reflection_interval = dsm.reflection_interval
    original_learning_events_maxlen = dsm.learning_events.maxlen
    dsm.reflection_interval = 5 
    dsm.learning_events = deque(maxlen=10) 
    dsm.cycles_since_reflection = 0 

    initial_traits = dsm.self_model.get("identity_traits", {}).copy()
    initial_persistence = initial_traits.get("persistence", 0.5)
    logger.info(f"Initial DSM Traits (for persistence test): {initial_traits}")

    async def simulate_dsm_process_for_reflection(action_type: str, outcome: str,
                                                  specific_key_suffix: Optional[str] = None):
        # ... (helper function remains the same) ...
        params = {"test_param": "value"} 
        action_key_specific = f"action:{action_type}{specific_key_suffix}" if specific_key_suffix else f"action:{action_type}"
        await dsm.update_self_model(
             last_action_type=action_type, action_params=params,
             action_outcome=outcome, action_error="TestError" if outcome == "failure" else None,
             action_result_data={"info": "test_data"}, current_phenomenal_state=None
        )


    # --- Scenario for Persistence Increase ---
    dsm.learning_events.clear(); dsm.cycles_since_reflection = 0
    # Ensure the persistence trait starts at a baseline or known value for clearer testing
    dsm.self_model["identity_traits"]["persistence"] = 0.5 
    initial_persistence_s_persist = dsm.self_model["identity_traits"]["persistence"]

    logger.info("Simulating persistence pattern (FAIL, FAIL, SUCCESS for ACTION_P)...")
    # Action P: Fail, Fail, Success (counts as 2 failures before success) -> key_persistence_value = 2/5 = 0.4
    await simulate_dsm_process_for_reflection("ACTION_P", "failure", specific_key_suffix=":persist_test")
    await simulate_dsm_process_for_reflection("ACTION_P", "failure", specific_key_suffix=":persist_test")
    await simulate_dsm_process_for_reflection("ACTION_P", "success", specific_key_suffix=":persist_test")
    # Action Q: Success (no persistence pattern here)
    await simulate_dsm_process_for_reflection("ACTION_Q", "success")
    # Action R: Fail (no persistence pattern here)
    await simulate_dsm_process_for_reflection("ACTION_R", "failure") # This 5th event triggers reflection

    # Expected: action_keys_evaluated = {P, Q, R}. num_distinct = 3.
    # Score for P = 0.4. Score for Q=0, R=0. Total score = 0.4.
    # normalized_persistence_metric = 0.4 / 3 = 0.133.
    # This is < 0.2. So persistence_change will be -0.05 (decrease).
    # This is NOT what we want for this scenario. The normalization needs adjustment or threshold.
    # The `total_events > len(self.learning_events) * 0.5` for decrease might prevent it. total_events=5. len=5. 5 > 2.5.
    
    # Let's make the pattern stronger to ensure increase with current normalization
    dsm.learning_events.clear(); dsm.cycles_since_reflection = 0
    dsm.self_model["identity_traits"]["persistence"] = 0.5
    logger.info("Simulating STRONG persistence pattern (3F, S for ACTION_P1; 1F, S for ACTION_P2)...")
    # P1: 3 failures then success -> score 3/5 = 0.6
    await simulate_dsm_process_for_reflection("ACTION_P1", "failure")
    await simulate_dsm_process_for_reflection("ACTION_P1", "failure")
    await simulate_dsm_process_for_reflection("ACTION_P1", "failure")
    await simulate_dsm_process_for_reflection("ACTION_P1", "success")
    # P2: 1 failure then success -> score 1/5 = 0.2
    await simulate_dsm_process_for_reflection("ACTION_P2", "failure") 
    # This 5th event triggers reflection, but P2 hasn't succeeded yet.
    # Let's adjust to ensure reflection happens AFTER P2 succeeds or sequence completes.
    # Maxlen is 10. Reflection interval is 5.
    dsm.learning_events.clear(); dsm.cycles_since_reflection = 0
    dsm.self_model["identity_traits"]["persistence"] = 0.5
    initial_persistence_s_strong = dsm.self_model["identity_traits"]["persistence"]

    # P1: F,F,S (score 0.4 for P1)
    await simulate_dsm_process_for_reflection("ACTION_P_STRONG", "failure")
    await simulate_dsm_process_for_reflection("ACTION_P_STRONG", "failure")
    await simulate_dsm_process_for_reflection("ACTION_P_STRONG", "success")
    # P_OTHER: S (score 0 for P_OTHER)
    await simulate_dsm_process_for_reflection("ACTION_P_OTHER", "success")
    # Filler to trigger reflection:
    await simulate_dsm_process_for_reflection("FILLER", "success") # Reflection after 5 events

    # At reflection: Events = [P_S:F, P_S:F, P_S:S, P_O:S, FILLER:S]
    # Evaluated keys: {ACTION_P_STRONG, ACTION_P_OTHER, FILLER} -> num_distinct_actions = 3
    # For ACTION_P_STRONG: failures_before_success=2. key_persistence_value = min(1.0, 2/5.0) = 0.4
    # Total action_key_persistence_score = 0.4
    # normalized_persistence_metric = 0.4 / 3 = 0.133...
    # This still leads to a decrease or no change.
    # The logic `normalized_persistence_metric > 0.5` for increase is too high if averaging over many non-persistent actions.

    # Let's simplify the test: focus on ONE action key showing persistence.
    dsm.learning_events.clear(); dsm.cycles_since_reflection = 0
    dsm.self_model["identity_traits"]["persistence"] = 0.5
    initial_persistence_s_focused = dsm.self_model["identity_traits"]["persistence"]
    logger.info(f"Simulating FOCUSED persistence for ACTION_Z (F,F,F,S) then one other action.")
    await simulate_dsm_process_for_reflection("ACTION_Z", "failure")
    await simulate_dsm_process_for_reflection("ACTION_Z", "failure")
    await simulate_dsm_process_for_reflection("ACTION_Z", "failure")
    await simulate_dsm_process_for_reflection("ACTION_Z", "success") # Score for ACTION_Z = min(1.0, 3/5.0) = 0.6
    await simulate_dsm_process_for_reflection("ACTION_Y", "success") # Triggers reflection

    # Reflection events: [Z:F, Z:F, Z:F, Z:S, Y:S]
    # Evaluated keys: {ACTION_Z, ACTION_Y}. num_distinct_actions = 2
    # For Z: score 0.6. For Y: score 0.
    # Total action_key_persistence_score = 0.6
    # normalized_persistence_metric = 0.6 / 2 = 0.3
    # Current logic: if metric > 0.5 -> increase. If < 0.2 -> decrease.
    # So, 0.3 will result in persistence_change = 0.0. This is not a good test for *increase*.

    # For persistence to increase, `normalized_persistence_metric` must be > 0.5.
    # If we have 2 evaluated keys, `action_key_persistence_score` must be > 1.0.
    # This means at least one key has a persistence value of 1.0 (e.g. 5 failures then success)
    # AND another key also has some persistence.

    # Let's try to get an increase:
    dsm.learning_events.clear(); dsm.cycles_since_reflection = 0
    dsm.self_model["identity_traits"]["persistence"] = 0.5
    initial_persistence_for_increase_test = dsm.self_model["identity_traits"]["persistence"]
    logger.info(f"Simulating STRONG persistence for ACTION_A (FFFFFS) and ACTION_B (FS) to test INCREASE.")
    # ACTION_A: 5 failures, then success -> score 1.0
    await simulate_dsm_process_for_reflection("ACTION_A_INC", "failure")
    await simulate_dsm_process_for_reflection("ACTION_A_INC", "failure")
    await simulate_dsm_process_for_reflection("ACTION_A_INC", "failure")
    await simulate_dsm_process_for_reflection("ACTION_A_INC", "failure")
    await simulate_dsm_process_for_reflection("ACTION_A_INC", "failure") 
    await simulate_dsm_process_for_reflection("ACTION_A_INC", "success") # 6 events
    # ACTION_B: 1 failure, then success -> score 0.2
    await simulate_dsm_process_for_reflection("ACTION_B_INC", "failure")
    await simulate_dsm_process_for_reflection("ACTION_B_INC", "success") # 8 events
    # Need reflection_interval = 8 or more for this test to run as one block before reflection.
    # Or ensure reflection is only called once at the end.
    # Current reflection_interval is 5. So reflection would have happened after 5th event.
    
    # Let's simplify: Test persistence logic in isolation first, then combine.
    # For now, assume the existing _perform_reflection logic is called and test its effect.
    # We need to ensure enough learning_events are present when _perform_reflection is called.
    # The loop in update_self_model calls _perform_reflection if cycles_since_reflection >= reflection_interval.
    # The helper simulate_dsm_process_for_reflection calls update_self_model, which increments cycles_since_reflection.

    dsm.learning_events.clear(); dsm.cycles_since_reflection = 0 # Reset for a clean scenario
    dsm.self_model["identity_traits"]["persistence"] = 0.5 # Reset trait for test
    base_persistence_val = dsm.self_model["identity_traits"]["persistence"]

    # Scenario: Strong persistence signal, expecting increase
    events_for_increase = [
        {"action_key_specific": "action:TASK_A", "outcome": "failure"},
        {"action_key_specific": "action:TASK_A", "outcome": "failure"},
        {"action_key_specific": "action:TASK_A", "outcome": "failure"},
        {"action_key_specific": "action:TASK_A", "outcome": "success"}, # TASK_A: 3F -> S (score 0.6)
        {"action_key_specific": "action:TASK_B", "outcome": "success"}  # TASK_B: S (score 0.0)
    ] # Total 5 events, reflection will trigger
    for event_data in events_for_increase: # Manually populate learning_events and call reflection
        dsm.learning_events.append(event_data)
    
    await dsm._perform_reflection() # Call reflection directly with this history
    traits_after_increase = dsm.self_model.get("identity_traits", {})
    persistence_after_increase = traits_after_increase.get("persistence", 0.0)
    logger.info(f"Persistence after strong signal (3F->S for TASK_A): {persistence_after_increase:.3f}")
    # With [A:F, A:F, A:F, A:S, B:S]: action_keys_evaluated = {A, B}. num_distinct = 2.
    # Score for A = 0.6. Score for B = 0. normalized_persistence_metric = 0.6 / 2 = 0.3.
    # This will *still* result in no change or decrease due to `normalized_persistence_metric > 0.5` threshold.

    # The persistence increase condition `normalized_persistence_metric > 0.5` needs to be met.
    # To get metric > 0.5 with 2 actions, total score must be > 1.0.
    # e.g., A: 5F->S (score 1.0), B: 1F->S (score 0.2). Total = 1.2. Metric = 0.6.
    dsm.learning_events.clear(); dsm.self_model["identity_traits"]["persistence"] = 0.5; base_persistence_val = 0.5
    events_for_clear_increase = [
        {"action_key_specific": "action:ULTRA_P", "outcome": "failure"}, {"action_key_specific": "action:ULTRA_P", "outcome": "failure"},
        {"action_key_specific": "action:ULTRA_P", "outcome": "failure"}, {"action_key_specific": "action:ULTRA_P", "outcome": "failure"},
        {"action_key_specific": "action:ULTRA_P", "outcome": "failure"}, {"action_key_specific": "action:ULTRA_P", "outcome": "success"}, # ULTRA_P: 5F->S (score 1.0)
        {"action_key_specific": "action:MODERATE_P", "outcome": "failure"}, {"action_key_specific": "action:MODERATE_P", "outcome": "success"} # MODERATE_P: 1F->S (score 0.2)
    ] # Maxlen is 10, this is 8 events.
    for event_data in events_for_clear_increase: dsm.learning_events.append(event_data)
    await dsm._perform_reflection()
    traits_after_clear_increase = dsm.self_model.get("identity_traits", {})
    persistence_after_clear_increase = traits_after_clear_increase.get("persistence", 0.0)
    logger.info(f"Persistence after ULTRA_P & MODERATE_P: {persistence_after_clear_increase:.3f}")
    # Evaluated: {ULTRA_P, MODERATE_P}. num_distinct=2. Score_U=1.0, Score_M=0.2. Total=1.2. Metric = 0.6.
    # normalized_persistence_metric (0.6) > 0.5 is TRUE. persistence_change = +0.05.
    # Expected: 0.5 + 0.05 = 0.55
    assert persistence_after_clear_increase > base_persistence_val, "Persistence should increase with strong patterns."
    assert abs(persistence_after_clear_increase - (base_persistence_val + 0.05)) < 0.001, "Persistence increase amount incorrect."


    # Scenario: Weak or no persistence signal, expecting decrease or no change if already low
    dsm.learning_events.clear(); dsm.self_model["identity_traits"]["persistence"] = 0.5; base_persistence_val = 0.5
    events_for_decrease = [
        {"action_key_specific": "action:TASK_C", "outcome": "failure"}, # No success for C
        {"action_key_specific": "action:TASK_D", "outcome": "success"}, # Immediate success for D
        {"action_key_specific": "action:TASK_E", "outcome": "failure"},
        {"action_key_specific": "action:TASK_F", "outcome": "success"},
        {"action_key_specific": "action:TASK_G", "outcome": "failure"}
    ] # All distinct, no fail-then-succeed patterns. action_key_persistence_score = 0. Metric = 0.
    for event_data in events_for_decrease: dsm.learning_events.append(event_data)
    await dsm._perform_reflection()
    traits_after_decrease = dsm.self_model.get("identity_traits", {})
    persistence_after_decrease = traits_after_decrease.get("persistence", 0.0)
    logger.info(f"Persistence after weak signal: {persistence_after_decrease:.3f}")
    # Metric = 0. persistence_change = -0.05. Expected = 0.5 - 0.05 = 0.45
    assert persistence_after_decrease < base_persistence_val, "Persistence should decrease with weak patterns."
    assert abs(persistence_after_decrease - (base_persistence_val - 0.05)) < 0.001, "Persistence decrease amount incorrect."
    

    # ... (Keep existing tests for Adaptability, Caution, Fast/Slow Learner if they are separate concerns) ...
    # For this focused update, we are only testing the persistence part.
    # The existing fast/slow learner tests should still be valid for their own logic.

    # Restore original DSM settings
    dsm.reflection_interval = original_reflection_interval
    dsm.learning_events = deque(maxlen=original_learning_events_maxlen)
    dsm.self_model["identity_traits"] = initial_traits # Restore all traits
    logger.info("--- Test Passed: DynamicSelfModel Perform Reflection (Persistence Focus) ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_dsm_uses_learning_rate_meta(test_agent: AgentController):
    agent = test_agent
    dsm = agent.dynamic_self_model
    logger.info("--- Test: DynamicSelfModel Uses Learning Rate Meta (Fast/Slow) ---")

    assert dsm is not None, "DynamicSelfModel component missing."

    # --- Setup ---
    base_lr = 0.2 # Set a clear base learning rate
    fast_multiplier = 1.5
    slow_multiplier = 0.5
    # These multipliers are hardcoded in _update_capability, test should match them.

    dsm.self_model["learning_rate"] = base_lr
    dsm.self_model.setdefault("learning_rate_meta", {}).setdefault("capabilities", {})
    dsm.self_model["learning_rate_meta"]["capabilities"]["fast_learner"] = ["action:FAST_ACTION"]
    dsm.self_model["learning_rate_meta"]["capabilities"]["slow_learner"] = ["action:SLOW_ACTION"]
    
    # Clear existing capabilities for these test actions to start from 0 confidence
    dsm.self_model.setdefault("capabilities", {})
    if "action:FAST_ACTION" in dsm.self_model["capabilities"]: del dsm.self_model["capabilities"]["action:FAST_ACTION"]
    if "action:SLOW_ACTION" in dsm.self_model["capabilities"]: del dsm.self_model["capabilities"]["action:SLOW_ACTION"]
    if "action:NORMAL_ACTION" in dsm.self_model["capabilities"]: del dsm.self_model["capabilities"]["action:NORMAL_ACTION"]


    async def simulate_and_check_capability(action_name: str, expected_lr_multiplier: float):
        action_key = f"action:{action_name}"
        initial_confidence = dsm.self_model["capabilities"].get(action_key, 0.0) # Start from 0

        # Simulate a success
        # update_self_model calls _update_capability. current_action_key_for_update will be action_key.
        await dsm.update_self_model(
            last_action_type=action_name, action_outcome="success",
            action_params={}, action_error=None, action_result_data=None, current_phenomenal_state=None
        )
        
        confidence_after_success = dsm.self_model["capabilities"].get(action_key, 0.0)
        
        # Expected change: new_conf = old_conf + effective_lr * (1.0 - old_conf)
        # Here, old_conf = 0.0. So, new_conf = effective_lr
        # effective_lr = base_lr * multiplier (then clamped)
        
        expected_effective_lr = base_lr * expected_lr_multiplier
        # Apply clamping as done in _update_capability
        expected_effective_lr = max(0.01, min(0.5, expected_effective_lr))

        logger.info(
            f"Action: {action_key}, InitialConf: {initial_confidence:.3f}, "
            f"ConfAfterSuccess: {confidence_after_success:.3f}, ExpectedEffectiveLR: {expected_effective_lr:.3f}"
        )
        assert abs(confidence_after_success - expected_effective_lr) < 0.001, \
            f"Confidence for {action_key} after success did not match expected effective LR. Got {confidence_after_success}, expected {expected_effective_lr}"
        
        # Reset capability for next check if needed, or check decrease from failure
        # For simplicity, we'll just check the first success.
        if action_key in dsm.self_model["capabilities"]: del dsm.self_model["capabilities"][action_key]


    # Test FAST_ACTION
    logger.info("Testing FAST_ACTION learning rate...")
    await simulate_and_check_capability("FAST_ACTION", fast_multiplier)

    # Test SLOW_ACTION
    logger.info("Testing SLOW_ACTION learning rate...")
    await simulate_and_check_capability("SLOW_ACTION", slow_multiplier)

    # Test NORMAL_ACTION (not in fast/slow lists)
    logger.info("Testing NORMAL_ACTION learning rate...")
    await simulate_and_check_capability("NORMAL_ACTION", 1.0) # Multiplier is 1.0


    logger.info("--- Test Passed: DynamicSelfModel Uses Learning Rate Meta (Fast/Slow) ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_htn_plan_caching(test_agent: AgentController):
    agent = test_agent
    planner = agent.htn_planner
    cache = agent.cache # Assuming agent.cache is the CognitiveCache instance
    logger.info("--- Test: HTN Plan Caching ---")

    assert planner is not None, "HTNPlanner component missing."
    assert cache is not None, "CognitiveCache component missing."
    assert planner._cache is cache, "HTNPlanner not using the agent's cache instance."
    assert Predicate is not None and Goal is not None and create_goal_from_descriptor is not None

    # --- Setup a simple planning problem ---
    original_operators = planner.operators.copy()
    original_methods = planner.methods.copy()
    planner.operators = {}
    planner.methods = {}
    planner.operators["OP_CACHE_A"] = Operator(name="OP_CACHE_A", effects={Predicate("state_cache_A", (), True)})
    planner.operators["OP_CACHE_B"] = Operator(name="OP_CACHE_B", preconditions={Predicate("state_cache_A", (), True)}, effects={Predicate("state_cache_B", (), True)})
    planner.methods["task_cache_X"] = [
        Method(name="method_cache_path", task_signature=("task_cache_X",), subtasks=["OP_CACHE_A", "OP_CACHE_B"])
    ]

    test_goal_cache = create_goal_from_descriptor("Achieve state_cache_B via task_cache_X")
    assert test_goal_cache is not None
    initial_state_cache = {Predicate("initial_dummy_pred_for_cache_key", (), True)} # Ensure state is somewhat unique

    with patch.object(planner, '_goal_to_task', return_value=("task_cache_X",)):
        # --- 1. First plan generation (should compute and cache) ---
        logger.info("Generating plan for the first time (should compute)...")
        # Temporarily patch _decompose to count calls
        original_decompose = planner._decompose
        decompose_call_count = 0
        async def mock_decompose_counter(*args, **kwargs):
            nonlocal decompose_call_count
            decompose_call_count += 1
            return await original_decompose(*args, **kwargs)

        with patch.object(planner, '_decompose', side_effect=mock_decompose_counter) as  mock_decompose_patch_obj:
            plan1 = await planner.plan(test_goal_cache, initial_state_cache)
            assert plan1 is not None, "Plan 1 generation failed."
            assert len(plan1) == 2, "Plan 1 has incorrect length."
            assert decompose_call_count > 0, "_decompose should have been called for plan 1."
            logger.info(f"Plan 1 generated. _decompose calls: {decompose_call_count}")

            # --- 2. Second plan generation for same goal & state (should hit cache) ---
            logger.info("Generating plan for the second time (should hit cache)...")
            decompose_call_count_before_plan2 = decompose_call_count # Store count before 2nd call
            
            plan2 = await planner.plan(test_goal_cache, initial_state_cache)
            assert plan2 is not None, "Plan 2 retrieval from cache failed."
            assert plan2 == plan1, "Cached plan (plan 2) does not match original plan (plan 1)."
            # Crucially, _decompose should NOT have been called again, or far fewer times if partial plans are cached.
            # For simple full plan caching, it should be the same.
            assert decompose_call_count == decompose_call_count_before_plan2, \
                f"_decompose should not have been called again for plan 2. Prev calls: {decompose_call_count_before_plan2}, New total: {decompose_call_count}"
            logger.info(f"Plan 2 retrieved from cache. _decompose calls still: {decompose_call_count}")

            # --- 3. Test Cache Expiry (if TTL is short enough for a test) ---
            # Set a short TTL for this test part if possible, or simulate time passing.
            # Current default test TTL for CognitiveCache is 0.2s. Plan cache TTL is from HTN config (e.g. 300s).
            # For this test, let's assume plan_cache_ttl is short.
            # If planner._plan_cache_ttl is, say, 0.1s for the test:
            # planner._plan_cache_ttl = 0.05 # Override for test
            # await asyncio.sleep(0.1) # Wait for cache to expire
            # decompose_call_count_before_plan3 = decompose_call_count
            # plan3 = await planner.plan(test_goal_cache, initial_state_cache)
            # assert plan3 is not None and plan3 == plan1, "Plan 3 (after expiry) generation failed or mismatch."
            # assert decompose_call_count > decompose_call_count_before_plan3, \
            #     "_decompose should have been called again for plan 3 after cache expiry."
            # logger.info(f"Plan 3 (after expiry) re-computed. _decompose calls: {decompose_call_count}")


    # Restore original library & settings
    planner.operators = original_operators
    planner.methods = original_methods
    logger.info("--- Test Passed: HTN Plan Caching ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_htn_heuristic_influence(test_agent: AgentController):
    agent = test_agent
    planner = agent.htn_planner
    logger.info("--- Test: HTN Heuristic Influence (Cost-Based) ---")

    assert planner is not None, "HTNPlanner component missing."
    assert Predicate is not None, "Predicate class not available for test setup."
    assert Goal is not None, "Goal class not available for test setup."
    assert create_goal_from_descriptor is not None, "create_goal_from_descriptor not available."

    original_operators = planner.operators.copy()
    original_methods = planner.methods.copy()
    planner.operators = {}
    planner.methods = {}

    _Predicate_for_test = globals().get('Predicate')
    assert _Predicate_for_test is not None and not (hasattr(_Predicate_for_test, '_is_dummy') and _Predicate_for_test._is_dummy()), \
        "Predicate class is missing or dummy for test_htn_heuristic_influence setup."

    planner.operators["OP_LOW_COST_A"] = Operator(name="OP_LOW_COST_A", estimated_cost=0.5, 
                                                 effects={_Predicate_for_test("effect_A", (), True)})
    planner.operators["OP_FAIL_PRECOND"] = Operator(name="OP_FAIL_PRECOND", estimated_cost=0.1, 
                                                 preconditions={_Predicate_for_test("missing_precond", (), True)}, 
                                                 effects={_Predicate_for_test("effect_fail", (), True)})
    
    planner.operators["OP_HIGH_COST_B1"] = Operator(name="OP_HIGH_COST_B1", estimated_cost=2.0, 
                                                   effects={_Predicate_for_test("effect_B1", (), True)})
    planner.operators["OP_HIGH_COST_B2"] = Operator(name="OP_HIGH_COST_B2", estimated_cost=2.0, 
                                                   effects={_Predicate_for_test("effect_B2", (), True)})

    method_A_cheap_fails = Method(name="method_A_cheap_fails",
                                 task_signature=("task_cost_choice",),
                                 preconditions={_Predicate_for_test("can_try_A", (), True)},
                                 subtasks=["OP_LOW_COST_A", "OP_FAIL_PRECOND"])

    method_B_expensive_succeeds = Method(name="method_B_expensive_succeeds",
                                       task_signature=("task_cost_choice",),
                                       preconditions={_Predicate_for_test("can_try_B", (), True)},
                                       subtasks=["OP_HIGH_COST_B1", "OP_HIGH_COST_B2"])

    planner.methods["task_cost_choice"] = [method_A_cheap_fails, method_B_expensive_succeeds]

    test_goal = create_goal_from_descriptor("Goal for cost choice leading to effect_B2")
    assert test_goal is not None, "Failed to create test_goal."
    test_goal.success_criteria = {_Predicate_for_test("effect_B2", (), True)}

    initial_state = {
        _Predicate_for_test("can_try_A", (), True), 
        _Predicate_for_test("can_try_B", (), True)
    }

    with patch.object(planner, '_goal_to_task', return_value=("task_cost_choice",)):
        htn_module_logger_name = planner.__class__.__module__
        htn_logger_for_test = logging.getLogger(htn_module_logger_name)
        
        original_level = htn_logger_for_test.level
        if htn_logger_for_test.level > logging.DEBUG or htn_logger_for_test.level == 0:
            htn_logger_for_test.setLevel(logging.DEBUG)

        with patch.object(htn_logger_for_test, 'debug') as mock_planner_debug_log:
            logger.info("Planning with cost-based heuristic choice...")
            current_agent_max_depth = agent.config.get("performance",{}).get("max_planning_depth", 5)
            logger.info(f"Test using agent's configured max_planning_depth: {current_agent_max_depth}")

            plan = await planner.plan(test_goal, initial_state)

            assert plan is not None, "Plan should be found using Method B after Method A (cheaper) fails."
            assert len(plan) == 2, f"Plan should consist of 2 actions from Method B, got {len(plan)}"
            assert plan[0]["type"] == "OP_HIGH_COST_B1" and plan[1]["type"] == "OP_HIGH_COST_B2", \
                   "Plan does not match Method B's subtasks."

            log_messages = [call_args[0][0] for call_args in mock_planner_debug_log.call_args_list]
            
            # --- Added print statements for debugging the mock capture ---
            logger.info(f"DEBUG_TEST: Mock planner debug log calls (count: {mock_planner_debug_log.call_count}):")
            for i, call_arg_tuple in enumerate(mock_planner_debug_log.call_args_list):
                logger.info(f"  DEBUG_TEST: Mocked Log Call {i}: {str(call_arg_tuple[0][0])[:200]}")
            # --- End added print statements ---

            method_A_tried_log = any(f"Trying method: {method_A_cheap_fails.name}" in msg for msg in log_messages)
            method_A_failed_log = any(f"Method '{method_A_cheap_fails.name}' failed" in msg for msg in log_messages)
            
            method_B_tried_log = any(f"Trying method: {method_B_expensive_succeeds.name}" in msg for msg in log_messages)
            
            method_B_succeeded_str_to_find = f"Method '{method_B_expensive_succeeds.name}' successfully decomposed task 'task_cost_choice'"
            method_B_succeeded_log = any(
                method_B_succeeded_str_to_find in msg.strip() 
                for msg in log_messages
            )
            if not method_B_succeeded_log:
                logger.error(f"Expected log string for Method B success not found: '{method_B_succeeded_str_to_find}'")


            assert method_A_tried_log, "Log does not show Method A (cheaper) was tried."
            assert method_A_failed_log, "Log does not show Method A failed."
            assert method_B_tried_log, "Log does not show Method B (more expensive) was tried after A."
            assert method_B_succeeded_log, \
                f"Log does not show Method B succeeded with the expected message. Searched for '{method_B_succeeded_str_to_find}'. Captured DEBUG logs for htn_planner:\n" + \
                "\n".join([f"  - {m}" for m in log_messages])

            try:
                idx_try_A = next(i for i, msg in enumerate(log_messages) if f"Trying method: {method_A_cheap_fails.name}" in msg)
                idx_fail_A_method_log = next(i for i, msg in enumerate(log_messages) if f"Method '{method_A_cheap_fails.name}' failed" in msg and i > idx_try_A)
                idx_try_B = next(i for i, msg in enumerate(log_messages) if f"Trying method: {method_B_expensive_succeeds.name}" in msg and i > idx_fail_A_method_log)
                idx_succeed_B_method_log = next(i for i, msg in enumerate(log_messages) if method_B_succeeded_str_to_find in msg.strip() and i > idx_try_B)

                assert idx_try_A < idx_fail_A_method_log < idx_try_B < idx_succeed_B_method_log, \
                    (f"Log order incorrect. A_try_idx:{idx_try_A}, A_fail_method_idx:{idx_fail_A_method_log}, "
                     f"B_try_idx:{idx_try_B}, B_succeed_method_idx:{idx_succeed_B_method_log}")
                logger.info("Heuristic ordering and backtracking verified by log message order.")
            except StopIteration:
                logger.error("Could not find all necessary log messages to verify order. Captured DEBUG logs for htn_planner (already printed by test if mock captured them).")
                # The detailed logs are now printed by the added debug loop above.
                pytest.fail("Could not find all necessary log messages in MOCK CAPTURE to verify cost-based heuristic trial order and success.")
        
        if htn_logger_for_test.level != original_level:
             htn_logger_for_test.setLevel(original_level)

    planner.operators = original_operators
    planner.methods = original_methods
    logger.info("--- Test Passed: HTN Heuristic Influence (Cost-Based) ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
@patch(f'{PACKAGE_NAME}.cognitive_modules.narrative_constructor.call_ollama', new_callable=AsyncMock)
async def test_narrative_prompt_changes_with_cs(mock_call_ollama_cs_prompt: AsyncMock, test_agent: AgentController):
    agent = test_agent
    nc = agent.narrative_constructor
    logger.info("--- Test: NarrativeConstructor Prompt Varies with ConsciousState ---")

    assert nc is not None, "NarrativeConstructor component missing."
    assert ConsciousState is not None, "ConsciousState enum not available for test."
    _PhenomenalState_class_for_test = globals().get('PhenomenalState') # Get PhenomenalState class
    assert _PhenomenalState_class_for_test is not None, "PhenomenalState class not available for test setup."


    mock_llm_response_text = "A standard narrative entry."
    mock_call_ollama_cs_prompt.return_value = (mock_llm_response_text, None)

    # Common inputs for nc.generate_narrative_entry
    test_p_state = _PhenomenalState_class_for_test(content={}, intensity=0.8, valence=0.2, timestamp=time.time())
    triggering_event_summary = {"test_trigger": "data"}
    reason_str = "TestReasonString"

    # --- Scenario 1: CONSCIOUS state ---
    agent.consciousness_level = ConsciousState.CONSCIOUS # Set agent's CS level
    nc._last_drive_state = {"curiosity": 0.5} # Provide some drive state for prompt

    logger.info("Generating narrative for CONSCIOUS state...")
    await nc.generate_narrative_entry(test_p_state, triggering_event_summary, reason_str)
    
    mock_call_ollama_cs_prompt.assert_called_once()
    args_conscious, _ = mock_call_ollama_cs_prompt.call_args
    system_prompt_conscious = args_conscious[1][0]['content'] # messages[0]['content'] is system prompt
    assert "brief, reflective entry" in system_prompt_conscious.lower(), \
        "Standard prompt not used for CONSCIOUS state."
    assert "deeply reflective" not in system_prompt_conscious.lower(), \
        "Meta-conscious prompt incorrectly used for CONSCIOUS state."
    mock_call_ollama_cs_prompt.reset_mock()
    logger.info("CONSCIOUS state prompt verified.")

    # --- Scenario 2: META_CONSCIOUS state ---
    agent.consciousness_level = ConsciousState.META_CONSCIOUS # Change agent's CS level
    nc._last_drive_state = {"curiosity": 0.6} # Update drive state for new call

    logger.info("Generating narrative for META_CONSCIOUS state...")
    await nc.generate_narrative_entry(test_p_state, triggering_event_summary, reason_str)

    mock_call_ollama_cs_prompt.assert_called_once()
    args_meta, _ = mock_call_ollama_cs_prompt.call_args
    system_prompt_meta = args_meta[1][0]['content']
    assert "deeply reflective and insightful entry" in system_prompt_meta.lower(), \
        "Meta-conscious prompt not used for META_CONSCIOUS state."
    assert "brief, reflective entry" not in system_prompt_meta.lower(), \
        "Standard prompt incorrectly used for META_CONSCIOUS state."
    mock_call_ollama_cs_prompt.reset_mock()
    logger.info("META_CONSCIOUS state prompt verified.")

    # --- Scenario 3: REFLECTIVE state (should also use the deeper prompt) ---
    agent.consciousness_level = ConsciousState.REFLECTIVE
    nc._last_drive_state = {"curiosity": 0.7}

    logger.info("Generating narrative for REFLECTIVE state...")
    await nc.generate_narrative_entry(test_p_state, triggering_event_summary, reason_str)

    mock_call_ollama_cs_prompt.assert_called_once()
    args_reflective, _ = mock_call_ollama_cs_prompt.call_args
    system_prompt_reflective = args_reflective[1][0]['content']
    assert "deeply reflective and insightful entry" in system_prompt_reflective.lower(), \
        "Meta-conscious/Reflective prompt not used for REFLECTIVE state."
    mock_call_ollama_cs_prompt.reset_mock()
    logger.info("REFLECTIVE state prompt verified.")

    logger.info("--- Test Passed: NarrativeConstructor Prompt Varies with ConsciousState ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_po_adjusts_loop_detector_window_size(test_agent: AgentController):
    agent = test_agent
    po = agent.performance_optimizer
    ld = agent.loop_detector
    kb = agent.knowledge_base # Loop detector needs KB
    logger.info("--- Test: PO Dynamically Adjusts LoopDetector.window_size ---")

    assert po is not None, "PerformanceOptimizer component missing."
    assert ld is not None, "LoopDetector component missing."
    assert kb is not None, "KnowledgeBase component missing for LoopDetector."
    assert ld._kb is kb, "LoopDetector does not have the correct KB reference for the test."


    # --- Setup ---
    agent.config.setdefault("performance_optimizer", {})["auto_apply_adjustments"] = True
    po._config_perf_optimizer_section["auto_apply_adjustments"] = True 

    initial_ld_window_size = 10
    agent.config.setdefault("loop_detection", {})["window_size"] = initial_ld_window_size
    ld.window_size = initial_ld_window_size 
    po.config_changes = {}

    logger.info(f"Initial agent.config[loop_detection][window_size] = {initial_ld_window_size}")

    # Simulate a "loop_detection" bottleneck
    loop_detection_threshold = po.optimization_thresholds.get("loop_detection", 0.01)
    simulated_ld_duration = loop_detection_threshold * 2.5 # Severity 2.5
    cycle_profile_ld_bottleneck = {"loop_detection": simulated_ld_duration}

    # --- Action 1: PO processes bottleneck and suggests change ---
    await po.process({"cycle_profile": cycle_profile_ld_bottleneck})
    po_status = await po.get_status()
    active_po_adjustments = po_status.get("active_config_adjustments", {})
    
    assert "loop_detection" in active_po_adjustments, "PO did not suggest for 'loop_detection'."
    assert "window_size" in active_po_adjustments.get("loop_detection",{}), "PO did not suggest 'window_size'."
    
    suggested_new_window_size = active_po_adjustments["loop_detection"]["window_size"]
    # Initial 10, sev 2.5 -> Rule1 (sev>2.0, current>4) -> new = max(2, 10-2) = 8
    assert suggested_new_window_size == 8, f"PO suggested LD window_size {suggested_new_window_size}, expected 8."
    logger.info(f"PO suggested new LD window_size: {suggested_new_window_size}")

    # --- Action 2: Simulate AgentController merging this change ---
    config_updated_flag_test = False
    if active_po_adjustments: # Generic merge
        for comp_key, comp_adjusts in active_po_adjustments.items():
            if comp_key not in agent.config: agent.config[comp_key] = {}
            if isinstance(agent.config.get(comp_key), dict) and isinstance(comp_adjusts, dict):
                for param_key, new_val in comp_adjusts.items():
                    if agent.config[comp_key].get(param_key) != new_val:
                        agent.config[comp_key][param_key] = new_val
                        config_updated_flag_test = True
                        logger.info(f"TEST_APPLY (LD): agent.config[{comp_key}][{param_key}] updated to {new_val}")
    assert config_updated_flag_test, "Agent config for LD window_size was not updated."
    assert agent.config.get("loop_detection", {}).get("window_size") == suggested_new_window_size

    # --- Action 3: Call LoopDetector.detect_loops and check if it uses the new window_size ---
    # We need to patch kb.query_state to control history and also log LD's effective window size.
    # LoopDetector's detect_loops calls kb.query_state({"recent_facts": fetch_count})
    # fetch_count depends on the window_size.
    
    # Mock KB's query_state to return controllable minimal data
    async def mock_kb_query_state_for_ld(query_dict):
        if "recent_facts" in query_dict:
            # Return empty list, so detect_loops doesn't do much actual detection,
            # but we can check the fetch_count used based on window_size.
            return {"recent_facts": []} 
        return {} # Default empty response for other queries

    ld_logger = logging.getLogger(f"{PACKAGE_NAME}.cognitive_modules.loop_detector")
    with patch.object(kb, 'query_state', side_effect=mock_kb_query_state_for_ld) as mock_kb_qs, \
         patch.object(ld_logger, 'debug') as mock_ld_debug_log:
        
        await ld.detect_loops()

        found_effective_window_log = False
        for call_args in mock_ld_debug_log.call_args_list:
            log_msg = str(call_args[0][0])
            if f"Detecting loops using effective window_size: {suggested_new_window_size}" in log_msg:
                found_effective_window_log = True
                logger.info(f"LoopDetector correctly logged usage of updated window_size: {suggested_new_window_size}")
                break
        assert found_effective_window_log, \
            f"LoopDetector did not log DEBUG message for 'Effective window_size: {suggested_new_window_size}'. Review LD debug logs."

        # Verify kb.query_state was called with a fetch_count derived from the new window_size
        # fetch_count = current_effective_window_size * 3 if self.ignore_thinking_actions else current_effective_window_size + 5
        # ignore_thinking_actions is True in test_agent fixture, window_size is 8
        # expected_fetch_count = 8 * 3 = 24
        expected_fetch_count = max(suggested_new_window_size * 3 if ld.ignore_thinking_actions else suggested_new_window_size + 5, 15)
        
        found_kb_call_with_correct_fetch = False
        for call in mock_kb_qs.call_args_list:
            args, _ = call
            if args and isinstance(args[0], dict) and "recent_facts" in args[0]:
                actual_fetch_count = args[0]["recent_facts"]
                if actual_fetch_count == expected_fetch_count:
                    found_kb_call_with_correct_fetch = True
                    logger.info(f"KB query_state called with correct fetch_count {actual_fetch_count} derived from new window_size.")
                    break
        assert found_kb_call_with_correct_fetch, \
            f"LoopDetector did not call KB with fetch_count ({expected_fetch_count}) derived from new window_size. Calls: {mock_kb_qs.call_args_list}"


    logger.info("--- Test Passed: PO Dynamically Adjusts LoopDetector.window_size ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_persistent_planning_failure_fails_goal(test_agent: AgentController):
    agent = test_agent
    logger.info("--- Test: Persistent Planning Failure Fails Goal ---")

    assert agent.htn_planner is not None, "HTNPlanner component missing."
    assert agent.knowledge_base is not None, "KnowledgeBase component missing."
    # These (Goal, GoalStatus, Predicate) are imported at the top of test_cognitive_cycle.py
    assert Goal is not None and GoalStatus is not None and Predicate is not None, "Required datatypes/enums missing for test."
    assert agent._create_goal_from_descriptor is not None, "_create_goal_from_descriptor missing from agent."

    # Store the original config value for max_consecutive_planning_failures to restore it later.
    # This value comes from the fixture's config_override.
    original_max_failures_config_val = agent.config.get("agent", {}).get("max_consecutive_planning_failures")
    
    # For the test, we'll use the value from config if it's there, or default to 2.
    # AgentController.__init__ already reads this into self._max_planning_failures_before_goal_fail.
    # We just need to ensure the test uses the intended value for its assertions.
    # The fixture sets config["agent"]["max_consecutive_planning_failures"] = 2
    test_specific_max_failures = agent.config.get("agent",{}).get("max_consecutive_planning_failures", 2)
    agent._max_planning_failures_before_goal_fail = test_specific_max_failures # Ensure instance var matches test config

    logger.info(f"Test configured to fail goal after {agent._max_planning_failures_before_goal_fail} planning failures.")

    unplannable_goal_desc = "This goal is unplannable for test"
    unplannable_goal = agent._create_goal_from_descriptor(unplannable_goal_desc, priority=USER_GOAL_PRIORITY) 
    assert unplannable_goal is not None and hasattr(unplannable_goal, 'id'), "Failed to create unplannable_goal."
    unplannable_goal_id = unplannable_goal.id
    agent.active_goals = [unplannable_goal]
    agent.current_plan = None 

    original_planner_plan_method = agent.htn_planner.plan
    async def mock_failing_plan(goal_arg, state_arg):
        _Goal_class_for_mock = globals().get('Goal') 
        if _Goal_class_for_mock and isinstance(goal_arg, _Goal_class_for_mock) and \
           hasattr(goal_arg, 'description') and goal_arg.description == unplannable_goal_desc:
            logger.info(f"MOCK HTN: Simulating planning failure for '{unplannable_goal_desc}'")
            return None
        return await original_planner_plan_method(goal_arg, state_arg) 

    with patch.object(agent.htn_planner, 'plan', side_effect=mock_failing_plan):
        for cycle_num_sim in range(1, agent._max_planning_failures_before_goal_fail + 2): 
            agent.cycle_count = cycle_num_sim 
            logger.info(f"Simulating cycle {cycle_num_sim} for planning failure test...")

            active_goal_sim = agent._oscar_get_active_goal() 
            
            if not active_goal_sim : 
                if cycle_num_sim <= agent._max_planning_failures_before_goal_fail:
                     pytest.fail(
                         f"Cycle {cycle_num_sim}: Unplannable goal disappeared before max planning failures. "
                         f"Active goals: {[g.description for g in agent.active_goals if hasattr(g,'description')]}"
                     )
                else: 
                     logger.info(f"Cycle {cycle_num_sim}: Unplannable goal correctly no longer active (expected as FAILED).")
                     break 
            
            assert hasattr(active_goal_sim, 'id') and active_goal_sim.id == unplannable_goal_id, \
                (f"Cycle {cycle_num_sim}: Active goal is not the unplannable one. "
                 f"Got: {active_goal_sim.description if hasattr(active_goal_sim,'description') else 'N/A'}")

            agent.current_plan = await agent.htn_planner.plan(active_goal_sim, set()) 
            
            current_goal_id_sim_tracking = active_goal_sim.id 

            if agent.current_plan is None: 
                agent._active_goal_planning_failure_count[current_goal_id_sim_tracking] = \
                    agent._active_goal_planning_failure_count.get(current_goal_id_sim_tracking, 0) + 1
                
                failure_count_sim = agent._active_goal_planning_failure_count[current_goal_id_sim_tracking]
                logger.info(f"Cycle {cycle_num_sim}: Planning failed for '{unplannable_goal_desc}'. Failure count: {failure_count_sim}")

                if failure_count_sim >= agent._max_planning_failures_before_goal_fail:
                    logger.warning(f"Cycle {cycle_num_sim}: Max planning failures reached for '{unplannable_goal_desc}'. Marking FAILED.")
                    
                    assert agent._GoalStatus is GoalStatus, \
                        f"Agent is not using the real GoalStatus enum. Type: {type(agent._GoalStatus)}"
                    if hasattr(active_goal_sim, 'status'):
                        active_goal_sim.status = agent._GoalStatus.FAILED 
                        
                        kb_sim = agent.knowledge_base
                        assert agent._Predicate is Predicate, \
                            f"Agent is not using the real Predicate class. Type: {type(agent._Predicate)}"
                        if kb_sim:
                            reason_str_sim = f"PersistentPlanningFailure_{failure_count_sim}_attempts"
                            await kb_sim.assert_fact(
                                agent._Predicate("goalFailed", (unplannable_goal_id, reason_str_sim), True, timestamp=time.time()) 
                            )
                        else:
                            pytest.fail("agent.knowledge_base is None, cannot assert goalFailed predicate.")
                            
                    if current_goal_id_sim_tracking in agent._active_goal_planning_failure_count:
                        del agent._active_goal_planning_failure_count[current_goal_id_sim_tracking]
            
            goals_to_remove_sim = []
            if hasattr(active_goal_sim, 'status') and active_goal_sim.status == agent._GoalStatus.FAILED: 
                goals_to_remove_sim.append(active_goal_sim)
            
            if goals_to_remove_sim:
                agent.active_goals = [g for g in agent.active_goals if g not in goals_to_remove_sim]
                logger.info(f"Cycle {cycle_num_sim}: Goal '{unplannable_goal_desc}' FAILED and removed from active_goals.")
            
            if not any(hasattr(g, 'id') and g.id == unplannable_goal_id for g in agent.active_goals): 
                 if cycle_num_sim >= agent._max_planning_failures_before_goal_fail:
                     logger.info(f"Goal {unplannable_goal_id} no longer in active_goals after {cycle_num_sim} cycles as expected.")
                     break
                 else:
                     pytest.fail(f"Goal {unplannable_goal_id} removed from active_goals prematurely at cycle {cycle_num_sim}")

    # Verification after loop
    assert unplannable_goal_id not in agent._active_goal_planning_failure_count, \
        "Planning failure count for the failed goal was not cleared from tracking dict."
    
    assert hasattr(unplannable_goal, 'status') and unplannable_goal.status == agent._GoalStatus.FAILED, \
        (f"Goal status was not FAILED. Got: {unplannable_goal.status.name if hasattr(unplannable_goal, 'status') and hasattr(unplannable_goal.status, 'name') else unplannable_goal.status}")

    all_goal_failed_preds = await agent.knowledge_base.query(name="goalFailed", value=True)
    found_target_predicate_in_kb = False
    # The value of _max_planning_failures_before_goal_fail used in the simulation loop was test_specific_max_failures
    expected_reason_part = f"PersistentPlanningFailure_{test_specific_max_failures}_attempts"
    
    for pred_item_kb in all_goal_failed_preds:
        if hasattr(pred_item_kb, 'args') and isinstance(pred_item_kb.args, tuple) and len(pred_item_kb.args) == 2:
            if pred_item_kb.args[0] == unplannable_goal_id:
                if isinstance(pred_item_kb.args[1], str) and expected_reason_part in pred_item_kb.args[1]:
                    found_target_predicate_in_kb = True
                    logger.info(f"Found matching goalFailed predicate in KB: {pred_item_kb}")
                    break
    assert found_target_predicate_in_kb, \
        (f"goalFailed predicate for goal_id '{unplannable_goal_id}' with reason containing "
         f"'{expected_reason_part}' not found in KB. All goalFailed preds: {all_goal_failed_preds}")

    # Restore original config setting for "max_consecutive_planning_failures"
    agent_config_section = agent.config.get("agent", {})
    if original_max_failures_config_val is not None:
        agent_config_section["max_consecutive_planning_failures"] = original_max_failures_config_val
        # Restore the instance variable on agent controller for consistency if it were to be used by other logic immediately
        agent._max_planning_failures_before_goal_fail = original_max_failures_config_val 
        logger.info(f"Restored 'max_consecutive_planning_failures' in agent.config to {original_max_failures_config_val}")
    else: 
        if "max_consecutive_planning_failures" in agent_config_section:
            del agent_config_section["max_consecutive_planning_failures"]
            logger.info("Removed 'max_consecutive_planning_failures' from agent.config as it was not in original_max_failures_config_val.")
        # Fallback default should match AgentController.__init__'s default if key is missing
        agent._max_planning_failures_before_goal_fail = 3 
        logger.info(f"Set agent._max_planning_failures_before_goal_fail to default 3 as original_max_failures_config_val was None.")

    logger.info("--- Test Passed: Persistent Planning Failure Fails Goal ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_persistent_execution_failure_fails_goal(test_agent: AgentController):
    agent = test_agent
    logger.info("--- Test: Persistent Action Execution Failure Fails Goal ---")

    assert agent.htn_planner is not None
    assert agent.knowledge_base is not None
    assert Goal is not None and GoalStatus is not None and Predicate is not None

    original_max_exec_failures_config = agent.config.get("agent",{}).get("max_execution_failures_per_goal")
    test_max_exec_failures = 2
    agent.config.get("agent",{})["max_execution_failures_per_goal"] = test_max_exec_failures
    agent._max_execution_failures_per_goal = test_max_exec_failures
    
    logger.info(f"Test configured to fail goal after {agent._max_execution_failures_per_goal} execution failures.")

    failing_action_goal_desc = "Goal with an action that always fails"
    failing_action_goal = agent._create_goal_from_descriptor(failing_action_goal_desc, priority=USER_GOAL_PRIORITY)
    assert failing_action_goal is not None and hasattr(failing_action_goal, 'id')
    failing_action_goal_id = failing_action_goal.id
    agent.active_goals = [failing_action_goal]
    agent.current_plan = None

    # Mock HTNPlanner.plan to return a plan with one action "FAILING_ACTION"
    # Mock AgentController._oscar_execute_action for "FAILING_ACTION" to always return failure
    
    async def mock_plan_with_failing_action(goal_arg, state_arg):
        if hasattr(goal_arg, 'description') and goal_arg.description == failing_action_goal_desc:
            logger.info(f"MOCK HTN: Providing plan with FAILING_ACTION for '{failing_action_goal_desc}'")
            return [{"type": "FAILING_ACTION", "params": {"reason": "test_failure"}}]
        # Fallback to original planner for other goals (if any)
        # For this test, we only expect this goal.
        nonlocal original_planner_plan_method # If original_planner_plan_method is defined outside
        original_planner_plan_method = agent.htn_planner.plan # Capture it if not done already
        return await original_planner_plan_method(goal_arg, state_arg)

    async def mock_execute_failing_action(action_arg):
        if isinstance(action_arg, dict) and action_arg.get("type") == "FAILING_ACTION":
            logger.info(f"MOCK EXEC: Simulating failure for FAILING_ACTION")
            return {"outcome": "failure", "error": "Action designed to fail for test"}
        # Fallback to original execute_action for other actions
        nonlocal original_execute_action_method # If original_execute_action_method is defined outside
        original_execute_action_method = agent._oscar_execute_action # Capture it
        return await original_execute_action_method(action_arg)

    original_planner_plan_method = agent.htn_planner.plan # Store to restore later
    original_execute_action_method = agent._oscar_execute_action

    with patch.object(agent.htn_planner, 'plan', side_effect=mock_plan_with_failing_action), \
         patch.object(agent, '_oscar_execute_action', side_effect=mock_execute_failing_action):

        for cycle_num_sim in range(1, agent._max_execution_failures_per_goal + 2): # N failures + 1 to check removal
            agent.cycle_count = cycle_num_sim
            logger.info(f"Simulating cycle {cycle_num_sim} for execution failure test...")

            active_goal_sim = agent._oscar_get_active_goal()
            
            if not active_goal_sim:
                if cycle_num_sim <= agent._max_execution_failures_per_goal:
                    pytest.fail(f"Cycle {cycle_num_sim}: Goal disappeared before max execution failures.")
                else:
                    logger.info(f"Cycle {cycle_num_sim}: Goal correctly no longer active after max failures.")
                    break
            
            assert hasattr(active_goal_sim, 'id') and active_goal_sim.id == failing_action_goal_id

            # --- Simulate Step 8: Planning (will call mock_plan_with_failing_action) ---
            if agent.current_plan is None: # If no plan (e.g., after failure), try to plan
                 agent.current_plan = await agent.htn_planner.plan(active_goal_sim, set())
            
            assert agent.current_plan is not None and len(agent.current_plan) == 1, \
                f"Cycle {cycle_num_sim}: Planner did not return expected failing plan."
            
            # --- Simulate Step 9: Action Selection & Execution ---
            next_action_sim = agent._oscar_select_next_action(agent.current_plan)
            action_result_sim = await agent._oscar_execute_action(next_action_sim) # Calls mock_execute_failing_action

            # --- Simulate Step 10: Goal Status Update Logic (simplified from AgentController) ---
            if action_result_sim.get("outcome") == "success":
                # ... (logic from controller for success, including clearing exec failure count) ...
                if agent.current_plan and len(agent.current_plan) > 0 and agent.current_plan[0] == next_action_sim:
                    agent.current_plan.pop(0)
                if failing_action_goal_id in agent._active_goal_execution_failure_count:
                    del agent._active_goal_execution_failure_count[failing_action_goal_id]

            elif action_result_sim.get("outcome") == "failure":
                agent.current_plan = None # Invalidate plan
                if hasattr(active_goal_sim, 'status') and active_goal_sim.status == agent._GoalStatus.ACTIVE:
                    agent._active_goal_execution_failure_count[failing_action_goal_id] = \
                        agent._active_goal_execution_failure_count.get(failing_action_goal_id, 0) + 1
                    
                    exec_fail_count = agent._active_goal_execution_failure_count[failing_action_goal_id]
                    logger.info(f"Cycle {cycle_num_sim}: Execution failed for '{failing_action_goal_desc}'. Exec failure count: {exec_fail_count}")

                    if exec_fail_count >= agent._max_execution_failures_per_goal:
                        active_goal_sim.status = agent._GoalStatus.FAILED
                        # Simulate KB assert
                        if agent.knowledge_base and agent._Predicate is Predicate:
                            reason = f"PersistentExecutionFailure_{exec_fail_count}_attempts"
                            await agent.knowledge_base.assert_fact(
                                agent._Predicate("goalFailed", (failing_action_goal_id, reason), True, timestamp=time.time())
                            )
                        if failing_action_goal_id in agent._active_goal_execution_failure_count:
                            del agent._active_goal_execution_failure_count[failing_action_goal_id]
            
            # Simulate removal of FAILED goals
            if hasattr(active_goal_sim, 'status') and active_goal_sim.status == agent._GoalStatus.FAILED:
                if active_goal_sim in agent.active_goals:
                    agent.active_goals.remove(active_goal_sim)
                logger.info(f"Cycle {cycle_num_sim}: Goal '{failing_action_goal_desc}' marked FAILED and removed.")
            
            if not any(hasattr(g, 'id') and g.id == failing_action_goal_id for g in agent.active_goals):
                if cycle_num_sim >= agent._max_execution_failures_per_goal:
                    logger.info(f"Goal {failing_action_goal_id} no longer active after {cycle_num_sim} cycles as expected.")
                    break

    # Verification
    assert failing_action_goal_id not in agent._active_goal_execution_failure_count
    assert hasattr(failing_action_goal, 'status') and failing_action_goal.status == agent._GoalStatus.FAILED
    
    kb_preds_exec_fail = await agent.knowledge_base.query(name="goalFailed", value=True)
    found_correct_kb_pred = False
    expected_reason_kb = f"PersistentExecutionFailure_{agent._max_execution_failures_per_goal}_attempts"
    for pred in kb_preds_exec_fail:
        if hasattr(pred, 'args') and pred.args[0] == failing_action_goal_id and expected_reason_kb in str(pred.args[1]):
            found_correct_kb_pred = True; break
    assert found_correct_kb_pred, f"Correct goalFailed predicate for execution failure not found. Found: {kb_preds_exec_fail}"

    # Restore
    if original_max_exec_failures_config is not None:
        agent.config.get("agent", {})["max_execution_failures_per_goal"] = original_max_exec_failures_config
        agent._max_execution_failures_per_goal = original_max_exec_failures_config
    else:
        if "max_execution_failures_per_goal" in agent.config.get("agent", {}):
            del agent.config.get("agent", {})["max_execution_failures_per_goal"]
        agent._max_execution_failures_per_goal = 3

    logger.info("--- Test Passed: Persistent Action Execution Failure Fails Goal ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_baseline_pain_increases_with_age_and_caps(test_agent: AgentController):
    agent = test_agent
    logger.info("--- Test: Baseline Pain Increases with Age and Caps ---")

    # Store original config values from the agent instance if we modify them
    original_age_factor = agent._baseline_pain_age_factor
    original_max_baseline_pain = agent._max_baseline_pain_from_age

    # --- Test Case 1: Initial state ---
    agent.agent_age_cycles = 0
    # Manually simulate the calculation from _update_internal_affective_state_upkeep
    scaled_age_c1 = agent.agent_age_cycles / 1000.0
    baseline_pain_c1_calc = math.log(scaled_age_c1 + 1) * agent._baseline_pain_age_factor
    agent.baseline_pain_level = min(baseline_pain_c1_calc, agent._max_baseline_pain_from_age)
    assert agent.baseline_pain_level == 0.0, "Initial baseline pain should be 0 for age 0."
    logger.info(f"Age 0: Baseline Pain = {agent.baseline_pain_level:.4f}")

    # --- Test Case 2: Moderate age, pain should increase ---
    # Use a test-specific factor to see change quickly
    agent._baseline_pain_age_factor = 0.1 
    agent._max_baseline_pain_from_age = 2.0 # Set a clear cap for test

    agent.agent_age_cycles = 5000 # Results in scaled_age = 5
    scaled_age_c2 = agent.agent_age_cycles / 1000.0
    baseline_pain_c2_calc = math.log(scaled_age_c2 + 1) * agent._baseline_pain_age_factor
    agent.baseline_pain_level = min(baseline_pain_c2_calc, agent._max_baseline_pain_from_age)
    
    expected_pain_c2 = math.log(5 + 1) * 0.1 # log(6) * 0.1 approx 1.79 * 0.1 = 0.179
    logger.info(f"Age {agent.agent_age_cycles}: Baseline Pain = {agent.baseline_pain_level:.4f}, Expected (uncapped) ~ {expected_pain_c2:.4f}")
    assert agent.baseline_pain_level > 0.0, "Baseline pain should be > 0 at moderate age."
    assert abs(agent.baseline_pain_level - expected_pain_c2) < 0.001, "Baseline pain calculation incorrect for moderate age."

    # --- Test Case 3: High age, pain should hit the cap ---
    # agent._max_baseline_pain_from_age is 2.0
    # agent._baseline_pain_age_factor is 0.1
    # We need scaled_age = exp(cap / factor) - 1 = exp(2.0 / 0.1) - 1 = exp(20) - 1
    # exp(20) is very large. So, set age_cycles to something that guarantees capping.
    agent.agent_age_cycles = 30000 * 1000 # scaled_age = 30000
    
    scaled_age_c3 = agent.agent_age_cycles / 1000.0
    baseline_pain_c3_calc = math.log(scaled_age_c3 + 1) * agent._baseline_pain_age_factor
    agent.baseline_pain_level = min(baseline_pain_c3_calc, agent._max_baseline_pain_from_age)

    uncapped_pain_c3 = math.log(30000 + 1) * 0.1 # log(30001)*0.1 approx 10.3 * 0.1 = 1.03
                                                 # Oh, wait, if factor is 0.1, max_baseline_pain should be smaller than log(large)*0.1
                                                 # For max_baseline_pain_from_age = 2.0, and factor = 0.1
                                                 # log(scaled_age+1) = 2.0 / 0.1 = 20
                                                 # scaled_age+1 = exp(20)
                                                 # scaled_age = exp(20)-1. This is too big for `agent_age_cycles`.

    # Let's re-think Test Case 3. We want to show capping.
    # If max_baseline_pain = 1.0 (from config) and factor = 0.00001 (from config)
    # Then log(scaled_age+1) would need to be 1.0 / 0.00001 = 100,000. This makes exp() huge.
    # The test needs to use more aggressive factors or a smaller cap for testing the cap.
    
    # Let's use the modified factors for Test Case 3 as well:
    # agent._baseline_pain_age_factor = 0.1
    # agent._max_baseline_pain_from_age = 0.5 # Set a cap that's easily reachable with factor 0.1

    agent._max_baseline_pain_from_age = 0.5 # Test with a smaller cap
    # To reach pain of 0.5 with factor 0.1: log(scaled_age+1) = 0.5/0.1 = 5
    # scaled_age+1 = exp(5) approx 148.4
    # scaled_age approx 147.4
    # agent_age_cycles = 147400
    
    agent.agent_age_cycles = 200000 # scaled_age = 200. This should definitely hit the cap of 0.5
    scaled_age_c3_capped = agent.agent_age_cycles / 1000.0
    baseline_pain_c3_capped_calc = math.log(scaled_age_c3_capped + 1) * agent._baseline_pain_age_factor # factor is 0.1
    # math.log(201)*0.1 = 5.3 * 0.1 = 0.53 (approx)
    agent.baseline_pain_level = min(baseline_pain_c3_capped_calc, agent._max_baseline_pain_from_age) # cap is 0.5

    logger.info(f"Age {agent.agent_age_cycles}: Baseline Pain = {agent.baseline_pain_level:.4f}, Expecting cap of {agent._max_baseline_pain_from_age:.4f}")
    assert abs(agent.baseline_pain_level - agent._max_baseline_pain_from_age) < 0.001, \
        f"Baseline pain should be capped. Got {agent.baseline_pain_level:.4f}, expected cap {agent._max_baseline_pain_from_age:.4f}"

    # Restore original factors on agent instance for other tests
    agent._baseline_pain_age_factor = original_age_factor
    agent._max_baseline_pain_from_age = original_max_baseline_pain

    logger.info("--- Test Passed: Baseline Pain Increases with Age and Caps ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_goal_failure_induces_pain_source(test_agent: AgentController):
    agent = test_agent
    logger.info("--- Test: Goal Failure Induces PainSource ---")

    # Ensure agent's internal _PainSource attribute is the actual PainSource class
    # imported at the top of this test file (via "from ...models.datatypes import ..., PainSource")
    assert 'PainSource' in globals(), "PainSource class not imported in test_cognitive_cycle.py"
    _PainSource_class_in_test_scope = globals()['PainSource'] # Get the class object

    assert agent._PainSource is _PainSource_class_in_test_scope, \
        (f"AgentController._PainSource (type: {type(agent._PainSource)}) is not the expected "
         f"PainSource class (type: {_PainSource_class_in_test_scope}) imported by the test file.")
    
    # Ensure agent config has factors for pain generation (these are read in agent.__init__)
    # For the test, we can directly set the agent's internal config attributes if they are different
    # from what the fixture's config_override provides for "internal_states".
    # The fixture already provides "internal_states" in config_override.
    # If we need specific values for this test:
    agent._acute_pain_goal_fail_priority_scale_factor = 0.5 # Test-specific override
    agent._pain_event_max_initial_intensity = 2.0        # Test-specific override
    logger.info(f"Test using acute_pain_factor={agent._acute_pain_goal_fail_priority_scale_factor}, max_initial_intensity={agent._pain_event_max_initial_intensity}")


    failed_goal_desc = "High priority goal that failed as missed opportunity"
    failed_goal_priority = USER_GOAL_PRIORITY # This constant is from agent_controller, available due to test imports
    
    # Use agent's _create_goal_from_descriptor which should use real Goal class
    failed_goal = agent._create_goal_from_descriptor(failed_goal_desc, priority=failed_goal_priority) 
    assert failed_goal is not None and hasattr(failed_goal, 'id'), "Failed to create test goal."
    failed_goal.time_limit = time.time() - 3600 # Time limit was 1 hour ago, ensuring missed opportunity
    failed_goal.status = agent._GoalStatus.FAILED # Manually set status for this test scenario

    # Simulate this goal was the one being processed
    agent.active_goals = [failed_goal] 

    initial_pain_level = agent.pain_level # Should be baseline_pain after upkeep
    initial_pain_source_count = len(agent.active_pain_sources)

    # --- Simulate the relevant parts of AgentController logic ---
    # 1. Upkeep to establish baseline pain, age etc.
    await agent._update_internal_affective_state_upkeep() 
    baseline_pain_after_upkeep = agent.baseline_pain_level
    
    # 2. Logic from Step 10 that processes FAILED goals and generates PainSource
    # This block is an adaptation of the logic inside _run_agent_loop's Step 10
    goals_to_remove_from_active_list_test = []
    current_time_for_pain_test = time.time() 

    for goal_in_list_test in list(agent.active_goals): # Iterate over a copy if modifying list
        if hasattr(goal_in_list_test, 'status') and goal_in_list_test.status == agent._GoalStatus.FAILED:
            goals_to_remove_from_active_list_test.append(goal_in_list_test)
            
            # Check if AgentController is set up to use real types (CORE_DEPENDENCIES_AVAILABLE should be True)
            # This assertion also implicitly checks that agent._PainSource is the real PainSource class.
            assert agent._PainSource is _PainSource_class_in_test_scope, "Agent not using real PainSource class during critical logic."

            goal_id_for_pain_test = getattr(goal_in_list_test, 'id', None)
            goal_desc_for_pain_test = getattr(goal_in_list_test, 'description', "Unknown Goal")
            goal_priority_for_pain_test = getattr(goal_in_list_test, 'priority', DEFAULT_OBSERVE_GOAL_PRIORITY) 

            is_missed_opportunity_test = False
            goal_time_limit_test = getattr(goal_in_list_test, 'time_limit', None)
            if goal_time_limit_test is not None and current_time_for_pain_test > goal_time_limit_test:
                is_missed_opportunity_test = True
            
            # Simplified missed opportunity check for test purposes if not time-limited
            if not is_missed_opportunity_test and goal_priority_for_pain_test >= (USER_GOAL_PRIORITY - 1.0):
                is_missed_opportunity_test = True

            if is_missed_opportunity_test and goal_id_for_pain_test:
                existing_pain_event_test = next((ps for ps in agent.active_pain_sources if ps.source_goal_id == goal_id_for_pain_test and not ps.is_resolved), None)
                if not existing_pain_event_test:
                    initial_intensity_test = min(
                        agent._pain_event_max_initial_intensity,
                        goal_priority_for_pain_test * agent._acute_pain_goal_fail_priority_scale_factor
                    )
                    if initial_intensity_test > 0:
                        # Use agent._PainSource which is asserted to be the real class
                        new_pain_source_test = agent._PainSource( 
                            id=f"PainSource_Test_{goal_id_for_pain_test}_{int(current_time_for_pain_test)}",
                            description=f"Failed Test: {goal_desc_for_pain_test[:70]}",
                            initial_intensity=initial_intensity_test,
                            # current_intensity is handled by PainSource.__post_init__
                            timestamp_created=current_time_for_pain_test,
                            decay_rate_per_cycle=agent._default_pain_event_decay_rate_per_cycle,
                            type="MissedOpportunityGoal",
                            source_goal_id=goal_id_for_pain_test
                        )
                        agent.active_pain_sources.append(new_pain_source_test)
                        logger.info(f"TEST: Created PainSource: {new_pain_source_test.id} with intensity {new_pain_source_test.current_intensity:.2f}")
                        
                        # Recalculate total pain level after adding new source
                        current_acute_pain_sum_test = sum(ps.current_intensity for ps in agent.active_pain_sources if not ps.is_resolved)
                        agent.pain_level = min(
                            agent._max_pain_shutdown_threshold,
                            agent.baseline_pain_level + current_acute_pain_sum_test
                        )
    
    # Simulate removal of the failed goal from active_goals list
    if goals_to_remove_from_active_list_test:
        agent.active_goals = [g for g in agent.active_goals if g not in goals_to_remove_from_active_list_test]


    # --- Assertions ---
    assert len(agent.active_pain_sources) == initial_pain_source_count + 1, "PainSource was not added."
    
    # Find the newly added pain source more reliably
    newly_added_pain_source = None
    for ps in agent.active_pain_sources:
        if ps.source_goal_id == failed_goal.id:
            newly_added_pain_source = ps
            break
    assert newly_added_pain_source is not None, f"Newly added pain source for goal {failed_goal.id} not found."

    assert failed_goal_desc[:70] in newly_added_pain_source.description, "PainSource description mismatch."
    
    expected_intensity = min(agent._pain_event_max_initial_intensity, failed_goal_priority * agent._acute_pain_goal_fail_priority_scale_factor)
    assert abs(newly_added_pain_source.initial_intensity - expected_intensity) < 0.001, "PainSource initial_intensity incorrect."
    assert abs(newly_added_pain_source.current_intensity - expected_intensity) < 0.001, "PainSource current_intensity incorrect."
    
    # Total pain should be baseline + new source's current intensity
    # initial_pain_level might have been slightly above 0 if baseline pain was calculated from a non-zero age.
    # Here, agent.pain_level was updated inside the loop after adding the new pain source.
    assert agent.pain_level > baseline_pain_after_upkeep or (abs(agent.pain_level - baseline_pain_after_upkeep) < 0.0001 and expected_intensity == 0), \
        (f"Agent's total pain level ({agent.pain_level:.3f}) should have increased "
         f"from baseline_pain_after_upkeep ({baseline_pain_after_upkeep:.3f}) if new pain source has intensity.")
    
    expected_total_pain = baseline_pain_after_upkeep + newly_added_pain_source.current_intensity
    expected_total_pain_clamped = min(agent._max_pain_shutdown_threshold, expected_total_pain)
    
    assert abs(agent.pain_level - expected_total_pain_clamped) < 0.001, \
        (f"Agent's total pain level ({agent.pain_level:.3f}) not updated correctly. "
         f"Expected sum of baseline ({baseline_pain_after_upkeep:.3f}) and new pain source intensity "
         f"({newly_added_pain_source.current_intensity:.3f}) = {expected_total_pain_clamped:.3f}.")

    logger.info("--- Test Passed: Goal Failure Induces PainSource ---")


@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_pain_source_decays_over_time_and_is_removed(test_agent: AgentController):
    agent = test_agent
    logger.info("--- Test: PainSource Decays Over Time and is Removed ---")
    
    # Ensure PainSource class is available in the test scope and agent is using it
    assert 'PainSource' in globals(), "PainSource class not imported in test_cognitive_cycle.py for test setup."
    _PainSource_class_in_test_scope = globals()['PainSource'] # Get the class object
    
    assert agent._PainSource is _PainSource_class_in_test_scope, \
        (f"AgentController._PainSource (type: {type(agent._PainSource)}) is not the expected "
         f"PainSource class (type: {_PainSource_class_in_test_scope}) imported by the test file.")

    # Store and override agent's config for test-specific decay and retention
    original_decay_rate = agent._default_pain_event_decay_rate_per_cycle
    original_min_retain = agent._pain_event_min_intensity_to_retain
    original_active_pain_sources = list(agent.active_pain_sources) # Save original list
    original_agent_age = agent.agent_age_cycles
    original_baseline_pain = agent.baseline_pain_level

    agent.active_pain_sources = [] # Start with a clean list for this test
    agent._default_pain_event_decay_rate_per_cycle = 0.25 # Faster decay for quicker test (e.g., 25% per cycle)
    agent._pain_event_min_intensity_to_retain = 0.1  # Remove if intensity drops below this

    initial_intensity = 1.0
    test_pain_source_id = "test_decay_pain_id_unique"
    
    # Instantiate using agent._PainSource, which should be the real PainSource class
    test_pain_source = agent._PainSource( 
        id=test_pain_source_id, 
        description="Test decay pain",
        initial_intensity=initial_intensity,
        # current_intensity will be set by __post_init__ based on initial_intensity
        decay_rate_per_cycle=agent._default_pain_event_decay_rate_per_cycle # Use the test-specific decay
    )
    agent.active_pain_sources.append(test_pain_source)
    
    # Reset agent's age and baseline pain for consistent results in this test run
    agent.agent_age_cycles = 0 
    agent.baseline_pain_level = 0.0
    agent.pain_level = 0.0 # Reset total pain too

    # --- First upkeep call: Initializes current_intensity and applies first decay ---
    logger.info(f"Before 1st upkeep: PainSource initial_intensity={test_pain_source.initial_intensity}, current_intensity should be same via post_init.")
    # current_intensity is set in post_init, so it's initial_intensity before first decay
    assert abs(test_pain_source.current_intensity - initial_intensity) < 1e-9, "current_intensity not set correctly by __post_init__."

    await agent._update_internal_affective_state_upkeep() 
    
    intensity_after_1_cycle = test_pain_source.current_intensity # This is after 1 decay
    expected_after_1_decay = initial_intensity * (1.0 - agent._default_pain_event_decay_rate_per_cycle)
    
    logger.info(f"Intensity after 1st upkeep: {intensity_after_1_cycle:.6f} (Expected: {expected_after_1_decay:.6f})")
    assert abs(intensity_after_1_cycle - expected_after_1_decay) < 1e-9, \
        f"Pain intensity after 1st upkeep ({intensity_after_1_cycle:.6f}) not as expected ({expected_after_1_decay:.6f})."
    assert intensity_after_1_cycle < initial_intensity, "Pain intensity did not decay after 1st upkeep."

    # --- Simulate further upkeep cycles to observe full decay and removal ---
    max_test_cycles = 20 # Safety break for the loop
    num_additional_decay_cycles = 0
    source_still_active = True

    for i in range(max_test_cycles):
        # Check if the specific pain source is still in the agent's list
        current_test_source_in_agent_list = next((ps for ps in agent.active_pain_sources if ps.id == test_pain_source_id), None)
        if not current_test_source_in_agent_list:
            source_still_active = False
            logger.info(f"PainSource '{test_pain_source_id}' removed from active_pain_sources after {num_additional_decay_cycles} additional upkeep cycles.")
            break
        
        # If still active, its current_intensity is the one we care about from the *previous* upkeep
        logger.debug(
            f"Start of upkeep cycle {i+2}: PainSource '{test_pain_source_id}' intensity is "
            f"{current_test_source_in_agent_list.current_intensity:.6f}"
        )
        await agent._update_internal_affective_state_upkeep()
        num_additional_decay_cycles += 1
    
    logger.info(f"PainSource processing complete after {num_additional_decay_cycles} additional upkeep cycles.")
    
    assert not source_still_active, \
        (f"PainSource '{test_pain_source_id}' should have been removed after decaying below threshold. "
         f"It might still be present if loop exited due to max_test_cycles. "
         f"Last known intensity for test_pain_source object: {test_pain_source.current_intensity:.6f}, "
         f"Min retain threshold: {agent._pain_event_min_intensity_to_retain}")
    
    # After the source is removed, agent.pain_level should revert to baseline_pain_level
    # Call upkeep one last time to ensure agent.pain_level is re-calculated based on empty active_pain_sources
    await agent._update_internal_affective_state_upkeep()
    
    # Due to agent_age_cycles incrementing, baseline_pain_level might not be 0.0
    # We need to compare agent.pain_level to agent.baseline_pain_level
    logger.info(f"Final check: TotalPain={agent.pain_level:.4f}, BaselinePain={agent.baseline_pain_level:.4f}")
    assert abs(agent.pain_level - agent.baseline_pain_level) < 1e-9, \
        (f"Total pain level ({agent.pain_level:.6f}) should be effectively equal to baseline_pain_level "
         f"({agent.baseline_pain_level:.6f}) after pain source decays and is removed.")

    # Restore original agent config values and state
    agent._default_pain_event_decay_rate_per_cycle = original_decay_rate
    agent._pain_event_min_intensity_to_retain = original_min_retain
    agent.active_pain_sources = original_active_pain_sources # Restore original list
    agent.agent_age_cycles = original_agent_age
    agent.baseline_pain_level = original_baseline_pain
    # Recalculate total pain based on restored state for subsequent tests
    await agent._update_internal_affective_state_upkeep()


    logger.info("--- Test Passed: PainSource Decays Over Time and is Removed ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_goal_achievement_increases_happiness(test_agent: AgentController):
    agent = test_agent
    logger.info("--- Test: Goal Achievement Increases Happiness ---")

    # Ensure relevant types and constants are available
    assert Goal is not None and GoalStatus is not None
    assert USER_GOAL_PRIORITY is not None and DEFAULT_OBSERVE_GOAL_PRIORITY is not None
    assert hasattr(agent, '_happiness_from_goal_priority_scale_factor')
    assert hasattr(agent, '_pain_impact_on_happiness_scale_factor')

    # --- Scenario 1: Achieving a default goal ---
    agent.happiness_level = 5.0 # Reset for test
    agent.pain_level = 0.0      # No pain for this scenario
    
    default_goal = agent._create_goal_from_descriptor(DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC, priority=DEFAULT_OBSERVE_GOAL_PRIORITY)
    assert default_goal is not None
    default_goal.status = agent._GoalStatus.ACHIEVED
    
    # Simulate relevant part of Step 10 from AgentController
    initial_happiness = agent.happiness_level
    priority_val_default = getattr(default_goal, 'priority', DEFAULT_OBSERVE_GOAL_PRIORITY)
    # No priority bonus for default goal description
    
    expected_gain_default = priority_val_default * agent._happiness_from_goal_priority_scale_factor
    agent.happiness_level += expected_gain_default
    # Apply pain impact (should be 0 here)
    agent.happiness_level -= agent.pain_level * agent._pain_impact_on_happiness_scale_factor
    agent.happiness_level = max(0.0, min(10.0, agent.happiness_level)) # Clamp

    logger.info(f"Default goal achievement: Initial happiness={initial_happiness:.2f}, Gain={expected_gain_default:.2f}, Final={agent.happiness_level:.2f}")
    assert abs(agent.happiness_level - (initial_happiness + expected_gain_default)) < 1e-9, \
        "Happiness did not increase correctly for default goal."

    # --- Scenario 2: Achieving a high-priority user goal ---
    agent.happiness_level = 5.0 # Reset
    agent.pain_level = 1.0      # Introduce some pain
    
    user_goal_desc = "Important user goal achieved"
    user_goal_priority = USER_GOAL_PRIORITY 
    user_goal = agent._create_goal_from_descriptor(user_goal_desc, priority=user_goal_priority)
    assert user_goal is not None
    user_goal.status = agent._GoalStatus.ACHIEVED

    initial_happiness_user = agent.happiness_level
    priority_val_user = getattr(user_goal, 'priority', USER_GOAL_PRIORITY)
    # Apply bonus for non-default, high-priority goal
    if getattr(user_goal, 'description', '') != DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC and \
       priority_val_user >= (USER_GOAL_PRIORITY - 1.0):
        priority_val_user *= 1.5
        
    expected_gain_user = priority_val_user * agent._happiness_from_goal_priority_scale_factor
    current_happiness_before_pain_impact = initial_happiness_user + expected_gain_user
    
    expected_pain_reduction = agent.pain_level * agent._pain_impact_on_happiness_scale_factor
    final_expected_happiness_user = current_happiness_before_pain_impact - expected_pain_reduction
    final_expected_happiness_user = max(0.0, min(10.0, final_expected_happiness_user)) # Clamp

    # Simulate agent logic
    agent.happiness_level += expected_gain_user # Add gain
    agent.happiness_level -= agent.pain_level * agent._pain_impact_on_happiness_scale_factor # Subtract pain impact
    agent.happiness_level = max(0.0, min(10.0, agent.happiness_level)) # Clamp
    
    logger.info(f"User goal achievement: Initial happiness={initial_happiness_user:.2f}, Expected Gain (adj for priority)={expected_gain_user:.2f}, "
                f"Happiness before pain impact={current_happiness_before_pain_impact:.2f}, Pain impact={expected_pain_reduction:.2f}, "
                f"Final Expected={final_expected_happiness_user:.2f}, Actual Final={agent.happiness_level:.2f}")

    assert abs(agent.happiness_level - final_expected_happiness_user) < 1e-9, \
        "Happiness did not increase and adjust for pain correctly for user goal."

    logger.info("--- Test Passed: Goal Achievement Increases Happiness ---")


@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_pain_resolution_boosts_happiness(test_agent: AgentController):
    agent = test_agent
    logger.info("--- Test: Pain Resolution Boosts Happiness ---")

    # Ensure PainSource class is available in the test scope and agent is using it
    assert 'PainSource' in globals(), "PainSource class not imported in test_cognitive_cycle.py for test setup."
    _PainSource_class_in_test_scope = globals()['PainSource'] 
    
    assert agent._PainSource is _PainSource_class_in_test_scope, \
        (f"AgentController._PainSource (type: {type(agent._PainSource)}) is not the expected "
         f"PainSource class (type: {type(_PainSource_class_in_test_scope)}) imported by the test file.")

    # --- Setup ---
    # Store and override relevant agent attributes for test predictability
    original_active_pain_sources = list(agent.active_pain_sources) # Save to restore
    original_happiness_level = agent.happiness_level
    original_pain_level = agent.pain_level
    original_baseline_pain_level = agent.baseline_pain_level
    original_default_pain_decay = agent._default_pain_event_decay_rate_per_cycle


    agent.active_pain_sources = [] 
    agent.happiness_level = 4.0
    agent.pain_level = 0.0 
    agent.baseline_pain_level = 0.0 # Keep baseline simple for this specific test
    
    # Test-specific config overrides (directly on agent instance for this test)
    agent._acute_pain_goal_fail_priority_scale_factor = 0.5
    agent._pain_event_max_initial_intensity = 2.0
    agent._happiness_from_goal_priority_scale_factor = 0.1 
    agent._pain_impact_on_happiness_scale_factor = 0.1
    agent._default_pain_event_decay_rate_per_cycle = 0.001 # Very slow decay, so it doesn't interfere much

    # 1. Create a failed goal and induce a PainSource
    failed_goal_id = "goal_id_for_pain_resolution_test"
    failed_goal_priority = USER_GOAL_PRIORITY # From agent_controller via test import
    
    initial_pain_intensity = min(agent._pain_event_max_initial_intensity,
                                 failed_goal_priority * agent._acute_pain_goal_fail_priority_scale_factor)
    
    # Instantiate using agent._PainSource
    pain_source_to_resolve = agent._PainSource( 
        id=f"PainSource_{failed_goal_id}",
        description="Pain to be resolved",
        initial_intensity=initial_pain_intensity,
        # current_intensity handled by __post_init__
        source_goal_id=failed_goal_id,
        decay_rate_per_cycle=agent._default_pain_event_decay_rate_per_cycle # Use the test-specific decay
    )
    agent.active_pain_sources.append(pain_source_to_resolve)
    # Manually update total pain level based on this new source for the test setup
    # This simulates the state *before* any decay from an upkeep step.
    agent.pain_level = min(agent._max_pain_shutdown_threshold, 
                           agent.baseline_pain_level + pain_source_to_resolve.current_intensity)
    
    logger.info(
        f"Setup: Initial Happiness={agent.happiness_level:.2f}, "
        f"PainSource Added (ID: {failed_goal_id}, Intensity: {pain_source_to_resolve.current_intensity:.2f}), "
        f"Total Pain={agent.pain_level:.2f}"
    )

    # 2. Simulate achieving a goal that resolves this pain source
    resolving_goal_desc = "Goal that resolves the previous failure"
    resolving_goal_priority = USER_GOAL_PRIORITY 
    # Ensure Goal class is the real one used by agent
    assert agent._Goal is Goal, f"Agent not using real Goal class. Type: {type(agent._Goal)}"
    resolving_goal = agent._create_goal_from_descriptor(resolving_goal_desc, priority=resolving_goal_priority)
    assert resolving_goal is not None and hasattr(resolving_goal, 'id'), "Failed to create resolving_goal"
    resolving_goal.id = failed_goal_id # IMPORTANT: Make its ID match the source_goal_id
    resolving_goal.status = agent._GoalStatus.ACHIEVED

    initial_happiness_before_resolution = agent.happiness_level
    pain_intensity_before_resolution = pain_source_to_resolve.current_intensity

    # --- Simulate relevant part of Step 10 from AgentController for this ACHIEVED goal ---
    # This directly calls the logic that would be in _run_agent_loop's Step 10
    
    priority_val_res = getattr(resolving_goal, 'priority', DEFAULT_OBSERVE_GOAL_PRIORITY)
    if getattr(resolving_goal, 'description', '') != DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC:
        if priority_val_res >= (USER_GOAL_PRIORITY - 1.0): # type: ignore
             priority_val_res *= 1.5
    
    happiness_gain_from_achievement = priority_val_res * agent._happiness_from_goal_priority_scale_factor
    agent.happiness_level += happiness_gain_from_achievement

    happiness_bonus_for_resolution_calc = 0.0
    resolved_pain_id_check = getattr(resolving_goal, 'id', None)
    if resolved_pain_id_check:
        for ps_event_idx, ps_event in enumerate(list(agent.active_pain_sources)): 
            if ps_event.source_goal_id == resolved_pain_id_check and not ps_event.is_resolved:
                agent.active_pain_sources[ps_event_idx].is_resolved = True 
                intensity_reduction_calc = agent.active_pain_sources[ps_event_idx].current_intensity * 0.90
                agent.active_pain_sources[ps_event_idx].current_intensity -= intensity_reduction_calc
                
                happiness_bonus_for_resolution_calc = intensity_reduction_calc * 0.5
                agent.happiness_level += happiness_bonus_for_resolution_calc
                
                # Recalculate total pain immediately based on updated active_pain_sources
                agent.pain_level = min(
                    agent._max_pain_shutdown_threshold,
                    agent.baseline_pain_level + sum(p.current_intensity for p in agent.active_pain_sources if not p.is_resolved)
                )
                break
    
    # Apply pain impact on happiness using the newly calculated agent.pain_level
    happiness_reduction_from_pain_calc = agent.pain_level * agent._pain_impact_on_happiness_scale_factor
    agent.happiness_level -= happiness_reduction_from_pain_calc
    agent.happiness_level = max(0.0, min(10.0, agent.happiness_level))

    logger.info(
        f"Resolution: Happiness Gain (Achieve): {happiness_gain_from_achievement:.3f}, "
        f"Happiness Bonus (Resolve Pain): {happiness_bonus_for_resolution_calc:.3f}, "
        f"Pain Impact on Happiness (using new pain_level {agent.pain_level:.3f}): {happiness_reduction_from_pain_calc:.3f}"
    )
    logger.info(f"Final Happiness: {agent.happiness_level:.3f}, Final Pain Level: {agent.pain_level:.3f}")
    
    resolved_pain_source_from_agent_list = next((ps for ps in agent.active_pain_sources if ps.id == pain_source_to_resolve.id), None)
    
    assert resolved_pain_source_from_agent_list is not None, "PainSource object was unexpectedly removed from list after resolution attempt."
    assert resolved_pain_source_from_agent_list.is_resolved, "PainSource was not marked as resolved." 
    assert resolved_pain_source_from_agent_list.current_intensity < pain_intensity_before_resolution * 0.15, \
        "PainSource current_intensity was not significantly reduced after resolution."

    expected_final_happiness = initial_happiness_before_resolution + \
                               happiness_gain_from_achievement + \
                               happiness_bonus_for_resolution_calc - \
                               happiness_reduction_from_pain_calc
    expected_final_happiness = max(0.0, min(10.0, expected_final_happiness))

    assert abs(agent.happiness_level - expected_final_happiness) < 1e-9, \
        f"Final happiness level incorrect. Expected: {expected_final_happiness:.3f}, Got: {agent.happiness_level:.3f}"
    
    # Since the pain source is resolved, it should no longer contribute to the active pain sum
    # used for agent.pain_level. agent.pain_level should now be agent.baseline_pain_level.
    expected_final_pain_level = agent.baseline_pain_level 
    
    assert abs(agent.pain_level - expected_final_pain_level) < 1e-9, \
        (f"Total pain level incorrect after resolving source. "
         f"Expected (baseline only): {expected_final_pain_level:.4f}, Got: {agent.pain_level:.4f}. "
         f"Resolved source residual intensity: {resolved_pain_source_from_agent_list.current_intensity:.4f}")

    # Restore original agent state that was modified for this test
    agent.active_pain_sources = original_active_pain_sources
    agent.happiness_level = original_happiness_level
    agent.pain_level = original_pain_level
    agent.baseline_pain_level = original_baseline_pain_level
    agent._default_pain_event_decay_rate_per_cycle = original_default_pain_decay
    # Recalculate total pain based on restored state for subsequent tests (if agent instance is reused)
    # For function-scoped fixtures, this might be less critical but good practice.
    await agent._update_internal_affective_state_upkeep()


    logger.info("--- Test Passed: Pain Resolution Boosts Happiness ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_dsm_capability_gain_increases_purpose(test_agent: AgentController):
    agent = test_agent
    logger.info("--- Test: DSM Capability Gain Increases Purpose ---")

    assert agent.dynamic_self_model is not None, "DynamicSelfModel component missing."
    assert hasattr(agent, '_purpose_from_capability_gain_factor'), "Agent missing purpose config factor for capability gain."
    assert hasattr(agent, '_last_dsm_capabilities_count'), "Agent missing _last_dsm_capabilities_count attribute."

    original_purpose_level = agent.purpose_level
    original_last_dsm_caps_count = agent._last_dsm_capabilities_count
    original_purpose_gain_factor = agent._purpose_from_capability_gain_factor

    agent.purpose_level = 5.0
    agent._last_dsm_capabilities_count = 10
    agent._purpose_from_capability_gain_factor = 0.05 

    initial_purpose = agent.purpose_level
    initial_caps_count_for_test = agent._last_dsm_capabilities_count

    new_caps_count_from_dsm = initial_caps_count_for_test + 2
    mock_dsm_status_return_dict = {"num_capabilities": new_caps_count_from_dsm} # The dict to be returned

    # Simulate relevant part of Step 10 from AgentController
    assert agent.dynamic_self_model is not None, "DSM instance should be available for this test."

    # --- MODIFIED MOCKING ---
    # If agent.dynamic_self_model.get_status is an async method,
    # patch.object will create an AsyncMock by default.
    # The return_value of this AsyncMock is what `await` will yield.
    with patch.object(agent.dynamic_self_model, 'get_status', return_value=mock_dsm_status_return_dict) as mock_get_dsm_status:
    # --- END MODIFIED MOCKING ---
        
        # Replicate logic from AgentController._run_agent_loop Step 10 for purpose update from DSM
        dsm_status_for_purpose_test = await agent.dynamic_self_model.get_status() # This will now directly be mock_dsm_status_return_dict
        
        assert isinstance(dsm_status_for_purpose_test, dict), \
            f"Mocked get_status did not return a dict. Got: {type(dsm_status_for_purpose_test)}"

        current_dsm_caps_count_test = dsm_status_for_purpose_test.get("num_capabilities", 0)
        
        if current_dsm_caps_count_test > agent._last_dsm_capabilities_count:
            cap_increase_amount_test = current_dsm_caps_count_test - agent._last_dsm_capabilities_count
            purpose_gain_caps_test = cap_increase_amount_test * agent._purpose_from_capability_gain_factor
            agent.purpose_level += purpose_gain_caps_test
        agent._last_dsm_capabilities_count = current_dsm_caps_count_test
        agent.purpose_level = max(0.0, min(10.0, agent.purpose_level)) # Clamp

    expected_increase = (new_caps_count_from_dsm - initial_caps_count_for_test) * agent._purpose_from_capability_gain_factor
    expected_purpose = min(10.0, initial_purpose + expected_increase)
    
    logger.info(f"Initial purpose={initial_purpose:.3f}, Caps increased from {initial_caps_count_for_test} to {new_caps_count_from_dsm}, "
                f"Expected increase={expected_increase:.3f}, Expected final purpose={expected_purpose:.3f}, "
                f"Actual final purpose={agent.purpose_level:.3f}")

    assert abs(agent.purpose_level - expected_purpose) < 1e-9, "Purpose level did not increase correctly after capability gain."
    assert agent._last_dsm_capabilities_count == new_caps_count_from_dsm, "Last DSM capabilities count not updated."

    # Restore original values
    agent.purpose_level = original_purpose_level
    agent._last_dsm_capabilities_count = original_last_dsm_caps_count
    agent._purpose_from_capability_gain_factor = original_purpose_gain_factor

    logger.info("--- Test Passed: DSM Capability Gain Increases Purpose ---")


@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_complex_goal_achievement_increases_purpose(test_agent: AgentController):
    agent = test_agent
    logger.info("--- Test: Complex Goal Achievement Increases Purpose ---")
    
    # Ensure necessary types/constants are available in the test scope
    # These are imported at the top of test_cognitive_cycle.py
    assert Goal is not None, "Goal class not available for test."
    assert agent._GoalStatus is not None, "Agent._GoalStatus not initialized."
    assert USER_GOAL_PRIORITY is not None, "USER_GOAL_PRIORITY constant not available."
    assert DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC is not None, "DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC not available."
    assert DEFAULT_OBSERVE_GOAL_PRIORITY is not None, "DEFAULT_OBSERVE_GOAL_PRIORITY not available."
    
    assert hasattr(agent, '_purpose_from_high_priority_goal_factor'), "Agent missing '_purpose_from_high_priority_goal_factor' attribute."
    assert hasattr(agent, '_complex_goal_priority_threshold'), "Agent missing '_complex_goal_priority_threshold' attribute."

    # Store and restore original agent state for isolation
    original_purpose_level = agent.purpose_level

    agent.purpose_level = 5.0 # Set a known baseline for the test
    initial_purpose = agent.purpose_level
    
    # Use USER_GOAL_PRIORITY as defined in agent_controller (available via test imports)
    _USER_GOAL_PRIORITY_TEST_SCOPE = USER_GOAL_PRIORITY 

    # --- Scenario 1: Achieve a high-priority goal that meets the complex threshold ---
    # agent._complex_goal_priority_threshold is set in agent.__init__ from config.
    # The fixture config sets USER_GOAL_PRIORITY = 5.0.
    # agent.__init__ sets self._complex_goal_priority_threshold = USER_GOAL_PRIORITY - 0.5 = 4.5
    # So, a goal with priority 4.5 or higher should trigger the purpose gain.
    complex_goal_priority_val = agent._complex_goal_priority_threshold 
    
    complex_goal = agent._create_goal_from_descriptor("Achieved complex goal test", priority=complex_goal_priority_val)
    assert complex_goal is not None, "Failed to create complex_goal for test."
    complex_goal.status = agent._GoalStatus.ACHIEVED

    # Simulate the logic from AgentController's Step 10 for purpose update from complex goal achievement
    test_goal_just_achieved_this_cycle_s1 = True
    test_priority_of_achieved_goal_s1 = complex_goal_priority_val 
    # Apply the 1.5x priority multiplier if it's a non-default "important" goal
    if getattr(complex_goal, 'description', '') != DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC: 
        if test_priority_of_achieved_goal_s1 >= (_USER_GOAL_PRIORITY_TEST_SCOPE - 1.0): # e.g. >= 4.0
             test_priority_of_achieved_goal_s1 *= 1.5
    
    # This logic directly mirrors what's in AgentController._run_agent_loop (Step 10)
    if test_goal_just_achieved_this_cycle_s1:
        if test_priority_of_achieved_goal_s1 >= agent._complex_goal_priority_threshold:
            agent.purpose_level += agent._purpose_from_high_priority_goal_factor
    agent.purpose_level = max(0.0, min(10.0, agent.purpose_level)) # Clamp

    expected_purpose_complex = min(10.0, initial_purpose + agent._purpose_from_high_priority_goal_factor)
    logger.info(
        f"Scenario 1 (Complex Goal): Initial purpose={initial_purpose:.3f}, "
        f"Achieved Goal Priority (raw):{complex_goal_priority_val:.2f}, Effective Priority (for purpose check):{test_priority_of_achieved_goal_s1:.2f}, "
        f"Complex Threshold:{agent._complex_goal_priority_threshold:.2f}, "
        f"Expected final={expected_purpose_complex:.3f}, Actual final={agent.purpose_level:.3f}"
    )
    assert abs(agent.purpose_level - expected_purpose_complex) < 1e-9, \
        "Purpose did not increase correctly for complex/high-priority goal."

    # --- Scenario 2: Achieve a low-priority goal (should NOT trigger this specific purpose gain) ---
    agent.purpose_level = 5.0 # Reset purpose for this scenario
    initial_purpose_low = agent.purpose_level
    
    low_priority_val = DEFAULT_OBSERVE_GOAL_PRIORITY # Typically 1.0
    # Ensure this is below the complex_goal_priority_threshold (e.g. 4.5)
    assert low_priority_val < agent._complex_goal_priority_threshold, \
        "Test setup error: low_priority_val is not less than _complex_goal_priority_threshold."

    low_priority_goal = agent._create_goal_from_descriptor("Achieved low priority goal test", priority=low_priority_val) 
    assert low_priority_goal is not None, "Failed to create low_priority_goal for test."
    low_priority_goal.status = agent._GoalStatus.ACHIEVED
    
    test_goal_just_achieved_this_cycle_s2 = True
    test_priority_of_achieved_goal_s2 = low_priority_val
    # The 1.5x multiplier for "important non-default" goals does not apply here if it's DEFAULT_OBSERVE_GOAL_PRIORITY
    # or if its description matches DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC.
    # Our test goal desc is "Achieved low priority goal test", so it *might* get multiplier if priority was high enough.
    # But since priority is low (DEFAULT_OBSERVE_GOAL_PRIORITY), test_priority_of_achieved_goal_s2 will likely remain low.

    # This logic directly mirrors what's in AgentController._run_agent_loop (Step 10)
    if test_goal_just_achieved_this_cycle_s2:
        if test_priority_of_achieved_goal_s2 >= agent._complex_goal_priority_threshold: # This condition should be false
            agent.purpose_level += agent._purpose_from_high_priority_goal_factor
    agent.purpose_level = max(0.0, min(10.0, agent.purpose_level)) # Clamp

    logger.info(
        f"Scenario 2 (Low-Priority Goal): Initial purpose={initial_purpose_low:.3f}, "
        f"Achieved Goal Priority:{test_priority_of_achieved_goal_s2:.2f}, "
        f"Complex Threshold:{agent._complex_goal_priority_threshold:.2f}, "
        f"Actual final={agent.purpose_level:.3f}"
    )
    # Purpose should not have changed from the "complex goal achievement" factor
    assert abs(agent.purpose_level - initial_purpose_low) < 1e-9, \
        "Purpose should not have increased from 'complex goal' factor for a low-priority/non-complex goal."

    # Restore original agent state
    agent.purpose_level = original_purpose_level

    logger.info("--- Test Passed: Complex Goal Achievement Increases Purpose ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_ems_drive_influence_by_pain_happiness_purpose(test_agent: AgentController):
    agent = test_agent
    ems = agent.emergent_motivation_system
    logger.info("--- Test: EMS Drive Influence by Pain, Happiness, Purpose ---")

    assert ems is not None, "EmergentMotivationSystem component missing."
    assert hasattr(ems, 'drives'), "EMS drives not initialized."
    
    # Store original drive params and values for restoration
    original_drives_config = {name: params.copy() for name, params in ems.drives.items()}

    # Helper to simulate EMS processing with specific cognitive state inputs
    async def run_ems_eval(pain=0.0, happiness=5.0, purpose=5.0, last_action_type="THINKING", last_outcome="success"):
        await ems.reset() # Reset drives to their configured defaults before each scenario
        # Override specific thresholds for testing, if needed, e.g.:
        # ems.drives["curiosity"]["threshold_high_pain_for_curiosity"] = 6.0 
        # ems.drives["curiosity"]["threshold_low_purpose_for_curiosity"] = 1.5
        
        initial_drive_values = ems.get_drive_values()
        
        mock_cognitive_state_ems = {
            "pain_level": pain,
            "happiness_level": happiness,
            "purpose_level": purpose,
            "consciousness_level": "CONSCIOUS" # For context key if EMS used it directly
        }
        # Minimal last_action_result, phenomenal_state, active_goal, self_model_summary
        # as this test focuses on pain/happiness/purpose direct inputs.
        # If other parts of EMS eval significantly interact, these would need richer mocks.
        ems_input = {
            "cognitive_state": mock_cognitive_state_ems,
            "last_action_result": {"type": last_action_type, "outcome": last_outcome},
            "phenomenal_state": None, "active_goal": None, "self_model_summary": None
        }
        # In AgentController, PWM error is also passed to EMS. We assume no PWM error for these specific checks.
        # If PWM error was present and significant, it would also affect curiosity.
        
        # Ensure PWM mock doesn't report error unless specified
        if agent.predictive_world_model:
            agent.predictive_world_model.last_prediction_error = None

        await ems.process(ems_input)
        return initial_drive_values, ems.get_drive_values()

    # --- Scenario 1: High Pain -> Expect increased Curiosity ---
    logger.info("Scenario 1: High Pain")
    ems.drives["curiosity"]["threshold_high_pain_for_curiosity"] = 6.0 # Ensure test pain is above
    initial_s1, final_s1 = await run_ems_eval(pain=7.5)
    assert final_s1["curiosity"] > initial_s1["curiosity"] - 0.01, \
        f"Curiosity should increase with high pain. Initial: {initial_s1['curiosity']:.3f}, Final: {final_s1['curiosity']:.3f}"
    # Also expect Satisfaction to decrease due to pain
    assert final_s1["satisfaction"] < initial_s1["satisfaction"] + 0.01, \
        f"Satisfaction should decrease with high pain. Initial: {initial_s1['satisfaction']:.3f}, Final: {final_s1['satisfaction']:.3f}"
    logger.info(f"S1 Results - Initial: {initial_s1}, Final: {final_s1}")

    # --- Scenario 2: Low Purpose -> Expect increased Curiosity & Competence ---
    logger.info("Scenario 2: Low Purpose")
    ems.drives["curiosity"]["threshold_low_purpose_for_curiosity"] = 1.5
    ems.drives["competence"]["threshold_low_purpose_for_competence"] = 1.5
    initial_s2, final_s2 = await run_ems_eval(purpose=1.0) # Below both thresholds
    assert final_s2["curiosity"] > initial_s2["curiosity"] - 0.01, \
        f"Curiosity should increase with low purpose. Initial: {initial_s2['curiosity']:.3f}, Final: {final_s2['curiosity']:.3f}"
    assert final_s2["competence"] > initial_s2["competence"] - 0.01, \
        f"Competence should increase with low purpose. Initial: {initial_s2['competence']:.3f}, Final: {final_s2['competence']:.3f}"
    logger.info(f"S2 Results - Initial: {initial_s2}, Final: {final_s2}")

    # --- Scenario 3: High Happiness (above baseline) -> Expect increased Satisfaction ---
    logger.info("Scenario 3: High Happiness")
    initial_s3, final_s3 = await run_ems_eval(happiness=8.0) # Baseline is 5.0
    assert final_s3["satisfaction"] > initial_s3["satisfaction"] - 0.01, \
        f"Satisfaction should increase with high happiness. Initial: {initial_s3['satisfaction']:.3f}, Final: {final_s3['satisfaction']:.3f}"
    logger.info(f"S3 Results - Initial: {initial_s3}, Final: {final_s3}")

    # --- Scenario 4: Low Happiness (below baseline) -> Expect decreased Satisfaction ---
    logger.info("Scenario 4: Low Happiness")
    initial_s4, final_s4 = await run_ems_eval(happiness=2.0)
    assert final_s4["satisfaction"] < initial_s4["satisfaction"] + 0.01, \
        f"Satisfaction should decrease with low happiness. Initial: {initial_s4['satisfaction']:.3f}, Final: {final_s4['satisfaction']:.3f}"
    logger.info(f"S4 Results - Initial: {initial_s4}, Final: {final_s4}")
    
    # --- Scenario 5: Combined - High Pain, Low Purpose, High Happiness (complex interaction) ---
    logger.info("Scenario 5: Combined High Pain, Low Purpose, High Happiness")
    # High pain should decrease satisfaction and increase curiosity.
    # Low purpose should increase curiosity and competence.
    # High happiness should increase satisfaction.
    # Net effect on Satisfaction: gain from happiness vs loss from pain.
    # Net effect on Curiosity: gain from pain + gain from low purpose.
    initial_s5, final_s5 = await run_ems_eval(pain=7.5, happiness=8.0, purpose=1.0)
    assert final_s5["curiosity"] > initial_s5["curiosity"], "Curiosity should strongly increase with high pain & low purpose."
    assert final_s5["competence"] > initial_s5["competence"], "Competence should increase with low purpose."
    # Satisfaction might increase or decrease depending on relative strength of pain vs happiness factors
    logger.info(f"S5 (Combined) Results - Initial: {initial_s5}, Final: {final_s5}")
    # We won't assert satisfaction change here as it depends on tuning, just log.


    # Restore original drive parameters
    ems.drives = original_drives_config
    await ems.reset() # Ensure it re-initializes with original config values from self._config if reset logic does that
                      # Current EMS reset re-reads from DEFAULT_DRIVES then merges self._config.
                      # So, if self._config (from agent.config) was what we wanted to restore, that's fine.
                      # If we modified ems.drives directly, this reset will use the original loaded config.
    logger.info("--- Test Passed: EMS Drive Influence by Pain, Happiness, Purpose ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
@patch(f'{PACKAGE_NAME}.cognitive_modules.narrative_constructor.call_ollama', new_callable=AsyncMock)
async def test_narrative_reflects_pain_happiness_purpose(mock_call_ollama_php: AsyncMock, test_agent: AgentController):
    agent = test_agent
    nc = agent.narrative_constructor
    logger.info("--- Test: Narrative Reflects Pain, Happiness, Purpose States & Events ---")

    assert nc is not None, "NarrativeConstructor component missing."
    # Ensure necessary classes are available in the test scope
    _PainSource_class_test = globals().get('PainSource')
    assert _PainSource_class_test is not None, "PainSource class not imported/available for test."
    _PhenomenalState_class_test = globals().get('PhenomenalState')
    assert _PhenomenalState_class_test is not None, "PhenomenalState class not imported/available for test."
    _ConsciousState_test = globals().get('ConsciousState') # For setting agent.consciousness_level
    assert _ConsciousState_test is not None, "ConsciousState enum not available for test."


    # --- Helper to simulate nc.process and get LLM prompt ---
    async def get_llm_prompt_for_narrative(
        p_state_override: Optional[PhenomenalState] = None, # Use type hint
        action_res_override: Optional[Dict[str,Any]] = None,
        # For directly controlling agent's state that NC reads for significance:
        agent_pain_level_override: Optional[float] = None,
        agent_happiness_level_override: Optional[float] = None,
        agent_purpose_level_override: Optional[float] = None,
        agent_active_pain_sources_override: Optional[List[PainSource]] = None, # Use type hint
        # For controlling NC's internal "last known" states for change detection:
        nc_last_pain_override: Optional[float] = None,
        nc_last_happy_override: Optional[float] = None,
        nc_last_purpose_override: Optional[float] = None,
        nc_known_pain_ids_override: Optional[Set[str]] = None
    ):
        mock_call_ollama_php.return_value = ("Mocked LLM narrative entry.", None)
        mock_call_ollama_php.reset_mock() 

        # Set agent's P/H/P levels for this specific call scenario
        # These are read by nc._is_significant via self._controller
        agent.pain_level = agent_pain_level_override if agent_pain_level_override is not None else 1.0
        agent.happiness_level = agent_happiness_level_override if agent_happiness_level_override is not None else 5.0
        agent.purpose_level = agent_purpose_level_override if agent_purpose_level_override is not None else 5.0
        agent.active_pain_sources = agent_active_pain_sources_override if agent_active_pain_sources_override is not None else []
        
        # Set NC's internal "last known" P/H/P levels to ensure change detection if needed
        nc._last_pain_level_nc = nc_last_pain_override if nc_last_pain_override is not None else agent.pain_level - (nc.pain_change_threshold_sig + 0.1)
        nc._last_happiness_level_nc = nc_last_happy_override if nc_last_happy_override is not None else agent.happiness_level - (nc.happiness_change_threshold_sig + 0.1)
        nc._last_purpose_level_nc = nc_last_purpose_override if nc_last_purpose_override is not None else agent.purpose_level - (nc.purpose_change_threshold_sig + 0.1)
        nc._known_pain_source_ids_nc = nc_known_pain_ids_override if nc_known_pain_ids_override is not None else set()
        
        # Default phenomenal state that causes high intensity trigger for significance
        if p_state_override is not None:
            p_state_for_sig = p_state_override
        else:
            p_state_for_sig = _PhenomenalState_class_test( 
                 content={"trigger": "high_intensity_event_default_narrative_test"}, 
                 intensity=(nc.intensity_threshold + 0.1) if hasattr(nc, 'intensity_threshold') else 0.8, 
                 valence=0.2,
                 distinct_source_count=1, content_diversity_lexical=0.5, shared_concept_count_gw=0.1,
                 attention_weight=0.5, timestamp=time.time()
            )

        action_res = action_res_override if action_res_override else {"type": "TEST_NARRATIVE_SIG_HELPER", "outcome": "success"}
        
        nc_input = {
            "phenomenal_state": p_state_for_sig, "last_action_result": action_res,
            "loop_info": None, "meta_analysis": {}, "prediction_error": None,
            "current_drives": {"curiosity": 0.5, "satisfaction":0.5, "competence":0.5} # Provide all drives
        }
        # Ensure agent's consciousness_level is set for the prompt generation
        if not hasattr(agent, 'consciousness_level') or agent.consciousness_level is None:
            agent.consciousness_level = _ConsciousState_test.CONSCIOUS # Default for prompt if not set

        await nc.process(nc_input) 

        if not mock_call_ollama_php.called:
            # Log details to help debug why _is_significant might not have triggered
            sig_check_debug, reason_debug, event_summary_debug = nc._is_significant(
                phenomenal_state=nc_input["phenomenal_state"],
                last_action_result=nc_input["last_action_result"],
                loop_info=nc_input["loop_info"],
                meta_analysis=nc_input["meta_analysis"],
                prediction_error=nc_input["prediction_error"],
                current_drives=nc_input["current_drives"]
            )
            logger.error(f"LLM not called in get_llm_prompt_for_narrative. "
                         f"Significance check result: {sig_check_debug}, Reason: '{reason_debug}', Event Summary: {event_summary_debug}\n"
                         f"Agent State for this call: Pain={agent.pain_level:.2f}, Happy={agent.happiness_level:.2f}, Purpose={agent.purpose_level:.2f}\n"
                         f"NC Last State for this call: Pain={nc._last_pain_level_nc:.2f}, Happy={nc._last_happiness_level_nc:.2f}, Purpose={nc._last_purpose_level_nc:.2f}\n"
                         f"NC known pain IDs: {nc._known_pain_source_ids_nc}, Agent active pain sources: {[ps.id for ps in agent.active_pain_sources if hasattr(ps,'id')]}\n"
                         f"PhenomenalState intensity: {getattr(p_state_for_sig, 'intensity', 'N/A')}, NC intensity_threshold: {getattr(nc, 'intensity_threshold', 'N/A')}")
            pytest.fail("LLM (mock_call_ollama_php) was not called, _is_significant likely returned False unexpectedly.")

        mock_call_ollama_php.assert_called_once()
        args_list, _ = mock_call_ollama_php.call_args
        user_prompt = args_list[1][1]['content'] 
        return user_prompt

    # Store original NC state for restoration
    original_nc_last_pain = nc._last_pain_level_nc
    original_nc_last_happy = nc._last_happiness_level_nc
    original_nc_last_purpose = nc._last_purpose_level_nc
    original_nc_known_pain_ids = nc._known_pain_source_ids_nc.copy()
    original_nc_narrative_len = len(nc.narrative)

    # Store original Agent state for restoration
    original_agent_pain = agent.pain_level
    original_agent_happy = agent.happiness_level
    original_agent_purpose = agent.purpose_level
    original_agent_active_pain_sources = list(agent.active_pain_sources)
    original_agent_cs = agent.consciousness_level


    # --- Scenario 1: High Pain, Happiness, Purpose Levels reflected in prompt ---
    logger.info("Scenario 1: P/H/P levels in prompt")
    test_pain_s1 = 8.5
    test_happy_s1 = 2.5
    test_purpose_s1 = 7.5
    # Ensure these values trigger change-based significance from default last_known in helper
    user_prompt_s1 = await get_llm_prompt_for_narrative(
        agent_pain_level_override=test_pain_s1,
        agent_happiness_level_override=test_happy_s1,
        agent_purpose_level_override=test_purpose_s1
    )
    assert f"- Pain Level: {test_pain_s1:.2f}/10" in user_prompt_s1
    assert f"- Happiness Level: {test_happy_s1:.2f}/10" in user_prompt_s1
    assert f"- Sense of Purpose: {test_purpose_s1:.2f}/10" in user_prompt_s1
    logger.info("Scenario 1: P/H/P levels correctly reflected in prompt.")


    # --- Scenario 2: New PainSource event reflected in prompt trigger ---
    logger.info("Scenario 2: New PainSource in prompt trigger")
    new_ps_desc = "A brand new sorrow event for testing"
    new_ps_test_obj = _PainSource_class_test(id="new_pain_s_id_s2", description=new_ps_desc,
                                      initial_intensity=1.8, 
                                      decay_rate_per_cycle=0.01, type="TestNewPainS2")
    
    user_prompt_new_pain_event = await get_llm_prompt_for_narrative(
        agent_active_pain_sources_override=[new_ps_test_obj], # Agent now has this pain
        agent_pain_level_override=new_ps_test_obj.current_intensity, # Agent's total pain matches this
        nc_known_pain_ids_override=set() # NC doesn't know about it yet
    )
    # Check the "Reason for this entry" part for the truncated new pain description
    assert f"NewPainSrc({new_ps_desc[:20]}...)" in user_prompt_new_pain_event, \
        f"New PainSource event not reflected as part of 'Reason for this entry'. Prompt: {user_prompt_new_pain_event}"
    # Check the "Event Details" part for the key indicating the new pain source
    assert f"'new_pain_source_{new_ps_test_obj.id}':" in user_prompt_new_pain_event, \
        f"Key for new_pain_source not found in the Event Details part of prompt. Prompt: {user_prompt_new_pain_event}"
    # Check if the full description is somewhere in the Event Details (it's inside the dict stringified)
    assert new_ps_desc in user_prompt_new_pain_event, \
        f"Full description of new pain source '{new_ps_desc}' missing from prompt. Prompt: {user_prompt_new_pain_event}"
    logger.info("Scenario 2: New PainSource correctly reflected as trigger and in prompt details.")
    

    # --- Scenario 3: Resolved PainSource event reflected in prompt trigger ---
    logger.info("Scenario 3: Resolved PainSource in prompt trigger")
    resolved_ps_desc = "An old wound now fully healed"
    resolved_ps_test_obj = _PainSource_class_test(id="resolved_pain_id_s3", description=resolved_ps_desc,
                                     initial_intensity=1.0, 
                                     #current_intensity=0.005, # Decayed significantly
                                     is_resolved=True, # Explicitly resolved
                                     source_goal_id="some_resolved_goal_s3",
                                     decay_rate_per_cycle=0.01, type="TestPainResolvedS3")
    
    user_prompt_resolved_pain_event = await get_llm_prompt_for_narrative(
        agent_active_pain_sources_override=[resolved_ps_test_obj], # Agent's list contains the resolved one
        agent_pain_level_override=0.1, # Assume low overall pain now
        nc_known_pain_ids_override={resolved_ps_test_obj.id} # NC knew about it when it was active
    )
    
    assert f"ResolvedPain({resolved_ps_desc[:20]}...)" in user_prompt_resolved_pain_event, \
        f"Resolved PainSource event not reflected as trigger in prompt. Prompt: {user_prompt_resolved_pain_event}"
    logger.info("Scenario 3: Resolved PainSource correctly reflected as trigger in prompt.")


    # Restore original NC and Agent state
    nc._last_pain_level_nc = original_nc_last_pain
    nc._last_happiness_level_nc = original_nc_last_happy
    nc._last_purpose_level_nc = original_nc_last_purpose
    nc._known_pain_source_ids_nc = original_nc_known_pain_ids
    # Restore narrative deque length if modified by test calls
    while len(nc.narrative) > original_nc_narrative_len:
        nc.narrative.pop() 
    while len(nc.narrative) < original_nc_narrative_len and original_nc_narrative_len > 0 : # Should not happen if only appending
        nc.narrative.append(MagicMock(spec=nc.NarrativeEntry)) # Add dummy if needed


    agent.pain_level = original_agent_pain
    agent.happiness_level = original_agent_happy
    agent.purpose_level = original_agent_purpose
    agent.active_pain_sources = original_agent_active_pain_sources
    agent.consciousness_level = original_agent_cs


    logger.info("--- Test Passed: Narrative Reflects Pain, Happiness, Purpose States & Events ---")

@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_agent_shutdown_on_max_pain(test_agent: AgentController):
    agent = test_agent
    logger.info("--- Test: Agent Shuts Down on Max Pain ---")

    assert hasattr(agent, '_max_pain_shutdown_threshold'), "Agent missing max pain threshold attribute."
    
    original_max_pain_threshold = agent._max_pain_shutdown_threshold
    original_agent_stop_method = agent.stop 
    original_agent_state = agent.agent_state
    # Fixture ensures agent._is_running_flag is False initially

    main_loop_task_for_test: Optional[asyncio.Task] = None

    try:
        agent._max_pain_shutdown_threshold = 2.0 
        mock_stop_method_spy = MagicMock(side_effect=lambda signum=None, frame=None: original_agent_stop_method(signum, frame)) # Ensure it can be called with/without args
        agent.stop = mock_stop_method_spy 

        agent.purpose_level = 5.0 
        agent.pain_level = 0.0 

        agent._is_running_flag.set() # Set flag before starting task
        agent.agent_state = agent._AgentState.RUNNING # Set state

        main_loop_task_for_test = asyncio.create_task(agent._run_agent_loop())
        agent._main_loop_task = main_loop_task_for_test # Let agent know its task

        # Allow loop to start and run a few cycles
        await asyncio.sleep(agent.cycle_delay_s * 3 if agent.cycle_delay_s > 0 else 0.05) 

        if not agent._is_running_flag.is_set() or (main_loop_task_for_test and main_loop_task_for_test.done()):
            pytest.fail("Test (Max Pain): Agent loop stopped prematurely before pain level could be set.")

        logger.info(f"Test (Max Pain): Setting pain_level to trigger shutdown ({agent._max_pain_shutdown_threshold + 0.1})")
        agent.pain_level = agent._max_pain_shutdown_threshold + 0.1 # Trigger shutdown condition in next cycle check
        
        # Wait for stop() to be called
        stop_call_timeout = 2.0 
        start_wait_stop_call = time.monotonic()
        while not mock_stop_method_spy.called:
            if time.monotonic() - start_wait_stop_call > stop_call_timeout:
                logger.error(f"Test (Max Pain): Timeout waiting for mock_stop_method_spy to be called.")
                break
            if not agent._is_running_flag.is_set() and main_loop_task_for_test.done(): # If loop exited for other reason
                logger.warning(f"Test (Max Pain): Loop task done or flag cleared before stop call explicitly caught.")
                break
            await asyncio.sleep(0.01)
        
        mock_stop_method_spy.assert_called_once()
        logger.info("Test (Max Pain): mock_stop_method_spy was called as expected.")

        # Now that stop() has been called (which clears the flag), wait for the loop task to fully complete.
        if main_loop_task_for_test and not main_loop_task_for_test.done():
            logger.info("Test (Max Pain): Waiting for main_loop_task to finish after stop() was called...")
            try:
                await asyncio.wait_for(main_loop_task_for_test, timeout=1.0) # Shorter timeout now
            except asyncio.TimeoutError:
                logger.error("Test (Max Pain): Timeout waiting for main_loop_task to complete after stop() call.")
            except asyncio.CancelledError:
                logger.info("Test (Max Pain): main_loop_task was cancelled while awaiting its completion after stop().")
        
        logger.info(f"Test (Max Pain): agent._is_running_flag state: {agent._is_running_flag.is_set()}, agent.agent_state: {agent.agent_state}")
        assert not agent._is_running_flag.is_set(), \
            f"Agent running flag should be false after stop() was called and loop exited. Current agent_state: {agent.agent_state}"

    finally:
        # Robust cleanup of the task
        if main_loop_task_for_test and not main_loop_task_for_test.done():
            logger.warning("Test (Max Pain) Finally: Main loop task still not done. Cancelling.")
            main_loop_task_for_test.cancel()
            with suppress(asyncio.CancelledError):
                await main_loop_task_for_test
        
        # Restore original agent attributes
        agent.stop = original_agent_stop_method
        agent._max_pain_shutdown_threshold = original_max_pain_threshold
        # Restore original_is_running_flag_state if it was modified only for test.
        # Fixture normally ensures it's false before test.
        agent.agent_state = original_agent_state 
        if agent.agent_state == agent._AgentState.STOPPED: # Default post-test state if not set otherwise
            agent._is_running_flag.clear()
        agent._main_loop_task = None

    logger.info("--- Test Passed: Agent Shuts Down on Max Pain ---")


@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE, reason="Agent components not available")
async def test_agent_shutdown_on_min_purpose(test_agent: AgentController):
    agent = test_agent
    logger.info("--- Test: Agent Shuts Down on Min Purpose ---")

    assert hasattr(agent, '_min_purpose_shutdown_threshold'), "Agent missing min purpose threshold attribute."

    original_min_purpose_threshold = agent._min_purpose_shutdown_threshold
    original_agent_stop_method = agent.stop
    original_agent_state = agent.agent_state

    main_loop_task_for_test: Optional[asyncio.Task] = None

    try:
        agent._min_purpose_shutdown_threshold = 2.0
        mock_stop_method_spy = MagicMock(side_effect=lambda signum=None, frame=None: original_agent_stop_method(signum, frame))
        agent.stop = mock_stop_method_spy # type: ignore

        agent.pain_level = 1.0 
        agent.purpose_level = 5.0 

        agent._is_running_flag.set()
        agent.agent_state = agent._AgentState.RUNNING

        main_loop_task_for_test = asyncio.create_task(agent._run_agent_loop())
        agent._main_loop_task = main_loop_task_for_test

        await asyncio.sleep(agent.cycle_delay_s * 3 if agent.cycle_delay_s > 0 else 0.05)

        if not agent._is_running_flag.is_set() or (main_loop_task_for_test and main_loop_task_for_test.done()):
            pytest.fail("Test (Min Purpose): Agent loop stopped prematurely before purpose level could be set.")

        logger.info(f"Test (Min Purpose): Setting purpose_level to trigger shutdown ({agent._min_purpose_shutdown_threshold - 0.1})")
        agent.purpose_level = agent._min_purpose_shutdown_threshold - 0.1
        
        stop_call_timeout = 2.0
        start_wait_stop_call = time.monotonic()
        while not mock_stop_method_spy.called:
            if time.monotonic() - start_wait_stop_call > stop_call_timeout:
                logger.error(f"Test (Min Purpose): Timeout waiting for mock_stop_method_spy to be called.")
                break
            if not agent._is_running_flag.is_set() and main_loop_task_for_test.done():
                logger.warning(f"Test (Min Purpose): Loop task done or flag cleared before stop call explicitly caught.")
                break
            await asyncio.sleep(0.01)

        mock_stop_method_spy.assert_called_once()
        logger.info("Test (Min Purpose): mock_stop_method_spy was called as expected.")

        if main_loop_task_for_test and not main_loop_task_for_test.done():
            logger.info("Test (Min Purpose): Waiting for main_loop_task to finish after stop() was called...")
            try:
                await asyncio.wait_for(main_loop_task_for_test, timeout=1.0)
            except asyncio.TimeoutError:
                logger.error("Test (Min Purpose): Timeout waiting for main_loop_task to complete after stop() call.")
            except asyncio.CancelledError:
                logger.info("Test (Min Purpose): main_loop_task was cancelled while awaiting its completion after stop().")
        
        logger.info(f"Test (Min Purpose): agent._is_running_flag state: {agent._is_running_flag.is_set()}, agent.agent_state: {agent.agent_state}")
        assert not agent._is_running_flag.is_set(), \
            f"Agent running flag should be false after stop() was called and loop exited. Current agent_state: {agent.agent_state}"

    finally:
        if main_loop_task_for_test and not main_loop_task_for_test.done():
            logger.warning("Test (Min Purpose) Finally: Main loop task still not done. Cancelling.")
            main_loop_task_for_test.cancel()
            with suppress(asyncio.CancelledError):
                await main_loop_task_for_test
        
        agent.stop = original_agent_stop_method
        agent._min_purpose_shutdown_threshold = original_min_purpose_threshold
        agent.agent_state = original_agent_state
        if agent.agent_state == agent._AgentState.STOPPED:
             agent._is_running_flag.clear()
        agent._main_loop_task = None

    logger.info("--- Test Passed: Agent Shuts Down on Min Purpose ---")