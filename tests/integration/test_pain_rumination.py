# --- START OF FILE test_pain_rumination.py ---

import pytest
import asyncio
import time
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import queue # For mocking AgentController's ui_queue
import pytest_asyncio
from typing import Optional, Dict, Any, List, Tuple, Set

# --- Imports for OSCAR-C components ---
CONTROLLER_AVAILABLE = False
AGENT_CONTROLLER_CLASS_FOR_TEST = None
PAINSOURCE_CLASS_FOR_TEST = None
AGENT_STATE_ENUM_FOR_TEST = None
CONSCIOUS_STATE_ENUM_FOR_TEST = None
GOAL_STATUS_ENUM_FOR_TEST = None
RECOVERY_MODE_ENUM_FOR_TEST = None
VALUE_CATEGORY_ENUM_FOR_TEST = None
GOAL_DATACLASS_FOR_TEST = None
PREDICATE_DATACLASS_FOR_TEST = None
PHENOMENAL_STATE_DATACLASS_FOR_TEST = None
VALUE_JUDGMENT_DATACLASS_FOR_TEST = None

try:
    from consciousness_experiment.agent_controller import AgentController
    from consciousness_experiment.agent_state import AgentState
    from consciousness_experiment.models.enums import (
        ConsciousState, GoalStatus, RecoveryMode, ValueCategory
    )
    from consciousness_experiment.models.datatypes import (
        Goal, Predicate, PhenomenalState, PainSource, ValueJudgment
    )

    CONTROLLER_AVAILABLE = True
    AGENT_CONTROLLER_CLASS_FOR_TEST = AgentController
    PAINSOURCE_CLASS_FOR_TEST = PainSource
    AGENT_STATE_ENUM_FOR_TEST = AgentState
    CONSCIOUS_STATE_ENUM_FOR_TEST = ConsciousState
    GOAL_STATUS_ENUM_FOR_TEST = GoalStatus
    RECOVERY_MODE_ENUM_FOR_TEST = RecoveryMode
    VALUE_CATEGORY_ENUM_FOR_TEST = ValueCategory
    GOAL_DATACLASS_FOR_TEST = Goal
    PREDICATE_DATACLASS_FOR_TEST = Predicate
    PHENOMENAL_STATE_DATACLASS_FOR_TEST = PhenomenalState
    VALUE_JUDGMENT_DATACLASS_FOR_TEST = ValueJudgment


    logging.info("test_pain_rumination: Successfully imported OSCAR-C components.")

except ImportError as e:
    logging.warning(f"test_pain_rumination: Failed to import OSCAR-C components: {e}. Some tests might be skipped or use mocks.")
    class MockPainSourceForTest:
        def __init__(self, id, description, initial_intensity, current_intensity=None, timestamp_created=None, decay_rate_per_cycle=0.0, type="Test", is_resolved=False, source_goal_id=None, timestamp_created_cycle=0):
            self.id = id; self.description = description; self.initial_intensity = initial_intensity
            self.current_intensity = current_intensity if current_intensity is not None else initial_intensity
            self.timestamp_created = timestamp_created if timestamp_created is not None else time.time()
            self.decay_rate_per_cycle = decay_rate_per_cycle; self.type = type; self.is_resolved = is_resolved
            self.source_goal_id = source_goal_id
            self.timestamp_created_cycle = timestamp_created_cycle
        def __hash__(self): return hash(self.id)
        def __eq__(self, other):
            if not isinstance(other, self.__class__): return NotImplemented
            return self.id == other.id
    PAINSOURCE_CLASS_FOR_TEST = MockPainSourceForTest

    class MockAgentController:
        def __init__(self, *args, **kwargs):
            self.config={"agent_data_paths": {}}; self.cycle_count=0; self._PainSource=PAINSOURCE_CLASS_FOR_TEST;
            self.active_pain_sources=[]; self._asyncio_loop = None; self._is_running_flag = asyncio.Event()
            self.agent_state = "MOCK_STOPPED"
            self.global_workspace_content = {}
            self._active_goal_modification_hints: Optional[Dict[str, Any]] = None
            self.ui_queue = MagicMock(spec=queue.Queue)
            self.DEFAULT_OBSERVE_GOAL_PRIORITY = 1.0
            self._AgentState = MagicMock()
            self._AgentState.PAUSED = "MOCK_PAUSED"
            self._AgentState.RUNNING = "MOCK_RUNNING"
            self._AgentState.STOPPED = "MOCK_STOPPED"
            self._oscar_gather_attention_candidates = MagicMock(return_value = asyncio.Future())
            self._oscar_gather_attention_candidates.return_value.set_result({})
            self.internal_state_manager = MagicMock()
            self.components = {}

        async def _run_agent_loop_single_cycle_for_test(self): await asyncio.sleep(0.001); self.cycle_count +=1
        async def initialize_components_for_test(self): pass
        async def shutdown_for_test(self): pass
        _initialized_components_logic_for_test = []
        def set_auto_pause(self, cycles: Optional[int]): pass
        def resume_agent(self): self.agent_state = self._AgentState.RUNNING
        def stop(self):
            self._is_running_flag.clear()
            if hasattr(self, '_main_loop_task') and self._main_loop_task: self._main_loop_task.cancel()
        def _initialize_components(self): pass
        async def _run_agent_loop(self):
             while self._is_running_flag.is_set():
                 self.cycle_count += 1
                 await asyncio.sleep(0.001)
        def _update_ui_state(self, new_state): self.agent_state = new_state
        def _log_to_ui(self, level, message): pass


    if AGENT_CONTROLLER_CLASS_FOR_TEST is None:
        AGENT_CONTROLLER_CLASS_FOR_TEST = MockAgentController


logger = logging.getLogger(__name__)

@pytest_asyncio.fixture
async def agent_fixture_with_config_rumination(tmp_path: Path):
    if not CONTROLLER_AVAILABLE or AGENT_CONTROLLER_CLASS_FOR_TEST is None:
        pytest.skip("AgentController component not available for rumination test.")

    agents_created_and_tasks = []

    async def _get_agent_factory(config_override_factory_arg: Optional[Dict[str, Any]] = None):
        nonlocal agents_created_and_tasks

        base_config = {
            "agent": {"pid_file_name": f"test_rumination_{time.time_ns()}.pid"},
            "agent_data_paths": {
                "pid_directory": str(tmp_path / "run"),
                "kb_db_path": str(tmp_path / f"kb_rum_test_{time.time_ns()}.db"),
                "narrative_log_path": str(tmp_path / f"narr_rum_test_{time.time_ns()}.jsonl"),
                "self_model_path": str(tmp_path / f"dsm_rum_test_{time.time_ns()}.json"),
                "predictive_model_learned_data_path": str(tmp_path / f"pwm_rum_test_{time.time_ns()}.csv"),
                "performance_adjustments_path": str(tmp_path / f"po_rum_test_{time.time_ns()}.json"),
                "htn_library_path": str(tmp_path / f"htn_lib_rum_test_{time.time_ns()}.json"),
            },
            "attention_controller": {
                "pain_rumination_threshold_cycles": 3,
                "pain_rumination_window_multiplier": 3,
                "pain_inactive_reset_cycles": 10,
                "pain_attention_distraction_factor": 0.2,
                "pain_rumination_suppression_factor": 0.1,
            },
            "performance": {"target_cycle_time": 0.1},
            "predictive_world_model": {"cbn_config_file": "data/pwm_cbn_config.json"}
        }
        current_config = base_config.copy()

        current_config.setdefault("performance", {})["target_cycle_time"] = 0.1
        current_config.setdefault("performance_optimizer", {})["auto_apply_adjustments"] = False
        current_config.setdefault("agent", {}).setdefault("ui_meter_update_interval_s", 1.0)

        if config_override_factory_arg:
            for section, params in config_override_factory_arg.items():
                if section not in current_config: current_config[section] = {}
                if isinstance(params, dict) and isinstance(current_config[section], dict):
                    current_config[section].update(params)
                else: current_config[section] = params

        mock_ui_queue = MagicMock(spec=queue.Queue)
        agent = None
        main_loop_task = None

        controller_module_path = AGENT_CONTROLLER_CLASS_FOR_TEST.__module__

        patch_targets_to_apply = {}
        try:
            from consciousness_experiment.agent_controller import component_classes as ac_component_classes_fixture
        except ImportError:
            ac_component_classes_fixture = {}
            logger.warning("Fixture: Could not import component_classes from agent_controller for selective patching.")

        components_to_quick_init = [
            "CognitiveCache", "GlobalWorkspaceManager", "ExperienceStream",
            "ConsciousnessLevelAssessor", "ErrorRecoverySystem", "DynamicSelfModel",
            "EmergentMotivationSystem", "StateHistoryLogger"
        ]
        for comp_key_fixture in components_to_quick_init:
            ComponentClass_fixture = ac_component_classes_fixture.get(comp_key_fixture)
            if ComponentClass_fixture:
                class_module_path_fixture = ComponentClass_fixture.__module__
                class_name_fixture = ComponentClass_fixture.__name__
                patch_targets_to_apply[f"{class_module_path_fixture}.{class_name_fixture}.process"] = MagicMock(return_value=asyncio.Future())
                patch_targets_to_apply[f"{class_module_path_fixture}.{class_name_fixture}.process"].return_value.set_result(None)
                logger.info(f"Fixture: Will patch process for {comp_key_fixture}")
            else:
                logger.warning(f"Fixture: Cannot find class for component key '{comp_key_fixture}' to patch its process.")


        load_config_patcher = patch.object(AGENT_CONTROLLER_CLASS_FOR_TEST, '_load_config', return_value=current_config)
        active_patchers = [load_config_patcher]

        for target_path, patch_obj in patch_targets_to_apply.items():
            try:
                p = patch(target_path, new=patch_obj)
                active_patchers.append(p)
                logger.info(f"Fixture: Successfully prepared patch for {target_path}")
            except Exception as e_patch:
                logger.warning(f"Skipping patch for '{target_path}' in fixture due to error: {e_patch}")

        try:
            for p in active_patchers: p.start()

            mock_cbn_config_path = tmp_path / "mock_pwm_cbn_config.json"
            mock_cbn_config_path.write_text('{"cbn_nodes": [], "cbn_structure": []}')
            if "predictive_world_model" not in current_config: current_config["predictive_world_model"] = {}
            current_config["predictive_world_model"]["cbn_config_file"] = str(mock_cbn_config_path)

            agent = AGENT_CONTROLLER_CLASS_FOR_TEST(model_name="test_rumination_model", ui_queue=mock_ui_queue)
            agent.config = current_config

            agent_loop = asyncio.get_event_loop()
            agent._asyncio_loop = agent_loop
            agent._is_running_flag.set()
            agent._is_paused_event.set()

            if not hasattr(agent, 'components') or not agent.components:
                 if hasattr(agent, '_initialize_components'):
                    agent._initialize_components()

            initialized_components_logic = []
            try:
                from consciousness_experiment.agent_controller import COMPONENT_INIT_ORDER_CTRL as INIT_ORDER
            except ImportError:
                INIT_ORDER = list(agent.components.keys()) if hasattr(agent, 'components') else []
                logger.warning("COMPONENT_INIT_ORDER_CTRL not found, using agent.components.keys() for init order.")


            for name in INIT_ORDER:
                if hasattr(agent, 'components') and name in agent.components:
                    comp = agent.components[name]
                    if hasattr(comp, 'initialize'):
                        success = await comp.initialize(agent.config, agent)
                        if success: initialized_components_logic.append(name)
            agent._initialized_components_logic_for_test = initialized_components_logic

            main_loop_task = agent_loop.create_task(agent._run_agent_loop())
            logger.info("Test Fixture: Agent main loop task created.")
            await asyncio.sleep(0.01)

            if hasattr(agent, 'InternalStateUpkeepManager') and not hasattr(agent, 'internal_state_manager'):
                 agent.internal_state_manager = agent.InternalStateUpkeepManager(agent)

            if hasattr(agent, '_AgentState') and hasattr(agent._AgentState, "RUNNING"):
                agent.agent_state = agent._AgentState.RUNNING
                if hasattr(agent, '_update_ui_state'):
                    agent._update_ui_state(agent._AgentState.RUNNING)
                logger.info("Test Fixture: Agent state explicitly set to RUNNING.")
            else:
                logger.error("Test Fixture: Could not set agent state to RUNNING, AgentState enum problematic.")


        except Exception as e_fixture_setup:
            logger.error(f"Error during agent setup in fixture factory: {e_fixture_setup}", exc_info=True)
            for p_item_err in reversed(active_patchers):
                try: p_item_err.stop()
                except RuntimeError: pass
            if main_loop_task and not main_loop_task.done(): main_loop_task.cancel()
            raise
        finally:
             for p_item_final in active_patchers:
                try: p_item_final.stop()
                except RuntimeError: pass


        agents_created_and_tasks.append((agent, main_loop_task))
        return agent

    yield _get_agent_factory

    logger.debug(f"Tearing down agent_fixture_with_config_rumination. Agents/Tasks created: {len(agents_created_and_tasks)}")
    for ag, task in agents_created_and_tasks:
        logger.debug(f"Cleaning up agent instance and task for {ag}")
        if hasattr(ag, 'stop') and callable(ag.stop):
            ag.stop()

        if task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.CancelledError:
                logger.debug("Agent main loop task successfully cancelled during teardown.")
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for agent main loop task to cancel during teardown.")
            except Exception as e_task_cancel:
                logger.error(f"Error during agent main loop task cancellation: {e_task_cancel}")

        if hasattr(ag, '_secondary_loop_task') and ag._secondary_loop_task and not ag._secondary_loop_task.done():
            logger.debug("Cancelling secondary_loop_task during teardown.")
            ag._secondary_loop_task.cancel()
            try:
                await asyncio.wait_for(ag._secondary_loop_task, timeout=0.5)
            except asyncio.CancelledError:
                logger.debug("Secondary loop task successfully cancelled during teardown.")
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for secondary loop task to cancel.")
            except Exception as e_sec_cancel:
                logger.error(f"Error cancelling secondary loop: {e_sec_cancel}")

        if hasattr(ag, '_shutdown_components') and hasattr(ag, '_initialized_components_logic_for_test'):
            try: await ag._shutdown_components(ag._initialized_components_logic_for_test)
            except Exception as e_shutdown: logger.error(f"Error shutting down components for agent: {e_shutdown}")

        if hasattr(ag, 'pid_file') and isinstance(ag.pid_file, Path) and ag.pid_file.exists():
            try: ag.pid_file.unlink()
            except OSError: pass

        if hasattr(ag, 'knowledge_base') and hasattr(ag.knowledge_base, 'db_path') and \
           isinstance(ag.knowledge_base.db_path, Path):
            if hasattr(ag.knowledge_base, '_connection') and ag.knowledge_base._connection:
                try: await ag.knowledge_base.shutdown()
                except Exception as e_kb_shutdown: logger.warning(f"Error during KB shutdown: {e_kb_shutdown}")
            if ag.knowledge_base.db_path.exists():
                try: ag.knowledge_base.db_path.unlink()
                except OSError as e_unlink: logger.warning(f"Could not unlink test KB DB {ag.knowledge_base.db_path}: {e_unlink}")


async def wait_for_agent_pause(agent, timeout_seconds=2.0):
    if not AGENT_STATE_ENUM_FOR_TEST or not hasattr(agent, '_AgentState') or not hasattr(agent._AgentState, "PAUSED"):
        raise RuntimeError("AgentState.PAUSED enum member not available for test.")

    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout_seconds:
        if agent.agent_state == agent._AgentState.PAUSED:
            logger.info(f"Agent entered PAUSED state at cycle {agent.cycle_count}.")
            return
        await asyncio.sleep(0.01)
    raise asyncio.TimeoutError(f"Agent did not enter PAUSED state within {timeout_seconds}s. Current state: {agent.agent_state}")


@pytest.mark.asyncio
@pytest.mark.skipif(not CONTROLLER_AVAILABLE or PAINSOURCE_CLASS_FOR_TEST is None or AGENT_STATE_ENUM_FOR_TEST is None,
                    reason="AgentController, PainSource, or AgentState not available for rumination test.")
async def test_pain_rumination_suppression_and_reset(agent_fixture_with_config_rumination, caplog):
    cycles_to_run_total = 12
    caplog.set_level(logging.INFO)

    rumination_threshold_config = 2
    inactive_reset_trigger_config = 4

    config_override_for_test = {
        "attention_controller": {
            "pain_rumination_threshold_cycles": rumination_threshold_config,
            "pain_rumination_window_multiplier": 3,
            "pain_inactive_reset_cycles": inactive_reset_trigger_config,
            "pain_attention_distraction_factor": 1.5,
            "pain_rumination_suppression_factor": 0.01,
            "hint_weight": 0.9,
            "recency_weight": 0.01, "novelty_bonus_weight": 0.01,
            "surprise_bonus_weight": 0.01, "goal_relevance_weight": 0.01,
        },
        "agent": {"ui_meter_update_interval_s": 1.0, "pid_file_name": f"test_rum_{time.time_ns()}.pid"},
        "performance": {"target_cycle_time": 0.1},
        "performance_optimizer": {"auto_apply_adjustments": False}
    }
    agent_factory = agent_fixture_with_config_rumination
    agent = await agent_factory(config_override_for_test)

    ps_rum_id = "PS_RUMINATE_INTEG_TEST"
    _PainSource_agent_runtime = agent._PainSource
    if not hasattr(_PainSource_agent_runtime, '__post_init__'):
        test_pain_source = _PainSource_agent_runtime(
            id=ps_rum_id, description="Rumination Test Pain", initial_intensity=7.0,
            current_intensity=7.0, timestamp_created=time.time()-1000,
            decay_rate_per_cycle=0.0001, type="TestRum", is_resolved=False,
            timestamp_created_cycle=agent.cycle_count # Initialize with current cycle
        )
    else:
        test_pain_source = _PainSource_agent_runtime(
            id=ps_rum_id, description="Rumination Test Pain", initial_intensity=7.0,
            timestamp_created=time.time()-1000, decay_rate_per_cycle=0.0001,
            type="TestRum", is_resolved=False
        )
        if hasattr(test_pain_source, 'timestamp_created_cycle'):
            test_pain_source.timestamp_created_cycle = agent.cycle_count

    if not hasattr(test_pain_source, 'current_intensity') or test_pain_source.current_intensity is None:
         test_pain_source.current_intensity = test_pain_source.initial_intensity

    agent.active_pain_sources.append(test_pain_source)
    logger.info(f"TEST_RUMINATE: Added PainSource '{ps_rum_id}' with current intensity {test_pain_source.current_intensity:.2f}")

    ps_rum_item_key_in_gwm = f"pain_event_{ps_rum_id}"

    phase = "GATHERING_INITIAL_SALIENCE"
    gwm_appearances_for_current_phase = 0
    # last_cycle_ps_rum_was_in_gwm is used to store the NEWEST GWM cycle in a streak
    last_cycle_ps_rum_was_in_gwm = 0
    actual_gwm_appearances_of_ps_rum = 0
    inactivity_cleanup_log_observed_on_agent_cycle = 0

    final_initial_salience_count = 0
    final_suppression_observed_at_cycle = 0
    final_cleanup_observed = False
    final_resurgence_observed_at_cycle = 0
    final_resuppression_observed_at_cycle = 0
    
    # Tracks the OLDEST GWM cycle in the current streak leading to suppression
    oldest_gwm_cycle_in_current_streak = 0
    # Tracks the OLDEST GWM cycle that started the very FIRST suppression streak
    oldest_gwm_at_first_suppression_trigger = 0


    original_gather_candidates_method = agent._oscar_gather_attention_candidates
    boost_applications_done = 0

    async def mock_gather_candidates_side_effect_for_patch(percepts_arg, active_goal_id_arg):
        nonlocal boost_applications_done
        agent_instance = agent
        current_processing_cycle = agent_instance.cycle_count

        logger.info(f"TEST_RUMINATE_PATCH_ENTRY: Mock gather for Agent Cycle {current_processing_cycle}. RumThreshCfg: {rumination_threshold_config}, BoostsDone: {boost_applications_done}")

        candidates = await original_gather_candidates_method(percepts_arg, active_goal_id_arg)
        logger.info(f"TEST_RUMINATE_PATCH_ORIG_CANDS: Cycle {current_processing_cycle} - Original candidates keys: {list(candidates.keys())}")

        if boost_applications_done < rumination_threshold_config:
            logger.info(f"TEST_RUMINATE_PATCH_CONDITION_MET: AgentCycle {current_processing_cycle}, BoostsDone {boost_applications_done} < RumThreshCfg {rumination_threshold_config}.")
            active_ps_exists = any(
                hasattr(ps, 'id') and ps.id == ps_rum_id for ps in agent_instance.active_pain_sources
                if isinstance(ps, PAINSOURCE_CLASS_FOR_TEST)
            )
            logger.info(f"TEST_RUMINATE_PATCH_ACTIVE_CHECK: Cycle {current_processing_cycle} - active_ps_exists ('{ps_rum_id}') = {active_ps_exists}")

            if active_ps_exists:
                candidate_key_to_boost = ps_rum_item_key_in_gwm
                pain_content_for_candidate = {
                        "type": getattr(test_pain_source, 'type', "TestRum"),
                        "description": getattr(test_pain_source, 'description', "Rumination Test Pain"),
                        "current_intensity": getattr(test_pain_source, 'current_intensity', 7.0),
                        "source_type": getattr(test_pain_source, 'type', "TestRum"),
                        "age_cycles": current_processing_cycle - getattr(test_pain_source, 'timestamp_created_cycle', agent.cycle_count)
                    }
                if candidate_key_to_boost in candidates:
                    candidates[candidate_key_to_boost]['weight_hint'] = 200.0
                    candidates[candidate_key_to_boost]['timestamp'] = time.time()
                    if isinstance(candidates[candidate_key_to_boost].get('content'), dict):
                        candidates[candidate_key_to_boost]['content'].update(pain_content_for_candidate)
                    else:
                        candidates[candidate_key_to_boost]['content'] = pain_content_for_candidate
                    logger.info(f"TEST_RUMINATE_PATCH_ACTION (MODIFIED): Cycle {current_processing_cycle} - Artificially boosted hint AND recency for existing candidate '{candidate_key_to_boost}'.")
                else:
                    ts_created_for_mock = time.time()
                    candidates[candidate_key_to_boost] = {
                        "content": pain_content_for_candidate,
                        "weight_hint": 200.0,
                        "timestamp": ts_created_for_mock
                    }
                    logger.info(f"TEST_RUMINATE_PATCH_ACTION (MODIFIED): Cycle {current_processing_cycle} - '{candidate_key_to_boost}' ADDED to candidates and boosted (hint & recency).")
                boost_applications_done += 1
        else:
             logger.info(f"TEST_RUMINATE_PATCH_CONDITION_NOT_MET: AgentCycle {current_processing_cycle}, BoostsDone {boost_applications_done} >= RumThreshCfg {rumination_threshold_config}. No boost applied.")

        logger.info(f"TEST_RUMINATE_PATCH_EXIT: Cycle {current_processing_cycle} - Final candidates keys after patch: {list(candidates.keys())}")
        return candidates

    with patch.object(agent, '_oscar_gather_attention_candidates',
                      side_effect=mock_gather_candidates_side_effect_for_patch) as mock_gather_call:
        for test_iteration_num in range(1, cycles_to_run_total + 1):
            logger.info(f"TEST_RUMINATE: --- Initiating Test Iteration {test_iteration_num} (Target Agent Cycle approx {agent.cycle_count + 1}) ---")
            agent.set_auto_pause(1)
            if agent.agent_state == agent._AgentState.PAUSED: agent.resume_agent()
            elif agent.agent_state != agent._AgentState.RUNNING: logger.warning(f"Test Iteration {test_iteration_num}: Agent not RUNNING or PAUSED. State: {agent.agent_state}")
            try: await wait_for_agent_pause(agent, timeout_seconds=5.0)
            except asyncio.TimeoutError: logger.error(f"TEST_RUMINATE: Timeout waiting for agent to pause in test iteration {test_iteration_num}. Current agent state: {agent.agent_state}"); raise

            completed_agent_cycle = agent.cycle_count
            current_gwm_keys = list(agent.global_workspace_content.keys())
            ps_rum_in_gwm_this_cycle = ps_rum_item_key_in_gwm in current_gwm_keys
            logger.info(f"TEST_RUMINATE: Agent has PAUSED after completing agent cycle {completed_agent_cycle}.")
            logger.info(f"TEST_RUMINATE: Agent Cycle {completed_agent_cycle} - PS_RUMINATE in GWM: {ps_rum_in_gwm_this_cycle}. Phase: {phase}. GWM appearances for this phase: {gwm_appearances_for_current_phase}")

            if phase == "GATHERING_INITIAL_SALIENCE":
                assert ps_rum_in_gwm_this_cycle, f"Cycle {completed_agent_cycle}: Expected PS_RUMINATE IN GWM (INITIAL_SALIENCE with boosted hint)"
                if ps_rum_in_gwm_this_cycle:
                    gwm_appearances_for_current_phase += 1
                    last_cycle_ps_rum_was_in_gwm = completed_agent_cycle
                    actual_gwm_appearances_of_ps_rum += 1
                    if gwm_appearances_for_current_phase == 1:
                        oldest_gwm_cycle_in_current_streak = completed_agent_cycle
                    logger.info(f"TEST_RUMINATE: Cycle {completed_agent_cycle} - PS_RUMINATE IN GWM. Count for this salience phase: {gwm_appearances_for_current_phase}. Oldest in streak: C{oldest_gwm_cycle_in_current_streak}")

                if gwm_appearances_for_current_phase == rumination_threshold_config:
                    final_initial_salience_count = gwm_appearances_for_current_phase
                    oldest_gwm_at_first_suppression_trigger = oldest_gwm_cycle_in_current_streak # Capture for final assert
                    phase = "EXPECTING_SUPPRESSION"
                    logger.info(f"TEST_RUMINATE: Met {rumination_threshold_config} GWM appearances. "
                                f"Oldest GWM in this streak: C{oldest_gwm_cycle_in_current_streak}. Newest: C{last_cycle_ps_rum_was_in_gwm}. "
                                f"Transition to EXPECTING_SUPPRESSION for Cycle {completed_agent_cycle + 1}")

            elif phase == "EXPECTING_SUPPRESSION":
                assert not ps_rum_in_gwm_this_cycle, \
                    (f"Cycle {completed_agent_cycle}: Expected PS_RUMINATE SUPPRESSED (threshold met in prior cycles), "
                     f"but was IN GWM. Total initial GWM appearances: {final_initial_salience_count}. GWM: {current_gwm_keys}")
                if not ps_rum_in_gwm_this_cycle:
                    final_suppression_observed_at_cycle = completed_agent_cycle
                    cycles_unseen_in_gwm_since_last_gwm = 1
                    phase = "OBSERVING_INACTIVITY_RESET"
                    logger.info(f"TEST_RUMINATE: Cycle {completed_agent_cycle} - Suppression correct. Transition to OBSERVING_INACTIVITY_RESET.")

            elif phase == "OBSERVING_INACTIVITY_RESET":
                logger.info(
                    f"TEST_RUMINATE_DEBUG_OBSERVE: Cycle {completed_agent_cycle}, "
                    f"Phase: OBSERVING_INACTIVITY_RESET. cycles_unseen_in_gwm_since_last_gwm = {cycles_unseen_in_gwm_since_last_gwm}"
                )
                expected_resurgence_cycle_for_phase_transition = oldest_gwm_cycle_in_current_streak + inactive_reset_trigger_config
                logger.info(
                    f"TEST_RUMINATE_TRANSITION_CHECK: Cycle {completed_agent_cycle}. "
                    f"Expecting resurgence around Agent Cycle {expected_resurgence_cycle_for_phase_transition} "
                    f"(Oldest GWM in supp. streak C{oldest_gwm_cycle_in_current_streak} + inactive_reset={inactive_reset_trigger_config})."
                )
                if completed_agent_cycle == expected_resurgence_cycle_for_phase_transition:
                    phase = "EXPECTING_RESURGENCE"
                    logger.info(f"TEST_RUMINATE: Transitioned to EXPECTING_RESURGENCE for Agent Cycle {completed_agent_cycle}. "
                                f"Cleanup in PainEventTracker should have occurred, allowing resurgence.")

                if phase == "OBSERVING_INACTIVITY_RESET":
                    assert not ps_rum_in_gwm_this_cycle, \
                        (f"Cycle {completed_agent_cycle}: PS_RUMINATE unexpectedly IN GWM during OBSERVING_INACTIVITY_RESET. "
                         f"Oldest in supp. streak C{oldest_gwm_cycle_in_current_streak}. Unseen streak for: {cycles_unseen_in_gwm_since_last_gwm} cycle(s). "
                         f"Resurgence expected C{expected_resurgence_cycle_for_phase_transition}.")
                    if not ps_rum_in_gwm_this_cycle:
                        cycles_unseen_in_gwm_since_last_gwm += 1
                        logger.info(f"TEST_RUMINATE: Cycle {completed_agent_cycle} - PS_RUMINATE remains unseen. Confirmed unseen count now: {cycles_unseen_in_gwm_since_last_gwm}")

            if phase == "EXPECTING_RESURGENCE":
                if not final_cleanup_observed:
                     for record in caplog.records:
                        if record.name == "consciousness_experiment.agent_helpers.cognitive_trackers" and \
                           "PET_CLEANUP" in record.message and \
                           "Pruned" in record.message and \
                           f"current agent cycle for check: {completed_agent_cycle}" in record.message:
                            final_cleanup_observed = True
                            inactivity_cleanup_log_observed_on_agent_cycle = completed_agent_cycle
                            logger.info(f"TEST_RUMINATE: Cycle {completed_agent_cycle} - Relevant general Cleanup log found from PainEventTracker: '{record.message}'")
                            break
                assert final_cleanup_observed, \
                    (f"Cycle {completed_agent_cycle}: Expected PET_CLEANUP log from PainEventTracker "
                     f"indicating pruning occurred relevant to this cycle (containing 'current agent cycle for check: {completed_agent_cycle}'), but not found. "
                     f"Make sure PainEventTracker._cleanup_inactive_entries is logging 'PET_CLEANUP ... Pruned ...' at INFO level. "
                     f"InactiveResetCfg: {inactive_reset_trigger_config}. Oldest GWM in first supp streak: C{oldest_gwm_at_first_suppression_trigger}. "
                     f"Test's_cycles_unseen_count_at_transition_was: {cycles_unseen_in_gwm_since_last_gwm}.")

                assert ps_rum_in_gwm_this_cycle, \
                    (f"Cycle {completed_agent_cycle}: Expected PS_RUMINATE to RESURGE (after cleanup and no suppression), but wasn't IN GWM. "
                     f"Cleanup log observed: {final_cleanup_observed}. Oldest GWM in first supp streak: C{oldest_gwm_at_first_suppression_trigger}. Unseen count before this cycle: {cycles_unseen_in_gwm_since_last_gwm}.")
                if ps_rum_in_gwm_this_cycle:
                    final_resurgence_observed_at_cycle = completed_agent_cycle
                    gwm_appearances_for_current_phase = 1 # Reset for the new streak
                    actual_gwm_appearances_of_ps_rum +=1
                    last_cycle_ps_rum_was_in_gwm = completed_agent_cycle
                    oldest_gwm_cycle_in_current_streak = completed_agent_cycle # Start of new streak
                    cycles_unseen_in_gwm_since_last_gwm = 0
                    phase = "POST_RESURGENCE_SALIENCE"
                    logger.info(f"TEST_RUMINATE: Resurgence correct at Cycle {completed_agent_cycle}. Transition to POST_RESURGENCE_SALIENCE for Cycle {completed_agent_cycle + 1}")

            elif phase == "POST_RESURGENCE_SALIENCE":
                assert ps_rum_in_gwm_this_cycle, f"Cycle {completed_agent_cycle}: Expected PS_RUMINATE IN GWM (POST_RESURGENCE_SALIENCE)"
                if ps_rum_in_gwm_this_cycle:
                    gwm_appearances_for_current_phase += 1
                    actual_gwm_appearances_of_ps_rum +=1
                    last_cycle_ps_rum_was_in_gwm = completed_agent_cycle
                    # oldest_gwm_cycle_in_current_streak was set when this phase was entered

                if gwm_appearances_for_current_phase == rumination_threshold_config:
                    # L_gwm_supp_start_base = last_cycle_ps_rum_was_in_gwm # No, L_gwm_supp_start_base not needed for this transition
                    phase = "EXPECTING_RE_SUPPRESSION"
                    logger.info(f"TEST_RUMINATE: Met {rumination_threshold_config} GWM appearances post-resurgence. "
                                f"Oldest GWM in this new streak: C{oldest_gwm_cycle_in_current_streak}. Newest: C{last_cycle_ps_rum_was_in_gwm}. "
                                f"Transition to EXPECTING_RE_SUPPRESSION for Cycle {completed_agent_cycle + 1}")

            elif phase == "EXPECTING_RE_SUPPRESSION":
                assert not ps_rum_in_gwm_this_cycle, f"Cycle {completed_agent_cycle}: Expected PS_RUMINATE RE-SUPPRESSED, but was IN GWM."
                if not ps_rum_in_gwm_this_cycle:
                    final_resuppression_observed_at_cycle = completed_agent_cycle
                    phase = "DONE_TESTING_PHASES"
                    logger.info(f"TEST_RUMINATE: Re-suppression observed at Cycle {completed_agent_cycle}. Test phases complete.")

            if test_iteration_num == cycles_to_run_total:
                logger.info(f"TEST_RUMINATE: Reached total test iterations ({cycles_to_run_total}). Stopping agent.")
                if agent.agent_state == agent._AgentState.PAUSED: agent.resume_agent()
                await asyncio.sleep(0.01)
                agent.stop()

    assert final_initial_salience_count == rumination_threshold_config, \
        (f"PS_RUMINATE did not appear in GWM exactly {rumination_threshold_config} times during initial salience phase. "
         f"Observed: {final_initial_salience_count}")

    expected_suppression_cycle = oldest_gwm_at_first_suppression_trigger + rumination_threshold_config if final_initial_salience_count == rumination_threshold_config else 0
    if final_initial_salience_count == rumination_threshold_config : # only check if initial salience was met
        expected_suppression_cycle = (oldest_gwm_at_first_suppression_trigger + rumination_threshold_config -1) +1 # Cycle after last GWM hit
        assert final_suppression_observed_at_cycle == expected_suppression_cycle, \
            (f"PS_RUMINATE suppression did not start at the correct agent cycle. "
             f"Oldest GWM in first streak was C{oldest_gwm_at_first_suppression_trigger}, threshold {rumination_threshold_config}. Last GWM hit was C{last_cycle_ps_rum_was_in_gwm if phase != 'GATHERING_INITIAL_SALIENCE' else oldest_gwm_at_first_suppression_trigger + rumination_threshold_config -1 }. "
             f"Expected suppression at C{expected_suppression_cycle}. Observed Start: C{final_suppression_observed_at_cycle}. ")

    assert final_cleanup_observed, "PainEventTracker cleanup log (relevant to the resurgence cycle) was not observed."

    expected_resurgence_cycle_final_assert = oldest_gwm_at_first_suppression_trigger + inactive_reset_trigger_config
    assert final_resurgence_observed_at_cycle == expected_resurgence_cycle_final_assert, \
        (f"PS_RUMINATE resurgence after inactivity reset was not observed at the correct cycle. "
         f"Oldest GWM of first suppression streak: C{oldest_gwm_at_first_suppression_trigger}, Inactive Trigger Cfg: {inactive_reset_trigger_config}, "
         f"Expected Resurgence Cycle: C{expected_resurgence_cycle_final_assert}. Observed Resurgence: C{final_resurgence_observed_at_cycle}")

    if final_resurgence_observed_at_cycle > 0 and \
       cycles_to_run_total >= (final_resurgence_observed_at_cycle + rumination_threshold_config):
        expected_resuppression_cycle = (oldest_gwm_cycle_in_current_streak + rumination_threshold_config -1) + 1 # Cycle after last GWM hit in 2nd streak
        if final_resuppression_observed_at_cycle > 0 :
             assert final_resuppression_observed_at_cycle == expected_resuppression_cycle, \
             f"PS_RUMINATE re-suppression not observed at correct cycle. Oldest GWM of second streak C{oldest_gwm_cycle_in_current_streak}, threshold {rumination_threshold_config}. Expected Re-Suppression: C{expected_resuppression_cycle}, Observed: C{final_resuppression_observed_at_cycle}"

    logger.info(
        f"TEST_RUMINATE: Test completed. Initial Salience Count: {final_initial_salience_count}, "
        f"Total GWM appearances for this pain_id during test (all phases): {actual_gwm_appearances_of_ps_rum}, "
        f"Oldest GWM at first suppression trigger: C{oldest_gwm_at_first_suppression_trigger}, "
        f"Suppression Start Cycle (Agent): {final_suppression_observed_at_cycle}, "
        f"Cleanup Log Observed Cycle (Agent): {inactivity_cleanup_log_observed_on_agent_cycle}, "
        f"Resurgence Cycle (Agent): {final_resurgence_observed_at_cycle}, "
        f"Re-Suppression Cycle (Agent): {final_resuppression_observed_at_cycle if final_resuppression_observed_at_cycle != 0 else 'N/A (not enough cycles or not reached)'}"
    )
# --- END OF FILE test_pain_rumination.py ---