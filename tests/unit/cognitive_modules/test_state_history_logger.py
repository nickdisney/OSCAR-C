# tests/unit/cognitive_modules/test_state_history_logger.py

import pytest
import asyncio
import time
from typing import Dict, Any, List
from collections import deque
from unittest.mock import MagicMock
import pytest_asyncio

# Attempt to import from the project structure
try:
    from consciousness_experiment.cognitive_modules.state_history_logger import StateHistoryLogger, DEFAULT_MAX_HISTORY_PER_COMPONENT, DEFAULT_MAX_CYCLE_SNAPSHOTS
    from consciousness_experiment.protocols import CognitiveComponent # For type checking
    SHL_MODELS_AVAILABLE = True
except ImportError:
    SHL_MODELS_AVAILABLE = False
    # Minimal fallbacks for the test structure to be parseable
    class CognitiveComponent: pass
    class StateHistoryLogger(CognitiveComponent): # type: ignore
        def __init__(self):
            self.max_history_per_component = 50
            self.max_cycle_snapshots = 100
            self.component_statuses_history: Dict[str, Deque[Dict[str, Any]]] = {}
            self.cycle_snapshots: Deque[Dict[str, Any]] = deque(maxlen=self.max_cycle_snapshots)
        async def initialize(self, config, controller): 
            sh_config = config.get("state_history_logger", {})
            self.max_history_per_component = int(sh_config.get("max_history_per_component", 50))
            self.max_cycle_snapshots = int(sh_config.get("max_cycle_snapshots", 100))
            self.cycle_snapshots = deque(maxlen=self.max_cycle_snapshots) # Re-init with new maxlen
            return True
        def log_component_status_update(self, cn, sd): 
            if cn not in self.component_statuses_history: self.component_statuses_history[cn] = deque(maxlen=self.max_history_per_component)
            self.component_statuses_history[cn].append({"timestamp": time.time(), "status": sd})
        def log_full_cycle_snapshot(self, cd): self.cycle_snapshots.append(cd)
        def get_component_status_history(self, cn, ws=None): return list(self.component_statuses_history.get(cn,[]))[-ws:] if ws else list(self.component_statuses_history.get(cn,[]))
        def get_latest_system_snapshot(self): return self.cycle_snapshots[-1] if self.cycle_snapshots else None
        def get_system_snapshot_at_cycle(self, cc): 
            for s in reversed(self.cycle_snapshots): 
                if s.get("cycle_count") == cc: return s
            return None
        async def reset(self): self.component_statuses_history.clear(); self.cycle_snapshots.clear(); self.cycle_snapshots = deque(maxlen=self.max_cycle_snapshots)
        async def get_status(self): return {}
        async def process(self, i=None): return None
        async def shutdown(self): pass

    DEFAULT_MAX_HISTORY_PER_COMPONENT = 50
    DEFAULT_MAX_CYCLE_SNAPSHOTS = 100


@pytest.fixture
def state_history_logger_config() -> Dict[str, Any]:
    return {
        "state_history_logger": {
            "max_history_per_component": 3, # Small for testing limits
            "max_cycle_snapshots": 5      # Small for testing limits
        }
    }

@pytest_asyncio.fixture
async def initialized_logger(state_history_logger_config: Dict[str, Any]):
    if not SHL_MODELS_AVAILABLE:
        pytest.skip("StateHistoryLogger or dependencies not available.")
    
    logger_instance = StateHistoryLogger()
    mock_controller = MagicMock() # Controller not heavily used by SHL directly in these tests
    await logger_instance.initialize(state_history_logger_config, mock_controller)
    return logger_instance


@pytest.mark.skipif(not SHL_MODELS_AVAILABLE, reason="StateHistoryLogger not available.")
class TestStateHistoryLogger:

    @pytest.mark.asyncio
    async def test_initialize_default_configs(self):
        logger_instance = StateHistoryLogger()
        await logger_instance.initialize({}, MagicMock()) # Empty config
        assert logger_instance.max_history_per_component == DEFAULT_MAX_HISTORY_PER_COMPONENT
        assert logger_instance.max_cycle_snapshots == DEFAULT_MAX_CYCLE_SNAPSHOTS
        assert isinstance(logger_instance.cycle_snapshots, deque)
        assert logger_instance.cycle_snapshots.maxlen == DEFAULT_MAX_CYCLE_SNAPSHOTS

    @pytest.mark.asyncio
    async def test_initialize_with_custom_configs(self, initialized_logger: StateHistoryLogger, state_history_logger_config: Dict[str, Any]):
        assert initialized_logger.max_history_per_component == state_history_logger_config["state_history_logger"]["max_history_per_component"]
        assert initialized_logger.max_cycle_snapshots == state_history_logger_config["state_history_logger"]["max_cycle_snapshots"]
        assert initialized_logger.cycle_snapshots.maxlen == state_history_logger_config["state_history_logger"]["max_cycle_snapshots"]

    def test_log_component_status_update_basic(self, initialized_logger: StateHistoryLogger):
        comp_name = "test_component_A"
        status_1 = {"value": 1, "state": "active"}
        status_2 = {"value": 2, "state": "idle"}

        initialized_logger.log_component_status_update(comp_name, status_1)
        assert comp_name in initialized_logger.component_statuses_history
        assert len(initialized_logger.component_statuses_history[comp_name]) == 1
        assert initialized_logger.component_statuses_history[comp_name][0]["status"] == status_1
        assert "timestamp" in initialized_logger.component_statuses_history[comp_name][0]

        initialized_logger.log_component_status_update(comp_name, status_2)
        assert len(initialized_logger.component_statuses_history[comp_name]) == 2
        assert initialized_logger.component_statuses_history[comp_name][1]["status"] == status_2

    def test_log_component_status_update_respects_maxlen(self, initialized_logger: StateHistoryLogger, state_history_logger_config: Dict[str, Any]):
        comp_name = "test_component_B"
        max_hist = state_history_logger_config["state_history_logger"]["max_history_per_component"] # Should be 3

        for i in range(max_hist + 2): # Log 5 items
            initialized_logger.log_component_status_update(comp_name, {"value": i})
        
        assert len(initialized_logger.component_statuses_history[comp_name]) == max_hist
        # Check that the oldest items were discarded
        assert initialized_logger.component_statuses_history[comp_name][0]["status"]["value"] == 2 # (0, 1 discarded)
        assert initialized_logger.component_statuses_history[comp_name][-1]["status"]["value"] == max_hist + 1 # (which is 4)

    def test_log_full_cycle_snapshot_basic(self, initialized_logger: StateHistoryLogger):
        cycle_data_1 = {
            "cycle_count": 1, "phenomenal_state_summary": {"intensity": 0.5},
            "workspace_content_snapshot": {"item1": "data1"},
            "all_component_statuses_this_cycle": {"compA": {"status": "ok"}}
        }
        initialized_logger.log_full_cycle_snapshot(cycle_data_1)
        assert len(initialized_logger.cycle_snapshots) == 1
        logged_snapshot = initialized_logger.cycle_snapshots[0]
        assert logged_snapshot["cycle_count"] == 1
        assert logged_snapshot["phenomenal_state_summary"] == {"intensity": 0.5}
        assert "timestamp" in logged_snapshot

    def test_log_full_cycle_snapshot_respects_maxlen(self, initialized_logger: StateHistoryLogger, state_history_logger_config: Dict[str, Any]):
        max_snaps = state_history_logger_config["state_history_logger"]["max_cycle_snapshots"] # Should be 5

        for i in range(max_snaps + 2): # Log 7 items
            initialized_logger.log_full_cycle_snapshot({"cycle_count": i})
        
        assert len(initialized_logger.cycle_snapshots) == max_snaps
        assert initialized_logger.cycle_snapshots[0]["cycle_count"] == 2 # (0, 1 discarded)
        assert initialized_logger.cycle_snapshots[-1]["cycle_count"] == max_snaps + 1 # (which is 6)

    def test_get_component_status_history(self, initialized_logger: StateHistoryLogger):
        comp_name = "history_test_comp"
        s1 = {"val": 1}; s2 = {"val": 2}; s3 = {"val": 3}; s4 = {"val": 4}
        initialized_logger.log_component_status_update(comp_name, s1)
        initialized_logger.log_component_status_update(comp_name, s2)
        initialized_logger.log_component_status_update(comp_name, s3)
        initialized_logger.log_component_status_update(comp_name, s4) # Max history per component is 3

        full_hist = initialized_logger.get_component_status_history(comp_name)
        assert len(full_hist) == 3 # Due to max_history_per_component = 3
        assert full_hist[0]["status"] == s2 # s1 was discarded
        assert full_hist[-1]["status"] == s4

        windowed_hist = initialized_logger.get_component_status_history(comp_name, window_size=2)
        assert len(windowed_hist) == 2
        assert windowed_hist[0]["status"] == s3
        assert windowed_hist[-1]["status"] == s4

        assert initialized_logger.get_component_status_history("non_existent_comp") == []

    def test_get_latest_system_snapshot(self, initialized_logger: StateHistoryLogger):
        assert initialized_logger.get_latest_system_snapshot() is None
        
        # Create snapshots with the structure log_full_cycle_snapshot builds
        snap1_input = {"cycle_count": 1, "phenomenal_state_summary": {"intensity": 0.1}}
        # What will actually be stored (minus the exact timestamp)
        expected_snap1_stored = {
            # "timestamp": time.time(), # We can't predict this exactly
            "cycle_count": 1,
            "phenomenal_state_summary": {"intensity": 0.1},
            "workspace_content_snapshot": {},
            "all_component_statuses_this_cycle": {},
            "php_levels_snapshot": {},
            "active_goal_snapshot": {},
            "last_action_result_snapshot": {}
        }

        snap2_input = {"cycle_count": 2, "workspace_content_snapshot": {"itemA": "contentA"}}
        expected_snap2_stored = {
            "cycle_count": 2,
            "phenomenal_state_summary": {},
            "workspace_content_snapshot": {"itemA": "contentA"},
            "all_component_statuses_this_cycle": {},
            "php_levels_snapshot": {},
            "active_goal_snapshot": {},
            "last_action_result_snapshot": {}
        }

        initialized_logger.log_full_cycle_snapshot(snap1_input)
        retrieved_snap1 = initialized_logger.get_latest_system_snapshot()
        assert retrieved_snap1 is not None
        assert retrieved_snap1["cycle_count"] == expected_snap1_stored["cycle_count"]
        assert retrieved_snap1["phenomenal_state_summary"] == expected_snap1_stored["phenomenal_state_summary"]
        assert "timestamp" in retrieved_snap1 # Check timestamp presence

        initialized_logger.log_full_cycle_snapshot(snap2_input)
        retrieved_snap2 = initialized_logger.get_latest_system_snapshot()
        assert retrieved_snap2 is not None
        assert retrieved_snap2["cycle_count"] == expected_snap2_stored["cycle_count"]
        assert retrieved_snap2["workspace_content_snapshot"] == expected_snap2_stored["workspace_content_snapshot"]
        assert "timestamp" in retrieved_snap2
    
    
    def test_get_system_snapshot_at_cycle(self, initialized_logger: StateHistoryLogger):
        # Define inputs that log_full_cycle_snapshot will transform
        snap10_input = {"cycle_count": 10, "all_component_statuses_this_cycle": {"compA": "s10"}}
        snap11_input = {"cycle_count": 11, "php_levels_snapshot": {"pain": 1}}
        snap12_input = {"cycle_count": 12, "active_goal_snapshot": {"id": "g1"}}

        initialized_logger.log_full_cycle_snapshot(snap10_input)
        initialized_logger.log_full_cycle_snapshot(snap11_input)
        initialized_logger.log_full_cycle_snapshot(snap12_input)

        retrieved_s11 = initialized_logger.get_system_snapshot_at_cycle(11)
        assert retrieved_s11 is not None
        assert retrieved_s11["cycle_count"] == 11
        assert retrieved_s11["php_levels_snapshot"] == {"pain": 1}
        assert retrieved_s11["phenomenal_state_summary"] == {} # Check default for other keys

        retrieved_s10 = initialized_logger.get_system_snapshot_at_cycle(10)
        assert retrieved_s10 is not None
        assert retrieved_s10["cycle_count"] == 10
        assert retrieved_s10["component_statuses_snapshot"] == {"compA": "s10"}

        retrieved_s12 = initialized_logger.get_system_snapshot_at_cycle(12)
        assert retrieved_s12 is not None
        assert retrieved_s12["cycle_count"] == 12
        assert retrieved_s12["active_goal_snapshot"] == {"id": "g1"}
        
        assert initialized_logger.get_system_snapshot_at_cycle(99) is None

    @pytest.mark.asyncio
    async def test_reset_logger(self, initialized_logger: StateHistoryLogger, state_history_logger_config: Dict[str, Any]):
        initialized_logger.log_component_status_update("comp_reset", {"val":1})
        initialized_logger.log_full_cycle_snapshot({"cycle_count": 1})
        
        assert len(initialized_logger.component_statuses_history["comp_reset"]) == 1
        assert len(initialized_logger.cycle_snapshots) == 1

        await initialized_logger.reset()

        assert not initialized_logger.component_statuses_history # Should be empty
        assert len(initialized_logger.cycle_snapshots) == 0
        # Check if deque was re-initialized with correct maxlen after reset
        assert initialized_logger.cycle_snapshots.maxlen == state_history_logger_config["state_history_logger"]["max_cycle_snapshots"]


    @pytest.mark.asyncio
    async def test_get_status_method(self, initialized_logger: StateHistoryLogger):
        initialized_logger.log_component_status_update("compA", {"v":1})
        initialized_logger.log_component_status_update("compA", {"v":2})
        initialized_logger.log_component_status_update("compB", {"v":1})
        initialized_logger.log_full_cycle_snapshot({"cycle_count":1})

        status = await initialized_logger.get_status()
        assert status["component"] == "StateHistoryLogger"
        assert status["current_cycle_snapshots_count"] == 1
        assert status["num_components_with_history"] == 2
        assert status["total_logged_component_status_updates"] == 3 # 2 for compA, 1 for compB