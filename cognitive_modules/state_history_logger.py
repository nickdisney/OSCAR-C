# consciousness_experiment/cognitive_modules/state_history_logger.py

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Deque
from collections import deque

try:
    from ..protocols import CognitiveComponent
    # Import other datatypes if directly stored/processed, e.g., PhenomenalState
    # from ..models.datatypes import PhenomenalState 
except ImportError:
    logging.warning("StateHistoryLogger: Relative imports failed. Using placeholders.")
    from typing import Protocol
    class CognitiveComponent(Protocol): pass
    # class PhenomenalState: pass


logger_state_history = logging.getLogger(__name__)

DEFAULT_MAX_HISTORY_PER_COMPONENT = 50
DEFAULT_MAX_CYCLE_SNAPSHOTS = 100

class StateHistoryLogger(CognitiveComponent):
    """
    Centralized component for tracking state history across all components
    and full cognitive cycle snapshots.
    """

    def __init__(self):
        self._controller: Optional[Any] = None
        self._config: Dict[str, Any] = {}
        
        self.max_history_per_component: int = DEFAULT_MAX_HISTORY_PER_COMPONENT
        self.max_cycle_snapshots: int = DEFAULT_MAX_CYCLE_SNAPSHOTS

        # component_name -> deque of {"timestamp": float, "status": Dict}
        self.component_statuses_history: Dict[str, Deque[Dict[str, Any]]] = {}
        
        # Deque of full cycle data snapshots
        # Each snapshot is a dict: {"timestamp": float, "cycle_count": int, 
        #                           "phenomenal_state": Dict, "workspace_content": Dict,
        #                           "component_statuses_this_cycle": Dict[str, Dict]}
        self.cycle_snapshots: Deque[Dict[str, Any]] = deque(maxlen=self.max_cycle_snapshots)

    async def initialize(self, config: Dict[str, Any], controller: Any) -> bool:
        self._controller = controller
        sh_config = config.get("state_history_logger", {}) # New config section
        self._config = sh_config

        self.max_history_per_component = int(sh_config.get("max_history_per_component", DEFAULT_MAX_HISTORY_PER_COMPONENT))
        self.max_cycle_snapshots = int(sh_config.get("max_cycle_snapshots", DEFAULT_MAX_CYCLE_SNAPSHOTS))
        
        # Re-initialize deques with configured maxlen
        self.cycle_snapshots = deque(maxlen=self.max_cycle_snapshots)
        # component_statuses_history will be populated dynamically

        logger_state_history.info(
            f"StateHistoryLogger initialized. Max per-component history: {self.max_history_per_component}, "
            f"Max cycle snapshots: {self.max_cycle_snapshots}."
        )
        return True

    def log_component_status_update(self, component_name: str, status_dict: Dict[str, Any]):
        """
        Logs a component's status dictionary. Called by AgentController after get_status().
        """
        if not isinstance(component_name, str) or not component_name:
            logger_state_history.warning("log_component_status_update: Invalid component_name.")
            return
        if not isinstance(status_dict, dict):
            logger_state_history.warning(f"log_component_status_update: Status for '{component_name}' is not a dict.")
            return

        if component_name not in self.component_statuses_history:
            self.component_statuses_history[component_name] = deque(maxlen=self.max_history_per_component)
        
        self.component_statuses_history[component_name].append({
            "timestamp": time.time(), # Log when status was received by logger
            "status": status_dict 
        })
        logger_state_history.debug(f"Logged status for component: {component_name}")

    def log_full_cycle_snapshot(self, cycle_data: Dict[str, Any]):
        """
        Logs a comprehensive snapshot of the agent's state at the end of a cognitive cycle.
        'cycle_data' should be a dictionary prepared by AgentController containing:
        - cycle_count: int
        - phenomenal_state: Dict (summary or full object)
        - workspace_content: Dict
        - all_component_statuses_this_cycle: Dict[str, Dict] (statuses gathered during this cycle)
        - (Optional) p_h_p_levels: Dict
        - (Optional) active_goal_info: Dict
        - (Optional) last_action_result: Dict
        """
        if not isinstance(cycle_data, dict):
            logger_state_history.warning("log_full_cycle_snapshot: cycle_data is not a dict.")
            return

        snapshot = {
            "timestamp": time.time(),
            "cycle_count": cycle_data.get("cycle_count", -1),
            "phenomenal_state_summary": cycle_data.get("phenomenal_state_summary", {}), # Expect summary
            "workspace_content_snapshot": cycle_data.get("workspace_content_snapshot", {}),
            "component_statuses_snapshot": cycle_data.get("all_component_statuses_this_cycle", {}),
            "php_levels_snapshot": cycle_data.get("php_levels_snapshot", {}),
            "active_goal_snapshot": cycle_data.get("active_goal_snapshot", {}),
            "last_action_result_snapshot": cycle_data.get("last_action_result_snapshot", {})
            # Add other key overall state items as needed
        }
        self.cycle_snapshots.append(snapshot)
        logger_state_history.debug(f"Logged full cycle snapshot for cycle {snapshot['cycle_count']}.")

    def get_component_status_history(self, component_name: str, window_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get status history for a specific component."""
        if component_name not in self.component_statuses_history:
            return []
        
        history_deque = self.component_statuses_history[component_name]
        if window_size is not None and window_size > 0:
            # Return last 'window_size' elements
            return list(history_deque)[-window_size:]
        return list(history_deque)

    def get_latest_system_snapshot(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent full cycle snapshot."""
        if not self.cycle_snapshots:
            return None
        return self.cycle_snapshots[-1]

    def get_system_snapshot_at_cycle(self, target_cycle_count: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves a full system snapshot for a specific cycle count.
        Searches backwards from most recent.
        """
        if not isinstance(target_cycle_count, int): return None
        for snapshot in reversed(self.cycle_snapshots):
            if snapshot.get("cycle_count") == target_cycle_count:
                return snapshot
        return None
        
    # --- CognitiveComponent Protocol Methods ---
    async def process(self, input_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        # This component is passive; data is logged to it by AgentController.
        # It could potentially perform analysis on its history during its process() turn,
        # e.g., for causal density calculation if that's not done elsewhere.
        # For now, make it a no-op.
        return None

    async def reset(self) -> None:
        self.component_statuses_history.clear()
        self.cycle_snapshots.clear()
        # Re-initialize with maxlen for cycle_snapshots
        self.cycle_snapshots = deque(maxlen=self.max_cycle_snapshots)
        logger_state_history.info("StateHistoryLogger reset (all histories cleared).")

    async def get_status(self) -> Dict[str, Any]:
        num_tracked_components = len(self.component_statuses_history)
        total_logged_component_updates = sum(len(hist) for hist in self.component_statuses_history.values())
        
        return {
            "component": "StateHistoryLogger",
            "status": "operational",
            "max_history_per_component": self.max_history_per_component,
            "max_cycle_snapshots": self.max_cycle_snapshots,
            "current_cycle_snapshots_count": len(self.cycle_snapshots),
            "num_components_with_history": num_tracked_components,
            "total_logged_component_status_updates": total_logged_component_updates
        }

    async def shutdown(self) -> None:
        # Could potentially save history to disk here if desired for persistence across agent runs
        # For now, history is in-memory for the session.
        logger_state_history.info("StateHistoryLogger shutting down. In-memory history will be lost.")