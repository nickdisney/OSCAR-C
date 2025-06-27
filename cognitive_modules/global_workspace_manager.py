# --- START OF cognitive_modules/global_workspace_manager.py (Modified Update Logic) ---

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Set, Tuple # Set, Tuple not used but kept for consistency

# --- Use standard relative imports ---
try:
    from ..protocols import WorkspaceManager
    from ..protocols import CognitiveComponent # For type checking if needed
except ImportError:
    logging.warning("GlobalWorkspaceManager: Relative imports failed, relying on globally defined types.")
    if 'WorkspaceManager' not in globals(): raise ImportError("WorkspaceManager not found globally")
    WorkspaceManager = globals().get('WorkspaceManager')

logger_global_workspace = logging.getLogger(__name__)

DEFAULT_CAPACITY = 7
DEFAULT_BROADCAST_THRESHOLD = 0.1 # Keep the lowered threshold from config as a primary filter
MIN_WORKSPACE_ITEMS_IF_ANY_ATTENDED = 1 # Ensure at least this many items if anything has attention

class GlobalWorkspaceManager(WorkspaceManager):
    def __init__(self):
        self._controller: Optional[Any] = None
        self._config: Dict[str, Any] = {}
        self.capacity: int = DEFAULT_CAPACITY
        self.broadcast_threshold: float = DEFAULT_BROADCAST_THRESHOLD
        self.workspace_content: Dict[str, Any] = {}
        self.workspace_weights: Dict[str, float] = {}
        self.last_broadcast_time: float = 0.0
        # New config item for the minimum items
        self.min_items_if_any_attended: int = MIN_WORKSPACE_ITEMS_IF_ANY_ATTENDED


    async def initialize(self, config: Dict[str, Any], controller: Any) -> bool:
        self._controller = controller
        ws_config = config.get("global_workspace", {})
        self._config = ws_config # Store component-specific config from initial load

        # self.capacity will be initially set from config here.
        # It will then be dynamically overridden in update_workspace if controller.config changes.
        self.capacity = ws_config.get("capacity", DEFAULT_CAPACITY)
        self.broadcast_threshold = ws_config.get("broadcast_threshold", DEFAULT_BROADCAST_THRESHOLD)
        self.min_items_if_any_attended = ws_config.get("min_items_if_any_attended", MIN_WORKSPACE_ITEMS_IF_ANY_ATTENDED)


        if self.capacity <= 0: self.capacity = DEFAULT_CAPACITY
        if not (0 <= self.broadcast_threshold <= 1.0): # Allow 0 for threshold
             self.broadcast_threshold = DEFAULT_BROADCAST_THRESHOLD
        if self.min_items_if_any_attended < 0: self.min_items_if_any_attended = 0
        if self.min_items_if_any_attended > self.capacity: # Check against current self.capacity
            logger_global_workspace.warning(f"min_items_if_any_attended ({self.min_items_if_any_attended}) "
                                            f"cannot exceed capacity ({self.capacity}). Setting to capacity.")
            self.min_items_if_any_attended = self.capacity


        logger_global_workspace.info(f"GlobalWorkspaceManager initialized. Initial Capacity: {self.capacity}, " # Log initial
                                     f"Broadcast Threshold: {self.broadcast_threshold:.2f}, "
                                     f"Min Items (if attended): {self.min_items_if_any_attended}")
        return True

    async def update_workspace(self, attention_weights: Dict[str, float], all_candidates_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        # --- DYNAMICALLY READ capacity FROM CONTROLLER'S CONFIG ---
        current_effective_capacity = self.capacity # Fallback to initialized value
        if self._controller and hasattr(self._controller, 'config') and \
           isinstance(self._controller.config, dict): # type: ignore
            gwm_config_live = self._controller.config.get("global_workspace", {}) # type: ignore
            if isinstance(gwm_config_live, dict):
                current_effective_capacity = gwm_config_live.get("capacity", self.capacity)
                if not isinstance(current_effective_capacity, int) or current_effective_capacity <= 0:
                    logger_global_workspace.warning(
                        f"Invalid capacity ({current_effective_capacity}) from live config. "
                        f"Using initialized/component default value: {self.capacity}"
                    )
                    current_effective_capacity = self.capacity
            # else: config section missing, use initialized self.capacity
        # --- END DYNAMIC READ ---


        if not attention_weights:
            if self.workspace_content:
                 logger_global_workspace.debug("Clearing workspace due to empty attention weights.")
                 self.workspace_content = {}; self.workspace_weights = {}
            return self.workspace_content

        logger_global_workspace.debug(
            f"Updating workspace. Input attention items: {len(attention_weights)}. "
            f"Effective capacity: {current_effective_capacity}." # Log effective capacity
        )
        
        sorted_all_by_weight = sorted(attention_weights.items(), key=lambda item: item[1], reverse=True)
        eligible_items_passing_threshold = {
            item_id: weight for item_id, weight in sorted_all_by_weight if weight >= self.broadcast_threshold
        }
        logger_global_workspace.debug(f"{len(eligible_items_passing_threshold)} items strictly above threshold {self.broadcast_threshold:.2f}")
        
        final_items_for_workspace = []

        if eligible_items_passing_threshold:
            final_items_for_workspace = list(eligible_items_passing_threshold.items())[:current_effective_capacity] # Use effective capacity
        elif sorted_all_by_weight and self.min_items_if_any_attended > 0:
            # Use effective_capacity for min_items_if_any_attended logic as well
            count_to_take = min(self.min_items_if_any_attended, current_effective_capacity)
            for item_id, weight in sorted_all_by_weight[:count_to_take]:
                if weight > 0: final_items_for_workspace.append((item_id, weight))
                else: break
            if final_items_for_workspace:
                logger_global_workspace.info(f"No items met threshold {self.broadcast_threshold:.2f}. "
                                             f"Forcing top {len(final_items_for_workspace)} item(s) into workspace due to min_items_if_any_attended={self.min_items_if_any_attended} (capacity limited to {current_effective_capacity}).")
        
        new_workspace_content: Dict[str, Any] = {}
        new_workspace_weights: Dict[str, float] = {}
        
        logger_global_workspace.debug(f"GWM_UPDATE_DEBUG: final_items_for_workspace (IDs and weights): {[(fid, fwt) for fid, fwt in final_items_for_workspace]}")
        logger_global_workspace.debug(f"GWM_UPDATE_DEBUG: all_candidates_data keys: {list(all_candidates_data.keys())}")

        for item_id, weight in final_items_for_workspace:
            if item_id in all_candidates_data:
                logger_global_workspace.debug(f"GWM_UPDATE_DEBUG: Adding '{item_id}' to workspace. Found in all_candidates_data.")
                item_content = all_candidates_data[item_id].get("content", f"Content missing for {item_id}")
                new_workspace_content[item_id] = item_content
                new_workspace_weights[item_id] = weight
            else:
                logger_global_workspace.warning(f"GWM_UPDATE_DEBUG: Item '{item_id}' selected for workspace but MISSING from all_candidates_data.")
        
        old_keys = set(self.workspace_content.keys()); new_keys = set(new_workspace_content.keys())
        if old_keys != new_keys:
            added = new_keys - old_keys; removed = old_keys - new_keys
            logger_global_workspace.info(f"Workspace updated. Size: {len(new_keys)}/{current_effective_capacity}. Added: {len(added)}, Removed: {len(removed)}") # Use current_effective_capacity
            if added: logger_global_workspace.debug(f"  Items added: {list(added)}")
            if removed: logger_global_workspace.debug(f"  Items removed: {list(removed)}")
        elif new_workspace_content and not self.workspace_content :
            logger_global_workspace.info(f"Workspace populated. Size: {len(new_keys)}/{current_effective_capacity}.") # Use current_effective_capacity

        self.workspace_content = new_workspace_content
        self.workspace_weights = new_workspace_weights
        self.last_broadcast_time = time.time()

        return self.workspace_content

    async def broadcast(self) -> Dict[str, Any]:
        logger_global_workspace.debug(f"Broadcasting workspace content ({len(self.workspace_content)} items).")
        return self.workspace_content

    async def process(self, input_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if (input_state and "attention_weights" in input_state and "all_candidates_data" in input_state):
            attention_weights = input_state["attention_weights"]
            all_candidates_data = input_state["all_candidates_data"]
            if isinstance(attention_weights, dict) and isinstance(all_candidates_data, dict):
                await self.update_workspace(attention_weights, all_candidates_data)
                broadcast_content = await self.broadcast()
                return {"broadcast_content": broadcast_content}
            else: logger_global_workspace.warning("GW process: Invalid type for inputs."); return None
        else: logger_global_workspace.debug("GW process: Missing inputs."); return None

    async def reset(self) -> None:
        self.workspace_content = {}; self.workspace_weights = {}; self.last_broadcast_time = 0.0
        logger_global_workspace.info("GlobalWorkspaceManager reset.")

    async def get_status(self) -> Dict[str, Any]:
        # Read live capacity for status reporting as well
        current_effective_capacity_for_status = self.capacity
        if self._controller and hasattr(self._controller, 'config') and \
           isinstance(self._controller.config, dict): # type: ignore
            gwm_config_live_status = self._controller.config.get("global_workspace", {}) # type: ignore
            if isinstance(gwm_config_live_status, dict):
                current_effective_capacity_for_status = gwm_config_live_status.get("capacity", self.capacity)

        return {
            "component": "GlobalWorkspaceManager", "status": "operational",
            "capacity": current_effective_capacity_for_status, # Report effective capacity
            "current_load": len(self.workspace_content),
            "broadcast_threshold": self.broadcast_threshold,
            "min_items_if_any_attended": self.min_items_if_any_attended, # Report new config
            "last_broadcast_time": self.last_broadcast_time,
            "current_items_ids": list(self.workspace_content.keys()) # Changed key for clarity
        }

    async def shutdown(self) -> None:
        logger_global_workspace.info("GlobalWorkspaceManager shutting down.")
        self.workspace_content = {}; self.workspace_weights = {}
# --- END OF cognitive_modules/global_workspace_manager.py (Modified Update Logic) ---