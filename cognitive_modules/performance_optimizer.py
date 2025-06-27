# --- START OF performance_optimizer.py

import asyncio
import json
import logging
import time
from pathlib import Path # Ensure Path is imported
from typing import Dict, Any, List, Optional, Deque
from collections import deque

# --- Use standard relative imports ---
try:
    from ..protocols import CognitiveComponent
    from ..models.enums import RecoveryMode
except ImportError:
    logging.warning("PerformanceOptimizer: Relative import failed, relying on globally defined types.")
    if 'CognitiveComponent' not in globals(): raise ImportError("CognitiveComponent not found via relative import or globally")
    if 'RecoveryMode' not in globals(): logging.warning("RecoveryMode enum not found globally.")
    CognitiveComponent = globals().get('CognitiveComponent')
    RecoveryMode = globals().get('RecoveryMode')

# --- Import default from AttentionController for fallback ---
try:
    from .attention_controller import DEFAULT_MAX_CANDIDATES as AC_DEFAULT_MAX_CANDIDATES
except ImportError:
    logging.getLogger(__name__).warning("PerformanceOptimizer: Could not import DEFAULT_MAX_CANDIDATES from attention_controller. Using hardcoded fallback.")
    AC_DEFAULT_MAX_CANDIDATES = 50 # Fallback if import fails
# --- Import default from GlobalWorkspaceManager for fallback ---
try:
    from .global_workspace_manager import DEFAULT_CAPACITY as GWM_DEFAULT_CAPACITY
except ImportError:
    logging.getLogger(__name__).warning("PerformanceOptimizer: Could not import DEFAULT_CAPACITY from global_workspace_manager. Using hardcoded fallback.")
    GWM_DEFAULT_CAPACITY = 7 # Fallback if import fails
# --- Import default from LoopDetector for fallback ---
try:
    from .loop_detector import DEFAULT_WINDOW_SIZE as LD_DEFAULT_WINDOW_SIZE
except ImportError:
    logging.getLogger(__name__).warning("PerformanceOptimizer: Could not import DEFAULT_WINDOW_SIZE from loop_detector. Using hardcoded fallback.")
    LD_DEFAULT_WINDOW_SIZE = 5 # Fallback if import fails


logger_perf_optimizer = logging.getLogger(__name__) # Use standard module logger name

# Default thresholds if not specified in config (can be tuned)
DEFAULT_OPTIMIZATION_THRESHOLDS_S = {
    "perception": 0.05,
    "prediction": 0.06,
    "attention": 0.03,
    "workspace": 0.04,
    "experience_integration": 0.05,
    "consciousness_assessment": 0.02,
    "meta_cognition": 0.03,
    "loop_detection": 0.01,
    "planning": 0.10,
    "action_selection": 0.01,
    "execution": 0.05,
    "model_updates": 0.04,
    "narrative": 0.03,
    "optimization": 0.01, # Self-monitoring
}

# Health thresholds
HEALTH_CRITICAL_THRESHOLD = 0.3
HEALTH_WARNING_THRESHOLD = 0.5

# Config persistence path name is now derived from config.toml
# CONFIG_ADJUSTMENTS_FILENAME = "perf_adjustments.json" # Not used directly anymore


class PerformanceOptimizer(CognitiveComponent):
    """Analyzes cycle performance and suggests/applies optimizations."""

    def __init__(self):
        # Use deque for efficient fixed-size history
        self.cycle_history: Deque[Dict[str, float]] = deque(maxlen=100)
        self.max_history: int = 100
        self.optimization_thresholds: Dict[str, float] = DEFAULT_OPTIMIZATION_THRESHOLDS_S.copy()
        self.config_changes: Dict[str, Any] = {} # Track applied adjustments
        self._controller: Optional[Any] = None
        self._config_main_performance_section: Dict[str, Any] = {} # Stores the main [performance] section from global config
        self._config_perf_optimizer_section: Dict[str, Any] = {} # Stores the [performance_optimizer] section
        self._target_cycle_time: float = 0.1 # Default, read from config
        self._adjustments_path: Optional[Path] = None
        # --- Store RecoveryMode enum reference ---
        self._RecoveryModeEnum = globals().get('RecoveryMode')


    async def initialize(self, config: Dict[str, Any], controller: Any) -> bool:
        """Initialize with configuration and load previous adjustments."""
        self._controller = controller
        self._config_main_performance_section = config.get("performance", {}) # Get performance section
        self._config_perf_optimizer_section = config.get("performance_optimizer", {}) # Specific config for this module

        self.max_history = self._config_perf_optimizer_section.get("history_size", 100)
        self.cycle_history = deque(maxlen=self.max_history) # Recreate deque with correct size

        thresholds_from_config = self._config_perf_optimizer_section.get("cycle_thresholds_s", {})
        if isinstance(thresholds_from_config, dict):
            self.optimization_thresholds.update(thresholds_from_config)
        else:
            logger_perf_optimizer.warning("cycle_thresholds_s in config is not a dictionary. Using defaults.")

        self._target_cycle_time = self._config_main_performance_section.get("target_cycle_time", 0.1)

        # --- Path Configuration using agent_root_path ---
        path_str_from_config = None
        if controller and hasattr(controller, 'agent_root_path'):
            agent_root = controller.agent_root_path
            agent_data_paths_config = config.get("agent_data_paths", {})
            # Get the relative path string from [agent_data_paths]
            path_str_from_config = agent_data_paths_config.get("performance_adjustments_path")

            if path_str_from_config:
                # If path_str_from_config is absolute, Path() handles it.
                self._adjustments_path = (Path(agent_root) / path_str_from_config).resolve()
                logger_perf_optimizer.info(f"PerformanceOptimizer adjustments path set to: {self._adjustments_path}")
                self._load_persisted_changes() # Uses self._adjustments_path
            else:
                logger_perf_optimizer.info("performance_adjustments_path not specified in [agent_data_paths]. Persisted adjustments disabled.")
                self._adjustments_path = None
        else:
            logger_perf_optimizer.error("PerformanceOptimizer: Controller or agent_root_path not available. Cannot determine adjustments path. Persistence disabled.")
            self._adjustments_path = None
        # --- End Path Configuration ---

        logger_perf_optimizer.info(
            f"PerformanceOptimizer initialized. Target cycle: {self._target_cycle_time}s. "
            f"History size: {self.max_history}."
        )
        logger_perf_optimizer.debug(f"Cycle thresholds: {self.optimization_thresholds}")
        if not self._RecoveryModeEnum:
            logger_perf_optimizer.warning("RecoveryMode enum not available. Recovery suggestions will be limited.")
        return True

    def _load_persisted_changes(self):
        """Load previously applied performance adjustments."""
        if not (self._adjustments_path and self._adjustments_path.exists()):
            logger_perf_optimizer.info("No persisted performance adjustments found or path not set.")
            self.config_changes = {} # Ensure it's an empty dict if no file
            return

        logger_perf_optimizer.info(f"Attempting to load persisted performance adjustments from {self._adjustments_path}")
        try:
            with open(self._adjustments_path, "r") as f:
                loaded_changes = json.load(f)
            if isinstance(loaded_changes, dict):
                self.config_changes = loaded_changes
                logger_perf_optimizer.info(f"Loaded {len(self.config_changes)} persisted performance adjustments.")
            else:
                logger_perf_optimizer.warning(f"Persisted adjustments file {self._adjustments_path} did not contain a dictionary. Ignoring.")
                self.config_changes = {}
        except json.JSONDecodeError:
            logger_perf_optimizer.error(f"Failed to decode JSON from {self._adjustments_path}. Ignoring persisted adjustments.")
            self.config_changes = {}
        except Exception as e:
            logger_perf_optimizer.error(f"Error loading persisted adjustments: {e}")
            self.config_changes = {}


    async def process(self, input_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Analyzes the cycle profile provided in the input state.
        Input state should contain {'cycle_profile': Dict[str, float]}.
        Returns optimization hints or None.
        """
        if not input_state or "cycle_profile" not in input_state:
            logger_perf_optimizer.debug("PerformanceOptimizer: No cycle_profile in input_state.")
            return None

        cycle_profile = input_state["cycle_profile"]
        if not isinstance(cycle_profile, dict):
            logger_perf_optimizer.warning("PerformanceOptimizer: Invalid cycle_profile type.")
            return None

        self.cycle_history.append(cycle_profile.copy())
        current_health_score = self._assess_health()

        optimizations: Dict[str, Any] = {
            "adjustments_applied_this_cycle": {},
            "suggested_adjustments": {},
            "identified_bottlenecks": [],
            "current_health_score": current_health_score,
            "recovery_mode_needed": None,
            "timestamp": time.time()
        }

        for component, duration in cycle_profile.items():
            if not isinstance(duration, (int,float)): continue
            base_threshold = self.optimization_thresholds.get(component, self._target_cycle_time * 0.1)

            if duration > base_threshold:
                severity = duration / base_threshold if base_threshold > 0 else float('inf')
                optimizations["identified_bottlenecks"].append({
                    "component": component,
                    "duration_s": round(duration, 5),
                    "threshold_s": round(base_threshold, 5),
                    "severity": round(severity, 2)
                })

        optimizations["identified_bottlenecks"].sort(key=lambda x: x["severity"], reverse=True)
        # auto_apply_adjustments from [performance_optimizer] section now
        auto_apply_adjustments = self._config_perf_optimizer_section.get("auto_apply_adjustments", False)

        if optimizations["identified_bottlenecks"]:
            suggested_adjustments = self._generate_adjustments(optimizations["identified_bottlenecks"])
            optimizations["suggested_adjustments"] = suggested_adjustments
            if auto_apply_adjustments and suggested_adjustments:
                 applied = self._apply_and_persist_adjustments(suggested_adjustments)
                 optimizations["adjustments_applied_this_cycle"] = applied

        suggested_recovery_enum_member = self._suggest_recovery_mode(current_health_score)

        if self._RecoveryModeEnum and isinstance(suggested_recovery_enum_member, self._RecoveryModeEnum):
            optimizations["recovery_mode_needed"] = suggested_recovery_enum_member
            logger_perf_optimizer.warning(
                f"HEALTH SCORE: {current_health_score:.3f}. Suggested recovery: {suggested_recovery_enum_member.name}"
            )
        elif suggested_recovery_enum_member is not None:
            optimizations["recovery_mode_needed"] = str(suggested_recovery_enum_member)
            logger_perf_optimizer.error(
                f"HEALTH SCORE: {current_health_score:.3f}. _suggest_recovery_mode returned non-enum type: {str(suggested_recovery_enum_member)} (Type: {type(suggested_recovery_enum_member)})"
            )
        elif current_health_score < HEALTH_CRITICAL_THRESHOLD:
             logger_perf_optimizer.warning(
                f"CRITICAL HEALTH: Score {current_health_score:.3f}, but no specific recovery mode suggested by _suggest_recovery_mode."
            )
        elif current_health_score < HEALTH_WARNING_THRESHOLD:
             logger_perf_optimizer.warning(f"WARNING HEALTH: Score {current_health_score:.3f}.")

        recovery_log_name = "None"
        if self._RecoveryModeEnum and isinstance(optimizations["recovery_mode_needed"], self._RecoveryModeEnum) :
            recovery_log_name = optimizations["recovery_mode_needed"].name # type: ignore
        elif optimizations["recovery_mode_needed"] is not None:
             recovery_log_name = str(optimizations["recovery_mode_needed"])

        logger_perf_optimizer.info(f"Perf Opt: Health={current_health_score:.3f}, Bottlenecks={len(optimizations['identified_bottlenecks'])}, Recovery={recovery_log_name}")

        return {"performance_analysis": optimizations}


    def _assess_health(self) -> float:
        """Assess overall cycle health (0.0 to 1.0) based on recent cycle times."""
        if not self.cycle_history:
            return 1.0

        total_times = []
        for cycle_prof_hist_entry in self.cycle_history:
            if isinstance(cycle_prof_hist_entry, dict):
                 try: total_times.append(sum(v for v in cycle_prof_hist_entry.values() if isinstance(v, (int, float))))
                 except TypeError: pass

        if not total_times: return 1.0

        avg_total_time = sum(total_times) / len(total_times)
        target_time = self._target_cycle_time if self._target_cycle_time > 0 else 0.1
        health = target_time / avg_total_time if avg_total_time > 0 else 1.0
        health = max(0.0, min(1.0, health))
        return health


    def _suggest_recovery_mode(self, health_score: float) -> Optional['RecoveryMode']: # type: ignore
        """Suggest appropriate recovery mode based on health score."""
        if not self._RecoveryModeEnum: return None

        if health_score < 0.1:
            return self._RecoveryModeEnum.HARD_RESET
        elif health_score < 0.2:
            return self._RecoveryModeEnum.MEDIUM_RESET
        elif health_score < HEALTH_CRITICAL_THRESHOLD:
            return self._RecoveryModeEnum.SOFT_RESET
        else:
            return None


    def _generate_adjustments(self, bottlenecks: List[Dict]) -> Dict[str, Any]:
        adjustments: Dict[str, Any] = {}
        logger_perf_optimizer.info(f"PO GenAdjust: START. Bottlenecks: {bottlenecks}")

        for bottleneck in bottlenecks:
            component_name_from_profile = bottleneck.get("component")
            severity = bottleneck.get("severity", 1.0)
            logger_perf_optimizer.info(f"PO GenAdjust: Eval Bottleneck: Comp='{component_name_from_profile}', Sev={severity:.2f}")

            if not component_name_from_profile:
                logger_perf_optimizer.warning("PO GenAdjust: Bottleneck entry missing 'component' name.")
                continue

            component_key_for_config = component_name_from_profile # e.g., "planning"

            if component_key_for_config == "planning":
                logger_perf_optimizer.info(f"PO GenAdjust: Processing 'planning' specific rules.")

                # --- MODIFIED: Get default depth from controller's live config ---
                planner_default_depth_from_main_config = 5 # Fallback default
                live_agent_config_performance_section = None
                if self._controller and hasattr(self._controller, 'config') and \
                   isinstance(self._controller.config, dict): # type: ignore
                    live_agent_config_performance_section = self._controller.config.get("performance", {}) # type: ignore
                    if isinstance(live_agent_config_performance_section, dict):
                        planner_default_depth_from_main_config = live_agent_config_performance_section.get("max_planning_depth", 5)
                    else:
                        logger_perf_optimizer.warning("PO GenAdjust: controller.config['performance'] is not a dict. Using fallback default for max_planning_depth.")
                else:
                    logger_perf_optimizer.warning("PO GenAdjust: Controller or controller.config not available. Using fallback default for max_planning_depth.")
                # --- END MODIFIED ---

                logger_perf_optimizer.info(f"PO GenAdjust: 'planning'. MainConfig 'max_planning_depth' (from controller.config): {planner_default_depth_from_main_config}")
                
                # --- DEFINE target_section and target_key HERE ---
                target_section = "performance" 
                target_key = "max_planning_depth"
                # --- END DEFINE ---
                
                po_tracked_changes_for_performance_section = self.config_changes.get("performance", {}) 
                if not isinstance(po_tracked_changes_for_performance_section, dict):
                    po_tracked_changes_for_performance_section = {}
                    logger_perf_optimizer.debug(f"PO GenAdjust: No prior changes tracked for 'performance' section or not a dict, initialized to empty dict.")
                else:
                    logger_perf_optimizer.debug(f"PO GenAdjust: Prior changes for 'performance' section: {po_tracked_changes_for_performance_section}")


                current_max_depth_from_po_changes = po_tracked_changes_for_performance_section.get("max_planning_depth")
                current_max_depth = current_max_depth_from_po_changes if current_max_depth_from_po_changes is not None else planner_default_depth_from_main_config

                logger_perf_optimizer.info(
                    f"PO GenAdjust 'planning': POTrackedForPerformance['max_planning_depth']={current_max_depth_from_po_changes}, "
                    f"MainConfigDefault={planner_default_depth_from_main_config}, Derived current_max_depth={current_max_depth}"
                )

                new_depth = float(current_max_depth) 
                rule_applied_msg = "No depth adjustment rule met"

                sev_float = float(severity)
                cmd_float = float(current_max_depth)

                if sev_float > 3.0 and cmd_float > 4.0: 
                    new_depth = float(max(4, int(cmd_float) - 2)) 
                    rule_applied_msg = f"Rule1 (sev>3,depth>4) -> new_depth={new_depth}"
                elif sev_float > 1.5 and cmd_float > 6.0:
                     new_depth = float(max(6, int(cmd_float) - 1))
                     rule_applied_msg = f"Rule2 (sev>1.5,depth>6) -> new_depth={new_depth}"

                logger_perf_optimizer.info(f"PO GenAdjust 'planning': {rule_applied_msg}. CurrentMaxDepth={cmd_float}, NewDepth={new_depth}")

                if abs(new_depth - cmd_float) > 1e-9 : 
                    adjustments.setdefault(target_section, {})[target_key] = int(new_depth) 
                    logger_perf_optimizer.info(f"PO GenAdjust: SUGGESTING '{target_section}.{target_key}': {cmd_float} -> {int(new_depth)}")
                else:
                    logger_perf_optimizer.debug(f"PO GenAdjust: NO CHANGE for 'planning' affecting '{target_section}.{target_key}'. Current={cmd_float}, New={new_depth}")
            
            elif component_key_for_config == "attention": 
                logger_perf_optimizer.info(f"PO GenAdjust: Processing 'attention' specific rules.")

                attn_config_section_in_agent_config = {}
                if self._controller and hasattr(self._controller, 'config') and \
                   isinstance(self._controller.config, dict): # type: ignore
                    attn_config_section_in_agent_config = self._controller.config.get("attention_controller", {}) # type: ignore
                
                default_max_cand_from_ac_config = AC_DEFAULT_MAX_CANDIDATES 
                if isinstance(attn_config_section_in_agent_config, dict):
                    default_max_cand_from_ac_config = attn_config_section_in_agent_config.get("max_candidates", AC_DEFAULT_MAX_CANDIDATES) 
                else:
                    logger_perf_optimizer.warning("PO GenAdjust: controller.config['attention_controller'] is not a dict. Using module default for max_candidates.")

                logger_perf_optimizer.info(f"PO GenAdjust 'attention': MainConfig 'max_candidates': {default_max_cand_from_ac_config}")

                po_tracked_changes_for_attn_ctrl = self.config_changes.get("attention_controller", {})
                if not isinstance(po_tracked_changes_for_attn_ctrl, dict):
                    po_tracked_changes_for_attn_ctrl = {}
                    logger_perf_optimizer.debug("PO GenAdjust: No prior changes tracked for 'attention_controller' or not dict, initialized empty.")
                
                current_max_cand_from_po_changes = po_tracked_changes_for_attn_ctrl.get("max_candidates")
                current_max_cand_effective = current_max_cand_from_po_changes if current_max_cand_from_po_changes is not None else default_max_cand_from_ac_config

                logger_perf_optimizer.info(
                    f"PO GenAdjust 'attention': POTracked['max_candidates']={current_max_cand_from_po_changes}, "
                    f"MainConfigDefault={default_max_cand_from_ac_config}, Derived current_max_cand_effective={current_max_cand_effective}"
                )

                new_max_cand = int(current_max_cand_effective) 
                rule_applied_msg_attn = "No max_candidates adjustment rule met"
                sev_float_attn = float(severity)

                min_allowable_max_candidates = 5 
                if sev_float_attn > 2.5 and new_max_cand > min_allowable_max_candidates + 10:
                    new_max_cand = max(min_allowable_max_candidates, new_max_cand - 10)
                    rule_applied_msg_attn = f"Rule1 (sev>2.5, current>{min_allowable_max_candidates + 10}) -> new_max_cand={new_max_cand}"
                elif sev_float_attn > 1.5 and new_max_cand > min_allowable_max_candidates + 5:
                    new_max_cand = max(min_allowable_max_candidates, new_max_cand - 5)
                    rule_applied_msg_attn = f"Rule2 (sev>1.5, current>{min_allowable_max_candidates + 5}) -> new_max_cand={new_max_cand}"
                
                logger_perf_optimizer.info(f"PO GenAdjust 'attention': {rule_applied_msg_attn}. CurrentMaxCand={current_max_cand_effective}, NewMaxCand={new_max_cand}")

                if new_max_cand != int(current_max_cand_effective):
                    adjustments.setdefault("attention_controller", {})["max_candidates"] = new_max_cand
                    logger_perf_optimizer.info(f"PO GenAdjust: SUGGESTING 'attention_controller.max_candidates': {current_max_cand_effective} -> {new_max_cand}")
                else:
                    logger_perf_optimizer.debug(f"PO GenAdjust: NO CHANGE for 'attention_controller.max_candidates'. Current={current_max_cand_effective}, New={new_max_cand}")

            elif component_key_for_config == "workspace": 
                logger_perf_optimizer.info(f"PO GenAdjust: Processing 'workspace' specific rules.")

                gwm_config_section_in_agent_config = {}
                if self._controller and hasattr(self._controller, 'config') and \
                   isinstance(self._controller.config, dict): # type: ignore
                    gwm_config_section_in_agent_config = self._controller.config.get("global_workspace", {}) # type: ignore
                
                default_capacity_from_gwm_config = GWM_DEFAULT_CAPACITY 
                if isinstance(gwm_config_section_in_agent_config, dict):
                    default_capacity_from_gwm_config = gwm_config_section_in_agent_config.get("capacity", GWM_DEFAULT_CAPACITY)
                else:
                    logger_perf_optimizer.warning("PO GenAdjust: controller.config['global_workspace'] is not a dict. Using fallback default for capacity.")

                logger_perf_optimizer.info(f"PO GenAdjust 'workspace': MainConfig 'capacity': {default_capacity_from_gwm_config}")

                po_tracked_changes_for_gwm = self.config_changes.get("global_workspace", {})
                if not isinstance(po_tracked_changes_for_gwm, dict):
                    po_tracked_changes_for_gwm = {}
                
                current_capacity_from_po_changes = po_tracked_changes_for_gwm.get("capacity")
                current_capacity_effective = current_capacity_from_po_changes if current_capacity_from_po_changes is not None else default_capacity_from_gwm_config

                logger_perf_optimizer.info(
                    f"PO GenAdjust 'workspace': POTracked['capacity']={current_capacity_from_po_changes}, "
                    f"MainConfigDefault={default_capacity_from_gwm_config}, Derived current_capacity_effective={current_capacity_effective}"
                )

                new_capacity = int(current_capacity_effective)
                rule_applied_msg_gwm = "No GWM capacity adjustment rule met"
                sev_float_gwm = float(severity)
                min_allowable_capacity = 3 

                if sev_float_gwm > 2.0 and new_capacity > min_allowable_capacity + 2:
                    new_capacity = max(min_allowable_capacity, new_capacity - 2)
                    rule_applied_msg_gwm = f"Rule1 (sev>2.0, current>{min_allowable_capacity+2}) -> new_capacity={new_capacity}"
                elif sev_float_gwm > 1.3 and new_capacity > min_allowable_capacity:
                    new_capacity = max(min_allowable_capacity, new_capacity - 1)
                    rule_applied_msg_gwm = f"Rule2 (sev>1.3, current>{min_allowable_capacity}) -> new_capacity={new_capacity}"

                logger_perf_optimizer.info(f"PO GenAdjust 'workspace': {rule_applied_msg_gwm}. CurrentCapacity={current_capacity_effective}, NewCapacity={new_capacity}")

                if new_capacity != int(current_capacity_effective):
                    adjustments.setdefault("global_workspace", {})["capacity"] = new_capacity
                    logger_perf_optimizer.info(f"PO GenAdjust: SUGGESTING 'global_workspace.capacity': {current_capacity_effective} -> {new_capacity}")
                else:
                    logger_perf_optimizer.debug(f"PO GenAdjust: NO CHANGE for 'global_workspace.capacity'. Current={current_capacity_effective}, New={new_capacity}")

            elif component_key_for_config == "loop_detection":
                logger_perf_optimizer.info(f"PO GenAdjust: Processing 'loop_detection' specific rules.")

                ld_config_section_in_agent_config = {}
                if self._controller and hasattr(self._controller, 'config') and \
                   isinstance(self._controller.config, dict): # type: ignore
                    ld_config_section_in_agent_config = self._controller.config.get("loop_detection", {}) # type: ignore
                
                default_window_size_from_ld_config = LD_DEFAULT_WINDOW_SIZE 
                if isinstance(ld_config_section_in_agent_config, dict):
                    default_window_size_from_ld_config = ld_config_section_in_agent_config.get("window_size", LD_DEFAULT_WINDOW_SIZE)
                else:
                    logger_perf_optimizer.warning("PO GenAdjust: controller.config['loop_detection'] is not a dict. Using module default for window_size.")

                logger_perf_optimizer.info(f"PO GenAdjust 'loop_detection': MainConfig 'window_size': {default_window_size_from_ld_config}")

                po_tracked_changes_for_ld = self.config_changes.get("loop_detection", {})
                if not isinstance(po_tracked_changes_for_ld, dict):
                    po_tracked_changes_for_ld = {}
                
                current_window_size_from_po_changes = po_tracked_changes_for_ld.get("window_size")
                current_window_size_effective = current_window_size_from_po_changes if current_window_size_from_po_changes is not None else default_window_size_from_ld_config

                logger_perf_optimizer.info(
                    f"PO GenAdjust 'loop_detection': POTracked['window_size']={current_window_size_from_po_changes}, "
                    f"MainConfigDefault={default_window_size_from_ld_config}, Derived current_window_size_effective={current_window_size_effective}"
                )

                new_window_size = int(current_window_size_effective)
                rule_applied_msg_ld = "No LoopDetector window_size adjustment rule met"
                sev_float_ld = float(severity)
                min_allowable_window_size = 2 

                if sev_float_ld > 2.0 and new_window_size > min_allowable_window_size + 2: # e.g. current > 4
                    new_window_size = max(min_allowable_window_size, new_window_size - 2)
                    rule_applied_msg_ld = f"Rule1 (sev>2.0, current>{min_allowable_window_size+2}) -> new_window_size={new_window_size}"
                elif sev_float_ld > 1.3 and new_window_size > min_allowable_window_size: # e.g. current > 2
                    new_window_size = max(min_allowable_window_size, new_window_size - 1)
                    rule_applied_msg_ld = f"Rule2 (sev>1.3, current>{min_allowable_window_size}) -> new_window_size={new_window_size}"

                logger_perf_optimizer.info(f"PO GenAdjust 'loop_detection': {rule_applied_msg_ld}. CurrentWindowSize={current_window_size_effective}, NewWindowSize={new_window_size}")

                if new_window_size != int(current_window_size_effective):
                    adjustments.setdefault("loop_detection", {})["window_size"] = new_window_size
                    logger_perf_optimizer.info(f"PO GenAdjust: SUGGESTING 'loop_detection.window_size': {current_window_size_effective} -> {new_window_size}")
                else:
                    logger_perf_optimizer.debug(f"PO GenAdjust: NO CHANGE for 'loop_detection.window_size'. Current={current_window_size_effective}, New={new_window_size}")


        logger_perf_optimizer.info(f"PO GenAdjust: FINAL Generated adjustments: {adjustments}")
        return adjustments

    def _apply_and_persist_adjustments(self, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """Merge adjustments into current config changes and save to disk."""
        applied_now : Dict[str, Any]= {}
        updated_config_map = False

        for component, changes in adjustments.items():
            if not isinstance(changes, dict): continue

            if component not in self.config_changes:
                self.config_changes[component] = {}

            current_component_config_in_po = self.config_changes[component]
            newly_applied_for_this_component = {}

            for key, value in changes.items():
                if current_component_config_in_po.get(key) != value:
                    current_component_config_in_po[key] = value
                    newly_applied_for_this_component[key] = value
                    updated_config_map = True

            if newly_applied_for_this_component:
                applied_now[component] = newly_applied_for_this_component

        if updated_config_map and self._adjustments_path:
            logger_perf_optimizer.info(f"Applying and persisting performance adjustments: {applied_now}")
            try:
                 self._adjustments_path.parent.mkdir(parents=True, exist_ok=True)
                 temp_path = self._adjustments_path.with_suffix(".tmp")
                 with open(temp_path, "w") as f:
                     json.dump(self.config_changes, f, indent=2, sort_keys=True)
                 Path(temp_path).replace(self._adjustments_path)
                 logger_perf_optimizer.info(f"Performance adjustments saved to {self._adjustments_path}")
            except Exception as e:
                logger_perf_optimizer.error(f"Failed to persist performance adjustments: {e}")
                if 'temp_path' in locals() and Path(temp_path).exists():
                    try: Path(temp_path).unlink()
                    except OSError: pass
        elif updated_config_map:
             logger_perf_optimizer.warning("Performance adjustments updated in memory but persistence path not set or invalid.")

        if updated_config_map and self._controller:
             logger_perf_optimizer.info(
                 "NOTE: Performance adjustments recorded by PerformanceOptimizer. "
                 "Live components need to be reconfigured or poll these changes to take effect."
             )
        return applied_now


    async def reset(self) -> None:
        """Reset performance history and loaded adjustments."""
        self.cycle_history.clear()
        self.config_changes = {} # Clears in-memory adjustments
        # Should we delete the persisted file on reset? Current behavior is no.
        # If desired, add:
        # if self._adjustments_path and self._adjustments_path.exists():
        #     try: self._adjustments_path.unlink()
        #     except OSError as e: logger_perf_optimizer.error(f"Could not delete persisted adjustments on reset: {e}")
        logger_perf_optimizer.info("PerformanceOptimizer reset.")


    async def get_status(self) -> Dict[str, Any]:
        """Get current status including health and bottlenecks."""
        health = self._assess_health()
        bottlenecks = []
        if self.cycle_history:
             last_profile = self.cycle_history[-1]
             if isinstance(last_profile, dict):
                 for component, duration in last_profile.items():
                     if not isinstance(duration, (int,float)): continue
                     threshold = self.optimization_thresholds.get(component, self._target_cycle_time * 0.1)
                     if duration > threshold:
                         severity = duration / threshold if threshold > 0 else float('inf')
                         bottlenecks.append({
                             "component": component,
                             "duration_s": round(duration, 5),
                             "severity": round(severity, 2)
                         })
                 bottlenecks.sort(key=lambda x: x["severity"], reverse=True)

        return {
            "component": "PerformanceOptimizer",
            "status": "operational",
            "current_health_score": round(health, 3),
            "history_size": len(self.cycle_history),
            "max_history_size": self.max_history,
            "identified_bottlenecks_last_cycle": bottlenecks[:3], # Report top 3
            "active_config_adjustments": self.config_changes
        }

    async def shutdown(self) -> None:
        """Ensure final adjustments are persisted."""
        logger_perf_optimizer.info("PerformanceOptimizer shutting down.")
        # No explicit save needed here as _apply_and_persist_adjustments saves when changes are made.
        # If there were unsaved changes not yet "applied", they would be lost.
        # Current design persists on application of new adjustments.

# --- END OF FILE performance_optimizer.py ---