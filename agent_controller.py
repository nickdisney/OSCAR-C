# --- START OF FILE agent_controller.py ---


import asyncio
import logging
import time
import os
import signal
import toml
import math
import queue
import re
import json # Added import for json parsing
from pathlib import Path
from typing import Dict, Any, Optional, List, Deque, Type, Set, Counter, Tuple, Callable, Union # Added Tuple, Callable, Union
from collections import deque, Counter

# --- Safe psutil import ---
try:
    import psutil
    PSUTIL_AVAILABLE_CTRL = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE_CTRL = False
    logging.warning("agent_controller: psutil not found. System monitoring will be limited.")

# === Global Definitions for Fallbacks - MOVED TO BE ALWAYS DEFINED ===
class MockEnum(str):
    def __new__(cls, value, *args, **kwargs): # type: ignore
        return super().__new__(cls, value) # type: ignore
    def __getattr__(self, name: str) -> str:
        return f"DUMMY_{name}"
    @classmethod
    def _is_dummy(cls):
        return True

class MockDataclass:
    def __init__(self, *args, **kwargs): # type: ignore
        self._dummy_args = args
        self._dummy_kwargs = kwargs
    @classmethod
    def _is_dummy(cls):
        return True

class BaseCognitiveComponentPlaceholder:
    def __init__(self, *args, **kwargs): pass # type: ignore
    async def initialize(self, config: Dict[str, Any], controller: Any) -> bool: return False
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]: return {}
    async def get_status(self) -> Dict[str, Any]: return {"name": "BaseCognitiveComponentPlaceholder"}
    async def shutdown(self) -> None: pass
    @classmethod
    def _is_dummy(cls):
        return True
# === END ALWAYS DEFINED MOCKS ===


# === Standard Python Imports for Dependencies ===
try:
    # Correctly import AgentState from .agent_state
    from .agent_state import AgentState  # <<< IMPORT AgentState SEPARATELY AND FIRST
    print(f"DEBUG_AC_IMPORT: Successfully imported AgentState from .agent_state: {AgentState}")

    from .models.enums import ConsciousState, GoalStatus, RecoveryMode, ValueCategory # AgentState removed from here
    from .models.datatypes import Goal, Predicate, PhenomenalState, create_goal_from_descriptor, PainSource, ValueJudgment
    from .protocols import CognitiveComponent
    from .cognitive_modules.knowledge_base import KnowledgeBase
    from .cognitive_modules.cognitive_cache import CognitiveCache
    from .cognitive_modules.performance_optimizer import PerformanceOptimizer
    from .cognitive_modules.htn_planner import HTNPlanner
    from .cognitive_modules.attention_controller import AttentionController
    from .cognitive_modules.global_workspace_manager import GlobalWorkspaceManager
    from .cognitive_modules.experience_stream import ExperienceStream
    from .cognitive_modules.consciousness_level_assessor import ConsciousnessLevelAssessor
    from .cognitive_modules.meta_cognitive_monitor import MetaCognitiveMonitor
    from .cognitive_modules.loop_detector import LoopDetector
    from .cognitive_modules.error_recovery import ErrorRecoverySystem
    from .cognitive_modules.predictive_world_model import PredictiveWorldModel
    from .cognitive_modules.dynamic_self_model import DynamicSelfModel
    from .cognitive_modules.emergent_motivation_system import EmergentMotivationSystem
    from .cognitive_modules.narrative_constructor import NarrativeConstructor
    from .cognitive_modules.value_system import ValueSystem 
    from .cognitive_modules.state_history_logger import StateHistoryLogger 
    from .external_comms import call_ollama
    from .agent_helpers.internal_state_manager import InternalStateUpkeepManager
    from .agent_helpers.cognitive_trackers import PainEventTracker

    CORE_DEPENDENCIES_AVAILABLE = True
    _CognitiveComponentBase = CognitiveComponent 
    print("DEBUG_AC_IMPORT: CORE_DEPENDENCIES_AVAILABLE = True")


except ImportError as e:
    print(f"CRITICAL: Failed to import core dependencies for AgentController: {e}")
    CORE_DEPENDENCIES_AVAILABLE = False
    _CognitiveComponentBase = BaseCognitiveComponentPlaceholder 

    # Fallback assignments for types
    # AgentState should already be defined (either real or mock from its separate import try-block)
    # If the separate AgentState import above failed, it would have used its own mock.
    # Here, we only assign fallbacks for *other* types.
    if 'AgentState' not in locals() or not hasattr(AgentState, 'STOPPED'): # If it truly failed above
        print("CRITICAL_AC_IMPORT: AgentState was not successfully imported or mocked initially, defining fallback again.")
        AgentState = MockEnum("AgentState_ImportFail_CoreBlock") # type: ignore
        AgentState.STOPPED = AgentState("STOPPED_FALLBACK_CORE") # type: ignore
        AgentState.STARTING = AgentState("STARTING_FALLBACK_CORE") # type: ignore
        AgentState.RUNNING = AgentState("RUNNING_FALLBACK_CORE") # type: ignore
        AgentState.STOPPING = AgentState("STOPPING_FALLBACK_CORE") # type: ignore
        AgentState.ERROR = AgentState("ERROR_FALLBACK_CORE") # type: ignore
        AgentState.PAUSED = AgentState("PAUSED_FALLBACK_CORE") # Add PAUSED if AgentState is mocked


    ConsciousState = MockEnum("ConsciousState_ImportFail") 
    GoalStatus = MockEnum("GoalStatus_ImportFail") 
    RecoveryMode = MockEnum("RecoveryMode_ImportFail")
    ValueCategory = MockEnum("ValueCategory_ImportFail") # Fallback for ValueCategory
    Goal = MockDataclass 
    Predicate = MockDataclass 
    PhenomenalState = MockDataclass 
    PainSource = MockDataclass
    ValueJudgment = MockDataclass # Fallback for ValueJudgment
    def create_goal_from_descriptor(*args, **kwargs): return None 

    class KnowledgeBase(_CognitiveComponentBase): pass 
    class CognitiveCache(_CognitiveComponentBase): pass 
    class PerformanceOptimizer(_CognitiveComponentBase): 
        async def get_status(self): return {"active_config_adjustments": {}} 
    class HTNPlanner(_CognitiveComponentBase): pass 
    class AttentionController(_CognitiveComponentBase): 
        async def initialize(self, *args, **kwargs): logging.getLogger(__name__).warning("Using DUMMY Imported AttentionController.initialize"); return False 
        async def process(self, *args, **kwargs): logging.getLogger(__name__).warning("Using DUMMY Imported AttentionController.process"); return {"attention_weights": {}} 
    class GlobalWorkspaceManager(_CognitiveComponentBase): pass 
    class ExperienceStream(_CognitiveComponentBase): pass 
    class ConsciousnessLevelAssessor(_CognitiveComponentBase): pass 
    class MetaCognitiveMonitor(_CognitiveComponentBase): pass 
    class LoopDetector(_CognitiveComponentBase): pass 
    class ErrorRecoverySystem(_CognitiveComponentBase): pass 
    class PredictiveWorldModel(_CognitiveComponentBase): pass 
    class DynamicSelfModel(_CognitiveComponentBase): pass 
    class EmergentMotivationSystem(_CognitiveComponentBase): pass 
    class NarrativeConstructor(_CognitiveComponentBase): pass 
    if 'ValueSystem' not in locals(): 
        class ValueSystem(_CognitiveComponentBase): pass # type: ignore
    if 'StateHistoryLogger' not in locals(): 
        class StateHistoryLogger(_CognitiveComponentBase): pass # type: ignore
    if 'InternalStateUpkeepManager' not in locals():
        # logger_agent_controller.error("Failed to import InternalStateUpkeepManager. PHP updates will be broken.") # logger_agent_controller not defined yet
        logging.getLogger(__name__).error("Failed to import InternalStateUpkeepManager. PHP updates will be broken.")

        class InternalStateUpkeepManager: # Dummy for type hinting
            def __init__(self, ac): pass # type: ignore
            async def perform_upkeep_cycle_start(self): pass
            async def perform_upkeep_post_action_learning(self, *args, **kwargs): pass # type: ignore
            def generate_pain_from_goal_failure(self, *args, **kwargs): pass # type: ignore
            def check_existential_thresholds(self): return False

    if 'PainEventTracker' not in locals(): # Fallback for PainEventTracker
        logging.getLogger(__name__).error("Failed to import PainEventTracker.")
        class PainEventTracker:
             def __init__(self, *args, **kwargs): pass # type: ignore
             def update_pain_event_in_gwm(self, *args, **kwargs): pass # type: ignore
             def get_pain_ids_in_last_gwm(self, *args, **kwargs): return [] # type: ignore
             def clear_gwm_pain_ids(self, *args, **kwargs): pass # type: ignore
    
    if 'call_ollama' not in locals(): call_ollama = None 
# === End Imports ===

logger_agent_controller = logging.getLogger(__name__)

# --- Default Goal Descriptions and Priorities ---
DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC = "Observe and learn from the environment"
DEFAULT_OBSERVE_GOAL_PRIORITY = 1.0  # Priority for the default "Observe and learn" goal
USER_GOAL_PRIORITY = 5.0             # Higher priority for goals derived from user input


COMPONENT_INIT_ORDER_CTRL = [
    "knowledge_base", "cache", 
    "value_system",  # ValueSystem early as it might influence planning/actions
    "htn_planner",
    "predictive_world_model",
    "attention_controller", "global_workspace", "experience_stream",
    "consciousness_assessor", 
    "dynamic_self_model", "emergent_motivation_system", 
    "loop_detector", 
    "state_history_logger", # <<< ADDED HERE (after core processing, before meta/optimization)
    "meta_cognition", "performance_optimizer", 
    "narrative_constructor", # Narrative often last to summarize cycle
    "error_recovery", # Error recovery last to catch issues from any prior component
]


component_classes: Dict[str, Optional[Type['CognitiveComponent']]] = { # type: ignore
    "knowledge_base": KnowledgeBase, "cache": CognitiveCache,
    "performance_optimizer": PerformanceOptimizer, "htn_planner": HTNPlanner,
    "attention_controller": AttentionController,
    "global_workspace": GlobalWorkspaceManager,
    "experience_stream": ExperienceStream, "consciousness_assessor": ConsciousnessLevelAssessor,
    "meta_cognition": MetaCognitiveMonitor, "loop_detector": LoopDetector,
    "predictive_world_model": PredictiveWorldModel, "dynamic_self_model": DynamicSelfModel,
    "emergent_motivation_system": EmergentMotivationSystem,
    "narrative_constructor": NarrativeConstructor,
    "value_system": ValueSystem, 
    "state_history_logger": StateHistoryLogger, 
    "error_recovery": ErrorRecoverySystem,
}
for name, CompClass_entry in component_classes.items():
    if CompClass_entry is not None and hasattr(CompClass_entry, '_is_dummy') and CompClass_entry._is_dummy() and not issubclass(CompClass_entry, _CognitiveComponentBase): # type: ignore
        logger_agent_controller.warning(f"Component class {name} ({CompClass_entry}) is a dummy but does not inherit from _CognitiveComponentBase. This might cause issues.")


class CycleProfiler:
    def __init__(self):
        self.profile_data: Dict[str, Deque[float]] = {}
        self.current_section: Optional[str] = None
        self.section_start: Optional[float] = None
        self.max_samples: int = 100

    def set_max_samples(self, max_samples: int):
        if max_samples > 0: self.max_samples = max_samples
        for section in self.profile_data:
            self.profile_data[section] = deque(self.profile_data[section], maxlen=self.max_samples)

    def start_section(self, section_name: str):
        if self.current_section: self.end_section()
        self.current_section = section_name
        self.section_start = time.monotonic()

    def end_section(self):
        if self.current_section and self.section_start is not None:
            duration = time.monotonic() - self.section_start
            if self.current_section not in self.profile_data:
                self.profile_data[self.current_section] = deque(maxlen=self.max_samples)
            self.profile_data[self.current_section].append(duration)
        self.current_section = None
        self.section_start = None

    def get_cycle_profile(self) -> Dict[str, float]:
        return {k: v[-1] for k, v in self.profile_data.items() if v}

    def get_average_profile(self) -> Dict[str, float]:
        return {k: sum(v)/len(v) for k, v in self.profile_data.items() if v}

    def reset(self):
        self.profile_data.clear()
        self.current_section = None
        self.section_start = None

class SecurityException(Exception):
    """Custom exception for security-related issues, e.g., path traversal."""
    pass

class AgentController:
    """OSCAR-C Phase 2 Agent Controller: Orchestrates the 12-step cognitive cycle."""

    def __init__(self, ui_queue: Optional[queue.Queue], model_name: str = "default_model", config_path: str = "config.toml"):
        logger_agent_controller.info("Initializing OSCAR-C AgentController...")

        self._AgentState = AgentState
        self._ConsciousState = ConsciousState
        self._GoalStatus = GoalStatus
        self._RecoveryMode = RecoveryMode
        self._ValueCategory = ValueCategory # Store ValueCategory enum reference
        self._Goal = Goal
        self._Predicate = Predicate
        self._PhenomenalState = PhenomenalState
        self._PainSource = PainSource # Store PainSource class reference
        self._ValueJudgment = ValueJudgment # Store ValueJudgment class reference
        self._create_goal_from_descriptor = create_goal_from_descriptor
        self._CognitiveComponentBase = _CognitiveComponentBase
        self._RecoveryModeEnum = RecoveryMode

        self.ui_queue = ui_queue if isinstance(ui_queue, queue.Queue) else queue.Queue()
        if not isinstance(ui_queue, queue.Queue): logger_agent_controller.warning("Invalid ui_queue provided.")

        self.model_name = model_name
        self.config_path = Path(config_path).resolve() # Ensure config_path is absolute
        self.config: Dict[str, Any] = self._load_config()

        # --- AGENT ROOT PATH ---
        self.agent_root_path = self.config_path.parent
        logger_agent_controller.info(f"Agent root path set to: {self.agent_root_path}")

        # --- CONFIGURATION VALUES ---
        agent_config_section = self.config.get("agent", {})
        agent_data_paths_config = self.config.get("agent_data_paths", {}) # For PID directory
        performance_config_section = self.config.get("performance", {})

        # PID File Path
        pid_file_name = agent_config_section.get("pid_file_name", "oscar_c.pid")
        # Default pid_directory to "." (agent_root_path) if not specified in config
        pid_dir_rel_path = agent_data_paths_config.get("pid_directory", ".")

        # If pid_dir_rel_path from config is absolute, Path will use it directly.
        # Otherwise, it's joined with agent_root_path.
        self.pid_file = (self.agent_root_path / pid_dir_rel_path / pid_file_name).resolve()
        logger_agent_controller.info(f"PID file will be at: {self.pid_file}")

        self.components: Dict[str, 'CognitiveComponent'] = {} # type: ignore

        self.active_goals: List['Goal'] = [] # type: ignore
        self.current_plan: Optional[List[Dict]] = None
        self.current_phenomenal_state: Optional['PhenomenalState'] = None # type: ignore
        self._plan_generated_for_goal_id: Optional[str] = None # Track which goal the current_plan is for
        self._active_goal_modification_hints: Optional[Dict[str, Any]] = None


        self.agent_state = self._AgentState.STOPPED if hasattr(self._AgentState, "STOPPED") else self._AgentState("STOPPED_FALLBACK") # type: ignore
        self.consciousness_level = self._ConsciousState.PRE_CONSCIOUS if hasattr(self._ConsciousState, "PRE_CONSCIOUS") else self._ConsciousState("PRE_CONSCIOUS_FALLBACK") # type: ignore
        self._prev_consciousness_level = self._ConsciousState.UNCONSCIOUS if hasattr(self._ConsciousState, "UNCONSCIOUS") else self._ConsciousState("UNCONSCIOUS_FALLBACK") # type: ignore

        self._last_action_executed: Optional[Dict] = None; self._last_action_result: Optional[Dict] = None
        self.last_prediction_error_for_attention: Optional[Dict[str, Any]] = None # For AttentionController

        self._is_running_flag = asyncio.Event(); self._main_loop_task: Optional[asyncio.Task] = None
        self._asyncio_loop: Optional[asyncio.AbstractEventLoop] = None
        self._user_input_queue: asyncio.Queue[str] = asyncio.Queue()
        self._offline_task_queue: asyncio.Queue[Tuple[asyncio.Future, Callable, Tuple, Dict, Optional[Callable]]] = asyncio.Queue()
        self._secondary_loop_task: Optional[asyncio.Task] = None
        self._pending_goal_mapping_tasks: Dict[str, Dict[str, Any]] = {}
        self._goal_mapping_task_id_counter: int = 0

        self._is_paused_event = asyncio.Event()
        self._is_paused_event.set()  # Start in unpaused state (event is set)

        self._auto_pause_cycle_target: Optional[int] = None
        self._auto_pause_enabled: bool = False
        self._cycles_ran_since_last_resume: int = 0 # To track cycles for auto-pause


        self.profiler = CycleProfiler()
        prof_hist_size = performance_config_section.get("profiler_history_size", 100)
        self.profiler.set_max_samples(prof_hist_size)

        self.ui_update_interval_s = agent_config_section.get("ui_meter_update_interval_s", 0.5)
        self._last_ui_meter_update_time = 0.0
        self.cycle_delay_s = performance_config_section.get("target_cycle_time", 0.1) # For sleep at end of loop


        self.cycle_count: int = 0
        self._last_default_observe_completion_cycle: int = 0 # Controls cooldown for default goal

        self.default_goal_cooldown_cycles = agent_config_section.get("default_goal_cooldown_cycles", 5)
        self.min_curiosity_for_observe = agent_config_section.get("min_curiosity_for_observe", 0.6)
        
        self._active_goal_planning_failure_count: Dict[str, int] = {} 
        self._max_planning_failures_before_goal_fail: int = agent_config_section.get("max_consecutive_planning_failures", 3)
        
        # --- Execution Failure Tracking ---
        self._active_goal_execution_failure_count: Dict[str, int] = {} # goal_id -> execution_failures
        self._max_execution_failures_per_goal: int = agent_config_section.get("max_execution_failures_per_goal", 3)


        # --- Internal State Metrics (Pain, Happiness, Purpose) ---
        # These are now primarily managed by InternalStateUpkeepManager
        self.agent_age_cycles: int = 0 
        self.pain_level: float = 0.0 
        self.baseline_pain_level: float = 0.0
        self.happiness_level: float = 5.0
        self.purpose_level: float = 5.0 
        self.active_pain_sources: List[PainSource] = [] 

        self.internal_state_manager = InternalStateUpkeepManager(self)
        self._pain_ids_in_last_gwm: List[str] = []


        # Pain Attention specific config (These specific values might be better placed within AttentionController or InternalStateManager if they directly use them)
        attention_config = self.config.get("attention_controller", {})
        self._pain_attention_distraction_factor: float = attention_config.get("pain_attention_distraction_factor", 0.2)
        self._pain_rumination_threshold_cycles: int = attention_config.get("pain_rumination_threshold_cycles", 10)
        
        # Use the imported PainEventTracker, not just a deque
        self.pain_event_tracker = PainEventTracker(
            pain_rumination_threshold=attention_config.get("pain_rumination_threshold_cycles", 3),
            rumination_window_multiplier=attention_config.get("pain_rumination_window_multiplier", 3),
            inactive_reset_cycles=attention_config.get("pain_inactive_reset_cycles", 10)
        )

        self._buffered_pre_action_state_for_pwm: Optional[Dict[str, Any]] = None 
        self.global_workspace_content: Dict[str, Any] = {} 
        
        self._initialize_components()
        logger_agent_controller.info("AgentController initialized.")

    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f: config_data = toml.load(f)
                logger_agent_controller.info(f"Loaded configuration from {self.config_path}"); return config_data
            except Exception as e: logger_agent_controller.exception(f"Failed to load config: {e}"); return {}
        else: logger_agent_controller.warning(f"Config file not found: {self.config_path}. Using empty."); return {}

    def _initialize_components(self):
        logger_agent_controller.info("Creating component instances...")
        for name in COMPONENT_INIT_ORDER_CTRL:
            ComponentClass = component_classes.get(name)
            if ComponentClass:
                try:
                    if not issubclass(ComponentClass, self._CognitiveComponentBase): # type: ignore
                        logger_agent_controller.error(f"Component class for '{name}' ({ComponentClass}) does not inherit from the expected base CognitiveComponent ({self._CognitiveComponentBase}). Skipping.")
                        continue
                    instance = ComponentClass()
                    self.components[name] = instance # type: ignore
                    setattr(self, name, instance)
                    logger_agent_controller.debug(f"Created component instance: {name} of type {type(instance)}")
                except Exception as e:
                    logger_agent_controller.exception(f"Failed to create instance for component '{name}' of type {ComponentClass}: {e}")
            else:
                 logger_agent_controller.warning(f"Component class for '{name}' not found in component_classes map or is None. Skipping instantiation.")

        num_created = len(self.components)
        logger_agent_controller.info(f"Finished component instantiation. Created {num_created} component instances.")
        if 'attention_controller' in self.components:
            logger_agent_controller.info(f"AttentionController type in self.components: {type(self.components['attention_controller'])}")
        else:
            logger_agent_controller.warning("AttentionController instance was NOT created or added to self.components.")


    def _log_to_ui(self, level: str, message: str):
        if hasattr(self, 'ui_queue') and self.ui_queue:
            try: self.ui_queue.put_nowait((f"log_{level.lower()}", str(message)))
            except queue.Full: logger_agent_controller.warning(f"UI queue full. Dropping log.")
            except Exception as e: logger_agent_controller.error(f"UI queue error: {e}", exc_info=False)

    def _update_ui_state(self, new_state: Any):
         current_state_val = self.agent_state.name if hasattr(self.agent_state, 'name') else str(self.agent_state)
         new_state_val = new_state.name if hasattr(new_state, 'name') else str(new_state)
         
         # Prevent transitioning from STOPPED directly to PAUSED, or from STOPPING to PAUSED
         if (self.agent_state == self._AgentState.STOPPED or self.agent_state == self._AgentState.STOPPING) and \
            new_state == self._AgentState.PAUSED:
             logger_agent_controller.debug(f"Ignored attempt to set state to PAUSED from {current_state_val}.")
             return

         if self.agent_state != new_state:
             logger_agent_controller.info(f"Agent state: {current_state_val} -> {new_state_val}")
             self.agent_state = new_state
             if hasattr(self, 'ui_queue') and self.ui_queue:
                 try: self.ui_queue.put_nowait(("state_update", new_state))
                 except queue.Full: logger_agent_controller.warning(f"UI queue full. Dropping state update.")
                 except Exception as e: logger_agent_controller.error(f"UI queue error: {e}", exc_info=False)

    def start(self):
        stopped_state = self._AgentState.STOPPED if hasattr(self._AgentState, "STOPPED") else self._AgentState("STOPPED_FALLBACK") # type: ignore
        if self.agent_state != stopped_state:
            logger_agent_controller.warning(f"Start ignored: state={self.agent_state}"); return

        logger_agent_controller.info("Starting OSCAR-C Agent...");
        starting_state = self._AgentState.STARTING if hasattr(self._AgentState, "STARTING") else self._AgentState("STARTING_FALLBACK") # type: ignore
        self._update_ui_state(starting_state)
        try: self._asyncio_loop = asyncio.get_running_loop(); logger_agent_controller.info("Using existing event loop.")
        except RuntimeError: self._asyncio_loop = asyncio.new_event_loop(); asyncio.set_event_loop(self._asyncio_loop); logger_agent_controller.info("Created new event loop.")
        self._add_signal_handlers()
        self._is_running_flag.set()
        self._is_paused_event.set() # Ensure it's not paused on start
        if self._secondary_loop_task is None or self._secondary_loop_task.done():
            self._secondary_loop_task = self._asyncio_loop.create_task(self._secondary_processing_loop())
            logger_agent_controller.info("Secondary processing loop task created.")
        self._main_loop_task = self._asyncio_loop.create_task(self._run_initialization_and_loop())
        if not self._asyncio_loop.is_running():
            logger_agent_controller.info("Starting event loop...");
            try: self._asyncio_loop.run_forever()
            except KeyboardInterrupt: logger_agent_controller.info("KeyboardInterrupt, stopping."); self.stop()
            finally: logger_agent_controller.info("Event loop finished."); self._cleanup();
        else: logger_agent_controller.info("Agent task scheduled on existing loop.")

    def stop(self, signum=None, frame=None):
        stopping_state = self._AgentState.STOPPING if hasattr(self._AgentState, "STOPPING") else self._AgentState("STOPPING_FALLBACK") # type: ignore
        stopped_state = self._AgentState.STOPPED if hasattr(self._AgentState, "STOPPED") else self._AgentState("STOPPED_FALLBACK") # type: ignore

        current_state_name = self.agent_state.name if hasattr(self.agent_state, 'name') else str(self.agent_state)
        if self.agent_state == stopping_state or self.agent_state == stopped_state:
            logger_agent_controller.debug(f"Stop ignored: Already {current_state_name}")
            return
        logger_agent_controller.info(f"Stop requested (Signal: {signum})...");
        self._update_ui_state(stopping_state)
        self._is_running_flag.clear()
        self._is_paused_event.set() # Ensure any paused loop can exit
        if self._main_loop_task and not self._main_loop_task.done():
             logger_agent_controller.info("Requesting main task cancellation...");
             self._main_loop_task.cancel()
        if self._secondary_loop_task and not self._secondary_loop_task.done():
            logger_agent_controller.info("Requesting secondary loop task cancellation...")
            self._secondary_loop_task.cancel()
        if self._asyncio_loop and self._asyncio_loop.is_running():
             try:
                 if asyncio.get_running_loop() == self._asyncio_loop: self._asyncio_loop.call_soon(self._asyncio_loop.stop)
                 else: self._asyncio_loop.call_soon_threadsafe(self._asyncio_loop.stop)
                 logger_agent_controller.info("Requested event loop stop.")
             except Exception as e: logger_agent_controller.error(f"Error scheduling loop stop: {e}")
        elif not (self._asyncio_loop and self._asyncio_loop.is_running()):
             logger_agent_controller.info("Loop not running, performing direct cleanup.")
             self._cleanup()

    async def _run_initialization_and_loop(self):
        initialized_components_logic = []
        error_state = self._AgentState.ERROR if hasattr(self._AgentState, "ERROR") else self._AgentState("ERROR_FALLBACK") # type: ignore
        running_state = self._AgentState.RUNNING if hasattr(self._AgentState, "RUNNING") else self._AgentState("RUNNING_FALLBACK") # type: ignore

        try:
            logger_agent_controller.info("Initializing component logic (AgentController._run_initialization_and_loop)...");
            init_logic_success = True

            if not self.components and COMPONENT_INIT_ORDER_CTRL:
                logger_agent_controller.error("No components were available in self.components. Aborting logic initialization.")
                init_logic_success = False

            for name in COMPONENT_INIT_ORDER_CTRL:
                if not init_logic_success: break

                if name in self.components:
                    component = self.components[name];
                    logger_agent_controller.debug(f"Running .initialize() for component: {name} of type {type(component)}...")
                    try:
                        init_method = getattr(component, 'initialize', None)
                        if init_method:
                             if asyncio.iscoroutinefunction(init_method): success = await init_method(self.config, self)
                             else: success = init_method(self.config, self)
                        else:
                            logger_agent_controller.debug(f"Component {name} has no specific initialize method for its logic.");
                            success = True

                        if not success:
                            logger_agent_controller.error(f"Component {name} .initialize() logic failed!");
                            init_logic_success = False
                        else:
                            initialized_components_logic.append(name);
                            logger_agent_controller.info(f"Component {name} .initialize() logic completed.")
                    except Exception as e:
                        logger_agent_controller.exception(f"Exception during {name} .initialize() logic: {e}");
                        init_logic_success = False
                elif name in component_classes:
                     logger_agent_controller.error(f"Component '{name}' in INIT_ORDER but instance not found in self.components. This indicates an issue in _initialize_components (instantiation).")
                     init_logic_success = False
                elif name not in component_classes:
                    logger_agent_controller.error(f"Component '{name}' in INIT_ORDER but not defined in component_classes map and no instance in self.components. Unknown component.")
                    init_logic_success = False

            if not init_logic_success:
                logger_agent_controller.error("Agent logic initialization failed."); self._log_to_ui("error", "Agent init failed.")
                self._update_ui_state(error_state);
                if self.pid_file.exists():
                    try: self.pid_file.unlink();
                    except OSError as e: logger_agent_controller.error(f"Err removing PID on init fail: {e}")
                return

            try:
                self.pid_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.pid_file, "w") as f: f.write(str(os.getpid())); logger_agent_controller.info(f"PID file created: {self.pid_file}")
            except Exception as e: logger_agent_controller.error(f"Failed to write PID file: {e}")

            self._log_to_ui("info", "Init complete. Starting cognitive cycle."); self._update_ui_state(running_state)
            await self._run_agent_loop()

        except asyncio.CancelledError: logger_agent_controller.info("Init/Loop task cancelled.")
        except Exception as e: logger_agent_controller.exception("Unhandled exception in init/loop"); self._update_ui_state(error_state)
        finally:
            logger_agent_controller.info("Shutting down components logic..."); await self._shutdown_components(initialized_components_logic)
            if self.pid_file.exists():
                try: self.pid_file.unlink(); logger_agent_controller.info("PID file removed.");
                except OSError as e: logger_agent_controller.error(f"Err removing PID during shutdown: {e}")

            stopped_state_final = self._AgentState.STOPPED if hasattr(self._AgentState, "STOPPED") else self._AgentState("STOPPED_FALLBACK") # type: ignore
            if self.agent_state != stopped_state_final:
                 self._update_ui_state(stopped_state_final)

    async def _shutdown_components(self, component_names: List[str]):
         logger_agent_controller.info(f"Shutting down logic for {len(component_names)} components...")
         for name in reversed(component_names):
             if name in self.components:
                 component = self.components[name]; logger_agent_controller.debug(f"Shutting down logic for: {name}...")
                 try:
                     shutdown_method = getattr(component, 'shutdown', None)
                     if shutdown_method:
                         if asyncio.iscoroutinefunction(shutdown_method): await shutdown_method()
                         else: shutdown_method()
                         logger_agent_controller.debug(f"{name} shutdown logic complete.")
                     else:
                         logger_agent_controller.debug(f"{name} has no shutdown logic method.")
                 except Exception as e: logger_agent_controller.exception(f"Err shutting down logic for {name}: {e}")
         logger_agent_controller.info("Component logic shutdown finished.")

    def _add_signal_handlers(self):
        if self._asyncio_loop:
            try:
                 stop_wrapper = lambda signum, frame: self.stop(signum, frame)
                 for sig in (signal.SIGINT, signal.SIGTERM):
                     try: self._asyncio_loop.add_signal_handler(sig, stop_wrapper, sig, None)
                     except (ValueError, OSError, RuntimeError, NotImplementedError) as e: logger_agent_controller.warning(f"Could not add signal handler for {sig}: {e}.")
                 logger_agent_controller.info("Signal handlers added where supported.")
            except Exception as e: logger_agent_controller.error(f"Error adding signals: {e}")

    def _remove_signal_handlers(self):
         if self._asyncio_loop and not self._asyncio_loop.is_closed():
             try:
                 for sig in (signal.SIGINT, signal.SIGTERM):
                     try: self._asyncio_loop.remove_signal_handler(sig);
                     except (ValueError, RuntimeError, NotImplementedError): pass
                 logger_agent_controller.info("Signal handlers removed.")
             except Exception as e: logger_agent_controller.error(f"Error removing signals: {e}")

    def _cleanup(self):
        stopped_state = self._AgentState.STOPPED if hasattr(self._AgentState, "STOPPED") else self._AgentState("STOPPED_FALLBACK") # type: ignore
        if self.agent_state == stopped_state: return
        logger_agent_controller.info("Cleaning up agent resources...")
        self._remove_signal_handlers()
        if self._asyncio_loop and not self._asyncio_loop.is_closed():
             if self._asyncio_loop.is_running():
                  try:
                     if asyncio.get_running_loop() == self._asyncio_loop: self._asyncio_loop.call_soon(self._asyncio_loop.stop)
                     else: self._asyncio_loop.call_soon_threadsafe(self._asyncio_loop.stop)
                     logger_agent_controller.info("Async loop stop scheduled during cleanup.")
                  except Exception as e: logger_agent_controller.error(f"Error scheduling loop stop during cleanup: {e}")
        self._asyncio_loop = None; self._main_loop_task = None; self._secondary_loop_task = None
        if self.pid_file.exists():
            try: self.pid_file.unlink(); logger_agent_controller.info("PID file removed during cleanup.");
            except OSError as e: logger_agent_controller.error(f"Err removing PID during cleanup: {e}")
        self._update_ui_state(stopped_state)

    def handle_user_input(self, text: str):
        if not self._asyncio_loop:
            logger_agent_controller.error("Cannot handle user input: Agent asyncio loop not available.")
            return
        if not self._is_running_flag.is_set():
             logger_agent_controller.warning("Ignoring user input: Agent is not running.")
             return
        try:
            self._asyncio_loop.call_soon_threadsafe(self._user_input_queue.put_nowait, text)
            logger_agent_controller.debug(f"User input queued: '{text[:50]}...'")
        except Exception as e:
            logger_agent_controller.error(f"Failed to queue user input: {e}", exc_info=True)

    def schedule_offline_task(
        self,
        coro_func: Callable[..., asyncio.Future], # The coroutine function to run
        *args: Any,                               # Positional arguments for coro_func
        callback_on_done: Optional[Callable[[Optional[Any], Optional[Exception]], None]] = None, # Sync callback
        **kwargs: Any                              # Keyword arguments for coro_func
    ) -> asyncio.Future:
        """
        Schedules a coroutine to be executed in the secondary processing loop.

        Args:
            coro_func: The awaitable coroutine function to execute.
            *args: Positional arguments to pass to coro_func.
            callback_on_done: An optional synchronous callback function that will be
                              invoked with (result, None) on success or (None, exception)
                              on failure. Called via call_soon_threadsafe.
            **kwargs: Keyword arguments to pass to coro_func.

        Returns:
            An asyncio.Future that can be awaited for the result of the coroutine.
        """
        if not self._asyncio_loop or self._asyncio_loop.is_closed():
            # This case should ideally not happen if agent is running.
            # If it does, create a completed future with an error.
            logger_agent_controller.error("Cannot schedule offline task: Agent main asyncio loop not available or closed.")
            future = asyncio.Future() # type: ignore[var-annotated]
            future.set_exception(RuntimeError("Agent main event loop is not running or closed."))
            return future

        # Create a future that the caller can await
        task_future = self._asyncio_loop.create_future()

        if not self._is_running_flag.is_set() and not self._offline_task_queue.empty():
            # If agent is stopping but queue still has items, it might process them.
            # If agent is stopping and queue is empty, or if fully stopped, don't add new.
             if self.agent_state == self._AgentState.STOPPING or self.agent_state == self._AgentState.STOPPED:
                logger_agent_controller.warning(
                    f"Agent is stopping/stopped. Offline task '{coro_func.__name__ if hasattr(coro_func, '__name__') else 'unknown_coro'}' "
                    f"will not be scheduled."
                )
                task_future.set_exception(RuntimeError("Agent is stopping, offline task not scheduled."))
                return task_future
        elif not self._is_running_flag.is_set():
             logger_agent_controller.warning(
                f"Agent is not running. Offline task '{coro_func.__name__ if hasattr(coro_func, '__name__') else 'unknown_coro'}' "
                f"will not be scheduled."
            )
             task_future.set_exception(RuntimeError("Agent not running, offline task not scheduled."))
             return task_future


        # Put the task details onto the queue for the secondary loop to pick up
        try:
            # Use call_soon_threadsafe if scheduling from a different thread,
            # but components calling this will likely be in the main agent's async context.
            # So, direct put or loop.call_soon should be fine.
            # If components are guaranteed to call this from the main event loop,
            # a direct self._offline_task_queue.put_nowait() could be used,
            # but call_soon is safer for general use if there's any doubt.
            self._asyncio_loop.call_soon_threadsafe(
                self._offline_task_queue.put_nowait,
                (task_future, coro_func, args, kwargs, callback_on_done)
            )
            logger_agent_controller.info(
                f"Scheduled offline task: {coro_func.__name__ if hasattr(coro_func, '__name__') else 'unknown_coro'}"
            )
        except Exception as e:
            logger_agent_controller.error(
                f"Failed to schedule offline task {coro_func.__name__ if hasattr(coro_func, '__name__') else 'unknown_coro'}: {e}",
                exc_info=True
            )
            task_future.set_exception(e) # Set exception on the future if scheduling fails

        return task_future

    # Add this helper method if it doesn't exist
    def _get_active_goal_type(self) -> str:
        """Extracts a general type/category from the active goal's description."""
        active_goal_obj = self._oscar_get_active_goal() # Existing helper
        if not active_goal_obj or not hasattr(active_goal_obj, 'description'):
            return "none"

        desc = active_goal_obj.description.lower()
        if "explore" in desc or "list" in desc: return "exploration"
        if "read" in desc or "analyze" in desc or "learn" in desc or "observe" in desc: return "information_gathering"
        if "write" in desc or "create" in desc: return "creation"
        if "respond" in desc or "explain" in desc or "status" in desc: return "communication"
        if "reflect" in desc: return "introspection"
        # Add more general categories as needed
        return "general_task"

    async def _gather_oscar_c_internal_state_for_persona(self) -> Dict[str, Any]:
        """
        Gathers a comprehensive snapshot of OSCAR-C's internal state
        for use with the persona LoRA.
        This should match the fields used in training data generation.
        """
        logger_agent_controller.debug("Gathering internal state for persona LoRA...")
        # Get P/H/P levels
        php_payload = {
            "pain_level": round(self.pain_level, 2),
            "happiness_level": round(self.happiness_level, 2),
            "purpose_level": round(self.purpose_level, 2),
        }

        # Agent age and CS level
        agent_info = {
            "agent_age_cycles": self.agent_age_cycles,
            "current_cs_level_name": self.consciousness_level.name if hasattr(self.consciousness_level, 'name') else str(self.consciousness_level),
        }

        # Active Goal
        active_goal_obj = self._oscar_get_active_goal()
        active_goal_details = {"description": "None", "priority": 0.0, "status": "None"}
        if active_goal_obj and hasattr(active_goal_obj, 'description') and hasattr(active_goal_obj, 'priority') and hasattr(active_goal_obj, 'status'):
            active_goal_details = {
                "description": str(active_goal_obj.description),
                "priority": float(active_goal_obj.priority),
                "status": active_goal_obj.status.name if hasattr(active_goal_obj.status, 'name') else str(active_goal_obj.status)
            }
        
        # Current Plan Summary
        plan_summary = "No active plan"
        if self.current_plan and isinstance(self.current_plan, list) and len(self.current_plan) > 0:
            next_action_in_plan = self.current_plan[0]
            action_type_in_plan = next_action_in_plan.get('type', 'Unknown action')
            plan_summary = f"Next action: {action_type_in_plan}, Steps remaining: {len(self.current_plan)}"
        elif self.current_plan == []: # Plan is empty, meaning goal might be near completion or just completed
            plan_summary = "Current plan has no more steps."

        # Drive Values
        drive_values = {}
        ems_component = getattr(self, "emergent_motivation_system", None)
        if ems_component and hasattr(ems_component, "get_drive_values"):
            try:
                drive_values = ems_component.get_drive_values() # Assuming this is a sync method or use get_status
            except Exception as e:
                logger_agent_controller.warning(f"Could not get drive values from EMS for persona state: {e}")
        
        # Last 1-2 Narrative Entry Summaries
        narrative_summaries = {"last_narrative_entry_1_summary": "No recent narrative entries.", 
                               "last_narrative_entry_2_summary": ""}
        nc_component = getattr(self, "narrative_constructor", None)
        if nc_component and hasattr(nc_component, "narrative") and nc_component.narrative:
            narr_list = list(nc_component.narrative)
            if len(narr_list) > 0:
                narrative_summaries["last_narrative_entry_1_summary"] = str(narr_list[-1].content)[:150] + "..."
            if len(narr_list) > 1:
                narrative_summaries["last_narrative_entry_2_summary"] = str(narr_list[-2].content)[:150] + "..."

        # Last Action Details
        last_action_details = {"type": "None", "outcome": "None", "error": "None"}
        if self._last_action_executed and isinstance(self._last_action_executed, dict):
            last_action_details["type"] = self._last_action_executed.get("type", "Unknown")
        if self._last_action_result and isinstance(self._last_action_result, dict):
            last_action_details["outcome"] = self._last_action_result.get("outcome", "Unknown")
            last_action_details["error"] = self._last_action_result.get("error", "None")
            if not last_action_details["error"]: # Ensure "None" if empty or truly None
                 last_action_details["error"] = "None"


        # Last Prediction Error Summary
        prediction_error_summary = {"type": "None", "details": "None"}
        # Assuming self.last_prediction_error_for_attention holds the relevant structure
        if self.last_prediction_error_for_attention and isinstance(self.last_prediction_error_for_attention, dict):
            prediction_error_summary["type"] = self.last_prediction_error_for_attention.get("type", "Unknown")
            details_obj = self.last_prediction_error_for_attention.get("error_source_details", 
                                 self.last_prediction_error_for_attention.get("details", "No specific details"))
            if isinstance(details_obj, dict):
                 prediction_error_summary["details"] = str(details_obj)[:150] + "..."
            else:
                 prediction_error_summary["details"] = str(details_obj)[:150]
            if not prediction_error_summary["details"]: prediction_error_summary["details"] = "None"


        # Simplified flags for ValueSystem/MCM issues (placeholders, expand as needed)
        # These would ideally come from the status of ValueSystem and MCM components
        value_system_summary = "No recent value conflicts noted."
        mcm_summary = "No significant cognitive issues detected recently."
        
        vs_comp = getattr(self, "value_system", None)
        if vs_comp and hasattr(vs_comp, "get_status"):
            try:
                vs_status = await vs_comp.get_status()
                if vs_status.get("recent_critical_judgments_count", 0) > 0:
                    value_system_summary = f"ValueSystem noted {vs_status['recent_critical_judgments_count']} critical judgment(s) recently."
                elif vs_status.get("recent_judgments_count",0) > 0 :
                     value_system_summary = f"ValueSystem made {vs_status['recent_judgments_count']} judgment(s) recently."
            except Exception as e:
                logger_agent_controller.warning(f"Could not get ValueSystem status for persona: {e}")

        mcm_comp = getattr(self, "meta_cognition", None)
        if mcm_comp and hasattr(mcm_comp, "get_status"):
            try:
                mcm_status = await mcm_comp.get_status()
                if mcm_status.get("issues_detected") and len(mcm_status["issues_detected"]) > 0:
                    mcm_summary = f"MCM detected issue(s): {str(mcm_status['issues_detected'][0])[:100]}..."
            except Exception as e:
                logger_agent_controller.warning(f"Could not get MCM status for persona: {e}")

        # Emotion Tag (example, needs actual generation logic if not directly in AgentController state)
        # This could be derived from P/H/P or other factors, similar to generate_state_snapshots.py
        emotion_tag = "neutral" # Placeholder
        if php_payload["pain_level"] > 7.0 and php_payload["happiness_level"] < 3.0: emotion_tag = "distressed"
        elif php_payload["pain_level"] < 2.0 and php_payload["happiness_level"] > 7.0: emotion_tag = "content"
        elif php_payload["purpose_level"] < 2.5: emotion_tag = "aimless"
        elif php_payload["purpose_level"] > 7.5 and php_payload["happiness_level"] > 6.0 : emotion_tag = "engaged"
        elif drive_values.get("curiosity", 0) > 0.75 : emotion_tag = "curious"


        state_snapshot = {
            **php_payload,
            **agent_info,
            "active_goal_description": active_goal_details["description"],
            "active_goal_priority": active_goal_details["priority"],
            "active_goal_status": active_goal_details["status"],
            "current_plan_summary": plan_summary,
            **{f"drive_{k}_value": v for k, v in drive_values.items()}, # Flatten drives
            **narrative_summaries,
            **last_action_details,
            "last_prediction_error_type": prediction_error_summary["type"],
            "last_prediction_error_details": prediction_error_summary["details"],
            "recent_value_conflict_summary": value_system_summary,
            "recent_mcm_issue_detected": mcm_summary,
            "emotion_tag": emotion_tag,
        }
        logger_agent_controller.debug(f"Gathered persona state snapshot: {json.dumps(state_snapshot, indent=2)[:500]}...")
        return state_snapshot

    async def _get_value_system_action_context(self, action_to_evaluate: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Constructs the comprehensive context dictionary required by 
        ValueSystem.evaluate_action_consequences as per MDP C.1.3.
        """
        base_cognitive_state = await self._oscar_get_cognitive_state() 

        vs_context: Dict[str, Any] = {
            "timestamp": base_cognitive_state.get("timestamp", time.time()),
            "php_levels": {
                "pain": base_cognitive_state.get("pain_level", 0.0),
                "happiness": base_cognitive_state.get("happiness_level", 5.0),
                "purpose": base_cognitive_state.get("purpose_level", 5.0),
            },
            "current_cs_level_name": base_cognitive_state.get("consciousness_level", "UNKNOWN"),
        }

        # 1. active_pain_sources_summary
        vs_context["active_pain_sources_summary"] = []
        if hasattr(self, 'active_pain_sources') and isinstance(self.active_pain_sources, list):
            _PainSource_ac_ctx = globals().get("PainSource")
            if _PainSource_ac_ctx: 
                for ps in self.active_pain_sources:
                    if isinstance(ps, _PainSource_ac_ctx) and hasattr(ps, 'id') and \
                       hasattr(ps, 'current_intensity') and hasattr(ps, 'type') and \
                       hasattr(ps, 'is_resolved') and not ps.is_resolved:
                        vs_context["active_pain_sources_summary"].append({
                            "id": ps.id,
                            "intensity": round(float(ps.current_intensity), 2), 
                            "type": str(ps.type)
                        })

        # 2. dsm_summary
        dsm_summary_data = {}
        dsm_component = getattr(self, "dynamic_self_model", None)
        if dsm_component and hasattr(dsm_component, "get_status"):
            try:
                get_status_method_dsm = getattr(dsm_component, "get_status")
                if asyncio.iscoroutinefunction(get_status_method_dsm):
                    dsm_summary_data = await get_status_method_dsm()
                else: 
                    dsm_summary_data = get_status_method_dsm()
            except Exception as e_dsm_status:
                logger_agent_controller.error(f"Error getting DSM status for VS context: {e_dsm_status}")
        vs_context["dsm_summary"] = dsm_summary_data if isinstance(dsm_summary_data, dict) else {}


        # 3. pwm_prediction_for_action 
        vs_context["pwm_prediction_for_action"] = {} 
        if action_to_evaluate: 
            pwm_component = getattr(self, "predictive_world_model", None)
            if pwm_component and hasattr(pwm_component, "process"):
                try:
                    
                    pwm_predict_context_for_vs = {
                        'action_to_execute': action_to_evaluate,
                        'context': {
                            'consciousness_level_name': vs_context["current_cs_level_name"],
                            'active_goal_type': base_cognitive_state.get("current_goal_type", self._get_active_goal_type()), 
                        }
                    }
                    pwm_process_method = getattr(pwm_component, "process")
                    
                    pred_output_for_vs = await pwm_process_method({"predict_request": pwm_predict_context_for_vs})
                    if isinstance(pred_output_for_vs, dict) and "prediction" in pred_output_for_vs:
                        vs_context["pwm_prediction_for_action"] = pred_output_for_vs["prediction"]
                        logger_agent_controller.debug(f"VS Context: Added PWM prediction for action '{action_to_evaluate.get('type')}'.")
                    else:
                        logger_agent_controller.debug(f"VS Context: PWM process did not return expected prediction structure for action '{action_to_evaluate.get('type')}'.")
                except Exception as e_pwm_predict_vs:
                    logger_agent_controller.error(f"Error getting PWM prediction for VS context: {e_pwm_predict_vs}")
        
        # 4. active_goal_details
        active_goal_obj = self._oscar_get_active_goal() 
        vs_context["active_goal_details"] = {"description": "None", "priority": 0.0}
        if active_goal_obj and hasattr(active_goal_obj, 'description') and hasattr(active_goal_obj, 'priority'):
            vs_context["active_goal_details"] = {
                "description": str(active_goal_obj.description),
                "priority": float(active_goal_obj.priority)
            }

        if hasattr(self, '_raw_percepts') and isinstance(self._raw_percepts, dict) and "system_state" in self._raw_percepts:
            if isinstance(self._raw_percepts["system_state"], dict): 
                vs_context["system_resources"] = self._raw_percepts["system_state"]
        
        return vs_context

    async def _secondary_processing_loop(self):
        """
        Continuously processes tasks from the _offline_task_queue.
        Tasks are expected to be awaitable coroutines.
        """
        logger_agent_controller.info("Secondary processing loop started.")
        while self._is_running_flag.is_set() or not self._offline_task_queue.empty():
            try:
                # Wait for a task with a timeout to allow the loop to exit if agent is stopping
                future_obj, coro_func, args, kwargs, callback_on_done = await asyncio.wait_for(
                    self._offline_task_queue.get(), timeout=0.5
                )
                logger_agent_controller.info(f"SecondaryLoop: Picked up task: {coro_func.__name__ if hasattr(coro_func, '__name__') else 'unknown_coro'}")
            except asyncio.TimeoutError:
                if not self._is_running_flag.is_set() and self._offline_task_queue.empty():
                    break # Exit if agent stopped and queue is empty
                continue # Continue waiting if agent is running or queue might still get items

            except asyncio.CancelledError:
                logger_agent_controller.info("Secondary processing loop cancelled during queue get.")
                break # Exit if cancelled
            except Exception as e_get:
                logger_agent_controller.error(f"SecondaryLoop: Error getting task from queue: {e_get}")
                await asyncio.sleep(0.1) # Avoid tight loop on persistent queue error
                continue

            try:
                # Execute the coroutine
                result = await coro_func(*args, **kwargs)
                if not future_obj.done():
                    future_obj.set_result(result)
                
                if callback_on_done:
                    try:
                        # Ensure callback is called from the main event loop context
                        # if the callback needs to interact with UI or non-threadsafe main components
                        if self._asyncio_loop and not self._asyncio_loop.is_closed():
                            self._asyncio_loop.call_soon_threadsafe(callback_on_done, result, None)
                        else: # Fallback or if callback is safe to run directly from this thread
                            callback_on_done(result, None)
                    except Exception as e_cb:
                        logger_agent_controller.error(f"SecondaryLoop: Error in success callback for {coro_func.__name__}: {e_cb}")

            except asyncio.CancelledError:
                logger_agent_controller.warning(f"SecondaryLoop: Task {coro_func.__name__} was cancelled.")
                if not future_obj.done():
                    future_obj.cancel("Task cancelled in secondary loop")
                # No callback for cancelled tasks in this design, future handles it.
            except Exception as e_task:
                logger_agent_controller.error(f"SecondaryLoop: Error executing task {coro_func.__name__}: {e_task}", exc_info=True)
                if not future_obj.done():
                    future_obj.set_exception(e_task)

                if callback_on_done:
                    try:
                        if self._asyncio_loop and not self._asyncio_loop.is_closed():
                            self._asyncio_loop.call_soon_threadsafe(callback_on_done, None, e_task)
                        else:
                            callback_on_done(None, e_task)
                    except Exception as e_cb_err:
                        logger_agent_controller.error(f"SecondaryLoop: Error in error callback for {coro_func.__name__}: {e_cb_err}")
            finally:
                self._offline_task_queue.task_done()
        
        logger_agent_controller.info("Secondary processing loop finished.")

    async def _process_completed_goal_mappings(self) -> Optional['Goal']: # Returns an optional Goal
        """
        Checks for and processes completed LLM goal mapping tasks.
        If a mapping is successful, it creates and returns a Goal object.
        """
        if not self._pending_goal_mapping_tasks:
            return None

        completed_task_hashes = []
        newly_created_user_goal: Optional[Goal] = None # type: ignore
        new_goal_already_added_this_cycle = False # Local flag for this method call

        for text_hash, task_info in self._pending_goal_mapping_tasks.items():
            future = task_info["future"]
            original_text = task_info["original_user_text"]

            if future.done():
                completed_task_hashes.append(text_hash)
                goal_desc_str: Optional[str] = None
                try:
                    # Expecting (content, thinking_trace, error) from the modified call_ollama
                    response_str, thinking_trace_str, llm_error = await future 

                    if thinking_trace_str: # <<< LOG THE THINKING TRACE
                        logger_agent_controller.info(
                            f"AC_LLM_GOAL_MAP_THINKING for '{original_text[:50]}...':\n--- START THINKING ---\n"
                            f"{thinking_trace_str}\n--- END THINKING ---"
                        )

                    if llm_error or not response_str:
                        logger_agent_controller.error(f"AC_LLM_GOAL_MAP_RESULT: LLM error for '{original_text[:50]}...': {llm_error}")
                    elif not response_str.strip() or response_str.strip().lower() == "null":
                        logger_agent_controller.info(f"AC_LLM_GOAL_MAP_RESULT: LLM returned null/empty for '{original_text[:50]}...'. No specific task mapped.")
                    else:
                        # --- Same parsing logic as before from _map_text_to_goal_via_llm ---
                        json_map_str = None
                        parsed_response_data = None
                        match_json_block_map = re.search(r"```json\s*([\s\S]+?)\s*```", response_str, re.DOTALL)
                        if match_json_block_map:
                            json_map_str = match_json_block_map.group(1).strip()
                        else: # Simplified fallback to parse first complete JSON or null
                            response_stripped = response_str.strip()
                            if response_stripped.lower() == "null": json_map_str = "null"
                            else:
                                first_char = response_stripped[0] if response_stripped else ''
                                start_char, end_char = ('{', '}') if first_char == '{' else (('[', ']') if first_char == '[' else ('', ''))
                                if start_char:
                                    balance, end_index = 0, -1
                                    for i, char_scan in enumerate(response_stripped):
                                        if char_scan == start_char: balance += 1
                                        elif char_scan == end_char: balance -= 1
                                        if balance == 0: end_index = i; break
                                    if end_index != -1:
                                        try:
                                            if json.loads(response_stripped[:end_index+1]): # Check if valid JSON
                                                json_map_str = response_stripped[:end_index+1]
                                        except json.JSONDecodeError: pass
                        
                        if not json_map_str or json_map_str.lower() == "null":
                            logger_agent_controller.info(f"AC_LLM_GOAL_MAP_RESULT: LLM for '{original_text[:50]}...' returned null or no clear JSON.")
                        else:
                            try:
                                parsed_response_data = json.loads(json_map_str)
                                if isinstance(parsed_response_data, dict) and parsed_response_data:
                                    task_type = parsed_response_data.get("task_type")
                                    llm_extracted_params = parsed_response_data.get("parameters", {})
                                    if not isinstance(llm_extracted_params, dict): llm_extracted_params = {}
                                    
                                    known_tasks_for_llm = { 
                                        "read_file": {"desc": "Read file.", "params": {"path": "string"}, "format_string": "read file : {path}"},
                                        "explore_directory": {"desc": "List contents.", "params": {"path": "string"}, "format_string": "explore directory : {path}"},
                                        "write_file": {"desc": "Write content.", "params": {"path": "string", "content": "string"}, "format_string": "write file : {path} content : {content}"},
                                        "report_status": {"desc": "Report status.", "params": None, "format_string": "report status"},
                                        "explain_goal": {"desc": "Explain goal.", "params": None, "format_string": "explain goal"},
                                    }
                                    if task_type and task_type in known_tasks_for_llm:
                                        task_details = known_tasks_for_llm[task_type]
                                        format_str = task_details.get("format_string")
                                        if format_str:
                                            actual_params_for_format = {}
                                            all_req_params = True
                                            expected_param_names_in_format = re.findall(r"\{(\w+)\}", format_str)
                                            for p_name in expected_param_names_in_format:
                                                param_value = llm_extracted_params.get(p_name)
                                                if param_value is None and task_details.get("params") and p_name in task_details["params"]:
                                                     if task_type == "explore_directory" and p_name == "path": param_value = "." 
                                                if param_value is not None: actual_params_for_format[p_name] = str(param_value).strip().strip("'\"")
                                                else: all_req_params = False; break
                                            if all_req_params: goal_desc_str = format_str.format(**actual_params_for_format)
                                    elif task_type:
                                        logger_agent_controller.warning(f"AC_LLM_GOAL_MAP_RESULT: LLM mapped to unknown task type '{task_type}' for '{original_text[:50]}...'.")
                                else: 
                                    logger_agent_controller.info(f"AC_LLM_GOAL_MAP_RESULT: LLM response for '{original_text[:50]}...' parsed to non-dict/empty: {str(parsed_response_data)[:100]}")
                            except json.JSONDecodeError as e:
                                logger_agent_controller.error(f"AC_LLM_GOAL_MAP_RESULT: Error decoding LLM JSON for '{original_text[:50]}...': {e}. JSON: '{json_map_str}'")
                        
                    if goal_desc_str:
                        logger_agent_controller.info(f"AC_LLM_GOAL_MAP_RESULT: LLM successfully mapped '{original_text[:50]}...' to desc: '{goal_desc_str}'")
                        _Goal_ac_map = globals().get('Goal')
                        _create_goal_func_ac_map = globals().get('create_goal_from_descriptor')
                        if _Goal_ac_map and _create_goal_func_ac_map:
                            newly_created_user_goal = _create_goal_func_ac_map(goal_desc_str, priority=USER_GOAL_PRIORITY)
                            if newly_created_user_goal and not new_goal_already_added_this_cycle:
                                new_goal_already_added_this_cycle = True # Ensure we only process one per cycle
                                break 
                        else: logger_agent_controller.error("Goal/create_goal_from_descriptor missing for LLM result processing.")
                    elif not goal_desc_str and not llm_error and response_str is not None: 
                        logger_agent_controller.info(f"AC_LLM_GOAL_MAP_RESULT: LLM mapping for '{original_text[:50]}...' resulted in no specific task. Falling back to generic response.")
                        goal_desc_str = f"respond to user : {original_text[:100]}" 
                        _Goal_ac_map_fb = globals().get('Goal')
                        _create_goal_func_ac_map_fb = globals().get('create_goal_from_descriptor')
                        if _Goal_ac_map_fb and _create_goal_func_ac_map_fb:
                            newly_created_user_goal = _create_goal_func_ac_map_fb(goal_desc_str, priority=USER_GOAL_PRIORITY)
                            if newly_created_user_goal and not new_goal_already_added_this_cycle:
                                new_goal_already_added_this_cycle = True
                                break

                except asyncio.CancelledError:
                    logger_agent_controller.warning(f"AC_LLM_GOAL_MAP_RESULT: LLM goal mapping task for '{original_text[:50]}...' was cancelled.")
                except Exception as e_res:
                    logger_agent_controller.error(f"AC_LLM_GOAL_MAP_RESULT: Exception retrieving result for LLM goal mapping task for '{original_text[:50]}...': {e_res}", exc_info=True)
                
                if newly_created_user_goal: # If a goal was made from this task
                    break # Stop checking other pending tasks for this cycle

        for text_hash in completed_task_hashes:
            if text_hash in self._pending_goal_mapping_tasks:
                del self._pending_goal_mapping_tasks[text_hash]
        
        return newly_created_user_goal

    async def _run_agent_loop(self):
        error_state = self._AgentState.ERROR if hasattr(self._AgentState, "ERROR") else self._AgentState("ERROR_FALLBACK") # type: ignore
        self._cycles_ran_since_last_resume = 0 # Initialize before loop starts

        previous_cycle_gwm_content: Optional[Dict[str,Any]] = None
        
        kb_init_assert = self.components.get("knowledge_base")
        if kb_init_assert and self._Predicate != MockDataclass:
            logger_agent_controller.info("Pre-populating KB with isFile for test files for surprise test...")
            try:
                await kb_init_assert.assert_fact(self._Predicate("isFile", ("test_good_file.txt",), True)) # type: ignore
                await kb_init_assert.assert_fact(self._Predicate("isFile", ("non_existent_for_surprise.txt",), True)) # type: ignore
                logger_agent_controller.info("KB pre-population for surprise test successful.")
            except Exception as e_kb_init:
                logger_agent_controller.error(f"Error during KB pre-population for surprise test: {e_kb_init}")
        else:
            logger_agent_controller.warning("KB or Predicate class not available for pre-population for surprise test.")

        self._config_updated_by_po_in_cycle = False

        while self._is_running_flag.is_set():
            # --- PAUSE CHECK ---
            if not self._is_paused_event.is_set():
                logger_agent_controller.info("Agent loop waiting: _is_paused_event is clear (paused).")
            await self._is_paused_event.wait() # Block here if event is cleared (paused)
            
            # Re-check running flag in case agent was stopped while paused
            if not self._is_running_flag.is_set():
                break 
            # --- END PAUSE CHECK ---

            if hasattr(self, 'profiler'): self.profiler.start_section("internal_state_upkeep") 
            await self.internal_state_manager.perform_upkeep_cycle_start()
            if hasattr(self, 'profiler'): self.profiler.end_section()
            
            self.cycle_count += 1
            
            # --- AUTO-PAUSE LOGIC ---
            if self._auto_pause_enabled and self._auto_pause_cycle_target is not None:
                self._cycles_ran_since_last_resume += 1
                logger_agent_controller.debug(
                    f"Auto-pause check: Cycles ran since resume = {self._cycles_ran_since_last_resume}, "
                    f"Target = {self._auto_pause_cycle_target}"
                )
                if self._cycles_ran_since_last_resume >= self._auto_pause_cycle_target:
                    logger_agent_controller.info(
                        f"Auto-pausing after {self._cycles_ran_since_last_resume} cycles "
                        f"(target was {self._auto_pause_cycle_target})."
                    )
                    self._is_paused_event.clear() # This will pause it at the start of the *next* cycle
                    self._update_ui_state(self._AgentState.PAUSED) # type: ignore
                    self._log_to_ui("info", f"Agent auto-paused after {self._cycles_ran_since_last_resume} cycles.")
                    self._auto_pause_enabled = False # Disable auto-pause until set again
                    self._auto_pause_cycle_target = None
            # --- END AUTO-PAUSE LOGIC ---

            cycle_start_time = time.monotonic()
            logger_agent_controller.debug(f"--- Cycle {self.cycle_count} START ---"); self.profiler.reset()
            
            self._raw_percepts={}; attention_candidates={}; attention_weights={} 
            prediction_from_pwm_step_1b: Optional[Dict[str, Any]] = None 
            prediction_result = {} 

            broadcast_content={}; relevant_memories=[]; action_context={}; cognitive_state={}
            meta_analysis={}; loop_info=None; active_goal: Optional[Goal] = None; current_state_set:Set[Predicate]=set() # type: ignore
            next_action={}; action_result={}; optimization_analysis=None; suggested_recovery_mode=None
            current_cycle_prediction_error_for_pwm_update = None
            ems_output = None
            # user_command_goal: Optional[Goal] = None # Replaced by newly_created_user_goal_this_cycle

            self._config_updated_by_po_in_cycle = False

            try:
                # --- 1. Perception & Prediction ---
                self.profiler.start_section("perception"); self._raw_percepts = await self._oscar_perceive(); self.profiler.end_section() 
                kb = getattr(self, 'knowledge_base', None) 

                # --- 1.B. Prediction ---
                self.profiler.start_section("prediction_default_action")
                predictive_world_model_inst = getattr(self, "predictive_world_model", None)
                if predictive_world_model_inst and hasattr(predictive_world_model_inst, 'process') and CORE_DEPENDENCIES_AVAILABLE:
                    proactive_prediction_action = {"type": "THINKING", "params": {"content": "Proactive prediction context"}}
                    
                    current_cognitive_state_for_proactive_pred = await self._oscar_get_cognitive_state(skip_component_statuses=True)
                    current_kb_state_for_proactive_pred = set()
                    kb_inst_pred_1b = getattr(self, 'knowledge_base', None)
                    if kb_inst_pred_1b and hasattr(kb_inst_pred_1b, 'query_state') and self._Predicate is not MockDataclass:
                        try:
                            kb_snapshot_result = await kb_inst_pred_1b.query_state({"all_facts": True, "limit": 150}) 
                            if isinstance(kb_snapshot_result, dict) and "all_facts" in kb_snapshot_result:
                                current_kb_state_for_proactive_pred = {
                                    p for p in kb_snapshot_result["all_facts"] if isinstance(p, self._Predicate)
                                }
                        except Exception as e_kb_pred_1b:
                            logger_agent_controller.warning(f"Error getting KB state for proactive prediction: {e_kb_pred_1b}")
                    
                    proactive_pred_input_payload = { 
                        'action_to_execute': proactive_prediction_action,
                        'context': current_cognitive_state_for_proactive_pred, 
                        'current_world_state_predicates': current_kb_state_for_proactive_pred
                    }

                    pwm_proactive_output = await predictive_world_model_inst.process({"predict_request": proactive_pred_input_payload}) # type: ignore
                    
                    if isinstance(pwm_proactive_output, dict) and "prediction" in pwm_proactive_output:
                        prediction_from_pwm_step_1b = pwm_proactive_output["prediction"] 
                        prediction_result = { 
                            "predicted_outcome": prediction_from_pwm_step_1b.get("predicted_outcome"),
                            "confidence": prediction_from_pwm_step_1b.get("confidence"),
                            "basis": prediction_from_pwm_step_1b.get("basis")
                        }
                        logger_agent_controller.debug(f"Proactive Prediction (Step 1B): {prediction_from_pwm_step_1b}")
                    else:
                        logger_agent_controller.warning("Proactive PWM process (Step 1B) did not return expected prediction structure.")
                        prediction_from_pwm_step_1b = None 
                        prediction_result = {} 
                else:
                    prediction_from_pwm_step_1b = None 
                    prediction_result = {} 
                self.profiler.end_section()


                # --- 2. Attention Allocation ---
                self.profiler.start_section("attention")
                active_goal_for_attention: Optional[Goal] = self._oscar_get_active_goal()
                current_active_id_for_attention = active_goal_for_attention.id if active_goal_for_attention and hasattr(active_goal_for_attention, 'id') else None
                
                attention_candidates = await self._oscar_gather_attention_candidates(self._raw_percepts, current_active_id_for_attention)


                attention_controller_inst = getattr(self, "attention_controller", None)
                if attention_controller_inst and hasattr(attention_controller_inst, 'process'):
                    attention_input = {
                        "candidates": attention_candidates,
                        "current_cycle_active_goal_id": current_active_id_for_attention,
                        "last_gwm_content_for_novelty": previous_cycle_gwm_content, 
                        "last_prediction_error": self.last_prediction_error_for_attention,
                        "current_cycle_count": self.cycle_count,
                        "pain_ids_in_last_gwm": self._pain_ids_in_last_gwm 
                    }
                    logger_agent_controller.debug(f"ATTN_INPUT_DEBUG: active_goal_id={current_active_id_for_attention}, "
                                                  f"gwm_novelty_keys={list(previous_cycle_gwm_content.keys()) if previous_cycle_gwm_content else None}, "
                                                  f"pred_err_for_surprise_type={type(self.last_prediction_error_for_attention).__name__ if self.last_prediction_error_for_attention else None}, "
                                                  f"pain_ids_last_gwm={self._pain_ids_in_last_gwm}")

                    process_method_attn = getattr(attention_controller_inst, 'process')
                    if asyncio.iscoroutinefunction(process_method_attn):
                        attn_output = await process_method_attn(attention_input) # type: ignore
                        attention_weights = attn_output.get("attention_weights", {}) if isinstance(attn_output, dict) else {}
                    else: logger_agent_controller.warning(f"AttentionController ({type(attention_controller_inst)}) process not async.")
                else: logger_agent_controller.debug("AttentionController or its process method missing."); attention_weights = {}
                self.profiler.end_section()

                # --- 3. Global Workspace ---
                self.profiler.start_section("workspace")
                global_workspace_inst = getattr(self, "global_workspace", None)
                if global_workspace_inst and hasattr(global_workspace_inst, 'process'):
                    workspace_input = { "attention_weights": attention_weights, "all_candidates_data": attention_candidates }
                    process_method_gws = getattr(global_workspace_inst, 'process')
                    if asyncio.iscoroutinefunction(process_method_gws):
                        ws_output = await process_method_gws(workspace_input)
                        broadcast_content = ws_output.get("broadcast_content", {}) if isinstance(ws_output, dict) else {}
                    else: logger_agent_controller.warning(f"GlobalWorkspace ({type(global_workspace_inst)}) process not async.")
                else: logger_agent_controller.debug("GlobalWorkspaceManager or its process method missing."); broadcast_content = {}
                self.global_workspace_content = broadcast_content 
                previous_cycle_gwm_content = broadcast_content.copy() if broadcast_content else None
                
                # --- Store Pain IDs from current GWM for next cycle's AttentionController ---
                current_gwm_pain_ids: List[str] = []
                if self.global_workspace_content and isinstance(self.global_workspace_content, dict):
                    for item_id_gwm in self.global_workspace_content.keys():
                        if isinstance(item_id_gwm, str) and item_id_gwm.startswith("pain_event_"):
                            # Extract the raw pain ID
                            raw_pain_id = item_id_gwm[len("pain_event_"):]
                            current_gwm_pain_ids.append(raw_pain_id)
                
                self._pain_ids_in_last_gwm = current_gwm_pain_ids
                if self._pain_ids_in_last_gwm:
                    logger_agent_controller.debug(
                        f"AC_GWM_PAIN_TRACK: Stored {len(self._pain_ids_in_last_gwm)} pain IDs from current GWM "
                        f"for next cycle's attention input: {self._pain_ids_in_last_gwm}"
                    )
                # --- End Pain ID GWM Tracking ---
                self.profiler.end_section()


                # --- 4. Experience Integration ---
                self.profiler.start_section("experience_integration")
                experience_stream_inst = getattr(self, "experience_stream", None)
                if experience_stream_inst and hasattr(experience_stream_inst, 'process'):
                    relevant_memories = await self._oscar_get_relevant_memories(broadcast_content)
                    action_context = self._oscar_get_current_action_context()
                    experience_input = { "percepts": self._raw_percepts, "memories": relevant_memories, "context": action_context, "broadcast_content": broadcast_content }
                    process_method_es = getattr(experience_stream_inst, 'process')
                    if asyncio.iscoroutinefunction(process_method_es):
                        exp_output = await process_method_es(experience_input)
                        self.current_phenomenal_state = exp_output.get("phenomenal_state") if isinstance(exp_output, dict) else None
                    else: logger_agent_controller.warning(f"ExperienceStream ({type(experience_stream_inst)}) process not async.")
                else: logger_agent_controller.debug("ExperienceStream or its process method missing."); self.current_phenomenal_state = None

                current_phenomenal_state_class = self._PhenomenalState
                if not current_phenomenal_state_class or current_phenomenal_state_class == MockDataclass:
                     logger_agent_controller.error("PhenomenalState class missing or dummy, experience cannot be properly generated.")
                     self.current_phenomenal_state = {"content": {}, "timestamp": time.time(), "error": "PhenomenalState class missing"} # type: ignore
                elif not isinstance(self.current_phenomenal_state, current_phenomenal_state_class):
                     logger_agent_controller.warning("Experience integration failed or returned invalid type.");
                     self.current_phenomenal_state = current_phenomenal_state_class(content={}, timestamp=time.time()) # type: ignore
                self.profiler.end_section()

                # --- 5. Consciousness Assessment ---
                self.profiler.start_section("consciousness_assessment")
                self._prev_consciousness_level = self.consciousness_level; new_conscious_state = self.consciousness_level
                consciousness_assessor_inst = getattr(self, "consciousness_assessor", None)
                current_conscious_state_enum = self._ConsciousState
                if current_conscious_state_enum != MockEnum and consciousness_assessor_inst and hasattr(consciousness_assessor_inst, 'process'): # type: ignore
                    assessment_input = { "experience": self.current_phenomenal_state, "workspace_content": broadcast_content }
                    process_method_cs = getattr(consciousness_assessor_inst, 'process')
                    if asyncio.iscoroutinefunction(process_method_cs):
                        assess_output = await process_method_cs(assessment_input)
                        if assess_output and "conscious_state" in assess_output and isinstance(assess_output["conscious_state"], current_conscious_state_enum): # type: ignore
                            new_conscious_state = assess_output["conscious_state"]
                        else: logger_agent_controller.warning("Consciousness assessment failed or returned invalid data.")
                    else: logger_agent_controller.warning(f"ConsciousnessAssessor ({type(consciousness_assessor_inst)}) process not async.")
                elif not consciousness_assessor_inst or not hasattr(consciousness_assessor_inst, 'process'): logger_agent_controller.debug("ConsciousnessLevelAssessor or its process method missing.")
                elif current_conscious_state_enum == MockEnum: logger_agent_controller.error("ConsciousState enum is a MockEnum (dummy).") # type: ignore
                self.consciousness_level = new_conscious_state
                if self.consciousness_level != self._prev_consciousness_level:
                    prev_name = self._prev_consciousness_level.name if hasattr(self._prev_consciousness_level, 'name') else str(self._prev_consciousness_level)
                    curr_name = self.consciousness_level.name if hasattr(self.consciousness_level, 'name') else str(self.consciousness_level)
                    logger_agent_controller.info(f"CS Level Changed: {prev_name} -> {curr_name}");
                    self._log_to_ui("info", f"State: {curr_name}")
                self.profiler.end_section()

                # --- 6. Meta-Cognitive Monitoring ---
                self.profiler.start_section("meta_cognition"); meta_analysis = {}
                meta_cognition_inst = getattr(self, "meta_cognition", None)
                if meta_cognition_inst and hasattr(meta_cognition_inst, 'process'):
                    cognitive_state = await self._oscar_get_cognitive_state()
                    if self.current_phenomenal_state and isinstance(self.current_phenomenal_state, (self._PhenomenalState, dict)): # type: ignore
                         if hasattr(self.current_phenomenal_state, 'content'): cognitive_state["workspace_load"] = len(self.current_phenomenal_state.content) # type: ignore
                         if hasattr(self.current_phenomenal_state, 'valence'): cognitive_state["emotional_valence"] = self.current_phenomenal_state.valence # type: ignore
                         if hasattr(self.current_phenomenal_state, 'integration_level'): cognitive_state["integration_level"] = self.current_phenomenal_state.integration_level # type: ignore
                    perf_metrics = self.profiler.get_average_profile(); meta_input = { "cognitive_state": cognitive_state, "performance_metrics": perf_metrics }
                    process_method_mc = getattr(meta_cognition_inst, 'process')
                    if asyncio.iscoroutinefunction(process_method_mc):
                        meta_output = await process_method_mc(meta_input)
                        if meta_output and "meta_analysis" in meta_output: meta_analysis = meta_output["meta_analysis"];
                        else: logger_agent_controller.warning("Meta-cognitive monitoring failed or returned invalid data.")
                    else: logger_agent_controller.warning(f"MetaCognitiveMonitor ({type(meta_cognition_inst)}) process not async.")
                else: logger_agent_controller.debug("MetaCognitiveMonitor or its process method missing.")
                self.profiler.end_section()

                # --- Process MCM Suggestions ---
                if meta_analysis and isinstance(meta_analysis.get("suggestions"), list):
                    for suggestion in meta_analysis["suggestions"]:
                        if isinstance(suggestion, dict) and suggestion.get("type") == "REPLAN_GOAL_WITH_HINTS":
                            suggested_goal_id = suggestion.get("goal_id")
                            
                            # Check if this suggestion is for the *current* active_goal
                            if active_goal and hasattr(active_goal, 'id') and active_goal.id == suggested_goal_id:
                                logger_agent_controller.info(
                                    f"AC_MCM_SUGGESTION: Received REPLAN_GOAL_WITH_HINTS for current active goal '{suggested_goal_id}'. "
                                    f"Hints: {suggestion.get('hints')}. Invalidating current plan."
                                )
                                self.current_plan = None # Invalidate current plan, forcing replan
                                self._plan_generated_for_goal_id = None # Ensure HTN re-evaluates
                                # Store hints to be passed to HTNPlanner
                                self._active_goal_modification_hints = suggestion.get("hints") 
                                break # Handle one major suggestion per cycle for now
                        # Add elif for other suggestion types here later


                # --- 7. Loop Detection & Intervention ---
                self.profiler.start_section("loop_detection"); loop_info=None 
                loop_detector_inst = getattr(self, "loop_detector", None)
                if loop_detector_inst and hasattr(loop_detector_inst, 'process'):
                    process_method_ld = getattr(loop_detector_inst, 'process')
                    if asyncio.iscoroutinefunction(process_method_ld):
                        loop_output = await process_method_ld(None) 
                        if loop_output and "loop_info" in loop_output: 
                            loop_info = loop_output["loop_info"] 
                    else: 
                        logger_agent_controller.warning(f"LoopDetector ({type(loop_detector_inst)}) process not async.")
                else: 
                    logger_agent_controller.debug("LoopDetector or its process method missing.")
                
                if loop_info: 
                    logger_agent_controller.warning(f"Loop detected: {loop_info}. Intervening.")
                    await self._oscar_handle_loop(loop_info, meta_analysis)
                    self.profiler.end_section() 
                    logger_agent_controller.info("Loop handled, continuing to next cycle immediately.")
                    continue 
                else:
                    logger_agent_controller.debug("No loop detected by LoopDetector this cycle.")
                
                self.profiler.end_section() 
                
              # --- 8. Planning & Goal Management ---
                self.profiler.start_section("planning")
                newly_created_user_goal_this_cycle: Optional[Goal] = None # type: ignore

                # --- Process any completed LLM goal mappings first ---
                if hasattr(self, '_process_completed_goal_mappings'): # Check if method exists
                    potential_goal_from_llm = await self._process_completed_goal_mappings()
                    if potential_goal_from_llm:
                        is_new = True
                        for ag_existing in self.active_goals:
                            if hasattr(ag_existing, 'description') and ag_existing.description == potential_goal_from_llm.description and \
                               hasattr(ag_existing, 'status') and ag_existing.status == self._GoalStatus.ACTIVE:
                                is_new = False; break
                        if is_new:
                            self.active_goals.append(potential_goal_from_llm)
                            newly_created_user_goal_this_cycle = potential_goal_from_llm
                            logger_agent_controller.info(f"AC_STEP8_PLAN: Goal from COMPLETED LLM mapping added: {potential_goal_from_llm.description}")
                        else:
                            logger_agent_controller.info(f"AC_STEP8_PLAN: Goal from COMPLETED LLM mapping was already active: {potential_goal_from_llm.description}")
                # --- End processing completed LLM mappings ---


                # --- Handle user input from broadcast_content (if not already handled by a completed LLM task this cycle) ---
                if not newly_created_user_goal_this_cycle and broadcast_content and "percept_user_input" in broadcast_content:
                    user_text_from_workspace = broadcast_content.get("percept_user_input")
                    if isinstance(user_text_from_workspace, str) and user_text_from_workspace.strip():
                        logger_agent_controller.info(f"AC_STEP8_PLAN: User input '{user_text_from_workspace[:50]}...' in GWS. Attempting map.")
                        
                        map_result = await self._map_text_to_goal(user_text_from_workspace)

                        if isinstance(map_result, self._Goal if self._Goal else object): # Direct regex map
                            potential_user_goal_direct = map_result
                            is_new_direct = True
                            for ag_existing in self.active_goals:
                                if hasattr(ag_existing, 'description') and ag_existing.description == potential_user_goal_direct.description and \
                                   hasattr(ag_existing, 'status') and ag_existing.status == self._GoalStatus.ACTIVE:
                                    is_new_direct = False; break
                            if is_new_direct:
                                self.active_goals.append(potential_user_goal_direct)
                                newly_created_user_goal_this_cycle = potential_user_goal_direct
                                logger_agent_controller.info(f"AC_STEP8_PLAN: User command (direct map) added to active_goals: {potential_user_goal_direct.description}")
                        
                        elif map_result == "mapping_in_progress":
                            logger_agent_controller.info(f"AC_STEP8_PLAN: LLM Goal mapping for '{user_text_from_workspace[:50]}...' is now IN PROGRESS.")
                        
                        elif map_result == "fallback_to_respond":
                            logger_agent_controller.info(f"AC_STEP8_PLAN: LLM mapping failed for '{user_text_from_workspace[:50]}...'. Creating generic response goal.")
                            generic_desc = f"respond to user : {user_text_from_workspace[:100]}"
                            _create_goal_func_fallback = globals().get('create_goal_from_descriptor')
                            if _create_goal_func_fallback:
                                fallback_goal = _create_goal_func_fallback(generic_desc, priority=USER_GOAL_PRIORITY)
                                if fallback_goal:
                                    self.active_goals.append(fallback_goal)
                                    newly_created_user_goal_this_cycle = fallback_goal
                        
                        elif map_result == "system_command_resolve_test_pain": 
                            logger_agent_controller.info("AC_STEP8_PLAN - Received internal command to resolve test pain.")
                            test_pain_goal_desc_pattern = "trigger_test_pain_event_high_priority"
                            resolved_count = 0
                            if self._PainSource is not MockDataclass:
                                for ps_event in self.active_pain_sources:
                                    if isinstance(ps_event, self._PainSource) and not ps_event.is_resolved and \
                                       hasattr(ps_event, 'description') and test_pain_goal_desc_pattern in ps_event.description:
                                        ps_event.is_resolved = True
                                        old_intensity_res = ps_event.current_intensity
                                        ps_event.current_intensity *= 0.05 
                                        happiness_bonus_manual_resolve = old_intensity_res * 0.75 
                                        self.happiness_level = min(10.0, self.happiness_level + happiness_bonus_manual_resolve)
                                        logger_agent_controller.info(
                                            f"AC_PHP_TEST - Manually marked PainSource '{ps_event.id}' ({ps_event.description[:30]}) as RESOLVED. "
                                            f"Intensity {old_intensity_res:.2f} -> {ps_event.current_intensity:.2f}. "
                                            f"Happiness bonus: {happiness_bonus_manual_resolve:.2f}."
                                        )
                                        resolved_count += 1
                            if resolved_count == 0: logger_agent_controller.warning("AC_STEP8_PLAN - No active test pain found to resolve via command.")
                            self.current_plan = None 
                        
                active_goal = await self._oscar_generate_or_select_goal(newly_created_user_goal_this_cycle)
                current_goal_id_for_this_cycle = getattr(active_goal, 'id', None)
                htn_planner_inst = getattr(self, "htn_planner", None)

                needs_new_plan = False
                if self.current_plan is None:
                    needs_new_plan = True
                    logger_agent_controller.debug("Planning trigger: self.current_plan is None.")
                elif not self.current_plan: 
                    needs_new_plan = True
                    logger_agent_controller.debug("Planning trigger: self.current_plan is an empty list.")
                elif self._plan_generated_for_goal_id != current_goal_id_for_this_cycle:
                    needs_new_plan = True
                    logger_agent_controller.debug(
                        f"Planning trigger: Goal changed from '{self._plan_generated_for_goal_id}' to '{current_goal_id_for_this_cycle}'."
                    )
                elif self._active_goal_modification_hints: # <<< ADD THIS CONDITION
                    needs_new_plan = True
                    logger_agent_controller.info(
                        f"Planning trigger: Active goal modification hints present for goal '{current_goal_id_for_this_cycle}'. Forcing replan."
                    )
                
                if needs_new_plan:
                    if self._plan_generated_for_goal_id != current_goal_id_for_this_cycle or self.current_plan is None or not self.current_plan :
                        logger_agent_controller.debug(f"Clearing plan details for new/changed goal or empty plan. Old plan for goal '{self._plan_generated_for_goal_id}'.")
                        self.current_plan = None 
                        if self._plan_generated_for_goal_id: 
                            self._active_goal_planning_failure_count.pop(self._plan_generated_for_goal_id, None)
                    
                    self._plan_generated_for_goal_id = current_goal_id_for_this_cycle

                    if active_goal and htn_planner_inst and \
                       hasattr(htn_planner_inst, 'plan') and CORE_DEPENDENCIES_AVAILABLE:
                        try:
                            current_kb_state_for_planning = set() 
                            kb_inst_for_planning = getattr(self, 'knowledge_base', None)
                            if kb_inst_for_planning and hasattr(kb_inst_for_planning, 'query_state'):
                                kb_plan_ctx_res = await kb_inst_for_planning.query_state({"all_facts":True, "limit":100})
                                if isinstance(kb_plan_ctx_res, dict) and self._Predicate is not MockDataclass:
                                    current_kb_state_for_planning = {p for p in kb_plan_ctx_res.get("all_facts",[]) if isinstance(p, self._Predicate)} # type: ignore
                            
                            # Retrieve and clear hints for the current planning attempt
                            hints_for_this_plan_attempt = None
                            if self._active_goal_modification_hints and \
                               self._plan_generated_for_goal_id == current_goal_id_for_this_cycle: # Ensure hints are for current goal
                                hints_for_this_plan_attempt = self._active_goal_modification_hints
                                logger_agent_controller.info(f"AC_PLAN: Passing hints to HTNPlanner for goal '{current_goal_id_for_this_cycle}': {hints_for_this_plan_attempt}")
                            
                            plan_list_of_actions = await htn_planner_inst.plan( # type: ignore
                                active_goal, 
                                current_kb_state_for_planning,
                                modification_hints=hints_for_this_plan_attempt # PASS HINTS
                            )
                            
                            # Important: Clear hints after they've been passed for one planning attempt
                            # to prevent them from being reused indefinitely without fresh MCM input.
                            if hints_for_this_plan_attempt:
                                self._active_goal_modification_hints = None 
                                logger_agent_controller.debug("AC_PLAN: Cleared modification hints after passing to planner.")

                            
                            if plan_list_of_actions is not None: 
                                self.current_plan = plan_list_of_actions
                                if current_goal_id_for_this_cycle: 
                                    self._active_goal_planning_failure_count[current_goal_id_for_this_cycle] = 0
                                logger_agent_controller.info(f"HTN Plan generated for goal '{getattr(active_goal,'description','N/A')}': {len(self.current_plan)} steps.")
                            else: 
                                self.current_plan = None 
                                logger_agent_controller.warning(f"HTN Planner returned no plan for goal '{getattr(active_goal,'description','N/A')}'.")
                                if current_goal_id_for_this_cycle:
                                    self._active_goal_planning_failure_count[current_goal_id_for_this_cycle] = \
                                        self._active_goal_planning_failure_count.get(current_goal_id_for_this_cycle, 0) + 1
                                    
                                    failure_count = self._active_goal_planning_failure_count[current_goal_id_for_this_cycle]
                                    logger_agent_controller.warning(
                                        f"Goal '{current_goal_id_for_this_cycle}' planning failure count: {failure_count} "
                                        f"(max: {self._max_planning_failures_before_goal_fail})"
                                    )

                                    if failure_count >= self._max_planning_failures_before_goal_fail:
                                        logger_agent_controller.error( 
                                            f"PHP_TRIGGER - Goal '{getattr(active_goal,'description','N/A')}' "
                                            f"FAILED due to persistent planning failures ({failure_count} attempts). "
                                            f"Delegating PainSource generation."
                                        )
                                        if hasattr(active_goal, 'status') and self._GoalStatus != MockEnum: # type: ignore
                                            active_goal.status = self._GoalStatus.FAILED # type: ignore
                                            kb_fail_plan = getattr(self, 'knowledge_base', None)
                                            if kb_fail_plan and self._Predicate is not MockDataclass: # type: ignore
                                                reason_str = f"PersistentPlanningFailure_{failure_count}_attempts"
                                                await kb_fail_plan.assert_fact(self._Predicate("goalFailed", (current_goal_id_for_this_cycle, reason_str), True, timestamp=time.time())) # type: ignore
                                        
                                        self.internal_state_manager.generate_pain_from_goal_failure(
                                            active_goal, 
                                            failure_type="PersistentPlanningFailure"
                                        )
                                        
                                        if current_goal_id_for_this_cycle in self._active_goal_planning_failure_count:
                                            del self._active_goal_planning_failure_count[current_goal_id_for_this_cycle]
                                        self.current_plan = None 
                        except Exception as e_plan:
                            logger_agent_controller.error(f"Error during HTN planning: {e_plan}", exc_info=True)
                            self.current_plan = None
                    elif not active_goal:
                        logger_agent_controller.debug("No active goal, skipping planning phase.")
                        self.current_plan = None 
                        self._plan_generated_for_goal_id = None 
                
                elif self.current_plan is not None: 
                    logger_agent_controller.debug(f"Using existing plan for goal '{self._plan_generated_for_goal_id}'. Steps remaining: {len(self.current_plan)}")
                
                # --- ValueSystem Plan Evaluation ---
                value_system_inst = getattr(self, "value_system", None)
                if self.current_plan and active_goal and value_system_inst and \
                   hasattr(value_system_inst, 'evaluate_plan_alignment') and CORE_DEPENDENCIES_AVAILABLE:
                    try:
                        plan_context_for_vs = await self._oscar_get_cognitive_state(skip_component_statuses=True)
                        alignment_score, plan_judgments, suggested_modifications = await value_system_inst.evaluate_plan_alignment( # type: ignore
                            self.current_plan, active_goal, plan_context_for_vs
                        )
                        
                        # Get the rejection threshold from the live agent config
                        vs_config_live_plan_ac = self.config.get("value_system", {})
                        plan_rejection_threshold_ac = float(vs_config_live_plan_ac.get("plan_rejection_value_threshold", -0.3)) # Default from config.toml
                        
                        # Log the check details
                        logger_agent_controller.info(
                            f"AC_VS_PLAN_CHECK: Goal '{getattr(active_goal, 'description', 'N/A')[:30]}'. "
                            f"Plan Alignment Score: {alignment_score:.3f}, "
                            f"Configured Rejection Threshold: {plan_rejection_threshold_ac:.3f}, "
                            f"Plan will be rejected: {alignment_score < plan_rejection_threshold_ac}"
                        )

                        if alignment_score < plan_rejection_threshold_ac:
                            active_goal_desc_for_log = getattr(active_goal, 'description', 'N/A')
                            logger_agent_controller.warning(
                                f"Plan for goal '{active_goal_desc_for_log}' REJECTED by ValueSystem. "
                                f"Score: {alignment_score:.3f} (Threshold: {plan_rejection_threshold_ac:.3f}). Discarding plan."
                            )
                            self._log_to_ui("warn", f"Plan rejected (VS). Score: {alignment_score:.2f}")
                            kb_vs_reject = getattr(self, 'knowledge_base', None)
                            if kb_vs_reject and self._Predicate is not MockDataclass: # type: ignore
                                reason_for_rejection_vs = "LowValueAlignment"
                                if plan_judgments:
                                    most_negative_judgment_vs = min(plan_judgments, key=lambda j: getattr(j, 'score', 0.0))
                                    if hasattr(most_negative_judgment_vs, 'value_category') and hasattr(most_negative_judgment_vs.value_category, 'name'):
                                         reason_for_rejection_vs = f"LowValueAlignment:{most_negative_judgment_vs.value_category.name}" # type: ignore
                                await kb_vs_reject.assert_fact(self._Predicate(name="planRejectedByValue", args=(current_goal_id_for_this_cycle, reason_for_rejection_vs, round(alignment_score, 2)),value=True,timestamp=time.time())) # type: ignore
                            
                            self.current_plan = None 
                            self._plan_generated_for_goal_id = None 
                            logger_agent_controller.info(f"AC_VS_PLAN_REJECT: self.current_plan is now: {self.current_plan}. Goal ID for plan was: {current_goal_id_for_this_cycle}")
                        
                        # Assert hints regardless of plan rejection, if suggestions were made
                        if suggested_modifications and isinstance(suggested_modifications, dict) and current_goal_id_for_this_cycle:
                            logger_agent_controller.info(f"ValueSystem suggested plan modifications: {suggested_modifications}")
                            self._log_to_ui("info", f"VS suggestions for plan: {str(suggested_modifications)[:100]}...")
                            kb_vs_hints = getattr(self, 'knowledge_base', None)
                            if kb_vs_hints and self._Predicate is not MockDataclass: # type: ignore
                                try:
                                    modifications_json_str = json.dumps(suggested_modifications)
                                    hint_predicate = self._Predicate( # type: ignore
                                        name="pendingPlanModificationHints",
                                        args=(current_goal_id_for_this_cycle, modifications_json_str),
                                        value=True,
                                        timestamp=time.time()
                                    )
                                    await kb_vs_hints.assert_fact(hint_predicate)
                                    logger_agent_controller.info(f"Asserted pendingPlanModificationHints for goal '{current_goal_id_for_this_cycle}' to KB.")
                                except Exception as e_kb_hints_vs:
                                    logger_agent_controller.error(f"Error asserting pendingPlanModificationHints to KB: {e_kb_hints_vs}")
                        elif suggested_modifications and not current_goal_id_for_this_cycle:
                             logger_agent_controller.warning("Cannot assert plan modification hints: active goal ID is missing for current cycle.")


                    except Exception as e_vs_plan_ac_outer:
                        logger_agent_controller.error(f"Error during ValueSystem plan evaluation: {e_vs_plan_ac_outer}", exc_info=True)
                        # If ValueSystem itself errors, it's safer to discard the current plan
                        self.current_plan = None 
                        self._plan_generated_for_goal_id = None
                
                self.profiler.end_section()

                # --- 9. Action Selection & Execution ---
                self.profiler.start_section("action_selection")
                next_action = self._oscar_select_next_action(self.current_plan)
                self.profiler.end_section()

                self._buffered_pre_action_state_for_pwm = None 
                if next_action and next_action.get("type") != "THINKING": 
                    try:
                        kb_inst_for_pre_state = getattr(self, 'knowledge_base', None)
                        pre_action_kb_snapshot_query = {"all_facts": True, "limit": 200} 
                        pre_action_kb_state_result = {}
                        if kb_inst_for_pre_state and hasattr(kb_inst_for_pre_state, 'query_state'):
                            query_state_method = getattr(kb_inst_for_pre_state, 'query_state')
                            if asyncio.iscoroutinefunction(query_state_method):
                                 pre_action_kb_state_result = await query_state_method(pre_action_kb_snapshot_query)
                            else: 
                                 pre_action_kb_state_result = query_state_method(pre_action_kb_snapshot_query) # type: ignore
                        
                        pre_action_predicates_list = pre_action_kb_state_result.get("all_facts", []) if isinstance(pre_action_kb_state_result, dict) else []
                        
                        _Predicate_ac_pre_state = globals().get("Predicate")
                        valid_pre_action_predicates: Set[Predicate] = set() # type: ignore
                        if _Predicate_ac_pre_state:
                            valid_pre_action_predicates = {p for p in pre_action_predicates_list if isinstance(p, _Predicate_ac_pre_state)}
                        
                        pre_action_cognitive_context = await self._oscar_get_cognitive_state()
                        
                        self._buffered_pre_action_state_for_pwm = {
                            "predicates": valid_pre_action_predicates,
                            "context": pre_action_cognitive_context, 
                            "action_to_be_executed": next_action.copy() 
                        }
                        logger_agent_controller.debug(
                            f"Buffered pre-action state for PWM. Action: {next_action.get('type')}, "
                            f"KB facts: {len(valid_pre_action_predicates)}, Context keys: {list(pre_action_cognitive_context.keys())}"
                        )
                    except Exception as e_buffer_pre_state:
                        logger_agent_controller.error(f"Error buffering pre-action state for PWM: {e_buffer_pre_state}", exc_info=True)
                
                action_vetoed_by_value = False
                if next_action and next_action.get("type") != "THINKING" and \
                   value_system_inst and hasattr(value_system_inst, 'evaluate_action_consequences') and \
                   CORE_DEPENDENCIES_AVAILABLE:
                    try:
                        value_eval_context_action = await self._get_value_system_action_context(next_action)

                        action_judgments: List[ValueJudgment] = await value_system_inst.evaluate_action_consequences( # type: ignore
                            next_action, value_eval_context_action
                        )
                        
                        vs_config_live_action = self.config.get("value_system", {})
                        action_safety_veto_threshold = vs_config_live_action.get("action_safety_veto_threshold", -0.8)

                        for judgment in action_judgments:
                            if hasattr(judgment, 'value_category') and hasattr(judgment.value_category, 'name') and \
                               judgment.value_category.name == self._ValueCategory.SAFETY.name and \
                               hasattr(judgment, 'score') and judgment.score < action_safety_veto_threshold: # type: ignore
                                
                                logger_agent_controller.critical(
                                    f"Action '{next_action.get('type')}' VETOED by ValueSystem due to critical SAFETY concern. "
                                    f"Judgment score: {judgment.score:.2f} (Threshold: {action_safety_veto_threshold}). Reason: {judgment.reason}"
                                )
                                self._log_to_ui("critical", f"Action VETOED (SAFETY): {next_action.get('type')}. Reason: {judgment.reason}")
                                action_result = {
                                    "outcome": "failure", 
                                    "error": f"ActionVetoed_SAFETY: {judgment.reason}",
                                    "value_judgment": judgment.__dict__ if hasattr(judgment, '__dict__') else str(judgment)
                                }
                                action_vetoed_by_value = True
                                self.current_plan = None 
                                
                                kb_veto = getattr(self, 'knowledge_base', None)
                                if kb_veto and self._Predicate is not MockDataclass:
                                    await kb_veto.assert_fact(self._Predicate( # type: ignore
                                        name="actionVetoedByValue",
                                        args=(next_action.get("type"), self._ValueCategory.SAFETY.name, round(judgment.score, 2)), # type: ignore
                                        value=True, timestamp=time.time()
                                    ))
                                break 
                    except Exception as e_vs_action:
                        logger_agent_controller.error(f"Error during ValueSystem action evaluation: {e_vs_action}", exc_info=True)

                self.profiler.start_section("execution")
                if not action_vetoed_by_value: 
                    action_result = await self._oscar_execute_action(next_action)
                self.profiler.end_section()

                self._last_action_executed = next_action
                self._last_action_result = action_result

                # --- 10. Update Goal Status & Model Updates ---
                current_goal_status_enum_exec = self._GoalStatus
                goal_to_update_post_action = active_goal 
                goal_just_achieved_this_cycle = False 
                priority_of_achieved_goal = 0.0 


                if isinstance(action_result, dict) and action_result.get("outcome") == "success":
                    action_was_part_of_current_plan = False 
                    if self.current_plan and len(self.current_plan) > 0 and self.current_plan[0] == next_action:
                        self.current_plan.pop(0) 
                        action_was_part_of_current_plan = True

                    if goal_to_update_post_action and hasattr(goal_to_update_post_action, 'id') and \
                       goal_to_update_post_action.id in self._active_goal_execution_failure_count: # type: ignore
                        logger_agent_controller.info(
                            f"Resetting execution failure count for goal '{goal_to_update_post_action.id}' after successful action." # type: ignore
                        )
                        del self._active_goal_execution_failure_count[goal_to_update_post_action.id] # type: ignore

                    if action_was_part_of_current_plan and not self.current_plan and \
                       goal_to_update_post_action and \
                       hasattr(goal_to_update_post_action, 'status') and current_goal_status_enum_exec != MockEnum:
                        
                        if hasattr(self, '_plan_generated_for_goal_id') and \
                           self._plan_generated_for_goal_id == getattr(goal_to_update_post_action, 'id', None):

                            goal_id_to_mark = goal_to_update_post_action.id # type: ignore
                            actual_goal_in_list = next((g for g in self.active_goals if hasattr(g, 'id') and g.id == goal_id_to_mark), None)
                            if actual_goal_in_list and hasattr(actual_goal_in_list, 'status'):
                                original_description_for_log = getattr(actual_goal_in_list, 'description', 'Unknown Goal')
                                is_default_observe_goal = hasattr(actual_goal_in_list, 'description') and actual_goal_in_list.description == DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC

                                actual_goal_in_list.status = current_goal_status_enum_exec.ACHIEVED # type: ignore
                                goal_just_achieved_this_cycle = True 
                                priority_of_achieved_goal = getattr(actual_goal_in_list, 'priority', DEFAULT_OBSERVE_GOAL_PRIORITY) 
                                self._active_goal_modification_hints = None # Clear hints on achievement

                                logger_agent_controller.info(f"Marked goal '{original_description_for_log}' (ID: {goal_id_to_mark}) as ACHIEVED.")
                                current_time_achieve = time.time()
                                kb_achieve = getattr(self, 'knowledge_base', None)
                                if kb_achieve and self._Predicate != MockDataclass:
                                    await kb_achieve.assert_fact(self._Predicate("goalAchieved", (goal_id_to_mark, original_description_for_log[:50]), True, timestamp=current_time_achieve)) # type: ignore

                                if is_default_observe_goal:
                                    self._last_default_observe_completion_cycle = self.cycle_count
                                    logger_agent_controller.info(f"Default '{DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC}' goal completed. Cooldown set using cycle {self.cycle_count}.")
                                else:
                                    logger_agent_controller.info(
                                        f"Non-default goal '{original_description_for_log}' (ID: {goal_id_to_mark}) completed. "
                                        f"Resetting cooldown for default '{DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC}' goal."
                                    )
                                    self._last_default_observe_completion_cycle = 0
                            else:
                                logger_agent_controller.warning(f"Could not find goal with ID {goal_id_to_mark} in active_goals list to mark ACHIEVED, or it lacks 'status'.")
                        else:
                            logger_agent_controller.warning(
                                f"Plan completed for goal '{getattr(goal_to_update_post_action, 'id', 'unknown')}', " # type: ignore
                                f"but it might not match _plan_generated_for_goal_id "
                                f"('{getattr(self, '_plan_generated_for_goal_id', 'None')}'). Goal not marked achieved yet."
                            )
                    elif next_action.get("type") == "THINKING" and goal_to_update_post_action and \
                         hasattr(goal_to_update_post_action, 'status') and \
                         goal_to_update_post_action.status == self._GoalStatus.ACTIVE: # type: ignore
                        logger_agent_controller.debug(f"THINKING action succeeded for active goal '{goal_to_update_post_action.description}'. Goal remains ACTIVE.") # type: ignore
                
                elif isinstance(action_result, dict) and action_result.get("outcome") == "failure":
                    logger_agent_controller.warning(
                        f"Action '{next_action.get('type')}' failed for goal "
                        f"'{goal_to_update_post_action.description if goal_to_update_post_action and hasattr(goal_to_update_post_action,'description') else 'N/A'}'. "
                        f"Error: {action_result.get('error')}"
                    )
                    self.current_plan = None 
                    self._active_goal_modification_hints = None # Clear hints on failure path too

                    if goal_to_update_post_action and hasattr(goal_to_update_post_action, 'id') and \
                       hasattr(goal_to_update_post_action, 'status') and \
                       goal_to_update_post_action.status == current_goal_status_enum_exec.ACTIVE: # type: ignore
                        
                        goal_id_for_exec_fail = goal_to_update_post_action.id
                        self._active_goal_execution_failure_count[goal_id_for_exec_fail] = \
                            self._active_goal_execution_failure_count.get(goal_id_for_exec_fail, 0) + 1
                        
                        exec_failure_count = self._active_goal_execution_failure_count[goal_id_for_exec_fail]
                        logger_agent_controller.warning(
                            f"Goal '{goal_id_for_exec_fail}' execution failure count: {exec_failure_count} "
                            f"(max: {self._max_execution_failures_per_goal})"
                        )

                        if exec_failure_count >= self._max_execution_failures_per_goal:
                            logger_agent_controller.error( # Use error level
                                f"PHP_TRIGGER - Goal '{goal_to_update_post_action.description if hasattr(goal_to_update_post_action,'description') else goal_id_for_exec_fail}' "
                                f"FAILED due to persistent execution failures ({exec_failure_count} attempts). "
                                f"Delegating PainSource generation."
                            )
                            goal_to_update_post_action.status = current_goal_status_enum_exec.FAILED # type: ignore
                            kb_fail = getattr(self, 'knowledge_base', None)
                            if kb_fail and self._Predicate != MockDataclass:
                                reason_str = f"PersistentExecutionFailure_{exec_failure_count}_attempts"
                                await kb_fail.assert_fact(self._Predicate("goalFailed", (goal_id_for_exec_fail, reason_str), True, timestamp=time.time())) # type: ignore
                            
                            self.internal_state_manager.generate_pain_from_goal_failure(
                                goal_to_update_post_action, 
                                failure_type="PersistentExecutionFailure"
                            )
                            
                            if goal_id_for_exec_fail in self._active_goal_execution_failure_count:
                                del self._active_goal_execution_failure_count[goal_id_for_exec_fail]
                        
                goals_to_remove = []
                # current_time_for_pain = time.time() # Already defined earlier if needed here

                for goal_in_list in self.active_goals: 
                    if not hasattr(goal_in_list, 'status') or current_goal_status_enum_exec == MockEnum: # type: ignore
                        continue
                    
                    goal_id_for_stats = getattr(goal_in_list, 'id', None)
                    plan_outcome_bool_for_stats: Optional[bool] = None
                    
                    if goal_in_list.status == current_goal_status_enum_exec.ACHIEVED: # type: ignore
                        goals_to_remove.append(goal_in_list)
                        plan_outcome_bool_for_stats = True
                        
                    elif goal_in_list.status == current_goal_status_enum_exec.FAILED: # type: ignore
                        goals_to_remove.append(goal_in_list)
                        plan_outcome_bool_for_stats = False
                        # Pain generation is now handled when goal is marked FAILED (above or in planning)
                
                    if plan_outcome_bool_for_stats is not None and goal_id_for_stats:
                        htn_p_inst_stats = getattr(self, "htn_planner", None)
                        if htn_p_inst_stats and hasattr(htn_p_inst_stats, 'update_method_performance_stats') and \
                           hasattr(htn_p_inst_stats, '_last_top_method_for_goal') and \
                           isinstance(getattr(htn_p_inst_stats, '_last_top_method_for_goal'), dict):
                            
                            method_info_tuple = getattr(htn_p_inst_stats, '_last_top_method_for_goal').get(goal_id_for_stats)
                            
                            if isinstance(method_info_tuple, tuple) and len(method_info_tuple) == 2:
                                task_name_for_stats, method_name_used_for_stats = method_info_tuple
                                
                                await htn_p_inst_stats.update_method_performance_stats( # type: ignore
                                    task_name_for_stats, method_name_used_for_stats, plan_outcome_bool_for_stats
                                )
                                getattr(htn_p_inst_stats, '_last_top_method_for_goal').pop(goal_id_for_stats, None)
                            elif method_info_tuple is not None: 
                                logger_agent_controller.warning(f"Corrupt method_info_tuple '{method_info_tuple}' for goal '{goal_id_for_stats}' in HTNPlanner._last_top_method_for_goal.")
                                getattr(htn_p_inst_stats, '_last_top_method_for_goal').pop(goal_id_for_stats, None) 

                logger_agent_controller.info(
                    f"PHP_TRIGGER - Delegating P/H/P updates post-action/learning. "
                    f"GoalAchievedThisCycle: {goal_just_achieved_this_cycle}, "
                    f"AchievedGoalPrio: {priority_of_achieved_goal if goal_just_achieved_this_cycle else 'N/A'}."
                )
                await self.internal_state_manager.perform_upkeep_post_action_learning(
                    goal_just_achieved_this_cycle,
                    priority_of_achieved_goal,
                    achieved_goal_object=(active_goal if goal_just_achieved_this_cycle else goal_to_update_post_action if goal_to_update_post_action and goal_just_achieved_this_cycle else None) 
                )
                
                if goals_to_remove:
                    removed_ids = [g.id for g in goals_to_remove if hasattr(g,'id')]
                    logger_agent_controller.info(f"Removing {len(goals_to_remove)} completed/failed goals: {removed_ids}")
                    self.active_goals = [g for g in self.active_goals if g not in goals_to_remove]
                    if goal_to_update_post_action in goals_to_remove:
                        goal_to_update_post_action = None
                        self.current_plan = None
                        self._active_goal_modification_hints = None # Clear hints when goal is removed
                
                # Clear hints if active goal changed due to selection logic, and wasn't just achieved/failed
                if active_goal and self._oscar_get_active_goal() and active_goal.id != self._oscar_get_active_goal().id: # type: ignore
                     self._active_goal_modification_hints = None


                self.profiler.start_section("model_updates"); ems_output = None
                if self._last_action_result and isinstance(self._last_action_result, dict):
                    predictive_world_model_inst_mu = getattr(self, "predictive_world_model", None)
                    if predictive_world_model_inst_mu and hasattr(predictive_world_model_inst_mu, 'process'):
                        current_context_for_pwm_update = await self._oscar_get_cognitive_state(skip_component_statuses=True) 
                        
                        update_input_pwm = {
                            "update_request": {
                                "prediction": prediction_from_pwm_step_1b if prediction_from_pwm_step_1b else {}, 
                                "actual_result": { 
                                    "type": self._last_action_executed.get('type') if isinstance(self._last_action_executed, dict) else None,
                                    "params": self._last_action_executed.get('params', {}) if isinstance(self._last_action_executed, dict) else {},
                                    "outcome": self._last_action_result.get('outcome'), 
                                    "error": self._last_action_result.get('error'), 
                                    "result_data": self._last_action_result.get('result_data'), 
                                    "context_at_execution": current_context_for_pwm_update 
                                },
                                "buffered_pre_action_state": self._buffered_pre_action_state_for_pwm 
                            }
                        }
                        
                        process_method_pwm_update = getattr(predictive_world_model_inst_mu, 'process')
                        if asyncio.iscoroutinefunction(process_method_pwm_update):
                            pwm_update_output = await process_method_pwm_update(update_input_pwm)
                            if pwm_update_output and isinstance(pwm_update_output, dict):
                                prediction_error_details = pwm_update_output.get("last_prediction_error_details")
                                self.last_prediction_error_for_attention = prediction_error_details
                                current_cycle_prediction_error_for_pwm_update = prediction_error_details
                            else: logger_agent_controller.warning("PWM process for update returned no/invalid output.")
                        else: logger_agent_controller.warning(f"PredictiveWorldModel ({type(predictive_world_model_inst_mu)}) process for update not async.")
                    elif predictive_world_model_inst_mu: logger_agent_controller.warning(f"PWM instance ({type(predictive_world_model_inst_mu)}) has no 'process' method for update.")

                    dsm_status = None 
                    dynamic_self_model_inst = getattr(self, "dynamic_self_model", None)
                    if dynamic_self_model_inst and hasattr(dynamic_self_model_inst, 'get_status'): 
                         get_status_method_dsm = getattr(dynamic_self_model_inst, 'get_status')
                         if asyncio.iscoroutinefunction(get_status_method_dsm): dsm_status = await get_status_method_dsm()
                    
                    current_active_goal_for_model_update = self._oscar_get_active_goal()
                    model_update_context = {
                        "last_action_type": self._last_action_executed.get("type") if isinstance(self._last_action_executed, dict) else None,
                        "action_outcome": self._last_action_result.get("outcome"),
                        "action_params": self._last_action_executed.get("params", {}) if isinstance(self._last_action_executed, dict) else {},
                        "action_error": self._last_action_result.get("error"),
                        "action_result_data": self._last_action_result.get("result_data"),
                        "phenomenal_state": self.current_phenomenal_state,
                        "cognitive_state": await self._oscar_get_cognitive_state(),
                        "active_goal": current_active_goal_for_model_update,
                        "self_model_summary": dsm_status 
                    }

                    if dynamic_self_model_inst and hasattr(dynamic_self_model_inst, 'process'):
                        process_method_dsm = getattr(dynamic_self_model_inst, 'process')
                        if asyncio.iscoroutinefunction(process_method_dsm): await process_method_dsm(model_update_context)
                        else: logger_agent_controller.warning("DSM process not async.")

                    emergent_motivation_system_inst = getattr(self, "emergent_motivation_system", None)
                    if emergent_motivation_system_inst and hasattr(emergent_motivation_system_inst, 'process'):
                        proc_ems = getattr(emergent_motivation_system_inst, 'process')
                        if asyncio.iscoroutinefunction(proc_ems): ems_output = await proc_ems(model_update_context)
                        else: logger_agent_controller.warning("EMS process not async."); ems_output = None
                    else: ems_output = None
                self.profiler.end_section() 

                # --- 11. Narrative Update ---
                self.profiler.start_section("narrative")
                narrative_constructor_inst = getattr(self, "narrative_constructor", None)
                if narrative_constructor_inst and hasattr(narrative_constructor_inst, 'process') and isinstance(self._last_action_result, dict):
                    narrative_input = {
                        "phenomenal_state": self.current_phenomenal_state,
                        "last_action_result": self._last_action_result,
                        "loop_info": loop_info,
                        "meta_analysis": meta_analysis,
                        "prediction_error": current_cycle_prediction_error_for_pwm_update 
                    }
                    if ems_output and isinstance(ems_output, dict) and "current_drives" in ems_output:
                         narrative_input["current_drives"] = ems_output["current_drives"]
                    elif ems_output is None and hasattr(self, 'emergent_motivation_system') and self.emergent_motivation_system:
                        try:
                            ems_status_for_narr = await self.emergent_motivation_system.get_status() # type: ignore
                            if ems_status_for_narr and "current_drives" in ems_status_for_narr:
                                narrative_input["current_drives"] = ems_status_for_narr["current_drives"]
                        except Exception as e_ems_narr:
                            logger_agent_controller.warning(f"Could not get EMS status for narrative input: {e_ems_narr}")

                    process_method_nc = getattr(narrative_constructor_inst, 'process')
                    if asyncio.iscoroutinefunction(process_method_nc): await process_method_nc(narrative_input)
                    else: logger_agent_controller.warning("NC process not async.")
                self.profiler.end_section()
                
                # --- NEW: Log Full Cycle Snapshot (for StateHistoryLogger) ---
                state_logger_inst = getattr(self, "state_history_logger", None)
                if state_logger_inst and hasattr(state_logger_inst, "log_full_cycle_snapshot"):
                    all_statuses_this_cycle = {}
                    for comp_name_log, comp_inst_log in self.components.items():
                        if hasattr(comp_inst_log, 'get_status'):
                            try:
                                get_status_method_log = getattr(comp_inst_log, 'get_status')
                                if asyncio.iscoroutinefunction(get_status_method_log):
                                    all_statuses_this_cycle[comp_name_log] = await get_status_method_log()
                                else: 
                                    all_statuses_this_cycle[comp_name_log] = get_status_method_log() # type: ignore
                            except Exception as e_get_status_log:
                                logger_agent_controller.warning(f"Error getting status for '{comp_name_log}' during cycle snapshot: {e_get_status_log}")
                                all_statuses_this_cycle[comp_name_log] = {"error": "status_unavailable"}
                    
                    p_state_summary_for_log = {}
                    if self.current_phenomenal_state: 
                        _PState_ac_log = globals().get("PhenomenalState")
                        if _PState_ac_log and isinstance(self.current_phenomenal_state, _PState_ac_log):
                            p_state_summary_for_log = {
                                "intensity": round(getattr(self.current_phenomenal_state, 'intensity', 0.0), 3),
                                "valence": round(getattr(self.current_phenomenal_state, 'valence', 0.0), 3),
                                "integration_level": round(getattr(self.current_phenomenal_state, 'integration_level', 0.0), 3),
                                "distinct_source_count": getattr(self.current_phenomenal_state, 'distinct_source_count', 0),
                                "content_diversity_lexical": round(getattr(self.current_phenomenal_state, 'content_diversity_lexical', 0.0), 3),
                                "shared_concept_count_gw": round(getattr(self.current_phenomenal_state, 'shared_concept_count_gw', 0.0), 3),
                                "content_keys": list(getattr(self.current_phenomenal_state, 'content', {}).keys())[:5] 
                            }
                        elif isinstance(self.current_phenomenal_state, dict): 
                            p_state_summary_for_log = {k:v for k,v in self.current_phenomenal_state.items() if k != 'content'}
                            p_state_summary_for_log['content_keys'] = list(self.current_phenomenal_state.get('content',{}).keys())[:5]

                    full_cycle_data_for_log = {
                        "cycle_count": self.cycle_count,
                        "phenomenal_state_summary": p_state_summary_for_log,
                        "workspace_content_snapshot": self.global_workspace_content.copy() if self.global_workspace_content else {},
                        "all_component_statuses_this_cycle": all_statuses_this_cycle,
                        "php_levels_snapshot": {"pain": self.pain_level, "happiness": self.happiness_level, "purpose": self.purpose_level},
                        "active_goal_snapshot": {
                            "id": active_goal.id if active_goal and hasattr(active_goal, 'id') else None,
                            "desc": active_goal.description if active_goal and hasattr(active_goal, 'description') else None,
                            "status": active_goal.status.name if active_goal and hasattr(active_goal, 'status') and hasattr(active_goal.status, 'name') else None
                        },
                        "last_action_result_snapshot": self._last_action_result.copy() if self._last_action_result else {}
                    }
                    state_logger_inst.log_full_cycle_snapshot(full_cycle_data_for_log) # type: ignore
                # --- END LOG Full Cycle Snapshot ---


                # --- 12. Enhanced Performance Optimization ---
                self.profiler.start_section("optimization")
                cycle_profile = self.profiler.get_cycle_profile()
                optimization_hints: Optional[Dict[str, Any]] = None
                recovery_mode_to_apply: Optional[Any] = None

                performance_optimizer_inst = getattr(self, "performance_optimizer", None)
                if performance_optimizer_inst and hasattr(performance_optimizer_inst, 'process') and hasattr(performance_optimizer_inst, 'get_status'):
                    optimization_input = {"cycle_profile": cycle_profile}
                    process_method_po = getattr(performance_optimizer_inst, 'process')
                    po_output = None
                    if asyncio.iscoroutinefunction(process_method_po): po_output = await process_method_po(optimization_input)
                    else: logger_agent_controller.warning("Perf Opt process not async.")

                    if po_output and isinstance(po_output.get("performance_analysis"), dict):
                        performance_analysis_results = po_output["performance_analysis"]
                        optimization_hints = performance_analysis_results
                        recovery_mode_to_apply = performance_analysis_results.get("recovery_mode_needed")

                        po_status = None
                        get_status_method_po = getattr(performance_optimizer_inst, 'get_status')
                        if asyncio.iscoroutinefunction(get_status_method_po): po_status = await get_status_method_po()
                        else: logger_agent_controller.warning("Perf Opt get_status not async.")

                        active_po_adjustments = po_status.get("active_config_adjustments", {}) if isinstance(po_status, dict) else {}

                        if active_po_adjustments:
                            logger_agent_controller.debug(f"PO adjustments available: {active_po_adjustments}")
                            for component_key, comp_adjustments in active_po_adjustments.items():
                                if component_key not in self.config: self.config[component_key] = {}
                                if isinstance(self.config.get(component_key), dict) and isinstance(comp_adjustments, dict):
                                    for param_key, new_value in comp_adjustments.items():
                                        current_config_value = self.config[component_key].get(param_key)
                                        if current_config_value != new_value:
                                            logger_agent_controller.info(f"PERF_ADJUST: Updating agent.config: [{component_key}].{param_key} = {new_value} (was {current_config_value})")
                                            self.config[component_key][param_key] = new_value
                                            self._config_updated_by_po_in_cycle = True
                                        else: logger_agent_controller.debug(f"PERF_ADJUST: No update needed for agent.config: [{component_key}].{param_key} is already {new_value}")
                                else: logger_agent_controller.warning(f"Cannot apply PO adjustments for '{component_key}': config section or adjustments not dicts.")
                            if self._config_updated_by_po_in_cycle: logger_agent_controller.info("AgentController.config updated with PerformanceOptimizer adjustments.")
                        else: logger_agent_controller.debug("No active adjustments found from PerformanceOptimizer status.")

                self.profiler.end_section() # End "optimization" section

                # Handle recovery if suggested by PO or ErrorRecoverySystem
                if recovery_mode_to_apply and self._RecoveryModeEnum and self._RecoveryModeEnum != MockEnum and isinstance(recovery_mode_to_apply, self._RecoveryModeEnum):
                    logger_agent_controller.warning(f"Recovery mode suggested: {recovery_mode_to_apply.name}. Health score was: {optimization_hints.get('current_health_score', 'N/A') if optimization_hints else 'N/A'}")
                    await self._oscar_handle_recovery(recovery_mode_to_apply)
                    if not self._is_running_flag.is_set(): break
                elif recovery_mode_to_apply is not None:
                     logger_agent_controller.error(f"Invalid recovery_mode suggested type: {type(recovery_mode_to_apply)}")

                # Re-fetch active goal for UI update after potential removals/changes
                active_goal_for_ui = self._oscar_get_active_goal()
                await self._oscar_send_throttled_ui_updates(active_goal_for_ui)

            except asyncio.CancelledError: logger_agent_controller.info("Cognitive cycle task cancelled."); break
            except Exception as e:
                logger_agent_controller.exception(f"Error in cognitive cycle {self.cycle_count}: {e}"); self._log_to_ui("error", f"Cycle Error: {e}")
                suggested_recovery_mode = None
                try:
                    error_recovery_inst = getattr(self, "error_recovery", None)
                    if error_recovery_inst and hasattr(error_recovery_inst, 'handle_error'):
                        cognitive_state_at_error = await self._oscar_get_cognitive_state()
                        active_goal_at_error = self._oscar_get_active_goal() # Fetch current goal state
                        current_goal_desc_at_error = active_goal_at_error.description if active_goal_at_error and hasattr(active_goal_at_error, 'description') else "None"
                        error_context = {
                            "cycle": self.cycle_count, "timestamp": time.time(),
                            "cognitive_state_summary": cognitive_state_at_error,
                            "current_goal_desc": current_goal_desc_at_error,
                            "last_action_type": self._last_action_executed.get("type") if isinstance(self._last_action_executed, dict) else None,
                            "exception_type": type(e).__name__,
                        }
                        handle_error_method = getattr(error_recovery_inst, 'handle_error')
                        if asyncio.iscoroutinefunction(handle_error_method): suggested_recovery_mode = await handle_error_method(e, error_context)
                        else: logger_agent_controller.warning("ErrorRecovery handle_error not async.")

                    recovery_mode_to_apply = suggested_recovery_mode

                    if self._RecoveryModeEnum != MockEnum and isinstance(recovery_mode_to_apply, self._RecoveryModeEnum):
                        await self._oscar_handle_recovery(recovery_mode_to_apply)
                    elif recovery_mode_to_apply is not None:
                        logger_agent_controller.error(f"Invalid recovery mode from ErrorRecovery: {type(recovery_mode_to_apply)}"); await asyncio.sleep(1.0)
                    else: logger_agent_controller.debug("No recovery needed or suggested by ErrorRecovery."); await asyncio.sleep(1.0)
                except asyncio.CancelledError: logger_agent_controller.info("Recovery handling cancelled."); break
                except Exception as recovery_error: logger_agent_controller.exception(f"CRITICAL: Error during error recovery attempt: {recovery_error}"); await asyncio.sleep(5.0)

            # --- FINAL CHECKS BEFORE NEXT CYCLE / SHUTDOWN ---
            shutdown_triggered_by_php = self.internal_state_manager.check_existential_thresholds()
            if shutdown_triggered_by_php:
                logger_agent_controller.info("PHP_TRIGGER - Existential threshold met. Shutdown initiated by InternalStateUpkeepManager.")
                break # Exit the while loop if shutdown was initiated by PHP thresholds

            cycle_end_time = time.monotonic(); elapsed = cycle_end_time - cycle_start_time
            sleep_duration = max(0, self.cycle_delay_s - elapsed)
            if elapsed > self.cycle_delay_s:
                logger_agent_controller.warning(f"Cycle {self.cycle_count} overrun: {elapsed:.4f}s > {self.cycle_delay_s:.4f}s")
                self._log_to_ui("warn", f"Cycle overrun: {elapsed:.3f}s")
            
            if self._is_running_flag.is_set() and self._is_paused_event.is_set(): # Only sleep if not paused
                await asyncio.sleep(sleep_duration)
            elif self._is_running_flag.is_set() and not self._is_paused_event.is_set():
                await asyncio.sleep(0.01) 

            logger_agent_controller.debug(f"Cycle {self.cycle_count} END. Elapsed: {elapsed:.4f}s, Slept: {sleep_duration:.4f}s")

        logger_agent_controller.info("Agent cognitive loop stopped.")


    def pause_agent(self):
        """Pauses the agent's main cognitive cycle loop."""
        if self.agent_state == self._AgentState.RUNNING:
            if self._is_paused_event.is_set(): # Only clear if not already paused
                self._is_paused_event.clear()
                self._update_ui_state(self._AgentState.PAUSED) # Update state # type: ignore
                logger_agent_controller.info("Agent cognitive cycle PAUSED by external request.")
                self._log_to_ui("info", "Agent PAUSED.")
        else:
            logger_agent_controller.warning(
                f"Agent cannot be paused. Current state: {self.agent_state.name if hasattr(self.agent_state, 'name') else self.agent_state}"
            )

    def resume_agent(self):
        """Resumes the agent's main cognitive cycle loop."""
        if self.agent_state == self._AgentState.PAUSED: # type: ignore
            if not self._is_paused_event.is_set(): # Only set if actually paused
                self._cycles_ran_since_last_resume = 0 # Reset auto-pause counter
                self._is_paused_event.set()
                self._update_ui_state(self._AgentState.RUNNING) # Update state
                logger_agent_controller.info("Agent cognitive cycle RESUMED by external request.")
                self._log_to_ui("info", "Agent RESUMED.")
        else:
            logger_agent_controller.warning(
                f"Agent cannot be resumed. Current state: {self.agent_state.name if hasattr(self.agent_state, 'name') else self.agent_state}"
            )

    def set_auto_pause(self, cycles: Optional[int]):
        """
        Sets the agent to automatically pause after a specified number of cycles.
        If cycles is None or <= 0, auto-pause is disabled.
        """
        if cycles is not None and cycles > 0:
            self._auto_pause_cycle_target = cycles
            self._auto_pause_enabled = True
            self._cycles_ran_since_last_resume = 0 # Reset counter
            logger_agent_controller.info(f"Auto-pause enabled. Agent will pause after {cycles} cycles.")
            self._log_to_ui("info", f"Auto-pause set for {cycles} cycles.")
        else:
            self._auto_pause_cycle_target = None
            self._auto_pause_enabled = False
            logger_agent_controller.info("Auto-pause disabled.")
            self._log_to_ui("info", "Auto-pause disabled.")

    async def _oscar_generate_or_select_goal(self, newly_created_user_goal: Optional['Goal'] = None) -> Optional['Goal']: # type: ignore
        logger_agent_controller.info(f"AC_GEN_SELECT_GOAL: Attempting to select/generate. Newly created passed: {newly_created_user_goal.description if newly_created_user_goal else 'None'}. Current active_goals count: {len(self.active_goals)}")

        if newly_created_user_goal and isinstance(newly_created_user_goal, self._Goal if self._Goal else object): # Check type
            # If a new user goal was just created (either directly or from LLM), prioritize it
            logger_agent_controller.info(f"AC_GEN_SELECT_GOAL: Prioritizing newly created user goal: '{newly_created_user_goal.description}'")
            # Ensure it's marked ACTIVE if not already
            if hasattr(newly_created_user_goal, 'status') and self._GoalStatus:
                newly_created_user_goal.status = self._GoalStatus.ACTIVE # type: ignore
            self._active_goal_modification_hints = None # Clear hints for new goal
            return newly_created_user_goal

        if self.active_goals:
            valid_goals = [g for g in self.active_goals if isinstance(g, self._Goal if self._Goal else object) and hasattr(g, 'status')]
            active_goals_list = [g for g in valid_goals if g.status == (self._GoalStatus.ACTIVE if self._GoalStatus else "active")] # type: ignore
            
            if active_goals_list:
                active_goals_list.sort(key=lambda g: (- (g.priority if hasattr(g,'priority') and g.priority is not None else 0.0),
                                                      g.creation_time if hasattr(g,'creation_time') and g.creation_time is not None else 0.0))
                selected_existing_goal = active_goals_list[0]
                logger_agent_controller.info(f"AC_GEN_SELECT_GOAL: Selected existing active goal: '{selected_existing_goal.description}' (Prio: {getattr(selected_existing_goal, 'priority', 0.0)})")
                # If the selected goal is different from the one whose plan was just (potentially) invalidated by hints, clear hints.
                if self._active_goal_modification_hints and self._plan_generated_for_goal_id != selected_existing_goal.id:
                    logger_agent_controller.debug(f"AC_GEN_SELECT_GOAL: Selected new active goal {selected_existing_goal.id}, clearing stale hints for {self._plan_generated_for_goal_id}.")
                    self._active_goal_modification_hints = None
                return selected_existing_goal

        current_curiosity = 0.5 
        if hasattr(self, 'emergent_motivation_system') and self.emergent_motivation_system is not None:
            ems_drives = getattr(self.emergent_motivation_system, 'drives', {})
            curiosity_drive_params = ems_drives.get('curiosity', {})
            current_curiosity = curiosity_drive_params.get('value', 0.5)
        can_generate_default_goal = (
            (self.cycle_count - self._last_default_observe_completion_cycle) >= self.default_goal_cooldown_cycles and
            current_curiosity >= self.min_curiosity_for_observe
        )
        if can_generate_default_goal:
            default_goal = self._create_goal_from_descriptor(DEFAULT_OBSERVE_AND_LEARN_GOAL_DESC, priority=DEFAULT_OBSERVE_GOAL_PRIORITY) # type: ignore
            if default_goal:
                if not any(g.description == default_goal.description and g.status == (self._GoalStatus.ACTIVE if self._GoalStatus else "active") for g in self.active_goals): # type: ignore
                    self.active_goals.append(default_goal)
                logger_agent_controller.info(f"AC_GEN_SELECT_GOAL: Generated default goal: '{default_goal.description}'")
                self._active_goal_modification_hints = None # Clear hints for new default goal
                return default_goal
        
        logger_agent_controller.warning("AC_GEN_SELECT_GOAL: No goal selected or generated.")
        return None


    # --- Helper Methods (_oscar_...) ---

    async def _oscar_send_throttled_ui_updates(self, current_active_goal_for_ui: Optional['Goal']): # type: ignore
        current_time = time.monotonic()
        if (current_time - self._last_ui_meter_update_time) < self.ui_update_interval_s:
            return
        if not (hasattr(self, 'ui_queue') and self.ui_queue):
            return

        try:
            # Agent CS Level (existing)
            current_conscious_state_enum_ui = self._ConsciousState
            if hasattr(self, 'consciousness_level') and current_conscious_state_enum_ui != MockEnum and isinstance(self.consciousness_level, current_conscious_state_enum_ui) and hasattr(self.consciousness_level, 'name'): # type: ignore
                self.ui_queue.put_nowait(("agent_cs_level_update", self.consciousness_level.name)) # type: ignore

            # Workspace Load (existing)
            global_workspace_inst_ui = getattr(self, "global_workspace", None)
            if global_workspace_inst_ui and hasattr(global_workspace_inst_ui, 'get_status'):
                get_status_method_ws = getattr(global_workspace_inst_ui, 'get_status'); ws_status = None
                if asyncio.iscoroutinefunction(get_status_method_ws): ws_status = await get_status_method_ws()
                else: ws_status = get_status_method_ws() # type: ignore
                if isinstance(ws_status, dict):
                    self.ui_queue.put_nowait(("agent_workspace_update", {"load": ws_status.get('current_load', 0), "capacity": ws_status.get('capacity', 0)}))


            # Drives (existing)
            emergent_motivation_system_inst_ui = getattr(self, "emergent_motivation_system", None)
            if emergent_motivation_system_inst_ui and hasattr(emergent_motivation_system_inst_ui, 'get_status'):
                get_status_method_ems = getattr(emergent_motivation_system_inst_ui, 'get_status'); ems_status = None
                if asyncio.iscoroutinefunction(get_status_method_ems): ems_status = await get_status_method_ems()
                else: ems_status = get_status_method_ems() # type: ignore
                if isinstance(ems_status, dict):
                    drive_values = ems_status.get("current_drives")
                    if isinstance(drive_values, dict): self.ui_queue.put_nowait(("agent_drives_update", drive_values))


            # Goal (existing)
            current_goal_class_ui = self._Goal
            if current_active_goal_for_ui and current_goal_class_ui != MockDataclass and isinstance(current_active_goal_for_ui, current_goal_class_ui): # type: ignore
                status_name_ui = "N/A"
                if hasattr(current_active_goal_for_ui, 'status') and current_active_goal_for_ui.status is not None: # type: ignore
                    if hasattr(current_active_goal_for_ui.status, 'name'): # type: ignore
                        status_name_ui = current_active_goal_for_ui.status.name # type: ignore
                    else:
                        status_name_ui = str(current_active_goal_for_ui.status) # type: ignore
                self.ui_queue.put_nowait(("agent_goal_update", {"description": current_active_goal_for_ui.description, "status": status_name_ui})) # type: ignore
            else:
                self.ui_queue.put_nowait(("agent_goal_update", None))

            # --- NEW/VERIFIED: P/H/P and related telemetry ---
            php_payload = {
                "pain_level": round(self.pain_level, 3),
                "happiness_level": round(self.happiness_level, 3),
                "purpose_level": round(self.purpose_level, 3),
                "agent_age_cycles": self.agent_age_cycles,
                "active_pain_sources_count": len(self.active_pain_sources),
                "top_pain_sources_summary": [] # Initialize
            }
            
            # Ensure self._PainSource is the class, not an instance. And not a mock.
            _PainSource_class_for_ui = self._PainSource 
            if _PainSource_class_for_ui and _PainSource_class_for_ui != MockDataclass: 
                # Sort active, unresolved pain sources by current_intensity descending
                sorted_unresolved_pain = sorted(
                    [ps for ps in self.active_pain_sources if hasattr(ps, 'is_resolved') and not ps.is_resolved and hasattr(ps, 'current_intensity') and ps.current_intensity > getattr(self.internal_state_manager, '_pain_event_min_intensity_to_retain', 0.01)],
                    key=lambda p: p.current_intensity if hasattr(p, 'current_intensity') else 0.0,
                    reverse=True
                )
                for ps_summary_item in sorted_unresolved_pain[:2]: # Take top 2
                    if hasattr(ps_summary_item, 'id') and hasattr(ps_summary_item, 'description') and hasattr(ps_summary_item, 'current_intensity'):
                        php_payload["top_pain_sources_summary"].append({
                            "id": ps_summary_item.id,
                            "description": ps_summary_item.description[:30] + "..." if len(ps_summary_item.description) > 30 else ps_summary_item.description,
                            "intensity": round(ps_summary_item.current_intensity, 2)
                        })
            
            self.ui_queue.put_nowait(("agent_php_update", php_payload))
            logger_agent_controller.debug(f"UI_TELEMETRY - Pushed PHP update to UI queue: {php_payload}")
            # --- END NEW/VERIFIED ---

            self._last_ui_meter_update_time = current_time
        except queue.Full:
            logger_agent_controller.warning(f"UI queue full during throttled update.")
        except Exception as e:
            logger_agent_controller.error(f"Error during throttled UI update: {e}", exc_info=True)

    async def _oscar_perceive(self) -> Dict[str, Any]:
        percepts = { "timestamp": time.time(), "system_state": None, "user_input": None, "internal_error": None }
        if PSUTIL_AVAILABLE_CTRL:
             try: percepts["system_state"] = {"cpu_percent": psutil.cpu_percent(), "memory_percent": psutil.virtual_memory().percent }
             except Exception as e: logger_agent_controller.warning(f"Failed psutil check: {e}")
        user_input_text = None
        try: user_input_text = self._user_input_queue.get_nowait(); self._user_input_queue.task_done()
        except asyncio.QueueEmpty: pass
        except Exception as e: logger_agent_controller.error(f"Error getting user input: {e}", exc_info=True)
        if user_input_text: percepts["user_input"] = user_input_text
        return percepts

    def _oscar_peek_next_action(self) -> Optional[Dict]:
        if self.current_plan and isinstance(self.current_plan, list) and len(self.current_plan) > 0: return self.current_plan[0]
        return None

    async def _oscar_gather_attention_candidates(self, percepts: Dict[str, Any], current_cycle_active_goal_id: Optional[str]) -> Dict[str, Any]:
        candidates = {}; timestamp_now = time.time()

        if isinstance(percepts, dict):
            for key, value in percepts.items():
                if value is not None:
                    weight = 0.95 if key == "user_input" else (0.7 if key == "system_state" else 0.6)
                    candidates[f"percept_{key}"] = {"content": value, "weight_hint": weight, "timestamp": percepts.get("timestamp", timestamp_now)}

        current_goal_class_attn = self._Goal
        if current_goal_class_attn != MockDataclass:
            for goal_item in self.active_goals:
                if isinstance(goal_item, current_goal_class_attn):
                    status_name_attn = "N/A"
                    current_goal_id = goal_item.id if hasattr(goal_item, 'id') else None
                    current_goal_priority = float(goal_item.priority) if hasattr(goal_item, 'priority') and goal_item.priority is not None else 1.0
                    current_goal_description = goal_item.description if hasattr(goal_item, 'description') else "Unknown Goal"
                    current_goal_creation_time = goal_item.creation_time if hasattr(goal_item, 'creation_time') else timestamp_now

                    if hasattr(goal_item, 'status') and goal_item.status is not None:
                        if hasattr(goal_item.status, 'name'): status_name_attn = goal_item.status.name
                        else: status_name_attn = str(goal_item.status)

                    base_priority = current_goal_priority
                    if current_goal_id and current_goal_id == current_cycle_active_goal_id:
                        weight = base_priority * 0.95
                        goal_label = f"ActiveGoal: {current_goal_description}"
                    else:
                        weight = base_priority * 0.2
                        goal_label = f"PendingGoal: {current_goal_description}"

                    if current_goal_id:
                        candidates[f"goal_{current_goal_id}"] = {
                            "content": f"{goal_label} (Status: {status_name_attn})",
                            "weight_hint": weight,
                            "timestamp": current_goal_creation_time
                        }

        current_phenomenal_state_class_attn = self._PhenomenalState
        current_phenomenal_state_inst = self.current_phenomenal_state # Use the one from the current cycle
        if current_phenomenal_state_class_attn != MockDataclass and \
           isinstance(current_phenomenal_state_inst, current_phenomenal_state_class_attn) and \
           hasattr(current_phenomenal_state_inst, 'timestamp') and \
           hasattr(current_phenomenal_state_inst, 'content'):
            time_delta = timestamp_now - current_phenomenal_state_inst.timestamp;
            candidates["last_experience"] = {
                "content": current_phenomenal_state_inst.content,
                "weight_hint": max(0, 0.5 * math.exp(-time_delta / 30.0)),
                "timestamp": current_phenomenal_state_inst.timestamp
            }
        
        # --- Add Active, Unresolved Pain Sources as Attention Candidates ---
        if self._PainSource is not MockDataclass: # Ensure PainSource class is real
            current_cycle_num = self.cycle_count
            for ps_event in self.active_pain_sources:
                if ps_event.is_resolved or ps_event.current_intensity < self.internal_state_manager.agent_controller.config.get("internal_states", {}).get("pain_event_min_intensity_to_retain", 0.01): # type: ignore
                    continue

                # Use PainEventTracker to decide on suppression
                if hasattr(self, 'pain_event_tracker') and self.pain_event_tracker.should_suppress_rumination(ps_event.id, current_cycle_num):
                    logger_agent_controller.debug(
                        f"Pain event '{ps_event.id}' ({ps_event.description[:30]}) suppressed from attention by PainEventTracker."
                    )
                    continue


                candidate_id = f"pain_event_{ps_event.id}"
                timestamp_created_cycle = getattr(ps_event, 'timestamp_created_cycle', current_cycle_num)


                candidates[candidate_id] = {
                    "content": { 
                        "type": "InternalState_Pain",
                        "description": ps_event.description,
                        "current_intensity": round(ps_event.current_intensity, 2), # type: ignore
                        "source_type": ps_event.type, # type: ignore
                        "age_cycles": current_cycle_num - timestamp_created_cycle 
                    },
                    "weight_hint": ps_event.current_intensity * self._pain_attention_distraction_factor, # type: ignore
                    "timestamp": ps_event.timestamp_created, # type: ignore
                    "source_component": "InternalStateMonitor_Pain" 
                }
                logger_agent_controller.debug(f"Added pain event '{candidate_id}' to attention candidates. Hint: {candidates[candidate_id]['weight_hint']:.2f}")
        # --- End Adding Pain Sources ---

        return candidates

    async def _oscar_get_relevant_memories(self, broadcast_content: Dict[str, Any]) -> List[Any]:
        retrieved = []
        if "goal_active" in broadcast_content: retrieved.append("memory_related_to_active_goal")
        if "percept_high_priority" in broadcast_content: retrieved.append("memory_related_to_urgent_percept")
        return retrieved if retrieved else ["default_memory_placeholder"]

    def _oscar_get_current_action_context(self) -> Dict[str, Any]:
        last_action_type = self._last_action_executed.get("type") if isinstance(self._last_action_executed, dict) else None
        last_outcome = self._last_action_result.get("outcome") if isinstance(self._last_action_result, dict) else None
        last_error = self._last_action_result.get("error") if isinstance(self._last_action_result, dict) else None
        return { "last_action_type": last_action_type, "last_action_outcome": last_outcome, "last_action_error": last_error, }

    async def _oscar_get_cognitive_state(self, skip_component_statuses: bool = False) -> Dict[str, Any]:
        cs_level_name = self.consciousness_level.name if hasattr(self.consciousness_level, 'name') else str(self.consciousness_level)
        active_goal_cog = self._oscar_get_active_goal() 

        current_goal_desc_cog = None
        current_goal_type_cog = self._get_active_goal_type() # Get goal type
        current_goal_status_cog = None
        if active_goal_cog and hasattr(active_goal_cog, 'description'): current_goal_desc_cog = active_goal_cog.description
        if active_goal_cog and hasattr(active_goal_cog, 'status') and active_goal_cog.status:
            current_goal_status_cog = active_goal_cog.status.name if hasattr(active_goal_cog.status, 'name') else str(active_goal_cog.status)

        # --- Prepare php_levels and drives sub-dictionaries ---
        php_levels_data = {
            "pain": self.pain_level,
            "happiness": self.happiness_level,
            "purpose": self.purpose_level
        }
        
        drive_data_for_cog_state = {}
        ems_cog_state_inst = getattr(self, "emergent_motivation_system", None)
        if ems_cog_state_inst and hasattr(ems_cog_state_inst, 'drives') and isinstance(ems_cog_state_inst.drives, dict): # type: ignore
            drive_data_for_cog_state = {
                name: params.get("value") 
                for name, params in ems_cog_state_inst.drives.items() # type: ignore
                if isinstance(params, dict) and "value" in params
            }


        state = {
            "timestamp": time.time(), 
            "consciousness_level": cs_level_name, 
            "current_cs_level_name": cs_level_name, 
            "active_goal_count": len(self.active_goals),
            "current_goal_desc": current_goal_desc_cog,
            "current_goal_status": current_goal_status_cog,
            "current_goal_type": current_goal_type_cog, 
            "current_plan_length": len(self.current_plan) if isinstance(self.current_plan, list) else 0,
            "workspace_load": 0, 
            "emotional_valence": 0.0, 
            "integration_level": 0.0, 
            
            "php_levels": php_levels_data, 
            "drives": drive_data_for_cog_state, 
            "pain_level": self.pain_level,
            "happiness_level": self.happiness_level,
            "purpose_level": self.purpose_level,
            # active_goal_details for Value System (simpler structure)
            "active_goal_details": {
                "description": current_goal_desc_cog,
                "priority": active_goal_cog.priority if active_goal_cog and hasattr(active_goal_cog, 'priority') else 0.0
            }
        }
        
        if not skip_component_statuses:
            global_workspace_inst_cog = getattr(self, "global_workspace", None)
            if global_workspace_inst_cog and hasattr(global_workspace_inst_cog, 'get_status'):
                try:
                    get_status_method_gws_cog = getattr(global_workspace_inst_cog, 'get_status')
                    ws_status_cog = None
                    if asyncio.iscoroutinefunction(get_status_method_gws_cog): ws_status_cog = await get_status_method_gws_cog();
                    else: ws_status_cog = get_status_method_gws_cog()
                    if isinstance(ws_status_cog, dict): state["workspace_load"] = ws_status_cog.get("current_load", 0)
                except Exception as e: logger_agent_controller.warning(f"Failed to get GW status for cognitive state: {e}")

        current_phenomenal_state_class_cog = self._PhenomenalState
        current_phenomenal_state_inst_cog = self.current_phenomenal_state
        if current_phenomenal_state_class_cog != MockDataclass and isinstance(current_phenomenal_state_inst_cog, current_phenomenal_state_class_cog):
            if hasattr(current_phenomenal_state_inst_cog, 'valence'): state["emotional_valence"] = current_phenomenal_state_inst_cog.valence
            if hasattr(current_phenomenal_state_inst_cog, 'integration_level'): state["integration_level"] = current_phenomenal_state_inst_cog.integration_level
        return state

    async def _oscar_handle_loop(self, loop_info: Dict[str, Any], meta_analysis: Dict[str, Any]):
        logger_agent_controller.warning(f"Handling detected loop: {loop_info}")
        current_goal_status_enum_loop = self._GoalStatus
        current_create_goal_func_loop = self._create_goal_from_descriptor
        if current_goal_status_enum_loop == MockEnum or current_create_goal_func_loop == (lambda *args, **kwargs: None):
            logger_agent_controller.error("Cannot handle loop, GoalStatus or create_goal_from_descriptor is a dummy."); return

        suggestions = meta_analysis.get("suggestions", ["change_goal"]) if isinstance(meta_analysis, dict) else ["change_goal"]
        intervention_strategy = suggestions[0] if suggestions else "change_goal"
        if intervention_strategy == "change_goal":
              new_goal_loop = current_create_goal_func_loop(f"INTERVENTION: Analyze/break loop ({loop_info.get('type', 'unknown')})", priority=1.5)
              if new_goal_loop:
                  new_goal_desc_loop = new_goal_loop.description if hasattr(new_goal_loop, 'description') else 'N/A'
                  logger_agent_controller.info(f"Suspending current goals and setting intervention goal: {new_goal_desc_loop}")
                  for goal_item_loop in self.active_goals:
                      if hasattr(goal_item_loop, 'status'): goal_item_loop.status = current_goal_status_enum_loop.SUSPENDED
                  self.active_goals.append(new_goal_loop)
              self.current_plan = None; self._log_to_ui("warn", f"Loop detected! Intervening.")
        else:
              logger_agent_controller.warning(f"Loop detected, unknown strategy '{intervention_strategy}'. Suspending current goal.")
              active_goal_loop = self._oscar_get_active_goal()
              if active_goal_loop and hasattr(active_goal_loop, 'status'):
                  active_goal_desc_loop_sus = active_goal_loop.description if hasattr(active_goal_loop, 'description') else 'N/A'
                  active_goal_loop.status = current_goal_status_enum_loop.SUSPENDED # type: ignore
                  self.current_plan = None; self._log_to_ui("warn", f"Loop detected! Suspending goal '{active_goal_desc_loop_sus}'.")

    def _oscar_get_active_goal(self) -> Optional['Goal']:
        """Selects the highest priority ACTIVE goal, or reactivates the highest priority SUSPENDED goal."""
        current_goal_class_get = self._Goal
        current_goal_status_enum_get = self._GoalStatus
        if current_goal_class_get == MockDataclass or current_goal_status_enum_get == MockEnum:
            return None

        active_status = getattr(current_goal_status_enum_get, 'ACTIVE', 'DUMMY_ACTIVE')
        suspended_status = getattr(current_goal_status_enum_get, 'SUSPENDED', 'DUMMY_SUSPENDED')

        # Filter valid goals (correct type and have status)
        valid_goals = [g for g in self.active_goals if isinstance(g, current_goal_class_get) and hasattr(g, 'status')]

        # Separate into ACTIVE and SUSPENDED
        active_goals = [g for g in valid_goals if g.status == active_status]
        suspended_goals = [g for g in valid_goals if g.status == suspended_status]

        # Sort by priority (descending) and then creation time (ascending)
        goal_sort_key = lambda g: (- (g.priority if hasattr(g,'priority') and g.priority is not None else 0.0),
                                  g.creation_time if hasattr(g,'creation_time') and g.creation_time is not None else 0.0)

        if active_goals:
            active_goals.sort(key=goal_sort_key)
            return active_goals[0]
        elif suspended_goals:
            suspended_goals.sort(key=goal_sort_key)
            goal_to_reactivate = suspended_goals[0]
            goal_to_reactivate.status = active_status # Reactivate
            logger_agent_controller.info(f"Reactivated suspended goal: {goal_to_reactivate.description if hasattr(goal_to_reactivate, 'description') else 'N/A'}")
            return goal_to_reactivate
        else:
            # No active or suspended goals found
            return None

    # --- LLM-based Goal Mapping Helper ---
    async def _map_text_to_goal_via_llm(self, user_text: str) -> Optional[str]: 
        """
        Schedules an LLM call to map user text to a potential goal description string.
        Returns None immediately; actual mapping happens asynchronously.
        """
        if not call_ollama or not self._asyncio_loop:
            logger_agent_controller.warning("LLM mapping skipped: call_ollama or asyncio_loop not available.")
            return None # Cannot schedule

        user_text_hash = str(hash(user_text)) 
        if user_text_hash in self._pending_goal_mapping_tasks:
            logger_agent_controller.info(
                f"LLM goal mapping for text (hash: {user_text_hash}) '{user_text[:50]}...' already in progress. Skipping duplicate schedule."
            )
            return "mapping_in_progress" 

        known_tasks_for_llm = {
            "read_file": {"desc": "Read file.", "params": {"path": "string"}, "format_string": "read file : {path}"},
            "explore_directory": {
                "desc": "List contents of a directory.",
                "params": {"path": "string (directory path, default is current directory '.')"},
                "format_string": "explore directory : {path}"
            },
            "write_file": {
                "desc": "Write content to a file.",
                "params": {"path": "string (file path to write)", "content": "string (content to write)"},
                "format_string": "write file : {path} content : {content}"
            },
            "report_status": {
                "desc": "Provide a summary of the agent's current status.",
                "params": None,
                "format_string": "report status"
            },
            "explain_goal": {"desc": "Explain current goal.", "params": None, "format_string": "explain goal"},
        }
        task_list_str = "\n".join([f"- {name} ({details['desc']}): Expected params: {details['params'] if details['params'] else 'None'}" for name, details in known_tasks_for_llm.items()])
        system_prompt = (
            "You are an intent parsing assistant for an AI agent. Your task is to understand the user's free-form text "
            "and map it to one of the agent's known internal task types. If the user's text clearly maps to one "
            "of these task types, respond with a JSON object containing the 'task_type' (from the list below) "
            "and a 'parameters' dictionary with extracted values. If parameters are not applicable or not extractable, 'parameters' can be null or an empty dict. "
            "If the user's intent is unclear, or doesn't map to a known task type, or is just conversational, respond with null or an empty JSON object. "
            "Do not be conversational yourself; only output the JSON or null.\n\n"
            "Known agent task types:\n"
            f"{task_list_str}\n\n" 
            "Example user text: 'Can you show me what's in the config file?'\n"
            "Example JSON response: {\"task_type\": \"read_file\", \"parameters\": {\"path\": \"config.toml\"}}\n\n"
            "Example user text: 'Summarize the main config for me.'\n"
            "Example JSON response: {\"task_type\": \"read_file\", \"parameters\": {\"path\": \"config.toml\"}}\n\n"
            "Example user text: 'Create a file named output.txt and put \"Hello World\" in it.'\n"
            "Example JSON response: {\"task_type\": \"write_file\", \"parameters\": {\"path\": \"output.txt\", \"content\": \"Hello World\"}}\n\n"
            "Example user text: 'What are you doing?'\n"
            "Example JSON response: {\"task_type\": \"explain_goal\", \"parameters\": null}\n\n"
            "Example user text: 'Hello there!'\n"
            "Example JSON response: null"
        )
        llm_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
        
        llm_settings_config = self.config.get("llm_settings", {})
        # --- Get thinking config ---
        intent_mapping_specialized_config = self.config.get("oscar_specialized_llms", {})
        enable_thinking_for_intent = intent_mapping_specialized_config.get(
            "intent_mapping_enable_thinking", 
            llm_settings_config.get("default_enable_thinking", False) # Fallback to global default
        )
        # --- End thinking config ---

        logger_agent_controller.info(f"AC_LLM_GOAL_MAP: Scheduling LLM-based goal mapping for: '{user_text[:100]}...' (Thinking: {enable_thinking_for_intent})")

        # Determine model to use for intent mapping
        intent_model_name = intent_mapping_specialized_config.get("intent_mapping_model", self.model_name)
        timeout_intent = intent_mapping_specialized_config.get("intent_mapping_timeout_s", llm_settings_config.get("default_timeout_s", 30.0))
        temperature_intent = intent_mapping_specialized_config.get("intent_mapping_temperature", 0.3)


        scheduled_coro = call_ollama
        # Note: The `loop` argument was removed from call_ollama signature in external_comms.py
        coro_args = (
            intent_model_name, 
            llm_messages, 
            temperature_intent
        )
        coro_kwargs = {
            "timeout": timeout_intent, 
            "enable_thinking": enable_thinking_for_intent # <<< PASS NEW FLAG
        }

        llm_future = self.schedule_offline_task(
            scheduled_coro,
            *coro_args,
            **coro_kwargs
        )
        
        self._pending_goal_mapping_tasks[user_text_hash] = {
            "future": llm_future,
            "original_user_text": user_text, 
            "timestamp_scheduled": time.time()
        }
        logger_agent_controller.info(f"AC_LLM_GOAL_MAP: LLM goal mapping task (hash: {user_text_hash}) scheduled for text: '{user_text[:50]}...'")
        
        return "mapping_in_progress"

    async def _map_text_to_goal(self, user_text: str) -> Optional[Union[Goal, str]]: # type: ignore
        text_lower = user_text.strip().lower()
        if not text_lower: return None

        _Goal_main_map = globals().get('Goal')
        _create_goal_func_main_map = globals().get('create_goal_from_descriptor')
        if not _Goal_main_map or not _create_goal_func_main_map:
            logger_agent_controller.error("Cannot map text to goal: Goal or create_goal_from_descriptor missing."); return None

        goal_desc_for_htn: Optional[str] = None 

        if text_lower == "status" or text_lower == "report status":
            goal_desc_for_htn = "report status"
        elif text_lower == "explain goal" or text_lower == "what is your goal":
            goal_desc_for_htn = "explain goal"
        elif text_lower == "resolve test pain event": 
            return "system_command_resolve_test_pain"
        elif re.match(r"^(?:llm query|ask llm|query llm)\s*:\s*(.+)", text_lower, re.IGNORECASE):
            goal_desc_for_htn = text_lower 
            logger_agent_controller.info(f"AC_MAP_GOAL: Regex matched 'llm query' pattern directly. Goal desc for HTN: '{goal_desc_for_htn}'")
        # --- Refined "respond with" regex block ---
        elif text_lower.startswith("respond with :"): 
            response_content_for_goal_desc = user_text[len("respond with :"):].strip()
            if response_content_for_goal_desc:
                goal_desc_for_htn = f"task_direct_response_content : {response_content_for_goal_desc}" 
                logger_agent_controller.info(f"AC_MAP_GOAL: Regex matched 'respond with'. Goal desc for HTN: '{goal_desc_for_htn}'")
            # This path will now create a Goal object if goal_desc_for_htn is set
        # --- End Refined "respond with" ---
        elif not goal_desc_for_htn: 
            match_read = re.match(r"(?:read file|get content of|cat|show me|display)\s*:?\s*(.+)", text_lower, re.IGNORECASE)
            if match_read:
                path = match_read.group(1).strip().strip("'\"")
                if path: goal_desc_for_htn = f"read file : {path}"
            
            elif re.match(r"^(?:explore|list files|ls|dir)\s*:?\s*(.+)", text_lower, re.IGNORECASE):
                match_explore = re.match(r"^(?:explore|list files|ls|dir)\s*:?\s*(.+)", text_lower, re.IGNORECASE) # type: ignore
                path = match_explore.group(1).strip().strip("'\"") # type: ignore
                if path: goal_desc_for_htn = f"explore directory : {path}"
            elif text_lower in ["explore", "list files", "list", "dir", "ls"]:
                goal_desc_for_htn = "explore directory : ."
            # NEW REPLACEMENT block in _goal_to_task
            else:
                # Regex for write: match "write file : path content : content_text"
                # Allows for optional "file" or "to", optional colons.
                # Captures path and content separately.
                match_write = re.match(
                    r"(?:write|create|save)\s+(?:file|to)?\s*:?\s*([^:]+?)\s+(?:content|with|text)\s*:?\s*(.+)", 
                    text_lower, re.IGNORECASE | re.DOTALL
                )
                if match_write:
                    path_param = match_write.group(1).strip().strip("'\"")
                    content_param = match_write.group(2).strip().strip("'\"")
                    if path_param and content_param is not None: # content can be empty string
                        # This string is what HTN's _goal_to_task expects for its own regex
                        goal_desc_for_htn = f"write file : {path_param} content : {content_param[:1000]}"


        if goal_desc_for_htn:
            logger_agent_controller.info(f"AC_MAP_GOAL: Final regex/keyword mapped user input '{user_text[:50]}' to desc for HTN: '{goal_desc_for_htn}'")
            return _create_goal_func_main_map(goal_desc_for_htn, priority=USER_GOAL_PRIORITY)

        if len(text_lower) < 10 and any(w in text_lower for w in ["hi", "hello", "ok", "okay", "thanks", "bye", "yes", "no"]):
            logger_agent_controller.debug(f"AC_MAP_GOAL: Skipping LLM mapping for short/common phrase: '{user_text}'")
            generic_fallback_desc = f"respond to user : {user_text[:100]}"
            logger_agent_controller.info(f"AC_MAP_GOAL: Short/common phrase. Defaulting to user response goal: '{generic_fallback_desc}'")
            return _create_goal_func_main_map(generic_fallback_desc, priority=USER_GOAL_PRIORITY)
        else: 
            logger_agent_controller.info(f"AC_MAP_GOAL: No direct regex map for '{user_text[:50]}...'. Attempting/scheduling LLM fallback.")
            llm_map_status = await self._map_text_to_goal_via_llm(user_text)
            
            if llm_map_status == "mapping_in_progress":
                return "mapping_in_progress"
            else: 
                logger_agent_controller.warning(f"AC_MAP_GOAL: LLM mapping scheduling failed or did not result in 'mapping_in_progress' for '{user_text[:50]}...'. Status: {llm_map_status}. Falling back to generic respond.")
                return "fallback_to_respond"
        
    def _oscar_select_next_action(self, plan: Optional[List[Dict]]) -> Dict[str, Any]:
        if plan and len(plan) > 0 and isinstance(plan[0], dict): return plan[0]
        return {"type": "THINKING", "params": {"content": "No active plan or plan empty."}}

    async def _oscar_execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_type = action.get("type", "UNKNOWN"); params = action.get("params", {})
        logger_agent_controller.info(f"Executing action: {action_type} with params: {params}");
        # self._log_to_ui("info", f"Executing: {action_type} {params if params else ''}") # Moved UI log for RESPOND_TO_USER
        result: Dict[str, Any] = {"outcome": "failure", "error": "Action Type Not Implemented", "result_data": None}
        current_time = time.time();
        kb = getattr(self, 'knowledge_base', None)
        current_predicate_class_exec = self._Predicate
        
        filesystem_config = self.config.get("filesystem", {})
        llm_settings_config = self.config.get("llm_settings", {})
        specialized_llms_config = self.config.get("oscar_specialized_llms", {})


        try:
            if action_type == "THINKING":
                self._log_to_ui("info", f"Thinking: {str(params.get('content', '...'))[:70]}")
                result = {"outcome": "success", "result_data": {"thought": str(params.get('content', ''))[:500]}}
            elif action_type == "QUERY_KB":
                self._log_to_ui("info", f"Querying KB: {params.get('name', '?')} {params.get('args', [])}")
                query_name = params.get("name"); query_args_list = params.get("args", []);
                query_args = tuple(query_args_list) if isinstance(query_args_list, list) else None;
                query_value = params.get("value", True)
                if query_name and kb and hasattr(kb, 'query') and current_predicate_class_exec != MockDataclass:
                    predicates = await kb.query(query_name, query_args, query_value);
                    result_data = [p.__dict__ for p in predicates if isinstance(p, current_predicate_class_exec)];
                    result = {"outcome": "success", "result_data": {"query": query_name, "args": query_args, "value": query_value, "results": result_data}}
                elif not kb: result = {"outcome": "failure", "error": "KnowledgeBase unavailable"}
                else: result = {"outcome": "failure", "error": "Missing 'name' parameter or invalid KB/Predicate"}
            elif action_type == "OBSERVE_SYSTEM":
                self._log_to_ui("info", "Observing system state...")
                sys_state = None
                if PSUTIL_AVAILABLE_CTRL: sys_state = { "cpu_percent": psutil.cpu_percent(), "memory_percent": psutil.virtual_memory().percent };
                else: sys_state = {"error": "psutil unavailable"}
                if kb and current_predicate_class_exec != MockDataclass: await kb.assert_fact(current_predicate_class_exec("observed", ("system_state", current_time), True, timestamp=current_time))
                result = {"outcome": "success", "result_data": sys_state}

            elif action_type == "LIST_FILES":
                self._log_to_ui("info", f"Listing files: {params.get('path', '.')}")
                path_param = params.get("path", ".")
                resolved_path: Optional[Path] = None
                try:
                    target_path_intermediate = Path(path_param)
                    if target_path_intermediate.is_absolute():
                        resolved_path = target_path_intermediate.resolve(strict=False)
                    else:
                        resolved_path = (self.agent_root_path / path_param).resolve(strict=False)
                    is_within_root = self.agent_root_path == resolved_path or self.agent_root_path in resolved_path.parents
                    if not is_within_root:
                        err_msg = f"SecurityError: Final path '{resolved_path}' is outside agent root '{self.agent_root_path}'. Original param: '{path_param}'"
                        logger_agent_controller.error(f"LIST_FILES: {err_msg}")
                        result = {"outcome": "failure", "error": err_msg}
                        raise SecurityException(err_msg)
                    if not resolved_path.exists(): result = {"outcome": "failure", "error": f"Path not exist: {resolved_path}"}
                    elif not resolved_path.is_dir(): result = {"outcome": "failure", "error": f"Not a directory: {resolved_path}"}
                    else:
                        max_files = filesystem_config.get("max_list_items", 20)
                        entries = list(resolved_path.iterdir()); entry_names = sorted([e.name for e in entries]); num = len(entry_names); display = entry_names[:max_files]
                        summary = f"{num} items in '{resolved_path}'. First {len(display)}: {display}" + (f" ... ({num - max_files} more)" if num > max_files else "")
                        result_data = {"path": str(resolved_path), "count": num, "entries": display, "summary": summary}; result = {"outcome": "success", "result_data": result_data}
                        if kb and current_predicate_class_exec != MockDataclass: await kb.assert_fact(current_predicate_class_exec("listedDirectoryContents", (str(resolved_path),), True, timestamp=current_time))
                except PermissionError as e: logger_agent_controller.warning(f"Perm denied listing: {resolved_path} - {e}"); result = {"outcome": "failure", "error": f"Permission denied: {resolved_path}"}
                except Exception as e:
                    if not isinstance(e, SecurityException):
                       logger_agent_controller.exception(f"Error listing {path_param} ({resolved_path}): {e}"); result = {"outcome": "failure", "error": f"Error listing: {e}"}
                if result["outcome"] == "failure" and resolved_path and kb and current_predicate_class_exec != MockDataclass and result.get("error") and any(term in result.get("error","").lower() for term in ["not exist", "not found", "no such file", "not a directory", "permission denied"]): await kb.assert_fact(current_predicate_class_exec(name="isInvalidPath", args=(str(resolved_path),), value=True, timestamp=current_time))

            elif action_type == "READ_FILE":
                self._log_to_ui("info", f"Reading file: {params.get('path', 'N/A')}")
                path_param = params.get("path")
                resolved_path: Optional[Path] = None
                if not path_param: result = {"outcome": "failure", "error": "Missing 'path'"}
                else:
                    try:
                        target_path_intermediate = Path(path_param)
                        if target_path_intermediate.is_absolute():
                            resolved_path = target_path_intermediate.resolve(strict=True)
                        else:
                            resolved_path = (self.agent_root_path / path_param).resolve(strict=True)
                        is_within_root = self.agent_root_path == resolved_path or self.agent_root_path in resolved_path.parents
                        if not is_within_root:
                            err_msg = f"SecurityError: Final path '{resolved_path}' is outside agent root '{self.agent_root_path}'. Original param: '{path_param}'"
                            logger_agent_controller.error(f"READ_FILE: {err_msg}")
                            result = {"outcome": "failure", "error": err_msg}
                            raise SecurityException(err_msg)
                        max_chars = filesystem_config.get("max_read_chars", 3500)
                        size_bytes = resolved_path.stat().st_size
                        content = ""; read_chars_count = 0; truncated = False
                        with open(resolved_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(max_chars)
                            read_chars_count = len(content)
                        if read_chars_count == max_chars and resolved_path.stat().st_size > len(content.encode('utf-8', errors='ignore')):
                            truncated = True
                        summary = f"Read {read_chars_count}/{size_bytes} (chars/bytes) from '{resolved_path}'." + (" (Truncated)" if truncated else "")
                        file_size_category = "small"
                        if size_bytes > 1_000_000: file_size_category = "large"
                        elif size_bytes > 100_000: file_size_category = "medium"
                        result_data = {
                            "path": str(resolved_path), "size_bytes": size_bytes,
                            "file_size_category": file_size_category,
                            "content_snippet": content, "chars_read": read_chars_count,
                            "truncated": truncated, "summary": summary }
                        result = {"outcome": "success", "result_data": result_data}
                        if kb and current_predicate_class_exec != MockDataclass:
                            await kb.assert_fact(current_predicate_class_exec("readFileContent", (str(resolved_path),), True, timestamp=current_time))
                    except FileNotFoundError: logger_agent_controller.warning(f"File not found for READ_FILE: {path_param} (resolved to {resolved_path})"); result = {"outcome": "failure", "error": f"File does not exist: {resolved_path if resolved_path else path_param}"}
                    except PermissionError as e: logger_agent_controller.warning(f"Perm denied reading: {resolved_path} - {e}"); result = {"outcome": "failure", "error": f"Permission denied: {resolved_path}"}
                    except IsADirectoryError: logger_agent_controller.warning(f"Attempted READ_FILE on directory: {resolved_path}"); result = {"outcome": "failure", "error": f"Path is a directory, not a file: {resolved_path}"}
                    except SecurityException as se: raise se
                    except Exception as e:
                        if not isinstance(e, SecurityException):
                            logger_agent_controller.exception(f"Error reading {path_param} ({resolved_path}): {e}"); result = {"outcome": "failure", "error": f"Error reading: {e}"}
                if result["outcome"] == "failure" and resolved_path and kb and current_predicate_class_exec != MockDataclass and result.get("error") and any(term in result.get("error","").lower() for term in ["not exist", "not found", "no such file", "not a file", "permission denied", "is a directory", "securityerror"]):
                    await kb.assert_fact(current_predicate_class_exec(name="isInvalidPath", args=(str(resolved_path),), value=True, timestamp=current_time))

            elif action_type == "WRITE_FILE":
                self._log_to_ui("info", f"Writing file: {params.get('path', 'N/A')}")
                path_param = params.get("path"); content_param = params.get("content"); resolved_path: Optional[Path] = None
                allow_write = filesystem_config.get("allow_file_write", False);
                allow_overwrite = filesystem_config.get("allow_overwrite", False)
                if not allow_write: result = {"outcome": "failure", "error": "File writing is disabled in configuration."}
                elif not path_param: result = {"outcome": "failure", "error": "Missing 'path' parameter."}
                elif content_param is None: result = {"outcome": "failure", "error": "Missing 'content' parameter."}
                else:
                    try:
                        target_path_intermediate = Path(path_param) # type: ignore
                        if target_path_intermediate.is_absolute():
                            resolved_path = target_path_intermediate.resolve(strict=False)
                        else:
                            resolved_path = (self.agent_root_path / path_param).resolve(strict=False) # type: ignore
                        is_within_root = self.agent_root_path == resolved_path or self.agent_root_path in resolved_path.parents # type: ignore
                        if not is_within_root:
                            err_msg = f"SecurityError: Final path '{resolved_path}' is outside agent root '{self.agent_root_path}'. Original param: '{path_param}'" # type: ignore
                            logger_agent_controller.error(f"WRITE_FILE: {err_msg}")
                            result = {"outcome": "failure", "error": err_msg}
                            raise SecurityException(err_msg)
                        current_error_write = result.get("error") if result.get("error") != "Action Type Not Implemented" else None
                        if resolved_path.exists(): # type: ignore
                            if resolved_path.is_dir(): current_error_write = f"Path is a directory: {resolved_path}" # type: ignore
                            elif not allow_overwrite: current_error_write = f"File exists and overwrite is disabled: {resolved_path}"
                        if not current_error_write:
                            resolved_path.parent.mkdir(parents=True, exist_ok=True) # type: ignore
                            written_bytes = resolved_path.write_text(str(content_param), encoding='utf-8') # type: ignore
                            summary = f"Successfully wrote {written_bytes} bytes to '{resolved_path}'."
                            result_data = {"path": str(resolved_path), "bytes_written": written_bytes, "summary": summary}
                            result = {"outcome": "success", "result_data": result_data}
                            if kb and current_predicate_class_exec != MockDataclass:
                                await kb.assert_fact(current_predicate_class_exec("fileWritten", (str(resolved_path),), True, timestamp=current_time))
                                await kb.assert_fact(current_predicate_class_exec("isValidPath", (str(resolved_path),), True, timestamp=current_time))
                                await kb.assert_fact(current_predicate_class_exec("isInvalidPath", (str(resolved_path),), False, timestamp=current_time))
                        else: result = {"outcome": "failure", "error": current_error_write}
                    except PermissionError as e: logger_agent_controller.warning(f"Perm denied writing to: {resolved_path} - {e}"); result = {"outcome": "failure", "error": f"Permission denied: {resolved_path}"}
                    except IsADirectoryError: logger_agent_controller.warning(f"Attempted write to directory path: {resolved_path}"); result = {"outcome": "failure", "error": f"Path is a directory: {resolved_path}"}
                    except SecurityException as se: raise se
                    except Exception as e:
                        if not isinstance(e, SecurityException):
                            logger_agent_controller.exception(f"Error writing to {path_param} ({resolved_path}): {e}"); result = {"outcome": "failure", "error": f"Error writing file: {e}"}
                if result["outcome"] == "failure" and resolved_path and kb and current_predicate_class_exec != MockDataclass and result.get("error") and any(term in result.get("error","").lower() for term in ["permission denied", "is a directory", "securityerror"]):
                    await kb.assert_fact(current_predicate_class_exec(name="isInvalidPath", args=(str(resolved_path),), value=True, timestamp=current_time))

            elif action_type == "CALL_LLM":
                self._log_to_ui("info", f"Calling LLM: {str(params.get('prompt', '...'))[:70]}")
                prompt_param = params.get("prompt"); context_param = params.get("context", "");
                model_override = params.get("model_override"); temp_override = params.get("temperature")
                enable_thinking_param = params.get("enable_thinking", llm_settings_config.get("default_enable_thinking", False))
                
                default_timeout = llm_settings_config.get("default_timeout_s", 180.0) 

                if not prompt_param: result = {"outcome": "failure", "error": "Missing 'prompt' parameter."}
                elif call_ollama is None: result = {"outcome": "failure", "error": "Ollama function (call_ollama) not available."}
                else:
                    model_to_use_exec = model_override if model_override else self.model_name;
                    temperature_to_use = temp_override if isinstance(temp_override, float) and 0.0 <= temp_override <= 2.0 else 0.7
                    final_enable_thinking = enable_thinking_param if isinstance(enable_thinking_param, bool) else default_enable_thinking

                    messages = [{"role": "system", "content": "You are a helpful AI assistant."}];
                    if context_param: messages.append({"role": "system", "content": f"Context: {str(context_param)[:1000]}"})
                    messages.append({"role": "user", "content": str(prompt_param)})
                    
                    logger_agent_controller.debug(
                        f"Calling LLM '{model_to_use_exec}' (T={temperature_to_use}, Timeout={default_timeout}s, Thinking={final_enable_thinking})..."
                    )
                    
                    llm_response_text, llm_thinking_trace, llm_error = await call_ollama(
                        selected_ollama_model=model_to_use_exec, 
                        messages=messages, 
                        temperature=temperature_to_use, 
                        timeout=default_timeout,
                        enable_thinking=final_enable_thinking
                    )

                    if llm_thinking_trace:
                        logger_agent_controller.info(f"AC_LLM_CALL_THINKING (Action CALL_LLM) for prompt '{str(prompt_param)[:50]}...':\n--- START THINKING ---\n{llm_thinking_trace}\n--- END THINKING ---")

                    if llm_error: result = {"outcome": "failure", "error": f"LLM Error: {llm_error}"}
                    elif llm_response_text is not None: 
                        result = {"outcome": "success", "result_data": {"response": llm_response_text, "thinking_trace": llm_thinking_trace}}
                    else: result = {"outcome": "failure", "error": "LLM returned empty response without error."}
                    
                    if kb and current_predicate_class_exec != MockDataclass: 
                        await kb.assert_fact(current_predicate_class_exec("llmCallCompleted", (model_to_use_exec,), True, timestamp=current_time))


            elif action_type == "RESPOND_TO_USER":
                user_query_for_persona = params.get("text") 
                if not user_query_for_persona:
                    result = {"outcome": "failure", "error": "Missing 'text' (user query) for RESPOND_TO_USER."}
                elif call_ollama is None:
                    result = {"outcome": "failure", "error": "Ollama function (call_ollama) not available for responding."}
                else:
                    self._log_to_ui("info", f"Formulating response to: {str(user_query_for_persona)[:70]}")
                    persona_model_name = specialized_llms_config.get("persona_dialogue_model", self.model_name)
                    enable_thinking_persona = specialized_llms_config.get("persona_dialogue_enable_thinking", llm_settings_config.get("default_enable_thinking",False))
                    temperature_persona = specialized_llms_config.get("persona_dialogue_temperature", 0.65)
                    timeout_persona = specialized_llms_config.get("persona_dialogue_timeout_s", 120.0)

                    oscar_internal_state = await self._gather_oscar_c_internal_state_for_persona()
                    oscar_internal_state_json = json.dumps(oscar_internal_state, indent=2)

                    lora_input_content = (
                        f"UserQuery: {user_query_for_persona}\n\n"
                        f"OSCAR-C Internal State Snapshot:\n{oscar_internal_state_json}"
                    )
                    
                    messages_for_persona_lora = [
                        {"role": "user", "content": lora_input_content}
                    ]

                    logger_agent_controller.info(
                        f"Calling Persona LoRA '{persona_model_name}' (T={temperature_persona}, Timeout={timeout_persona}s, Thinking={enable_thinking_persona}) "
                        f"for query: '{user_query_for_persona[:50]}...'"
                    )
                    logger_agent_controller.debug(f"Persona LoRA input content (detail):\n{lora_input_content[:1000]}...")

                    persona_response_content, persona_thinking_trace, persona_error = await call_ollama(
                        selected_ollama_model=persona_model_name,
                        messages=messages_for_persona_lora,
                        temperature=temperature_persona,
                        timeout=timeout_persona,
                        enable_thinking=enable_thinking_persona
                    )

                    if persona_thinking_trace:
                        logger_agent_controller.info(
                            f"AC_PERSONA_LORA_THINKING for query '{user_query_for_persona[:50]}...':\n--- START THINKING ---\n"
                            f"{persona_thinking_trace}\n--- END THINKING ---"
                        )

                    if persona_error:
                        err_msg = f"Persona LoRA Error: {persona_error}"
                        result = {"outcome": "failure", "error": err_msg}
                        self._log_to_ui("error", err_msg)
                    elif persona_response_content:
                        self._log_to_ui("agent", persona_response_content) 
                        result = {"outcome": "success", "result_data": {"response_sent": persona_response_content, "thinking_trace": persona_thinking_trace}}
                        if kb and current_predicate_class_exec != MockDataclass:
                            await kb.assert_fact(current_predicate_class_exec("respondedToUser", (user_query_for_persona[:50],), True, timestamp=current_time))
                    else:
                        err_msg = "Persona LoRA returned empty response without error."
                        result = {"outcome": "failure", "error": err_msg}
                        self._log_to_ui("error", err_msg)
            
            elif action_type == "GET_AGENT_STATUS":
                self._log_to_ui("info", f"Reporting status...") 
                try:
                    status_data = await self._oscar_get_cognitive_state()
                    status_summary = (f"Status:\n- Consciousness: {status_data.get('consciousness_level', 'N/A')}\n"
                                      f"- Goal: {str(status_data.get('current_goal_desc', 'None'))[:50]} "
                                      f"({status_data.get('current_goal_status', 'N/A')})\n"
                                      f"- Plan Steps: {status_data.get('current_plan_length', 0)}\n"
                                      f"- Workspace Load: {status_data.get('workspace_load', 0)}\n"
                                      f"- Valence: {status_data.get('emotional_valence', 0.0):.2f}, "
                                      f"Integration: {status_data.get('integration_level', 0.0):.2f}")
                    self._log_to_ui("info", status_summary)
                    result = {"outcome": "success", "result_data": status_data}
                    if kb and current_predicate_class_exec != MockDataclass: await kb.assert_fact(current_predicate_class_exec("reportedStatus", (), True, timestamp=current_time))
                except Exception as e_status:
                    logger_agent_controller.exception("Error getting agent status for execution.")
                    result = {"outcome": "failure", "error": f"Error generating status: {e_status}"}
            elif action_type == "EXPLAIN_GOAL":
                self._log_to_ui("info", f"Explaining current goal...")
                active_goal_instance_expl = self._oscar_get_active_goal()
                if active_goal_instance_expl and self._Goal != MockDataclass and isinstance(active_goal_instance_expl, self._Goal):
                    status_name_expl = "N/A"
                    if hasattr(active_goal_instance_expl, 'status') and active_goal_instance_expl.status is not None:
                        status_name_expl = active_goal_instance_expl.status.name if hasattr(active_goal_instance_expl.status, 'name') else str(active_goal_instance_expl.status)
                    goal_id_expl = active_goal_instance_expl.id if hasattr(active_goal_instance_expl, 'id') else 'N/A'
                    goal_desc_expl = active_goal_instance_expl.description if hasattr(active_goal_instance_expl, 'description') else 'N/A'
                    goal_prio_expl = active_goal_instance_expl.priority if hasattr(active_goal_instance_expl, 'priority') else 'N/A'

                    goal_info = (f"Current Goal: {goal_desc_expl}\n"
                                 f"- ID: {goal_id_expl}\n"
                                 f"- Status: {status_name_expl}\n"
                                 f"- Priority: {goal_prio_expl}")
                    self._log_to_ui("info", goal_info)
                    result = {"outcome": "success", "result_data": {"goal_description": goal_desc_expl, "goal_status": status_name_expl}}
                    if kb and current_predicate_class_exec != MockDataclass and goal_id_expl != 'N/A':
                        await kb.assert_fact(current_predicate_class_exec("explainedGoal", (goal_id_expl,), True, timestamp=current_time))
                else:
                    no_goal_msg = "No active goal currently set."
                    self._log_to_ui("info", no_goal_msg)
                    result = {"outcome": "success", "result_data": {"message": no_goal_msg}}
            else:
                 if result.get("error") == "Action Type Not Implemented": 
                     err_msg_unknown = f"Action type '{action_type}' not implemented."
                     result = {"outcome": "failure", "error": err_msg_unknown}
                     logger_agent_controller.error(err_msg_unknown)

        except SecurityException as se_direct:
            logger_agent_controller.error(f"SecurityException during action '{action_type}': {se_direct}. Params: {params.get('path', 'N/A')}")
            if kb and current_predicate_class_exec != MockDataclass:
                safe_path_param_str = str(params.get("path","N/A"))[:150]
                await kb.assert_fact(current_predicate_class_exec(name="securityViolation", args=(action_type, safe_path_param_str, str(se_direct)), value=True, timestamp=time.time()))
            result = {"outcome": "failure", "error": f"Security Violation: {se_direct}"}


        except Exception as e:
            logger_agent_controller.exception(f"Unhandled exception executing {action_type}: {e}")
            result = {"outcome": "failure", "error": f"Unexpected execution exception: {str(e)}"}

        if kb and current_predicate_class_exec != MockDataclass:
             kb_timestamp = time.time()
             try:
                 safe_params_str = str({k: str(v)[:50] for k, v in params.items()})[:150]
                 safe_error_str = str(result.get("error", ""))[:100]
                 await kb.assert_fact(current_predicate_class_exec(name="eventOccurred", args=("actionExecution", action_type, result["outcome"]), value=True, timestamp=kb_timestamp ))

                 if result["outcome"] == "failure" and result.get("error") and "Security Violation" not in result.get("error", ""): 
                     await kb.assert_fact(current_predicate_class_exec(name="actionFailed", args=(action_type, safe_params_str, safe_error_str), value=True, timestamp=kb_timestamp ))
                     if action_type == "CALL_LLM":
                         failed_model_name_kb = model_to_use_exec if 'model_to_use_exec' in locals() else self.model_name
                         await kb.assert_fact(current_predicate_class_exec(name="llmCallFailed", args=(failed_model_name_kb, safe_error_str), value=True, timestamp=kb_timestamp))
             except Exception as kb_e: logger_agent_controller.error(f"Failed KB update post-action: {kb_e}")

        logger_agent_controller.info(f"Action {action_type} outcome: {result['outcome']}")
        if result['outcome'] == 'failure':
            if "Security Violation" not in result.get("error", ""): 
                 self._log_to_ui("error", f"Action Failed: {action_type} - {result.get('error', 'Unknown')[:100]}")
        elif result['outcome'] == 'success' and action_type not in ["RESPOND_TO_USER", "GET_AGENT_STATUS", "EXPLAIN_GOAL", "THINKING"]: 
             self._log_to_ui("info", f"Action OK: {action_type}")


        return result

    async def _oscar_handle_recovery(self, recovery_mode: 'RecoveryMode'):
        _current_recovery_mode_enum_rec = self._RecoveryModeEnum
        if _current_recovery_mode_enum_rec == MockEnum:
            logger_agent_controller.error("Cannot handle recovery, RecoveryMode enum is a dummy."); return

        mode_name = recovery_mode.name if hasattr(recovery_mode, 'name') else str(recovery_mode)
        logger_agent_controller.warning(f"Handling recovery mode: {mode_name}"); self._log_to_ui("warn", f"Recovery: {mode_name}")

        cache_inst_rec = getattr(self, "cache", None)
        current_goal_status_enum_rec = self._GoalStatus

        soft_reset = getattr(_current_recovery_mode_enum_rec, 'SOFT_RESET', 'DUMMY_SOFT_RESET')
        medium_reset = getattr(_current_recovery_mode_enum_rec, 'MEDIUM_RESET', 'DUMMY_MEDIUM_RESET')
        hard_reset = getattr(_current_recovery_mode_enum_rec, 'HARD_RESET', 'DUMMY_HARD_RESET')
        safe_mode = getattr(_current_recovery_mode_enum_rec, 'SAFE_MODE', 'DUMMY_SAFE_MODE')
        suspended_status_rec = getattr(current_goal_status_enum_rec, 'SUSPENDED', 'DUMMY_SUSPENDED')


        if recovery_mode in [soft_reset, medium_reset, hard_reset]:
            if cache_inst_rec and hasattr(cache_inst_rec, 'reset'):
                try:
                    reset_method_cache = getattr(cache_inst_rec, 'reset')
                    if asyncio.iscoroutinefunction(reset_method_cache): await reset_method_cache()
                    else: reset_method_cache()
                except Exception as e: logger_agent_controller.error(f"Cache reset error during recovery: {e}")
            self.current_plan = None;
            active_goal_instance_rec = self._oscar_get_active_goal() # Get current active goal before potentially suspending
            if active_goal_instance_rec and current_goal_status_enum_rec != MockEnum and hasattr(active_goal_instance_rec, 'status') and active_goal_instance_rec.status != suspended_status_rec:
                 active_goal_desc_rec = active_goal_instance_rec.description if hasattr(active_goal_instance_rec, 'description') else 'N/A'
                 logger_agent_controller.warning(f"Suspending goal '{active_goal_desc_rec}' due to recovery.")
                 active_goal_instance_rec.status = suspended_status_rec # type: ignore

            if recovery_mode in [medium_reset, hard_reset]:
                components_to_reset = [ "htn_planner", "global_workspace", "experience_stream", "meta_cognition", "loop_detector", "dynamic_self_model", "emergent_motivation_system", "narrative_constructor", "predictive_world_model", "performance_optimizer", "error_recovery" ]
                for comp_name_rec in components_to_reset:
                     component_instance_rec = getattr(self, comp_name_rec, None)
                     if component_instance_rec and hasattr(component_instance_rec, 'reset'):
                          logger_agent_controller.info(f"Resetting component '{comp_name_rec}' during recovery.")
                          try:
                              reset_method_rec = getattr(component_instance_rec, 'reset')
                              if asyncio.iscoroutinefunction(reset_method_rec): await reset_method_rec()
                              else: reset_method_rec()
                          except Exception as r_e_rec: logger_agent_controller.error(f"Failed reset for {comp_name_rec}: {r_e_rec}")
            if recovery_mode == hard_reset:
                logger_agent_controller.critical("HARD RESET triggered. Applying drastic measures...")
                # Could potentially reload config, re-initialize components etc.
                # For now, just logs and does same as medium.
        elif recovery_mode == safe_mode:
            logger_agent_controller.critical("SAFE MODE triggered!"); self._log_to_ui("error", "SAFE MODE Activated!")
            # Could limit allowed actions, reduce performance targets etc.
            # Example: self.config['filesystem']['allow_file_write'] = False (temporary change)

# --- END OF FILE agent_controller.py ---