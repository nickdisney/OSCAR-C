# --- START OF FINAL INTEGRATED agent_controller.py ---

import asyncio
import logging
import time
import os
import psutil
import signal # For handling signals
import toml # For loading config.toml
import math # For experience stream helpers
from pathlib import Path
from typing import Dict, Any, Optional, List, Deque, Type, Set, Counter # Added Deque, Type, Set, Counter
from collections import deque, Counter # Added Counter

# --- Core OSCAR-C Imports ---
from .models.enums import ConsciousState, GoalStatus, RecoveryMode
from .models.datatypes import Goal, Predicate, PhenomenalState, create_goal_from_descriptor
from .protocols import CognitiveComponent, Planner, AttentionMechanism, WorkspaceManager, ExperienceIntegrator, ConsciousnessAssessor
from .agent_state import AgentState
from . import agent_config # Keep potentially useful constants

# --- Cognitive Module Imports (ALL Components) ---
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
from .cognitive_modules.error_recovery import ErrorRecoverySystem          # <<< INTEGRATED
from .cognitive_modules.predictive_world_model import PredictiveWorldModel  # <<< INTEGRATED
from .cognitive_modules.dynamic_self_model import DynamicSelfModel          # <<< INTEGRATED
from .cognitive_modules.emergent_motivation_system import EmergentMotivationSystem # <<< INTEGRATED
from .cognitive_modules.narrative_constructor import NarrativeConstructor   # <<< INTEGRATED

# --- External Comms ---
from .external_comms import call_ollama, add_document_to_memory, retrieve_from_memory

# --- UI Communication ---
import queue

logger = logging.getLogger(__name__)

# Define Component Initialization Order (FINAL - Adjust if dependencies change)
COMPONENT_INIT_ORDER = [
    "knowledge_base",
    "cache",
    "performance_optimizer",
    "htn_planner",
    "attention_controller",
    "global_workspace",
    "experience_stream",
    "consciousness_assessor",
    "meta_cognition",
    "loop_detector",
    "predictive_world_model",       # <<< INTEGRATED
    "dynamic_self_model",           # <<< INTEGRATED
    "emergent_motivation_system",   # <<< INTEGRATED
    "narrative_constructor",        # <<< INTEGRATED
    "error_recovery",               # <<< INTEGRATED (Initialize last?)
]

# --- Helper Class for Profiling ---
# (Profiler class remains the same as before)
class CycleProfiler:
    def __init__(self):
        self.profile_data: Dict[str, Deque[float]] = {}
        self.current_section: Optional[str] = None
        self.section_start: Optional[float] = None
        self.max_samples = 100
    def set_max_samples(self, max_samples: int):
        if max_samples > 0: self.max_samples = max_samples
        for section in self.profile_data: self.profile_data[section] = deque(self.profile_data[section], maxlen=self.max_samples)
    def start_section(self, section_name: str):
        if self.current_section: self.end_section()
        self.current_section = section_name; self.section_start = time.monotonic()
    def end_section(self):
        if self.current_section and self.section_start is not None:
            duration = time.monotonic() - self.section_start
            if self.current_section not in self.profile_data: self.profile_data[self.current_section] = deque(maxlen=self.max_samples)
            self.profile_data[self.current_section].append(duration)
        self.current_section = None; self.section_start = None
    def get_cycle_profile(self) -> Dict[str, float]:
        profile = {};
        for section, times in self.profile_data.items():
             if times: profile[section] = times[-1]
        return profile
    def get_average_profile(self) -> Dict[str, float]:
         avg_profile = {};
         for section, times in self.profile_data.items():
             if times: avg_profile[section] = sum(times) / len(times)
         return avg_profile
    def reset(self):
        self.profile_data.clear(); self.current_section = None; self.section_start = None

# --- Main Agent Controller Class ---
class AgentController:
    """
    OSCAR-C Phase 2 Agent Controller.
    Orchestrates the 12-step cognitive cycle using modular components.
    """
    def __init__(self, ui_queue: queue.Queue, model_name: str = "default_model", config_path: str = "config.toml"):
        logger.info("Initializing OSCAR-C AgentController (Phase 2 - FINAL INTEGRATION)...")
        if not isinstance(ui_queue, queue.Queue): logger.warning("ui_queue is not a valid Queue."); self.ui_queue = queue.Queue()
        else: self.ui_queue = ui_queue
        self.model_name = model_name
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = self._load_config()
        self.components: Dict[str, CognitiveComponent] = {}
        self.active_goals: List[Goal] = []
        self.current_plan: Optional[List[Dict]] = None
        self.current_phenomenal_state: Optional[PhenomenalState] = None
        self.consciousness_level: ConsciousState = ConsciousState.PRE_CONSCIOUS
        self._prev_consciousness_level: ConsciousState = ConsciousState.UNCONSCIOUS
        self.agent_state: AgentState = AgentState.STOPPED
        self._last_action_executed: Optional[Dict] = None
        self._last_action_result: Optional[Dict] = None
        self._is_running_flag = asyncio.Event()
        self._main_loop_task: Optional[asyncio.Task] = None
        self._asyncio_loop: Optional[asyncio.AbstractEventLoop] = None
        self.pid_file = Path("/tmp/oscar_c.pid")
        self.profiler = CycleProfiler()
        prof_hist_size = self.config.get("performance", {}).get("profiler_history_size", 100)
        self.profiler.set_max_samples(prof_hist_size)
        self._initialize_components()
        logger.info("AgentController initialized.")

    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.exists() and self.config_path.is_file():
            try:
                with open(self.config_path, "r") as f: config_data = toml.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config_data
            except Exception as e: logger.exception(f"Failed to load config file {self.config_path}: {e}"); return {}
        else: logger.warning(f"Config file not found at {self.config_path}. Using empty config."); return {}

    def _initialize_components(self):
        logger.info("Creating cognitive component instances...")
        component_classes: Dict[str, Type[CognitiveComponent]] = {
            "knowledge_base": KnowledgeBase, "cache": CognitiveCache,
            "performance_optimizer": PerformanceOptimizer, "htn_planner": HTNPlanner,
            "attention_controller": AttentionController, "global_workspace": GlobalWorkspaceManager,
            "experience_stream": ExperienceStream, "consciousness_assessor": ConsciousnessLevelAssessor,
            "meta_cognition": MetaCognitiveMonitor, "loop_detector": LoopDetector,
            "predictive_world_model": PredictiveWorldModel,      # <<< INTEGRATED
            "dynamic_self_model": DynamicSelfModel,            # <<< INTEGRATED
            "emergent_motivation_system": EmergentMotivationSystem, # <<< INTEGRATED
            "narrative_constructor": NarrativeConstructor,     # <<< INTEGRATED
            "error_recovery": ErrorRecoverySystem,            # <<< INTEGRATED
        }
        for name in COMPONENT_INIT_ORDER:
            if name in component_classes:
                try:
                    ComponentClass = component_classes[name]; instance = ComponentClass()
                    self.components[name] = instance; setattr(self, name, instance)
                    logger.debug(f"Created component instance: {name}")
                except Exception as e: logger.exception(f"Failed to create component instance {name}: {e}")
            else:
                 if name in component_classes: logger.warning(f"Component '{name}' in INIT_ORDER but no class found.")
        logger.info(f"Created {len(self.components)} component instances.")

    # --- UI Communication Helpers ---
    def _log_to_ui(self, level: str, message: str):
        if hasattr(self, 'ui_queue') and self.ui_queue:
            try: self.ui_queue.put_nowait((f"log_{level.lower()}", str(message)))
            except queue.Full: logger.warning(f"UI queue full. Dropping log: {message}")
            except Exception as e: logger.error(f"Error putting log on UI queue: {e}", exc_info=False)
        else: logger.warning(f"UI queue not available. Cannot log: {message}")

    def _update_ui_state(self, new_state: AgentState):
         if self.agent_state != new_state:
             logger.info(f"Agent state changing: {self.agent_state.name} -> {new_state.name}")
             self.agent_state = new_state
             if hasattr(self, 'ui_queue') and self.ui_queue:
                 try: self.ui_queue.put_nowait(("state_update", new_state))
                 except queue.Full: logger.warning(f"UI queue full. Dropping state update: {new_state.name}")
                 except Exception as e: logger.error(f"Error putting state update on UI queue: {e}", exc_info=False)
             else: logger.warning(f"UI queue not available. Cannot update state: {new_state.name}")

    # --- Public Control Methods ---
    def start(self):
        """Starts the agent's asynchronous event loop and cognitive cycle."""
        if self.agent_state != AgentState.STOPPED: logger.warning(f"Start ignored: Agent not in STOPPED state (current: {self.agent_state.name})"); return
        logger.info("Starting OSCAR-C Agent..."); self._update_ui_state(AgentState.STARTING)
        try: self._asyncio_loop = asyncio.get_running_loop(); logger.info("Using existing running event loop.")
        except RuntimeError: self._asyncio_loop = asyncio.new_event_loop(); asyncio.set_event_loop(self._asyncio_loop); logger.info("Created new event loop.")
        self._add_signal_handlers()
        self._is_running_flag.set()
        self._main_loop_task = self._asyncio_loop.create_task(self._run_initialization_and_loop())
        if not self._asyncio_loop.is_running():
            logger.info("Starting event loop...")
            try: self._asyncio_loop.run_forever()
            except KeyboardInterrupt: logger.info("KeyboardInterrupt received, stopping loop."); self.stop()
            finally: logger.info("Event loop finished."); self._cleanup();
        else: logger.info("Agent initialization task scheduled on existing running loop.")

    def stop(self, signum=None, frame=None):
        """Initiates graceful shutdown of the agent."""
        if not self._is_running_flag.is_set() and self.agent_state != AgentState.STARTING:
            if self.agent_state != AgentState.STOPPING and self.agent_state != AgentState.STOPPED: logger.info("Stop ignored: Agent is not running or starting."); return
        logger.info(f"Stop requested (Signal: {signum})..."); self._update_ui_state(AgentState.STOPPING)
        self._is_running_flag.clear()
        if self._main_loop_task and not self._main_loop_task.done():
             logger.info("Requesting main loop task cancellation..."); self._main_loop_task.cancel()
             if self._asyncio_loop and self._asyncio_loop.is_running(): self._asyncio_loop.call_soon(self._asyncio_loop.stop); logger.info("Requested event loop stop.")
             else: logger.warning("Cannot request loop stop, loop not running or not available.")
        elif self._asyncio_loop and not self._asyncio_loop.is_running(): self._cleanup()

    async def _run_initialization_and_loop(self):
        """Handles component initialization and then runs the main cognitive loop."""
        initialized_components = []
        try:
            logger.info("Initializing components..."); init_success = True
            for name in COMPONENT_INIT_ORDER:
                if name in self.components:
                    component = self.components[name]; logger.debug(f"Initializing component: {name}...")
                    try:
                        success = await component.initialize(self.config, self)
                        if not success: logger.error(f"Component {name} initialization failed!"); init_success = False; break
                        else: initialized_components.append(name); logger.info(f"Component {name} initialized successfully.")
                    except Exception as e: logger.exception(f"Exception during {name} initialization: {e}"); init_success = False; break
            if not init_success:
                logger.error("Agent initialization failed."); self._log_to_ui("error", "Agent initialization failed. Check logs.")
                self._update_ui_state(AgentState.ERROR);
                if self.pid_file.exists(): try: self.pid_file.unlink(); except OSError as e: logger.error(f"Error removing PID file: {e}")
                return
            try:
                with open(self.pid_file, "w") as f: f.write(str(os.getpid())); logger.info(f"PID file created at {self.pid_file}")
            except Exception as e: logger.error(f"Failed to write PID file: {e}")
            self._log_to_ui("info", "Agent initialization complete. Starting cognitive cycle."); self._update_ui_state(AgentState.RUNNING)
            await self._run_agent_loop()
        except asyncio.CancelledError: logger.info("Initialization or main loop task was cancelled.")
        except Exception as e: logger.exception("Unhandled exception in _run_initialization_and_loop"); self._update_ui_state(AgentState.ERROR)
        finally:
            logger.info("Cognitive cycle finished or interrupted. Shutting down components..."); await self._shutdown_components(initialized_components)
            if self.pid_file.exists(): try: self.pid_file.unlink(); logger.info("PID file removed during shutdown."); except OSError as e: logger.error(f"Error removing PID file during shutdown: {e}")

    async def _shutdown_components(self, component_names: List[str]):
         logger.info(f"Shutting down {len(component_names)} components...")
         for name in reversed(component_names):
             if name in self.components:
                 component = self.components[name]; logger.debug(f"Shutting down component: {name}...")
                 try:
                     if hasattr(component, 'shutdown') and asyncio.iscoroutinefunction(component.shutdown): await component.shutdown()
                     logger.debug(f"Component {name} shutdown complete.")
                 except Exception as e: logger.exception(f"Error shutting down component {name}: {e}")
         logger.info("Component shutdown finished.")

    def _add_signal_handlers(self):
        if self._asyncio_loop:
            try:
                 for sig in (signal.SIGINT, signal.SIGTERM): self._asyncio_loop.add_signal_handler(sig, self.stop, sig, None)
                 logger.info("Added signal handlers for SIGINT and SIGTERM.")
            except NotImplementedError: logger.warning("Signal handlers not fully supported on this platform.")
            except Exception as e: logger.error(f"Error adding signal handlers: {e}")

    def _remove_signal_handlers(self):
         if self._asyncio_loop and not self._asyncio_loop.is_closed():
             try:
                 for sig in (signal.SIGINT, signal.SIGTERM):
                     try: removed = self._asyncio_loop.remove_signal_handler(sig);
                     except (ValueError, RuntimeError): pass # Ignore if not found or loop stopping
                 # logger.info("Attempted removal of signal handlers.") # Verbose
             except NotImplementedError: pass
             except Exception as e: logger.error(f"Error removing signal handlers: {e}")

    def _cleanup(self):
        """Cleans up resources like the event loop and PID file."""
        if self.agent_state == AgentState.STOPPED: return
        logger.info("Cleaning up agent resources...")
        self._remove_signal_handlers()
        if self._asyncio_loop and not self._asyncio_loop.is_closed():
             if self._asyncio_loop.is_running(): self._asyncio_loop.stop(); logger.info("Asyncio loop stopped during cleanup.")
             self._asyncio_loop.close(); logger.info("Asyncio loop closed during cleanup.")
        self._asyncio_loop = None; self._main_loop_task = None
        if self.pid_file.exists(): try: self.pid_file.unlink(); logger.info("PID file removed during cleanup."); except OSError as e: logger.error(f"Error removing PID file during cleanup: {e}")
        self._update_ui_state(AgentState.STOPPED)


    async def _run_agent_loop(self):
        """Main 12-step cognitive cycle."""
        cycle_count = 0
        target_cycle_time = self.config.get("performance", {}).get("target_cycle_time", 0.1)

        while self._is_running_flag.is_set():
            cycle_start_time = time.monotonic(); cycle_count += 1
            logger.debug(f"--- Cognitive Cycle {cycle_count} START ---"); self.profiler.reset()

            # --- State variables for the cycle ---
            raw_percepts = {}; prediction_result = {}; attention_candidates = {}; attention_weights = {}; broadcast_content = {}
            relevant_memories = []; action_context = {}; cognitive_state = {}; meta_analysis = {}
            loop_info = None; active_goal = None; current_state_set: Set[Predicate] = set(); next_action = {}
            action_result = {}; optimization_analysis = None

            try:
                # --- 1. Perception & Prediction ---
                self.profiler.start_section("perception"); raw_percepts = await self._oscar_perceive(); self.profiler.end_section()

                # <<< INTEGRATED >>>
                self.profiler.start_section("prediction")
                if hasattr(self, 'predictive_world_model'):
                    # Input needs refinement - what action/state are we predicting based on?
                    # Using a generic request for now.
                    predict_input = {"context": "pre-computation_prediction"}
                    prediction_result = await self.predictive_world_model.predict_next_state(predict_input)
                else: logger.error("PredictiveWorldModel component not available."); prediction_result = {}
                self.profiler.end_section()

                # --- 2. Attention Allocation ---
                self.profiler.start_section("attention"); attention_candidates = self._oscar_gather_attention_candidates(raw_percepts)
                if hasattr(self, 'attention_controller'): attention_weights_result = await self.attention_controller.process({"candidates": attention_candidates})
                else: logger.error("AttentionController component not available."); attention_weights_result = None
                attention_weights = attention_weights_result.get("attention_weights", {}) if attention_weights_result else {}; self.profiler.end_section()

                # --- 3. Global Workspace Competition & Broadcast ---
                self.profiler.start_section("workspace")
                if hasattr(self, 'global_workspace'): workspace_output = await self.global_workspace.process({ "attention_weights": attention_weights, "all_candidates_data": attention_candidates })
                else: logger.error("GlobalWorkspaceManager component not available."); workspace_output = None
                broadcast_content = workspace_output.get("broadcast_content", {}) if workspace_output else {}; self.profiler.end_section()

                # --- 4. Experience Integration ---
                self.profiler.start_section("experience_integration")
                if hasattr(self, 'experience_stream'):
                    relevant_memories = await self._oscar_get_relevant_memories(broadcast_content); action_context = self._oscar_get_current_action_context()
                    experience_input = { "percepts": raw_percepts, "memories": relevant_memories, "context": action_context, "broadcast_content": broadcast_content }
                    phenomenal_state_output = await self.experience_stream.process(experience_input)
                else: logger.error("ExperienceStream component not available."); phenomenal_state_output = None
                if phenomenal_state_output and "phenomenal_state" in phenomenal_state_output: self.current_phenomenal_state = phenomenal_state_output["phenomenal_state"]
                else: logger.warning("Experience integration failed. Creating default."); self.current_phenomenal_state = PhenomenalState(content={}, timestamp=time.time())
                self.profiler.end_section()

                # --- 5. Consciousness Assessment ---
                self.profiler.start_section("consciousness_assessment"); self._prev_consciousness_level = self.consciousness_level; new_conscious_state = self.consciousness_level
                if hasattr(self, 'consciousness_assessor'):
                    assessment_input = { "experience": self.current_phenomenal_state, "workspace_content": broadcast_content }
                    assessment_output = await self.consciousness_assessor.process(assessment_input)
                    if assessment_output and "conscious_state" in assessment_output: new_conscious_state = assessment_output["conscious_state"]
                    else: logger.warning("Consciousness assessment failed.")
                else: logger.error("ConsciousnessLevelAssessor component not available.")
                self.consciousness_level = new_conscious_state
                if self.consciousness_level != self._prev_consciousness_level: logger.info(f"Consciousness level transition: {self._prev_consciousness_level.name} -> {self.consciousness_level.name}"); self._log_to_ui("info", f"Consciousness: {self.consciousness_level.name}")
                self.profiler.end_section()

                # --- 6. Meta-Cognitive Monitoring ---
                self.profiler.start_section("meta_cognition"); meta_analysis = {} # Initialize for the cycle
                if hasattr(self, 'meta_cognition'):
                    cognitive_state = self._oscar_get_cognitive_state()
                    if self.current_phenomenal_state:
                        cognitive_state["workspace_load"] = len(self.current_phenomenal_state.content)
                        cognitive_state["emotional_valence"] = self.current_phenomenal_state.valence
                        cognitive_state["integration_level"] = self.current_phenomenal_state.integration_level
                    perf_metrics = self.profiler.get_average_profile(); meta_input = { "cognitive_state": cognitive_state, "performance_metrics": perf_metrics }
                    meta_output = await self.meta_cognition.process(meta_input)
                    if meta_output and "meta_analysis" in meta_output: meta_analysis = meta_output["meta_analysis"]; logger.debug(f"Meta-cognition analysis generated.") # Less verbose log
                    else: logger.warning("Meta-cognitive monitoring failed or returned invalid output.")
                else: logger.error("MetaCognitiveMonitor component not available.")
                self.profiler.end_section()

                # --- 7. Loop Detection & Intervention ---
                self.profiler.start_section("loop_detection"); loop_info = None
                if hasattr(self, 'loop_detector'):
                    loop_output = await self.loop_detector.process(None)
                    if loop_output and "loop_info" in loop_output: loop_info = loop_output["loop_info"]
                else: logger.error("LoopDetector component not available.")
                if loop_info: logger.warning(f"Loop detected: {loop_info}. Initiating intervention."); await self._oscar_handle_loop(loop_info, meta_analysis); self.profiler.end_section(); continue
                self.profiler.end_section()

                # --- 8. Planning & Goal Management ---
                self.profiler.start_section("planning"); active_goal = self._oscar_get_active_goal()
                if not active_goal: logger.info("No active goal found, selecting default goal."); active_goal = create_goal_from_descriptor("Observe and learn"); self.active_goals.append(active_goal)
                if active_goal and hasattr(self, 'htn_planner') and hasattr(self, 'knowledge_base'):
                    kb_query_result = await self.knowledge_base.query_state({"all_facts": True})
                    facts_list = kb_query_result.get("all_facts") if isinstance(kb_query_result, dict) else None
                    current_state_set = set(facts_list) if isinstance(facts_list, list) else set()
                    self.current_plan = await self.htn_planner.plan(active_goal, current_state_set)
                    if self.current_plan: logger.debug(f"Generated plan ({len(self.current_plan)} steps) for goal: {active_goal.description}")
                    else: logger.warning(f"Planning failed for goal: {active_goal.description}")
                elif not hasattr(self, 'htn_planner'): logger.error("HTNPlanner component not available."); self.current_plan = None
                elif not hasattr(self, 'knowledge_base'): logger.error("KnowledgeBase component not available for planner."); self.current_plan = None
                else: self.current_plan = None
                self.profiler.end_section()

                # --- 9. Action Selection & Execution ---
                self.profiler.start_section("action_selection"); next_action = self._oscar_select_next_action(self.current_plan); self.profiler.end_section()
                self.profiler.start_section("execution"); action_result = await self._oscar_execute_action(next_action)
                self._last_action_executed = next_action; self._last_action_result = action_result # Store for Step 10
                if action_result.get("outcome") == "success" and self.current_plan and len(self.current_plan) > 0:
                    if self.current_plan[0] == next_action: self.current_plan.pop(0); logger.debug("Removed successfully executed action from plan.")
                elif action_result.get("outcome") == "failure": logger.warning(f"Action {next_action.get('type')} failed. Clearing current plan."); self.current_plan = None
                self.profiler.end_section()


                # --- 10. Model Updates ---
                self.profiler.start_section("model_updates")
                if self._last_action_result:
                    # Update Predictive World Model
                    if hasattr(self, 'predictive_world_model'):
                         # Pass the prediction made in step 1 and the actual result from step 9
                         await self.predictive_world_model.update_model(prediction_result, self._last_action_result)
                    else: logger.error("PredictiveWorldModel not available for update.")
                    # Update Dynamic Self Model
                    if hasattr(self, 'dynamic_self_model'):
                         dsm_input = { "last_action_type": self._last_action_executed.get("type"), "action_outcome": self._last_action_result.get("outcome"),
                                       "action_params": self._last_action_executed.get("params", {}), "action_error": self._last_action_result.get("error"),
                                       "action_result_data": self._last_action_result.get("result_data"), "phenomenal_state": self.current_phenomenal_state }
                         await self.dynamic_self_model.process(dsm_input)
                    else: logger.error("DynamicSelfModel not available for update.")
                    # Update Emergent Motivation System
                    if hasattr(self, 'emergent_motivation_system'):
                         ems_input = { "cognitive_state": cognitive_state, "last_action_result": self._last_action_result, "phenomenal_state": self.current_phenomenal_state }
                         await self.emergent_motivation_system.process(ems_input)
                    else: logger.error("EmergentMotivationSystem not available for update.")
                else: logger.debug("Skipping model updates as no action was executed this cycle.")
                self.profiler.end_section()

                # --- 11. Narrative Update ---
                self.profiler.start_section("narrative")
                if hasattr(self, 'narrative_constructor') and self._last_action_result:
                    narrative_input = { "phenomenal_state": self.current_phenomenal_state, "last_action_result": self._last_action_result }
                    await self.narrative_constructor.process(narrative_input)
                elif not hasattr(self, 'narrative_constructor'): logger.error("NarrativeConstructor not available.")
                self.profiler.end_section()

                # --- 12. Performance Optimization ---
                self.profiler.start_section("optimization"); cycle_profile_data = self.profiler.get_cycle_profile()
                if hasattr(self, 'performance_optimizer') and self.performance_optimizer:
                     optimization_analysis = await self.performance_optimizer.process({"cycle_profile": cycle_profile_data})
                     if optimization_analysis:
                         perf_analysis = optimization_analysis.get("performance_analysis", {}); recovery_mode = perf_analysis.get("recovery_mode_needed")
                         if recovery_mode: logger.warning(f"Performance optimizer triggered recovery mode: {recovery_mode}"); await self._oscar_handle_recovery(recovery_mode)
                self.profiler.end_section()

            except asyncio.CancelledError: logger.info("Cognitive cycle task cancelled."); break
            except Exception as e:
                logger.exception(f"Error in cognitive cycle {cycle_count}: {e}"); self._log_to_ui("error", f"Cycle Error: {e}")
                # --- Error Recovery ---
                suggested_recovery_mode = None
                try:
                    if hasattr(self, 'error_recovery'): # <<< INTEGRATED Check
                        error_context = { "cycle": cycle_count, "timestamp": time.time(), "cognitive_state_summary": cognitive_state,
                                          "current_goal_desc": active_goal.description if active_goal else "None",
                                          "last_action_type": self._last_action_executed.get("type") if self._last_action_executed else None,
                                          "exception_type": type(e).__name__, }
                        suggested_recovery_mode = await self.error_recovery.handle_error(e, error_context) # <<< INTEGRATED Call
                    else: logger.error("Error recovery component not available."); await asyncio.sleep(2.0)
                    if suggested_recovery_mode: await self._oscar_handle_recovery(suggested_recovery_mode)
                    else: logger.debug("No specific recovery action suggested or component missing, delaying."); await asyncio.sleep(1.0)
                except asyncio.CancelledError: logger.info("Recovery handling cancelled."); raise
                except Exception as recovery_error: logger.error(f"CRITICAL: Error occurred *during* error recovery attempt: {recovery_error}"); await asyncio.sleep(5.0)


            # --- Cycle Timing Control ---
            cycle_end_time = time.monotonic(); elapsed = cycle_end_time - cycle_start_time; sleep_duration = max(0, target_cycle_time - elapsed)
            if elapsed > target_cycle_time: logger.warning(f"Cycle {cycle_count} overran target time. Elapsed: {elapsed:.4f}s, Target: {target_cycle_time:.4f}s"); self._log_to_ui("warn", f"Cycle overrun: {elapsed:.3f}s")
            if self._is_running_flag.is_set(): await asyncio.sleep(sleep_duration)
            logger.debug(f"Cycle {cycle_count} END. Elapsed: {elapsed:.4f}s, Slept: {sleep_duration:.4f}s")

        logger.info("Agent cognitive loop stopped.")


    # --- Helper Methods for Cognitive Cycle Steps ---
    # (These remain largely the same placeholders, but are called by the integrated cycle steps)
    async def _oscar_perceive(self) -> Dict[str, Any]:
        percepts = { "timestamp": time.time(), "system_state": {"cpu_percent": psutil.cpu_percent(), "memory_percent": psutil.virtual_memory().percent }, "user_input": None, "internal_error": None }
        return percepts

    def _oscar_gather_attention_candidates(self, percepts: Dict[str, Any]) -> Dict[str, Any]:
         candidates = {}
         for key, value in percepts.items():
             if value is not None: weight = 0.7 if key == "user_input" else 0.5; candidates[f"percept_{key}"] = {"content": value, "weight_hint": weight, "timestamp": percepts["timestamp"]}
         for goal in self.active_goals: weight = goal.priority * 0.8; candidates[f"goal_{goal.id}"] = {"content": goal.description, "weight_hint": weight, "timestamp": goal.creation_time}
         if self.current_phenomenal_state: time_delta = time.time() - self.current_phenomenal_state.timestamp; weight = max(0, 0.5 * math.exp(-time_delta / 30.0)); candidates["last_experience"] = {"content": self.current_phenomenal_state.content, "weight_hint": weight, "timestamp": self.current_phenomenal_state.timestamp}
         return candidates

    async def _oscar_get_relevant_memories(self, broadcast_content: Dict[str, Any]) -> List[Any]:
         return ["memory_placeholder_1"] # Placeholder

    def _oscar_get_current_action_context(self) -> Dict[str, Any]:
        return { "last_action_type": self._last_action_executed.get("type") if self._last_action_executed else None,
                 "last_action_outcome": self._last_action_result.get("outcome") if self._last_action_result else None,
                 "last_action_error": self._last_action_result.get("error") if self._last_action_result else None, }

    def _oscar_get_cognitive_state(self) -> Dict[str, Any]:
        state = { "timestamp": time.time(), "consciousness_level": self.consciousness_level.name, "active_goal_count": len(self.active_goals), "current_plan_length": len(self.current_plan) if self.current_plan else 0, "workspace_load": 0, "emotional_valence": 0.0, "integration_level": 0.0 }
        return state

    async def _oscar_handle_loop(self, loop_info: Dict[str, Any], meta_analysis: Dict[str, Any]):
         logger.warning(f"Step 7a: Handling detected loop: {loop_info}")
         new_goal_desc = f"Analyze and break loop involving '{loop_info.get('pattern', 'unknown')}'"
         self.active_goals = [create_goal_from_descriptor(new_goal_desc, priority=1.5)]
         self.current_plan = None; self._log_to_ui("warn", f"Loop detected! Setting goal: {new_goal_desc}")

    def _oscar_get_active_goal(self) -> Optional[Goal]:
         active = [g for g in self.active_goals if g.status == GoalStatus.ACTIVE]
         if not active: return None
         active.sort(key=lambda g: (-g.priority, g.creation_time)); return active[0]

    def _oscar_select_next_action(self, plan: Optional[List[Dict]]) -> Dict[str, Any]:
        if plan and len(plan) > 0: action = plan[0]; logger.info(f"Selected action: {action.get('type', 'UNKNOWN')}"); return action
        else: logger.info("No plan available, selecting default THINKING action."); return {"type": "THINKING", "params": {"content": "No active plan. Observing."}}

    async def _oscar_execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the selected action."""
        action_type = action.get("type", "UNKNOWN"); params = action.get("params", {})
        logger.info(f"Step 9b: Executing action: {action_type} with params: {params}"); self._log_to_ui("info", f"Executing: {action_type} {params if params else ''}")
        result: Dict[str, Any] = {"outcome": "failure", "error": "Not Implemented", "result_data": None}
        try:
            if action_type == "THINKING": result = {"outcome": "success", "result_data": f"Thought process recorded: {params.get('content', '')[:100]}"}
            elif action_type == "QUERY_KB":
                 query_name = params.get("name"); query_args_list = params.get("args", []); query_args = tuple(query_args_list) if isinstance(query_args_list, list) else None; query_value = params.get("value", True)
                 if query_name and hasattr(self, 'knowledge_base'): predicates = await self.knowledge_base.query(query_name, query_args, query_value); result = {"outcome": "success", "result_data": [p.__dict__ for p in predicates]}
                 else: result = {"outcome": "failure", "error": "Missing 'name' param or KB unavailable"}
            elif action_type == "OBSERVE_SYSTEM":
                 sys_state = { "cpu_percent": psutil.cpu_percent(), "memory_percent": psutil.virtual_memory().percent };
                 if hasattr(self, 'knowledge_base'): await self.knowledge_base.assert_fact(Predicate("observed", ("system_state", time.time()), True))
                 result = {"outcome": "success", "result_data": sys_state}
            else: result = {"outcome": "failure", "error": f"Action type '{action_type}' not implemented."}
        except Exception as e: logger.exception(f"Exception during execution of action {action_type}: {e}"); result = {"outcome": "failure", "error": str(e)}
        if hasattr(self, 'knowledge_base'):
             timestamp = time.time()
             try:
                 await self.knowledge_base.assert_fact(Predicate(name="eventOccurred", args=("actionExecution", action_type, result["outcome"]), value=True, timestamp=timestamp ))
                 if result["outcome"] == "failure" and result.get("error"): await self.knowledge_base.assert_fact(Predicate(name="actionFailed", args=(action_type, str(params)[:100], result["error"][:100]), value=True, timestamp=timestamp ))
             except Exception as kb_e: logger.error(f"Failed to update KB after action execution: {kb_e}")
        logger.info(f"Action {action_type} execution result: {result['outcome']}")
        if result['outcome'] == 'failure': self._log_to_ui("error", f"Action Failed: {action_type} - {result.get('error', 'Unknown reason')}")
        return result

    async def _oscar_handle_recovery(self, recovery_mode: RecoveryMode):
        """Handles performance degradation or errors requiring recovery."""
        logger.warning(f"Step 12a: Handling recovery mode: {recovery_mode.name}"); self._log_to_ui("warn", f"Entering recovery mode: {recovery_mode.name}")
        if recovery_mode in [RecoveryMode.SOFT_RESET, RecoveryMode.MEDIUM_RESET, RecoveryMode.HARD_RESET]:
             if hasattr(self, 'cache'): await self.cache.reset()
             self.current_plan = None; active_goal = self._oscar_get_active_goal()
             if active_goal: logger.warning(f"Suspending active goal '{active_goal.description}' due to {recovery_mode.name}."); active_goal.status = GoalStatus.SUSPENDED
             if recovery_mode in [RecoveryMode.MEDIUM_RESET, RecoveryMode.HARD_RESET]:
                 components_to_reset = ["htn_planner", "global_workspace", "experience_stream", "meta_cognition", "loop_detector", "dynamic_self_model", "emergent_motivation_system"] # Add more as needed
                 for comp_name in components_to_reset:
                      if hasattr(self, comp_name) and hasattr(getattr(self, comp_name), 'reset'): # Check reset exists
                           logger.info(f"Resetting component '{comp_name}' due to {recovery_mode.name} recovery.")
                           await getattr(self, comp_name).reset()
                 if recovery_mode == RecoveryMode.HARD_RESET: logger.critical("HARD RESET triggered.")
        elif recovery_mode == RecoveryMode.SAFE_MODE: logger.critical("SAFE MODE triggered!"); self._log_to_ui("error", "Entering SAFE MODE due to critical issues!")


# --- END OF FINAL INTEGRATED agent_controller.py ---