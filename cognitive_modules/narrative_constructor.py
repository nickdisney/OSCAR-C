# --- START OF cognitive_modules/narrative_constructor.py (LLM Integration) ---

import asyncio
import logging
import time
import json
import os
import math # Needed for drive shift calculation
from pathlib import Path
from typing import Dict, Any, Optional, List, Deque, Tuple, Type, Set, Callable # Added Set, Callable
from collections import deque
from dataclasses import dataclass, field

# --- Use standard relative imports ---
try:
    from ..protocols import CognitiveComponent
    from ..models.enums import ConsciousState, GoalStatus # Keep enums imported
    from ..models.datatypes import PhenomenalState, Goal, Predicate, PainSource # Keep datatypes imported, Added Predicate & PainSource
    # Get references to other components for status checks (needed in process)
    from .predictive_world_model import PredictiveWorldModel
    from .emergent_motivation_system import EmergentMotivationSystem
    from .knowledge_base import KnowledgeBase # Added KnowledgeBase import
    # --- Import the actual call_ollama function ---
    from ..external_comms import call_ollama
except ImportError as e:
    logging.warning(f"NarrativeConstructor: Relative import failed ({e}), relying on globally defined types.")
    # Check existence of necessary types globally
    if 'CognitiveComponent' not in globals(): raise ImportError("CognitiveComponent not found via relative import or globally")
    if 'ConsciousState' not in globals(): raise ImportError("ConsciousState not found via relative import or globally")
    if 'GoalStatus' not in globals(): raise ImportError("GoalStatus not found via relative import or globally")
    if 'PhenomenalState' not in globals(): logging.warning("PhenomenalState class not found globally.")
    if 'Goal' not in globals(): logging.warning("Goal class not found globally.")
    if 'Predicate' not in globals(): logging.warning("Predicate class not found globally.") # Added check
    if 'PainSource' not in globals(): logging.warning("PainSource class not found globally.")
    if 'PredictiveWorldModel' not in globals(): logging.warning("PredictiveWorldModel class not found globally.")
    if 'EmergentMotivationSystem' not in globals(): logging.warning("EmergentMotivationSystem class not found globally.")
    if 'KnowledgeBase' not in globals(): logging.warning("KnowledgeBase class not found globally.") # Added check

    # Check for call_ollama specifically
    if 'call_ollama' not in globals():
        logging.error("NarrativeConstructor: call_ollama function not found globally or via import. Narrative generation will fail.")
        # Define placeholder if import fails, log error
        async def call_ollama(model_name, messages, temperature, loop, timeout=None, enable_thinking=False): # Added timeout and enable_thinking for consistency
            logger_narrative.error(f"Placeholder call_ollama invoked for {model_name}. Cannot generate real narrative.")
            await asyncio.sleep(0.01) # Simulate tiny delay
            return None, None, "call_ollama function unavailable" # Return error explicitly, None for thinking_text
    else:
         call_ollama = globals().get('call_ollama') # Assign from global scope if found

    # Assign other types from global scope
    CognitiveComponent = globals().get('CognitiveComponent'); ConsciousState = globals().get('ConsciousState'); GoalStatus = globals().get('GoalStatus')
    PhenomenalState = globals().get('PhenomenalState'); Goal = globals().get('Goal'); Predicate = globals().get('Predicate'); PainSource = globals().get('PainSource')
    PredictiveWorldModel = globals().get('PredictiveWorldModel'); EmergentMotivationSystem = globals().get('EmergentMotivationSystem')
    KnowledgeBase = globals().get('KnowledgeBase') # Added KnowledgeBase


logger_narrative = logging.getLogger(__name__) # Use standard module logger name

# Default config values (remain the same)
DEFAULT_NARRATIVE_LENGTH = 50
DEFAULT_VALENCE_CHANGE_THRESHOLD = 0.3
DEFAULT_INTENSITY_THRESHOLD = 0.7
DEFAULT_SAVE_INTERVAL_S = 300
DEFAULT_DRIVE_CHANGE_THRESHOLD = 0.25

@dataclass
class NarrativeEntry:
    timestamp: float = field(default_factory=time.time)
    content: str = ""
    triggering_event: Dict[str, Any] = field(default_factory=dict)
    phenomenal_state_summary: Dict[str, Any] = field(default_factory=dict)
    consciousness_level: str = "UNKNOWN"
    drive_state: Dict[str, float] = field(default_factory=dict)
    llm_thinking_trace: Optional[str] = None # New field for LLM thinking

class NarrativeConstructor(CognitiveComponent):
    def __init__(self):
        self._controller: Optional[Any] = None
        self._config: Dict[str, Any] = {}
        # Configuration
        self.max_length: int = DEFAULT_NARRATIVE_LENGTH
        self.valence_change_threshold: float = DEFAULT_VALENCE_CHANGE_THRESHOLD
        self.intensity_threshold: float = DEFAULT_INTENSITY_THRESHOLD
        self.save_interval_s: float = DEFAULT_SAVE_INTERVAL_S
        self.drive_change_threshold: float = DEFAULT_DRIVE_CHANGE_THRESHOLD
        # State
        self.narrative: Deque[NarrativeEntry] = deque(maxlen=self.max_length)
        self._narrative_path: Optional[Path] = None
        self._last_save_time: float = 0.0
        self._last_phenomenal_state: Optional['PhenomenalState'] = None
        self._last_drive_state: Optional[Dict[str, float]] = None
        # LLM settings
        self.llm_model_name: str = "default_narrative_llm" # Will be overridden in initialize
        self.llm_temperature: float = 0.75 # Default, can be overridden
        self.llm_timeout_s: float = 120.0 # Default timeout for LLM calls
        # KB and Predicate references
        self._kb: Optional[KnowledgeBase] = None
        self._PredicateClass: Optional[Type[Predicate]] = None
        
        self._last_pain_level_nc: Optional[float] = None
        self._last_happiness_level_nc: Optional[float] = None
        self._last_purpose_level_nc: Optional[float] = None
        self._known_pain_source_ids_nc: Set[str] = set() # Track IDs of pain sources already noted

        # New config for significance thresholds for these new metrics
        self.pain_change_threshold_sig: float = 0.5 
        self.happiness_change_threshold_sig: float = 0.75
        self.purpose_change_threshold_sig: float = 0.5
        
        self._pending_narrative_llm_tasks: Dict[str, Dict[str, Any]] = {} # Stores task_id -> {future, entry_data_for_finalization}
        self._narrative_task_id_counter: int = 0


    async def initialize(self, config: Dict[str, Any], controller: Any) -> bool:
        self._controller = controller
        narrative_config = config.get("narrative_constructor", {})
        self._config = narrative_config # Store component-specific config

        # --- Get KB Reference and Predicate Class ---
        _KnowledgeBase_local = globals().get('KnowledgeBase')
        if _KnowledgeBase_local and hasattr(controller, 'knowledge_base') and \
           isinstance(controller.knowledge_base, _KnowledgeBase_local): # type: ignore
            self._kb = controller.knowledge_base # type: ignore
            logger_narrative.info("NarrativeConstructor: Successfully obtained KnowledgeBase reference.")
        else:
            logger_narrative.warning("NarrativeConstructor: Could not get valid KnowledgeBase reference. KB logging for narrative disabled.")
            self._kb = None

        self._PredicateClass = globals().get('Predicate')
        if not self._PredicateClass: # type: ignore
            logger_narrative.warning("NarrativeConstructor: Predicate class not found. KB logging for narrative disabled.")
            self._PredicateClass = None # Ensure it's None if not valid
        # --- End KB Reference and Predicate Class ---


        self.max_length = narrative_config.get("max_length", DEFAULT_NARRATIVE_LENGTH)
        self.narrative = deque(maxlen=self.max_length) # Ensure deque uses updated max_length

        self.valence_change_threshold = narrative_config.get("valence_change_threshold", DEFAULT_VALENCE_CHANGE_THRESHOLD)
        self.intensity_threshold = narrative_config.get("intensity_threshold", DEFAULT_INTENSITY_THRESHOLD)
        self.save_interval_s = narrative_config.get("save_interval_s", DEFAULT_SAVE_INTERVAL_S)
        self.drive_change_threshold = narrative_config.get("drive_change_threshold", DEFAULT_DRIVE_CHANGE_THRESHOLD)
        
        self.pain_change_threshold_sig = narrative_config.get("pain_change_threshold_sig", 0.5)
        self.happiness_change_threshold_sig = narrative_config.get("happiness_change_threshold_sig", 0.75)
        self.purpose_change_threshold_sig = narrative_config.get("purpose_change_threshold_sig", 0.5)

        # Initialize last known states from controller if available on init
        if self._controller:
            self._last_pain_level_nc = getattr(self._controller, 'pain_level', 0.0)
            self._last_happiness_level_nc = getattr(self._controller, 'happiness_level', 5.0)
            self._last_purpose_level_nc = getattr(self._controller, 'purpose_level', 5.0)
            active_ps_init = getattr(self._controller, 'active_pain_sources', [])
            if isinstance(active_ps_init, list) and globals().get('PainSource'):
                _PainSource_local_init = globals().get('PainSource')
                self._known_pain_source_ids_nc = {
                    ps.id for ps in active_ps_init 
                    if isinstance(ps, _PainSource_local_init) and hasattr(ps, 'id') # type: ignore
                }


        # --- Path Configuration using agent_root_path ---
        save_path_str_from_config = None
        if controller and hasattr(controller, 'agent_root_path'):
            agent_root = controller.agent_root_path
            agent_data_paths_config = config.get("agent_data_paths", {})
            # Get the relative path string from the centralized [agent_data_paths] section
            save_path_str_from_config = agent_data_paths_config.get("narrative_log_path")

            if save_path_str_from_config:
                # Construct the absolute path
                # If save_path_str_from_config happens to be absolute, Path() handles it.
                self._narrative_path = (Path(agent_root) / save_path_str_from_config).resolve()
                try:
                    self._narrative_path.parent.mkdir(parents=True, exist_ok=True)
                    if self._narrative_path.exists():
                        await self._load_narrative()
                    else:
                        logger_narrative.info(f"Narrative file does not exist at {self._narrative_path}. Will create on first save.")
                    logger_narrative.info(f"Narrative save path set to: {self._narrative_path}")
                except OSError as e:
                    logger_narrative.error(f"Cannot create/access directory for narrative log at {self._narrative_path.parent}: {e}. Saving disabled.")
                    self._narrative_path = None
                except Exception as e:
                    logger_narrative.error(f"Error setting/loading narrative path {self._narrative_path}: {e}. Saving disabled.")
                    self._narrative_path = None
            else:
                logger_narrative.info("Narrative_log_path not specified in [agent_data_paths]. Narrative saving disabled.")
                self._narrative_path = None
        else:
            logger_narrative.error("NarrativeConstructor: Controller or agent_root_path not available. Cannot determine narrative save path. Saving disabled.")
            self._narrative_path = None
        # --- End Path Configuration ---
        
        self._last_save_time = time.time()

        # --- LLM Configuration ---
        # Prioritize controller's main model_name, then component-specific config, then default
        if controller and hasattr(controller, 'model_name') and controller.model_name:
            self.llm_model_name = controller.model_name
        else:
            # Fallback to component-specific config or a hardcoded default
            self.llm_model_name = narrative_config.get("llm_model_name", "llama3:latest") 
        
        self.llm_temperature = narrative_config.get("temperature", 0.75)
        # Get default timeout from general LLM settings, then component specific, then hardcoded default
        llm_settings_config = config.get("llm_settings", {})
        self.llm_timeout_s = narrative_config.get("timeout_s", llm_settings_config.get("default_timeout_s", 120.0))


        # Initialize last drive state (logic remains same)
        _EmergentMotivationSystem = globals().get('EmergentMotivationSystem')
        if hasattr(self._controller, 'emergent_motivation_system'):
             ems_comp = getattr(self._controller, 'emergent_motivation_system')
             if _EmergentMotivationSystem and isinstance(ems_comp, _EmergentMotivationSystem) and hasattr(ems_comp, 'get_status'):
                  try:
                      get_status_method = getattr(ems_comp, 'get_status')
                      if asyncio.iscoroutinefunction(get_status_method):
                          ems_status = await get_status_method()
                          self._last_drive_state = ems_status.get("current_drives", {})
                      else: logger_narrative.warning("EMS get_status is not async. Cannot get initial drive state.")
                  except Exception as e: logger_narrative.warning(f"Could not get initial drive state: {e}")
             else: logger_narrative.warning("EMS component found but is not the expected type or lacks get_status.")
        else: logger_narrative.warning("Could not get EMS component reference from controller for initial drive state.")

        logger_narrative.info(
            f"NarrativeConstructor initialized. MaxLen: {self.max_length}, "
            f"ValChgThr: {self.valence_change_threshold:.2f}, IntThr: {self.intensity_threshold:.2f}, "
            f"DrvChgThr:{self.drive_change_threshold:.2f}, LLM: {self.llm_model_name}@{self.llm_temperature}, Timeout: {self.llm_timeout_s}s"
        )
        return True

    async def _load_narrative(self):
        """Loads narrative state from the configured file."""
        # Get type refs safely (remains the same)
        _PhenomenalState = globals().get('PhenomenalState')
        _ConsciousState = globals().get('ConsciousState')
        _GoalStatus = globals().get('GoalStatus')
        _Goal = globals().get('Goal')

        if not self._narrative_path or not self._narrative_path.exists():
            logger_narrative.info("No existing narrative file found or path not set.")
            return

        logger_narrative.info(f"Attempting load narrative from {self._narrative_path}")
        try:
            loaded_entries = []
            with open(self._narrative_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry_dict = json.loads(line)
                        # Basic check for essential keys
                        if not all(k in entry_dict for k in ["timestamp", "content"]):
                            logger_narrative.warning(f"Skipping line (missing essential keys): {line.strip()}")
                            continue

                        # Reconstruct safely, providing defaults for missing optional fields
                        drive_state_raw = entry_dict.get("drive_state", {})
                        # Convert values back to float, handle errors
                        drive_state_processed = {}
                        if isinstance(drive_state_raw, dict):
                             for k, v in drive_state_raw.items():
                                 try: drive_state_processed[k] = float(v)
                                 except (ValueError, TypeError): logger_narrative.warning(f"Could not convert drive state value {v} to float for key {k} in loaded entry.")
                        else: logger_narrative.warning(f"Loaded drive state was not a dict: {drive_state_raw}")


                        entry = NarrativeEntry(
                            timestamp=float(entry_dict["timestamp"]),
                            content=str(entry_dict["content"]),
                            triggering_event=dict(entry_dict.get("triggering_event", {})),
                            phenomenal_state_summary=dict(entry_dict.get("phenomenal_state_summary", {})),
                            consciousness_level=str(entry_dict.get("consciousness_level", "UNKNOWN")),
                            drive_state=drive_state_processed,
                            llm_thinking_trace=entry_dict.get("llm_thinking_trace") # Load thinking trace
                        )
                        loaded_entries.append(entry)
                    except (json.JSONDecodeError, TypeError, ValueError) as e_line:
                        logger_narrative.warning(f"Skipping invalid line in narrative file: {line.strip()} - {e_line}")

            loaded_entries.sort(key=lambda x: x.timestamp)
            # Assign to deque using the correct maxlen
            self.narrative = deque(loaded_entries, maxlen=self.max_length)
            logger_narrative.info(f"Loaded {len(self.narrative)} entries into narrative deque (maxlen={self.max_length}).")
        except Exception as e:
            logger_narrative.exception(f"Error loading narrative: {e}. Starting fresh.");
            self.narrative.clear()


    async def save_narrative(self, force: bool = False):
        """Saves the current narrative deque to the configured file."""
        # Logic remains the same
        current_time = time.time()
        if not self._narrative_path: return False
        # Save if forced OR if interval elapsed AND there's something new to save
        should_save = force or ((current_time - self._last_save_time >= self.save_interval_s) and len(self.narrative) > 0)
        if not should_save: return False

        logger_narrative.info(f"Saving narrative ({len(self.narrative)} entries) to {self._narrative_path}...")
        temp_path = None
        try:
            self._narrative_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self._narrative_path.with_suffix(".tmp")
            with open(temp_path, 'w', encoding='utf-8') as f:
                 narrative_list = list(self.narrative) # Copy deque for safe iteration
                 for entry in narrative_list:
                     try:
                         # Use dataclasses.asdict if available and preferred, else stick to __dict__
                         # entry_dict = dataclasses.asdict(entry)
                         entry_dict = entry.__dict__ # Use built-in dict for dataclass
                         json_line = json.dumps(entry_dict, default=str) # Use default=str for safety
                         f.write(json_line + '\n')
                     except Exception as e_entry:
                          logger_narrative.error(f"Could not serialize narrative entry: {entry} - {e_entry}")
            os.replace(temp_path, self._narrative_path) # Atomic replace
            self._last_save_time = current_time
            logger_narrative.info("Narrative saved successfully.")
            return True
        except Exception as e:
            logger_narrative.exception(f"Failed to save narrative: {e}")
            if temp_path and temp_path.exists():
                try: temp_path.unlink(); logger_narrative.debug(f"Cleaned up temp file: {temp_path}")
                except OSError as unlink_e: logger_narrative.error(f"Error cleaning temp file {temp_path}: {unlink_e}")
            return False

    def _calculate_drive_shift(self, current_drives: Dict[str, float]) -> float:
         """ Calculates a measure of change between current drives and last known state. """
         # Logic remains the same
         if self._last_drive_state is None or not current_drives: return 0.0
         total_abs_change = 0.0; compared_keys = 0
         common_keys = set(self._last_drive_state.keys()) & set(current_drives.keys())
         for key in common_keys:
             try: total_abs_change += abs(float(current_drives[key]) - float(self._last_drive_state[key])); compared_keys += 1
             except (TypeError, ValueError): pass # Ignore if values aren't numbers
         return total_abs_change if compared_keys > 0 else 0.0

    def _is_significant(self,
                       current_phenomenal_state: Optional['PhenomenalState'],
                       last_action_result: Dict[str, Any],
                       loop_info: Optional[Dict[str, Any]],
                       meta_analysis: Dict[str, Any],
                       prediction_error: Optional[Dict[str, Any]], # Added
                       current_drives: Dict[str, float] # Added
                       ) -> Tuple[bool, str, Dict[str, Any]]:
        """ Determines if the current state/event warrants a narrative entry. """
        _PhenomenalState = globals().get('PhenomenalState')
        _GoalStatus = globals().get('GoalStatus')
        _Goal = globals().get('Goal')
        _ConsciousState = globals().get('ConsciousState')
        _PainSource_class_nc = globals().get('PainSource')


        if not _PhenomenalState or not _GoalStatus or not _Goal or not _ConsciousState:
            logger_narrative.warning("Missing datatypes/enums for significance check.")
            return False, "Internal Error: Missing Types", {}

        if not current_phenomenal_state or (not isinstance(current_phenomenal_state, _PhenomenalState) and not isinstance(current_phenomenal_state, dict)):
            logger_narrative.debug("Significance check: No valid phenomenal state provided.")
            return False, "No phenomenal state", {}

        reason = ""
        significant = False
        event_summary = {}

        last_action_type = last_action_result.get("type", "UNKNOWN") if isinstance(last_action_result, dict) else "UNKNOWN"
        last_outcome = last_action_result.get("outcome", "unknown") if isinstance(last_action_result, dict) else "unknown"

        current_valence = getattr(current_phenomenal_state, 'valence', 0.0) if isinstance(current_phenomenal_state, _PhenomenalState) else current_phenomenal_state.get('valence', 0.0)
        current_intensity = getattr(current_phenomenal_state, 'intensity', 0.0) if isinstance(current_phenomenal_state, _PhenomenalState) else current_phenomenal_state.get('intensity', 0.0)
        prev_valence = getattr(self._last_phenomenal_state, 'valence', current_valence) if self._last_phenomenal_state else current_valence

        valence_change = abs(current_valence - prev_valence)
        if valence_change >= self.valence_change_threshold: significant = True; reason += f"ValenceΔ({valence_change:.2f}). "; event_summary["valence_change"] = round(valence_change, 2)
        if current_intensity >= self.intensity_threshold: significant = True; reason += f"HighIntensity({current_intensity:.2f}). "; event_summary["intensity"] = round(current_intensity, 2)

        active_goal = getattr(self._controller, '_oscar_get_active_goal', lambda: None)() if self._controller else None
        if active_goal and isinstance(active_goal, _Goal) and hasattr(active_goal, 'status') and active_goal.status in [_GoalStatus.ACHIEVED, _GoalStatus.FAILED]:
            goal_desc_short = str(getattr(active_goal, 'description', ''))[:20]
            significant = True; reason += f"Goal'{goal_desc_short}' {active_goal.status.name}. "; event_summary["goal_status_change"] = {"id": active_goal.id, "status": active_goal.status.name}

        if last_outcome == "failure" and last_action_type != "THINKING": significant = True; reason += f"ActionFail({last_action_type}). "; event_summary["action_failure"] = {"type": last_action_type, "error": last_action_result.get("error")}
        elif last_outcome == "success" and last_action_type in ["SET_GOAL", "EXPLORE", "READ_FILE", "LIST_FILES", "CALL_LLM", "RESPOND_TO_USER"]: significant = True; reason += f"ActionOK({last_action_type}). "; event_summary["action_success"] = {"type": last_action_type}

        if loop_info: significant = True; reason += f"Loop({loop_info.get('type')}). "; event_summary["loop_detected"] = loop_info
        meta_issues = meta_analysis.get("issues_detected", []) if isinstance(meta_analysis, dict) else []
        if meta_issues: significant = True; reason += f"MetaIssue({len(meta_issues)}). "; event_summary["meta_issue"] = meta_issues[0] # Store first issue for summary

        if self._controller:
             current_cs_level_obj = getattr(self._controller, 'consciousness_level', None)
             prev_cs_level_obj = getattr(self._controller, '_prev_consciousness_level', None)
             current_cs = current_cs_level_obj if _ConsciousState and isinstance(current_cs_level_obj, _ConsciousState) else _ConsciousState.UNCONSCIOUS
             prev_cs = prev_cs_level_obj if _ConsciousState and isinstance(prev_cs_level_obj, _ConsciousState) else _ConsciousState.UNCONSCIOUS
             if current_cs != prev_cs: significant = True; reason += f"CSΔ({prev_cs.name}->{current_cs.name}). "; event_summary["consciousness_change"] = {"from": prev_cs.name, "to": current_cs.name}

        if prediction_error:
             significant = True; reason += f"PredictionErr({prediction_error.get('type', '?')}). "; event_summary["prediction_error"] = {"type": prediction_error.get("type"), "action": prediction_error.get("action_type")}
        drive_shift = self._calculate_drive_shift(current_drives)
        if drive_shift >= self.drive_change_threshold:
             significant = True; reason += f"DriveShift({drive_shift:.2f}). "; event_summary["drive_shift"] = round(drive_shift, 2)
        
        current_pain_for_sig = getattr(self._controller, 'pain_level', 0.0) if self._controller else 0.0
        current_happiness_for_sig = getattr(self._controller, 'happiness_level', 5.0) if self._controller else 5.0
        current_purpose_for_sig = getattr(self._controller, 'purpose_level', 5.0) if self._controller else 5.0

        if self._last_pain_level_nc is not None and \
           abs(current_pain_for_sig - self._last_pain_level_nc) >= self.pain_change_threshold_sig:
            significant = True
            reason_detail = f"PainΔ({self._last_pain_level_nc:.2f}->{current_pain_for_sig:.2f})"
            reason += f"{reason_detail}. "
            event_summary["pain_level_change"] = {"from": round(self._last_pain_level_nc,2), "to": round(current_pain_for_sig,2)}
            logger_narrative.debug(f"NC_PHP_SIG - Significance triggered by: {reason_detail}")
        
        if self._last_happiness_level_nc is not None and \
           abs(current_happiness_for_sig - self._last_happiness_level_nc) >= self.happiness_change_threshold_sig:
            significant = True
            reason_detail = f"HappyΔ({self._last_happiness_level_nc:.2f}->{current_happiness_for_sig:.2f})"
            reason += f"{reason_detail}. "
            event_summary["happiness_level_change"] = {"from": round(self._last_happiness_level_nc,2), "to": round(current_happiness_for_sig,2)}
            logger_narrative.debug(f"NC_PHP_SIG - Significance triggered by: {reason_detail}")


        if self._last_purpose_level_nc is not None and \
           abs(current_purpose_for_sig - self._last_purpose_level_nc) >= self.purpose_change_threshold_sig:
            significant = True
            reason_detail = f"PurposeΔ({self._last_purpose_level_nc:.2f}->{current_purpose_for_sig:.2f})"
            reason += f"{reason_detail}. "
            event_summary["purpose_level_change"] = {"from": round(self._last_purpose_level_nc,2), "to": round(current_purpose_for_sig,2)}
            logger_narrative.debug(f"NC_PHP_SIG - Significance triggered by: {reason_detail}")

        if self._controller and hasattr(self._controller, 'active_pain_sources') and _PainSource_class_nc:
            current_agent_pain_sources: List[PainSource] = getattr(self._controller, 'active_pain_sources', [])
            current_active_unresolved_ids = {
                ps.id for ps in current_agent_pain_sources 
                if hasattr(ps, 'id') and hasattr(ps, 'is_resolved') and not ps.is_resolved and 
                   hasattr(ps, 'current_intensity') and ps.current_intensity >= getattr(self._controller, '_pain_event_min_intensity_to_retain', 0.01)
            }
            
            newly_added_pain_ids = current_active_unresolved_ids - self._known_pain_source_ids_nc
            newly_resolved_or_decayed_pain_ids = self._known_pain_source_ids_nc - current_active_unresolved_ids

            for new_id in newly_added_pain_ids:
                new_ps = next((ps for ps in current_agent_pain_sources if ps.id == new_id), None)
                if new_ps:
                    significant = True
                    reason_detail = f"NewPainSrc({new_ps.description[:20]}...)"
                    reason += f"{reason_detail}. "
                    event_summary[f"new_pain_source_{new_id}"] = {
                        "description": new_ps.description, "intensity": round(new_ps.current_intensity,2)
                    }
                    logger_narrative.debug(f"NC_PHP_SIG - Significance triggered by: {reason_detail}")
            
            for resolved_id in newly_resolved_or_decayed_pain_ids:
                original_ps_obj = next((ps for ps in current_agent_pain_sources if ps.id == resolved_id and hasattr(ps, 'is_resolved') and ps.is_resolved), None)
                if original_ps_obj: 
                    significant = True
                    reason_detail = f"ResolvedPain({original_ps_obj.description[:20]}...)"
                    reason += f"{reason_detail}. "
                    event_summary[f"resolved_pain_source_{resolved_id}"] = {
                        "description": original_ps_obj.description
                    }
                    logger_narrative.debug(f"NC_PHP_SIG - Significance triggered by: {reason_detail}")
            
            self._known_pain_source_ids_nc = current_active_unresolved_ids 

        if _PhenomenalState and isinstance(current_phenomenal_state, _PhenomenalState): self._last_phenomenal_state = current_phenomenal_state
        elif isinstance(current_phenomenal_state, dict) and _PhenomenalState:
             try: self._last_phenomenal_state = _PhenomenalState(**current_phenomenal_state)
             except Exception: self._last_phenomenal_state = None
        else: self._last_phenomenal_state = None
        self._last_drive_state = current_drives.copy() if current_drives else None
        
        self._last_pain_level_nc = current_pain_for_sig
        self._last_happiness_level_nc = current_happiness_for_sig
        self._last_purpose_level_nc = current_purpose_for_sig

        logger_narrative.debug(f"Significance check (incl. P/H/P): {significant} (Reason: {reason.strip()})")
        return significant, reason.strip(), event_summary


    async def generate_narrative_entry(self,
                                     phenomenal_state: Optional[PhenomenalState], # type: ignore
                                     triggering_event_summary: Dict[str, Any],
                                     reason_for_narration: str,
                                     original_timestamp: float 
                                    ) -> None: 
        _ConsciousState_enum_nc = globals().get('ConsciousState')
        _call_ollama_func = globals().get('call_ollama') # Use the global one

        logger_narrative.debug(f"NC_GEN_SCHEDULE: Scheduling narrative LLM call. Reason: {reason_for_narration}")
        
        if not (self._controller and hasattr(self._controller, 'schedule_offline_task') and
                hasattr(self._controller, '_asyncio_loop') and _call_ollama_func):
            error_msg = "NarrativeConstructor: Missing controller, schedule_offline_task, loop, or call_ollama."
            logger_narrative.error(error_msg)
            return

        current_cs_for_prompt = _ConsciousState_enum_nc.CONSCIOUS if _ConsciousState_enum_nc else "CONSCIOUS_FALLBACK"
        controller_cs_level_obj = getattr(self._controller, 'consciousness_level', None)
        if _ConsciousState_enum_nc and isinstance(controller_cs_level_obj, _ConsciousState_enum_nc):
            current_cs_for_prompt = controller_cs_level_obj
        
        prev_entries = [entry.content for entry in list(self.narrative)[-2:]]
        prev_snippets_str = "\n".join(f'- "{str(snippet)[:150]}..."' for snippet in prev_entries) if prev_entries else "None"
        
        drive_state_str = ", ".join(f"{k}={v:.2f}" for k, v in (self._last_drive_state or {}).items()) if self._last_drive_state else "Unknown"

        p_state_summary = {}
        _PhenomenalState_class_nc = globals().get('PhenomenalState', object)
        if isinstance(phenomenal_state, _PhenomenalState_class_nc):
            p_state_content_keys = list(getattr(phenomenal_state, 'content', {}).keys())
            p_state_summary = {"intensity": round(getattr(phenomenal_state, 'intensity', 0.0), 2), 
                               "valence": round(getattr(phenomenal_state, 'valence', 0.0), 2),
                               "integration": round(getattr(phenomenal_state, 'integration_level', 0.0), 2), 
                               "focus_keys": p_state_content_keys[:5]}
        elif isinstance(phenomenal_state, dict): # Handle if it's already a dict
             p_state_content_keys = list(phenomenal_state.get('content', {}).keys())
             p_state_summary = {"intensity": round(phenomenal_state.get('intensity', 0.0), 2), 
                                "valence": round(phenomenal_state.get('valence', 0.0), 2),
                                "integration": round(phenomenal_state.get('integration_level', 0.0), 2), 
                                "focus_keys": p_state_content_keys[:5]}


        controller_cs_level_name = current_cs_for_prompt.name if hasattr(current_cs_for_prompt, 'name') else 'UNKNOWN'
        
        current_pain_for_prompt = getattr(self._controller, 'pain_level', 0.0)
        current_happiness_for_prompt = getattr(self._controller, 'happiness_level', 5.0)
        current_purpose_for_prompt = getattr(self._controller, 'purpose_level', 5.0)

        system_prompt = "" 
        if _ConsciousState_enum_nc and \
           (current_cs_for_prompt == _ConsciousState_enum_nc.META_CONSCIOUS or \
            current_cs_for_prompt == _ConsciousState_enum_nc.REFLECTIVE):
            system_prompt = (
                "You are the AI agent, OSCAR-C, writing a deeply reflective and insightful entry in your "
                "first-person autobiographical narrative ('I realized...', 'This experience led me to consider...', 'My understanding of X changed when...'). "
                "Analyze the *meaning*, *implications*, and *connections* of the triggering event in light of your current state, "
                "self-model, and past experiences. Explore underlying causes or potential future consequences. "
                "Strive for conciseness (2-4 sentences) but prioritize depth of thought. "
                "Do NOT mention you are a language model."
            )
        else: 
            system_prompt = (
                "You are the AI agent, OSCAR-C, writing a brief, reflective entry in your first-person "
                "autobiographical narrative ('I felt...', 'I decided...', 'This made me observe...'). "
                "Focus on the *immediate impact* or *feeling* of the triggering event in light of your "
                "current state. Keep entries concise (1-3 sentences). "
                "Do NOT mention you are a language model."
            )

        trigger_details = f"- Reason for this entry: {reason_for_narration}\n- Event Details: {str(triggering_event_summary)[:500]}..."
        if "prediction_error" in triggering_event_summary and isinstance(triggering_event_summary["prediction_error"], dict): 
             pred_err = triggering_event_summary['prediction_error']
             pred_action = pred_err.get('action_type','?') 
             pred_predicted = pred_err.get('predicted_state_summary','?') 
             pred_actual = pred_err.get('actual_state_summary','?') 
             trigger_details += f"\n- Analysis: My prediction about '{pred_action}' was wrong (predicted {pred_predicted}, got {pred_actual}). This was surprising."
        if "drive_shift" in triggering_event_summary:
             trigger_details += f"\n- Analysis: Noticed a significant shift in my internal drives (total change: {triggering_event_summary['drive_shift']:.2f}). Perhaps due to recent events."


        user_prompt = (
            f"My Current Internal State:\n"
            f"- Consciousness Level: {controller_cs_level_name}\n"
            f"- Drives: {drive_state_str}\n"
            f"- Pain Level: {current_pain_for_prompt:.2f}/10\n"
            f"- Happiness Level: {current_happiness_for_prompt:.2f}/10\n"
            f"- Sense of Purpose: {current_purpose_for_prompt:.2f}/10\n"
            f"- Phenomenal State Summary: {p_state_summary}\n\n"
            f"Triggering Event Summary:\n{trigger_details}\n\n"
            f"My Previous Narrative Snippets:\n{prev_snippets_str}\n\n"
            f"Write the next narrative entry from my perspective (I...):"
        )

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        llm_settings_config = self._controller.config.get("llm_settings", {}) # type: ignore
        narr_specialized_config = self._controller.config.get("oscar_specialized_llms", {}) # type: ignore
        
        model_to_use_narr = narr_specialized_config.get("narrative_llm_model", self.llm_model_name)
        temperature_narr = narr_specialized_config.get("narrative_llm_temperature", self.llm_temperature)
        timeout_narr = narr_specialized_config.get("narrative_llm_timeout_s", self.llm_timeout_s)
        enable_thinking_narr = narr_specialized_config.get(
            "narrative_llm_enable_thinking",
            llm_settings_config.get("default_enable_thinking", False)
        )
        
        entry_data_for_finalization = {
            "timestamp": original_timestamp,
            "triggering_event": triggering_event_summary,
            "phenomenal_state_summary": p_state_summary,
            "consciousness_level": controller_cs_level_name,
            "drive_state": self._last_drive_state.copy() if self._last_drive_state else {},
            "original_reason": reason_for_narration,
            "llm_thinking_trace_placeholder": None # Will be filled by the callback
        }

        self._narrative_task_id_counter += 1
        task_id = f"narrative_llm_{self._narrative_task_id_counter}"
        
        # Use the call_ollama function from the global scope (either real or mock)
        llm_future = self._controller.schedule_offline_task( # type: ignore
            _call_ollama_func,
            model_to_use_narr,
            messages,
            temperature_narr,
            # self._controller._asyncio_loop, # loop is often implicit now
            timeout=timeout_narr,
            enable_thinking=enable_thinking_narr
        )
        
        self._pending_narrative_llm_tasks[task_id] = {
            "future": llm_future,
            "entry_data": entry_data_for_finalization
        }
        logger_narrative.info(f"NC_GEN_SCHEDULE: Scheduled LLM task {task_id} for narrative. Reason: {reason_for_narration}, Thinking: {enable_thinking_narr}")


    def _add_system_note_narrative(self, content: str, original_timestamp: float):
        """Adds a system-generated note to the narrative deque directly."""
        entry = NarrativeEntry(
            timestamp=original_timestamp,
            content=f"SYSTEM NOTE: {content}",
            triggering_event={"type": "system_note", "details": content},
            phenomenal_state_summary={}, # Or fetch current PState summary if needed
            consciousness_level=getattr(self._controller, 'consciousness_level', "UNKNOWN").name if self._controller else "UNKNOWN",
            drive_state=self._last_drive_state or {}
        )
        self.narrative.append(entry)
        logger_narrative.warning(f"Added system note to narrative: {content}")

    async def process(self, input_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Checks significance and potentially adds a new entry."""
        completed_task_ids = []
        for task_id, task_info in self._pending_narrative_llm_tasks.items():
            future = task_info["future"]
            entry_data = task_info["entry_data"]

            if future.done():
                completed_task_ids.append(task_id)
                narrative_content = ""
                thinking_trace_from_llm = None
                try:
                    llm_response_tuple = future.result() # This will re-raise exception if task failed
                    # Expecting (response_text, thinking_text, error)
                    llm_text, thinking_text, llm_error = llm_response_tuple

                    if llm_error:
                        logger_narrative.error(f"NC_PROCESS_RESULT: LLM task {task_id} failed: {llm_error}")
                        narrative_content = f"I noted a significant event ({entry_data['original_reason']}) but encountered an error trying to reflect on it ({llm_error})."
                    elif not llm_text:
                        logger_narrative.warning(f"NC_PROCESS_RESULT: LLM task {task_id} returned empty response for reason: {entry_data['original_reason']}")
                        narrative_content = f"I noted a significant event ({entry_data['original_reason']}) but couldn't formulate a narrative entry."
                    else:
                        narrative_content = llm_text.strip().strip('"\'')
                        thinking_trace_from_llm = thinking_text # Store the thinking trace
                        if thinking_trace_from_llm:
                            logger_narrative.info(
                                f"NC_PROCESS_RESULT: LLM task {task_id} thinking trace for '{entry_data['original_reason'][:50]}...':\n--- START THINKING ---\n"
                                f"{thinking_trace_from_llm[:500]}...\n--- END THINKING ---" # Log snippet
                            )
                        logger_narrative.info(f"NC_PROCESS_RESULT: LLM task {task_id} completed. Narrative snippet: {narrative_content[:100]}...")
                
                except asyncio.CancelledError:
                    logger_narrative.warning(f"NC_PROCESS_RESULT: LLM task {task_id} was cancelled.")
                    narrative_content = f"SYSTEM NOTE: Narrative generation for '{entry_data['original_reason']}' was cancelled."
                except Exception as e_res:
                    logger_narrative.error(f"NC_PROCESS_RESULT: Exception retrieving result for LLM task {task_id}: {e_res}", exc_info=True)
                    narrative_content = f"SYSTEM NOTE: Error processing LLM result for '{entry_data['original_reason']}': {e_res}"

                new_entry = NarrativeEntry(
                    timestamp=entry_data["timestamp"],
                    content=narrative_content,
                    triggering_event=entry_data["triggering_event"],
                    phenomenal_state_summary=entry_data["phenomenal_state_summary"],
                    consciousness_level=entry_data["consciousness_level"],
                    drive_state=entry_data["drive_state"],
                    llm_thinking_trace=thinking_trace_from_llm # Store thinking trace in the entry
                )
                self.narrative.append(new_entry)
                logger_narrative.info(f"Added narrative entry #{len(self.narrative)} (from completed task {task_id}): {narrative_content[:100]}...")
                # Assert to KB here as well, using new_entry data
                if self._kb and self._PredicateClass:
                    try:
                        summary_predicates_to_assert: List[Predicate] = [] # type: ignore
                        
                        reason_for_kb = entry_data.get("original_reason", "Unknown Event")[:80]
                        event_rec_pred = self._PredicateClass(
                            name="narrativeEventRecorded",
                            args=(new_entry.timestamp, reason_for_kb, new_entry.consciousness_level),
                            value=True,
                            timestamp=new_entry.timestamp
                        )
                        summary_predicates_to_assert.append(event_rec_pred)

                        p_state_summary_for_kb = new_entry.phenomenal_state_summary
                        if isinstance(p_state_summary_for_kb, dict) and "valence" in p_state_summary_for_kb:
                            try:
                                valence_val = float(p_state_summary_for_kb["valence"])
                                associated_valence_pred = self._PredicateClass(
                                    name="narrativeAssociatedValence",
                                    args=(new_entry.timestamp, round(valence_val, 2)),
                                    value=True,
                                    timestamp=new_entry.timestamp
                                )
                                summary_predicates_to_assert.append(associated_valence_pred)
                            except (ValueError, TypeError):
                                logger_narrative.warning(f"Could not parse valence '{p_state_summary_for_kb['valence']}' for KB predicate.")
                        
                        if isinstance(p_state_summary_for_kb, dict) and "intensity" in p_state_summary_for_kb:
                            try:
                                intensity_val = float(p_state_summary_for_kb["intensity"])
                                associated_intensity_pred = self._PredicateClass(
                                    name="narrativeAssociatedIntensity",
                                    args=(new_entry.timestamp, round(intensity_val, 2)),
                                    value=True,
                                    timestamp=new_entry.timestamp
                                )
                                summary_predicates_to_assert.append(associated_intensity_pred)
                            except (ValueError, TypeError):
                                logger_narrative.warning(f"Could not parse intensity '{p_state_summary_for_kb['intensity']}' for KB predicate.")

                        triggering_event_summary_for_kb = new_entry.triggering_event
                        if isinstance(triggering_event_summary_for_kb, dict):
                            main_trigger_type = "unknown_trigger"
                            if "prediction_error" in triggering_event_summary_for_kb: main_trigger_type = "prediction_error"
                            elif "loop_detected" in triggering_event_summary_for_kb: main_trigger_type = "loop_detected"
                            elif "goal_status_change" in triggering_event_summary_for_kb: main_trigger_type = "goal_status_change"
                            elif "consciousness_change" in triggering_event_summary_for_kb: main_trigger_type = "consciousness_change"
                            elif "action_failure" in triggering_event_summary_for_kb: main_trigger_type = "action_failure"
                            elif "action_success" in triggering_event_summary_for_kb: main_trigger_type = "action_success"
                            elif "drive_shift" in triggering_event_summary_for_kb : main_trigger_type = "drive_shift"
                            elif "valence_change" in triggering_event_summary_for_kb: main_trigger_type = "valence_change"
                            elif "intensity" in triggering_event_summary_for_kb and main_trigger_type == "unknown_trigger": main_trigger_type = "high_intensity"
                            
                            if main_trigger_type != "unknown_trigger":
                                 trigger_type_pred = self._PredicateClass(
                                     name="narrativeTriggerType",
                                     args=(new_entry.timestamp, main_trigger_type),
                                     value=True,
                                     timestamp=new_entry.timestamp
                                 )
                                 summary_predicates_to_assert.append(trigger_type_pred)
                    
                        if summary_predicates_to_assert: 
                            assert_tasks = [self._kb.assert_fact(pred) for pred in summary_predicates_to_assert]
                            await asyncio.gather(*assert_tasks)
                            logger_narrative.info(f"Asserted {len(summary_predicates_to_assert)} summary predicates to KB for completed narrative task {task_id}.")
                    except Exception as e_kb_narr_async:
                        logger_narrative.error(f"Failed to assert async narrative summary to KB: {e_kb_narr_async}")
        
        for task_id in completed_task_ids:
            del self._pending_narrative_llm_tasks[task_id]

        if not input_state: return None
        phenomenal_state = input_state.get("phenomenal_state")
        last_action_result = input_state.get("last_action_result", {})
        loop_info = input_state.get("loop_info")
        meta_analysis = input_state.get("meta_analysis", {})

        _PhenomenalState = globals().get('PhenomenalState')
        _GoalStatus = globals().get('GoalStatus')
        _Goal = globals().get('Goal')
        _ConsciousState = globals().get('ConsciousState')
        _PredictiveWorldModel = globals().get('PredictiveWorldModel')
        _EmergentMotivationSystem = globals().get('EmergentMotivationSystem')

        if not _PhenomenalState or not _GoalStatus or not _Goal or not _ConsciousState:
             logger_narrative.error("Cannot process narrative, missing dependent types.")
             return None

        prediction_error = input_state.get("prediction_error") 
        current_drives = input_state.get("current_drives", {}) # Get current_drives from input_state

        is_sig, reason, event_summary = self._is_significant( phenomenal_state, last_action_result, loop_info, meta_analysis, prediction_error, current_drives )

        if is_sig:
            logger_narrative.info(f"Significant event detected (reason: {reason}), scheduling narrative generation.")
            event_timestamp = getattr(phenomenal_state, 'timestamp', time.time()) if isinstance(phenomenal_state, globals().get('PhenomenalState', object)) else (phenomenal_state.get('timestamp', time.time()) if isinstance(phenomenal_state, dict) else time.time())
            await self.generate_narrative_entry(phenomenal_state, event_summary, reason, event_timestamp)
            await self.save_narrative()
        return None


    async def reset(self) -> None:
        """Reset narrative state."""
        self.narrative.clear()
        self._last_phenomenal_state = None
        self._last_save_time = 0.0
        self._last_drive_state = None
        self._last_pain_level_nc = None 
        self._last_happiness_level_nc = None
        self._last_purpose_level_nc = None
        self._known_pain_source_ids_nc.clear()
        
        # Cancel any pending LLM tasks on reset
        for task_id, task_info in self._pending_narrative_llm_tasks.items():
            future = task_info["future"]
            if not future.done():
                future.cancel(f"NarrativeConstructor reset: Cancelling task {task_id}")
                logger_narrative.info(f"Cancelled pending narrative LLM task {task_id} during reset.")
        self._pending_narrative_llm_tasks.clear()
        self._narrative_task_id_counter = 0


        # Re-initialize from controller if possible after reset
        if self._controller:
            self._last_pain_level_nc = getattr(self._controller, 'pain_level', 0.0)
            self._last_happiness_level_nc = getattr(self._controller, 'happiness_level', 5.0)
            self._last_purpose_level_nc = getattr(self._controller, 'purpose_level', 5.0)
            active_ps_init = getattr(self._controller, 'active_pain_sources', [])
            if isinstance(active_ps_init, list) and globals().get('PainSource'):
                _PainSource_local_reset = globals().get('PainSource')
                self._known_pain_source_ids_nc = {
                    ps.id for ps in active_ps_init 
                    if isinstance(ps, _PainSource_local_reset) and hasattr(ps, 'id') #type: ignore
                }

        logger_narrative.info("NarrativeConstructor reset (including P/H/P tracking and pending LLM tasks).")

    async def get_status(self) -> Dict[str, Any]:
        """Return status of the narrative constructor."""
        # Logic remains the same
        last_entry_ts = self.narrative[-1].timestamp if self.narrative else None
        return { "component": "NarrativeConstructor", "status": "operational", "narrative_length": len(self.narrative),
                 "max_length": self.max_length, "last_entry_timestamp": last_entry_ts, "last_save_time": self._last_save_time,
                 "save_path": str(self._narrative_path) if self._narrative_path else None,
                 "llm_model": self.llm_model_name,
                 "pending_llm_tasks": len(self._pending_narrative_llm_tasks) } # Added pending task count

    async def shutdown(self) -> None:
        """Perform cleanup, ensuring narrative is saved."""
        logger_narrative.info("NarrativeConstructor shutting down...");
        # Attempt to wait for any very quickly finishing tasks, but don't block shutdown excessively
        # This is a simple approach; more robust would involve signaling tasks to finish early.
        if self._pending_narrative_llm_tasks:
            logger_narrative.info(f"Waiting briefly for {len(self._pending_narrative_llm_tasks)} pending LLM tasks before final save...")
            # Wait for all futures, but with a very short timeout for each
            # This is tricky as futures might be handled by a different loop that's also shutting down.
            # For now, primarily rely on the secondary_loop's own shutdown logic.
            # We will just ensure one final save attempt.
        
        await self.save_narrative(force=True)
        logger_narrative.info("NarrativeConstructor shutdown complete.")


# --- END OF cognitive_modules/narrative_constructor.py (LLM Integration) ---