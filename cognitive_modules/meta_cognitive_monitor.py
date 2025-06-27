# --- START OF CORRECTED meta_cognitive_monitor.py ---

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Deque, Tuple
from collections import deque
import json # Added for parsing hints

# --- Use standard relative imports ---
try:
    from ..protocols import CognitiveComponent
    # Import Enums and Datatypes needed
    from ..models.enums import ConsciousState, GoalStatus
    from ..models.datatypes import Goal, Predicate # Added Predicate
    from .knowledge_base import KnowledgeBase # Added KnowledgeBase for type hinting
except ImportError:
    # Fallback for different execution context (e.g., combined script)
    logging.warning("MetaCognitiveMonitor: Relative imports failed, relying on globally defined types.")
    if 'CognitiveComponent' not in globals(): raise ImportError("CognitiveComponent not found via relative import or globally")
    if 'ConsciousState' not in globals(): raise ImportError("ConsciousState not found via relative import or globally")
    if 'GoalStatus' not in globals(): raise ImportError("GoalStatus not found via relative import or globally")
    if 'Goal' not in globals(): raise ImportError("Goal not found via relative import or globally")
    if 'Predicate' not in globals(): logging.warning("Predicate not found globally.") # Allow if not strictly needed
    if 'KnowledgeBase' not in globals(): logging.warning("KnowledgeBase not found globally.")

    CognitiveComponent = globals().get('CognitiveComponent')
    ConsciousState = globals().get('ConsciousState')
    GoalStatus = globals().get('GoalStatus')
    Goal = globals().get('Goal')
    Predicate = globals().get('Predicate') # type: ignore
    KnowledgeBase = globals().get('KnowledgeBase') # type: ignore


logger_meta_cognition = logging.getLogger(__name__) # Use standard module logger name

# Default configuration values
DEFAULT_STAGNATION_THRESHOLD_S = 120 # Time in seconds before considering a goal potentially stuck
DEFAULT_FAILURE_RATE_THRESHOLD = 0.6 # e.g., if >60% of recent actions for a goal failed
DEFAULT_LOW_CONSCIOUSNESS_DURATION_S = 30 # Time agent can stay below CONSCIOUS before flagging
DEFAULT_HISTORY_SIZE = 20 # How many recent states/metrics to consider

# --- Inherit correctly from CognitiveComponent ---
class MetaCognitiveMonitor(CognitiveComponent):
    """
    Observes the agent's cognitive state and performance for anomalies and inefficiencies.
    Can detect issues like goal stagnation, repetitive failures, etc.
    """

    def __init__(self):
        self._controller: Optional[Any] = None
        self._config: Dict[str, Any] = {}
        # Thresholds
        self.stagnation_threshold_s: float = DEFAULT_STAGNATION_THRESHOLD_S
        self.failure_rate_threshold: float = DEFAULT_FAILURE_RATE_THRESHOLD
        self.low_consciousness_threshold_s: float = DEFAULT_LOW_CONSCIOUSNESS_DURATION_S
        # History tracking (can be expanded)
        self.history_size: int = DEFAULT_HISTORY_SIZE
        self.recent_goal_progress: Deque[Dict[str, Any]] = deque(maxlen=self.history_size)
        # Ensure type hint uses quotes or defined type
        _ConsciousState = globals().get('ConsciousState', Any) # Get type ref safely
        self.recent_consciousness_levels: Deque[Tuple[float, _ConsciousState]] = deque(maxlen=self.history_size)
        
        # --- ADD KB REFERENCE ---
        self._kb: Optional[KnowledgeBase] = None


    async def initialize(self, config: Dict[str, Any], controller: Any) -> bool:
        """Initialize monitor with configuration."""
        self._controller = controller
        mc_config = config.get("meta_cognition", {})
        self._config = mc_config

        self.stagnation_threshold_s = mc_config.get("stagnation_threshold_s", DEFAULT_STAGNATION_THRESHOLD_S)
        self.failure_rate_threshold = mc_config.get("failure_rate_threshold", DEFAULT_FAILURE_RATE_THRESHOLD)
        self.low_consciousness_threshold_s = mc_config.get("low_consciousness_threshold_s", DEFAULT_LOW_CONSCIOUSNESS_DURATION_S)
        self.history_size = mc_config.get("history_size", DEFAULT_HISTORY_SIZE)

        # Recreate deques with configured size
        self.recent_goal_progress = deque(maxlen=self.history_size)
        _ConsciousState_local = globals().get('ConsciousState', Any) # Get type ref safely for local use
        self.recent_consciousness_levels = deque(maxlen=self.history_size) # Type hint handled at definition

        # --- GET KB REFERENCE ---
        _KnowledgeBaseClass_local = globals().get('KnowledgeBase') # Local var for type check
        if _KnowledgeBaseClass_local and hasattr(controller, 'knowledge_base') and \
           isinstance(controller.knowledge_base, _KnowledgeBaseClass_local):
            self._kb = controller.knowledge_base # type: ignore
            logger_meta_cognition.info("MetaCognitiveMonitor initialized with KB reference.")
        else:
            logger_meta_cognition.error("MetaCognitiveMonitor: Could not get valid KnowledgeBase reference. Goal stagnation checks will be limited.")
            # self._kb remains None, checks needing it will gracefully degrade.
        # --- END KB REFERENCE ---


        logger_meta_cognition.info(f"MetaCognitiveMonitor initialized. Stagnation Threshold: {self.stagnation_threshold_s}s, "
                    f"Failure Threshold: {self.failure_rate_threshold:.2f}, History Size: {self.history_size}")
        return True

    async def monitor_cognition(self,
                                cognitive_state: Dict[str, Any],
                                performance_metrics: Dict[str, float]
                               ) -> Dict[str, Any]:
        """
        Analyzes the current cognitive state and performance metrics.
        Returns a dictionary containing the analysis results (meta_analysis).
        """
        # Get necessary types safely
        _ConsciousState_local = globals().get('ConsciousState')
        _Goal_local = globals().get('Goal')
        _GoalStatus_local = globals().get('GoalStatus')
        _Predicate_local = globals().get('Predicate') 

        if not _ConsciousState_local or not _Goal_local or not _GoalStatus_local:
             logger_meta_cognition.error("Cannot monitor cognition, missing dependent types (ConsciousState, Goal, GoalStatus).")
             return {"timestamp": time.time(), "issues_detected": [{"type": "initialization_error", "details": "Missing types"}], "suggestions": [], "confidence": 0.0}


        logger_meta_cognition.debug("Performing meta-cognitive monitoring...")
        current_time = time.time()
        analysis: Dict[str, Any] = {
            "timestamp": current_time,
            "issues_detected": [],
            "suggestions": [],
            "confidence": 1.0 # Overall confidence in current strategy
        }

        # --- Update History ---
        # Track consciousness level over time
        cs_level_str = cognitive_state.get("consciousness_level", "UNCONSCIOUS")
        current_cs_level = _ConsciousState_local.UNCONSCIOUS # Default
        try:
            if _ConsciousState_local : current_cs_level = _ConsciousState_local[cs_level_str] # type: ignore
        except KeyError:
            logger_meta_cognition.warning(f"Invalid consciousness level string '{cs_level_str}', defaulting to UNCONSCIOUS.")
        self.recent_consciousness_levels.append((current_time, current_cs_level))

        # --- Perform Checks ---

        # 1. Check for Goal Stagnation
        active_goal: Optional[Goal] = None # type: ignore
        if self._controller and hasattr(self._controller, '_oscar_get_active_goal'):
            active_goal_getter = getattr(self._controller, '_oscar_get_active_goal', lambda: None)
            active_goal = active_goal_getter()

        if active_goal and isinstance(active_goal, _Goal_local) and hasattr(active_goal, 'creation_time') and hasattr(active_goal, 'status'): # type: ignore
            goal_age = current_time - active_goal.creation_time # type: ignore
            
            # Only check for stagnation if goal is ACTIVE and old enough
            if active_goal.status == _GoalStatus_local.ACTIVE and goal_age > self.stagnation_threshold_s: # type: ignore
                actual_failure_rate = 0.0
                actions_for_goal_count = 0

                if self._kb and _Predicate_local: # Proceed only if KB and Predicate class are available
                    try:
                        # Query for eventOccurred predicates: ("actionExecution", action_type, outcome)
                        # This relies on the KB storing Predicate objects or dicts that can be parsed.
                        # We'll fetch all "eventOccurred" of type "actionExecution" and filter by timestamp.
                        # This is less efficient than a direct timestamped query but works with current KB.query_state.
                        
                        # Fetch a reasonable number of recent facts
                        # Estimate based on goal age and typical cycle time
                        estimated_facts_to_fetch = 500 # Max facts to check for performance
                        if self._controller and hasattr(self._controller, 'config'):
                             target_cycle_time_kb = self._controller.config.get("performance", {}).get("target_cycle_time", 0.1)
                             if target_cycle_time_kb > 0:
                                 estimated_facts_to_fetch = int(goal_age / target_cycle_time_kb * 1.5) + 50 # x1.5 for buffer, +50 base
                                 estimated_facts_to_fetch = max(50, min(1000, estimated_facts_to_fetch)) # Cap fetching
                        
                        kb_query_result = await self._kb.query_state({"recent_facts": estimated_facts_to_fetch})
                        recent_facts_dicts = kb_query_result.get("recent_facts", [])
                        
                        relevant_actions_outcomes = []
                        for fact_data in recent_facts_dicts:
                            # Assuming recent_facts returns list of dicts from KB
                            if isinstance(fact_data, dict) and \
                               fact_data.get("name") == "eventOccurred" and \
                               fact_data.get("timestamp", 0) >= active_goal.creation_time: # type: ignore
                                
                                args = fact_data.get("args")
                                if isinstance(args, (list, tuple)) and len(args) == 3 and args[0] == "actionExecution":
                                    # args[1] is action_type, args[2] is outcome ("success" or "failure")
                                    relevant_actions_outcomes.append(args[2])
                        
                        actions_for_goal_count = len(relevant_actions_outcomes)
                        if actions_for_goal_count > 0:
                            failures = sum(1 for outcome in relevant_actions_outcomes if outcome == "failure")
                            actual_failure_rate = failures / actions_for_goal_count
                            logger_meta_cognition.debug(
                                f"Goal '{str(active_goal.description)[:30]}...': Age={goal_age:.0f}s, " # type: ignore
                                f"Analyzed {actions_for_goal_count} action outcomes since creation. "
                                f"Failures: {failures}, Actual Fail Rate: {actual_failure_rate:.2f}"
                            )
                        else:
                            logger_meta_cognition.debug(f"Goal '{str(active_goal.description)[:30]}...' old but no action outcomes recorded in KB since creation.") # type: ignore
                            # If goal is old but no actions, it's also stagnating.
                            actual_failure_rate = 1.0 # Treat as 100% failure if no actions attempted for an old goal

                    except Exception as e_kb:
                        logger_meta_cognition.exception(f"Error querying KB for goal action history: {e_kb}")
                        actual_failure_rate = -1 # Indicate error in calculation
                else: # No KB or Predicate class
                    logger_meta_cognition.warning("KB or Predicate class not available for goal stagnation check.")
                    actual_failure_rate = -1 # Cannot determine

                if actual_failure_rate > self.failure_rate_threshold:
                    issue_details = (
                        f"Goal '{str(active_goal.description)[:30]}...' potentially stagnated. " # type: ignore
                        f"Age: {goal_age:.0f}s. Observed Failure Rate: {actual_failure_rate:.2f} "
                        f"(Threshold: {self.failure_rate_threshold:.2f}) based on {actions_for_goal_count} actions."
                    )
                    logger_meta_cognition.warning(f"MetaCognition: {issue_details}")
                    analysis["issues_detected"].append({"type": "goal_stagnation", "details": issue_details, "goal_id": active_goal.id}) # type: ignore
                    analysis["suggestions"].extend(["review_goal_viability", "try_alternative_plan_method"])
                    analysis["confidence"] *= 0.7
                elif actual_failure_rate == -1: # Error during calculation
                     logger_meta_cognition.warning(f"Could not determine failure rate for goal '{str(active_goal.description)[:30]}...' due to KB query error or missing KB.") # type: ignore


        # 2. Check for Persistent Low Consciousness
        # time_below_conscious = 0 # This variable definition was redundant
        if len(self.recent_consciousness_levels) >= self.history_size: 
            first_timestamp_in_window = self.recent_consciousness_levels[0][0]
            duration_in_history_window = current_time - first_timestamp_in_window
            
            if duration_in_history_window > self.low_consciousness_threshold_s and _ConsciousState_local: # type: ignore
                 is_persistently_low = all(level.value < _ConsciousState_local.CONSCIOUS.value for _, level in self.recent_consciousness_levels) # type: ignore
                 if is_persistently_low:
                      issue = f"Persistently low consciousness (below CONSCIOUS for >{duration_in_history_window:.0f}s)"
                      logger_meta_cognition.warning(f"MetaCognition: {issue}")
                      analysis["issues_detected"].append({"type": "low_consciousness", "details": issue})
                      analysis["suggestions"].append("increase_stimulation_or_check_sensory_input")
                      analysis["confidence"] *= 0.8

        # 3. Check for High Performance Bottlenecks
        avg_cycle_time = performance_metrics.get("average_cycle_time", 0.0)
        if avg_cycle_time == 0.0 and performance_metrics: 
            total_duration_in_profile = sum(v for v in performance_metrics.values() if isinstance(v, (int, float)))
            avg_cycle_time = total_duration_in_profile 

        target_cycle_time = 0.1
        if self._controller and hasattr(self._controller, 'config'):
             target_cycle_time = self._controller.config.get("performance", {}).get("target_cycle_time", 0.1)
        
        if avg_cycle_time > target_cycle_time * 1.5: 
            issue = f"Average cycle time ({avg_cycle_time:.3f}s) significantly exceeds target ({target_cycle_time:.3f}s)"
            logger_meta_cognition.warning(f"MetaCognition: {issue}")
            analysis["issues_detected"].append({"type": "performance_bottleneck", "details": issue})
            analysis["suggestions"].append("review_performance_profile_and_optimizer_settings")

        # --- NEW CHECK: Meta-Conscious State Active ---
        # Ensure _ConsciousState_local is available (it's fetched at the start of the method)
        if _ConsciousState_local and hasattr(_ConsciousState_local, "META_CONSCIOUS"):
            meta_conscious_enum_val = _ConsciousState_local.META_CONSCIOUS
            # current_cs_level is already derived and available from the history update part
            # (or directly from cognitive_state.get("consciousness_level") if preferred)
            
            # Use current_cs_level which is already an enum member
            if current_cs_level == meta_conscious_enum_val: # type: ignore
                issue_meta_conscious = {
                    "type": "meta_conscious_state_active",
                    "details": "Agent is in a meta-conscious state, indicating deep self-awareness or processing.",
                    "current_level": current_cs_level.name # type: ignore
                }
                logger_meta_cognition.info(f"MetaCognition: Detected META_CONSCIOUS state.")
                analysis["issues_detected"].append(issue_meta_conscious)
                # Optionally add a suggestion if specific actions are desired in this state
                # analysis["suggestions"].append("initiate_deep_reflection_task")
                # analysis["confidence"] might be adjusted here too if needed
        # --- END NEW CHECK ---

        # --- NEW: Check for recent plan rejections by ValueSystem and modification hints ---
        if self._kb and _Predicate_local and active_goal and hasattr(active_goal, 'id'): # type: ignore
            goal_id_mcm = active_goal.id # type: ignore
            try:
                # Check if the plan for the current active goal was recently rejected
                rejection_preds = await self._kb.query(
                    name="planRejectedByValue",
                    args=(goal_id_mcm,), # Query by goal_id, other args will be part of the result
                    value=True
                )
                latest_rejection_pred = max(rejection_preds, key=lambda p: p.timestamp, default=None)

                if latest_rejection_pred and (current_time - latest_rejection_pred.timestamp < self.stagnation_threshold_s / 2): # Check if rejection is recent
                    rejection_reason_mcm = latest_rejection_pred.args[1] if len(latest_rejection_pred.args) > 1 else "UnknownReason" # type: ignore
                    rejection_score_mcm = latest_rejection_pred.args[2] if len(latest_rejection_pred.args) > 2 else "N/A" # type: ignore
                    
                    issue_details_plan_reject = (
                        f"Plan for current goal '{str(active_goal.description)[:30]}...' (ID: {goal_id_mcm}) " # type: ignore
                        f"was recently rejected by ValueSystem. Reason: {rejection_reason_mcm}, Score: {rejection_score_mcm}."
                    )
                    logger_meta_cognition.warning(f"MetaCognition: {issue_details_plan_reject}")
                    analysis["issues_detected"].append({
                        "type": "plan_value_rejection", 
                        "details": issue_details_plan_reject, 
                        "goal_id": goal_id_mcm
                    })
                    analysis["confidence"] *= 0.8

                    # Now look for modification hints associated with this rejected plan/goal
                    hint_preds = await self._kb.query(
                        name="pendingPlanModificationHints",
                        args=(goal_id_mcm,),
                        value=True
                    )
                    latest_hint_pred_mcm = max(hint_preds, key=lambda p: p.timestamp, default=None)
                    
                    parsed_hints_for_suggestion = None
                    if latest_hint_pred_mcm and len(latest_hint_pred_mcm.args) > 1: # type: ignore
                        try:
                            parsed_hints_for_suggestion = json.loads(latest_hint_pred_mcm.args[1]) # type: ignore
                        except json.JSONDecodeError:
                            logger_meta_cognition.error(f"MCM: Failed to parse hints JSON: {latest_hint_pred_mcm.args[1]}") # type: ignore

                    suggestion_for_replan = {
                        "type": "REPLAN_GOAL_WITH_HINTS",
                        "goal_id": goal_id_mcm,
                        "reason": "value_system_rejection",
                        "current_plan_rejected": True # Flag that the current plan (if any) for this goal is bad
                    }
                    if parsed_hints_for_suggestion:
                        suggestion_for_replan["hints"] = parsed_hints_for_suggestion
                        logger_meta_cognition.info(f"MCM: Suggesting REPLAN_GOAL_WITH_HINTS for goal '{goal_id_mcm}' due to VS rejection. Hints provided.")
                    else:
                        logger_meta_cognition.info(f"MCM: Suggesting REPLAN_GOAL (no specific hints found/parsed) for goal '{goal_id_mcm}' due to VS rejection.")
                    
                    analysis["suggestions"].append(suggestion_for_replan)

            except Exception as e_kb_mcm_vs:
                logger_meta_cognition.error(f"MCM: Error querying KB for VS rejection/hints: {e_kb_mcm_vs}")
        # --- END NEW ---

        logger_meta_cognition.info(f"Meta-cognitive analysis complete. Issues: {len(analysis['issues_detected'])}, Suggestions: {len(analysis['suggestions'])}, Confidence: {analysis['confidence']:.2f}")
        return analysis


    # --- CognitiveComponent Implementation ---

    async def process(self, input_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Performs meta-cognitive monitoring based on input state.
        Input keys: 'cognitive_state' (Dict), 'performance_metrics' (Dict)
        """
        if not input_state:
            logger_meta_cognition.warning("MetaCognitiveMonitor process: Missing input state.")
            return None

        cognitive_state = input_state.get("cognitive_state")
        performance_metrics = input_state.get("performance_metrics", {}) # Use empty dict if missing

        if not isinstance(cognitive_state, dict):
            logger_meta_cognition.error("MetaCognitiveMonitor process: Invalid or missing 'cognitive_state'.")
            return None
        if not isinstance(performance_metrics, dict):
             logger_meta_cognition.error("MetaCognitiveMonitor process: Invalid type for 'performance_metrics'.")
             return None


        # Call the core monitoring logic
        meta_analysis = await self.monitor_cognition(cognitive_state, performance_metrics)

        # Return the result wrapped in a dict
        return {"meta_analysis": meta_analysis}

    async def reset(self) -> None:
        """Reset monitor state, clearing history."""
        self.recent_goal_progress.clear()
        self.recent_consciousness_levels.clear()
        logger_meta_cognition.info("MetaCognitiveMonitor reset.")

    async def get_status(self) -> Dict[str, Any]:
        """Return status of the meta-cognitive monitor."""
        last_cs_level_name = "N/A"
        if self.recent_consciousness_levels:
             last_cs_level_name = self.recent_consciousness_levels[-1][1].name

        return {
            "component": "MetaCognitiveMonitor",
            "status": "operational",
            "history_size": len(self.recent_consciousness_levels), # Example metric
            "last_recorded_consciousness": last_cs_level_name,
            "config_thresholds": {
                "stagnation_s": self.stagnation_threshold_s,
                "failure_rate": self.failure_rate_threshold,
                "low_cs_s": self.low_consciousness_threshold_s
            }
        }

    async def shutdown(self) -> None:
        """Perform any necessary cleanup."""
        logger_meta_cognition.info("MetaCognitiveMonitor shutting down.")

# --- END OF CORRECTED meta_cognitive_monitor.py ---