# consciousness_experiment/agent_helpers/internal_state_manager.py
import logging
import math
import time # Though not directly used in perform_upkeep, good to have if timestamps are added
from typing import TYPE_CHECKING, List, Any, Dict, Optional 
import asyncio

# --- Add Fallback MockDataclass definition at the top, like in AgentController ---
# This ensures MockDataclass is defined in this module's scope for the comparison.
class MockDataclassFwd: 
    def __init__(self, *args, **kwargs): pass
    @classmethod
    def _is_dummy(cls): return True
# --- End Fallback ---

# Import necessary enums and datatypes.
# These will be resolved if AgentController and other modules are in sys.path
# or if running in a context where they are globally available.
try:
    from ..models.datatypes import PainSource
    from ..models.enums import GoalStatus # Assuming GoalStatus might be needed for context later
    # If AgentController type hint is needed for self.agent
    if TYPE_CHECKING:
        from ..agent_controller import AgentController
except ImportError:
    logging.getLogger(__name__).warning("InternalStateManager: Relative imports failed. Using placeholders.")
    # Fallback for standalone or partial execution context
    if 'PainSource' not in globals():
        class PainSource(MockDataclassFwd): pass # type: ignore
    if 'GoalStatus' not in globals():
        class GoalStatus(MockDataclassFwd): pass # type: ignore 
    if TYPE_CHECKING:
        class AgentController: pass # type: ignore


logger_isum = logging.getLogger(__name__) # Internal State Upkeep Manager

class InternalStateUpkeepManager:
    def __init__(self, agent_controller: 'AgentController'):
        self.agent_controller = agent_controller
        self._last_dsm_capabilities_count: int = 0
        self._dsm_status_initialized: bool = False 
        logger_isum.info("InternalStateUpkeepManager initialized.")

    async def perform_upkeep_cycle_start(self):
        """Handles per-cycle updates for age, baseline pain, and active pain source decay."""
        agent = self.agent_controller
        internal_states_config = agent.config.get("internal_states", {}) 

        agent.agent_age_cycles += 1
        logger_isum.debug(f"PHP_UPKEEP - Age: {agent.agent_age_cycles}")

        # Baseline Pain from Age
        baseline_pain_age_factor = float(internal_states_config.get("baseline_pain_age_factor", 0.00001)) 
        max_baseline_pain_from_age = float(internal_states_config.get("max_baseline_pain_from_age", 1.0)) 
        
        scaled_age_for_pain = agent.agent_age_cycles / 1000.0
        current_baseline_pain = math.log(scaled_age_for_pain + 1) * baseline_pain_age_factor 
        agent.baseline_pain_level = min(current_baseline_pain, max_baseline_pain_from_age) 
        logger_isum.debug(f"PHP_UPKEEP - BaselinePain: {agent.baseline_pain_level:.4f} (ScaledAge: {scaled_age_for_pain:.4f}, Factor: {baseline_pain_age_factor}, Max: {max_baseline_pain_from_age})")

        # Decay Active Pain Sources & Sum Acute Pain
        current_acute_pain_sum = 0.0
        active_pain_sources_next_cycle: List[PainSource] = []
        
        pain_event_min_intensity_to_retain = float(internal_states_config.get("pain_event_min_intensity_to_retain", 0.01)) 

        is_pain_source_real_class = agent._PainSource and not (hasattr(agent._PainSource, '_is_dummy') and agent._PainSource._is_dummy()) # type: ignore

        if is_pain_source_real_class:
            for pain_event in agent.active_pain_sources:
                if not isinstance(pain_event, agent._PainSource): # type: ignore
                    logger_isum.warning(f"Skipping pain_event of unexpected type: {type(pain_event)}")
                    continue

                if not pain_event.is_resolved: # type: ignore
                    old_intensity = pain_event.current_intensity # type: ignore
                    pain_event.current_intensity *= (1.0 - pain_event.decay_rate_per_cycle) # type: ignore
                    logger_isum.debug(f"PHP_UPKEEP - PainSrc Decay: ID='{pain_event.id}', Type='{pain_event.type}', Desc='{pain_event.description[:30]}...', OldIntensity={old_intensity:.3f} -> NewIntensity={pain_event.current_intensity:.3f} (DecayRate: {pain_event.decay_rate_per_cycle})") # type: ignore
                    if pain_event.current_intensity > pain_event_min_intensity_to_retain: # type: ignore
                        active_pain_sources_next_cycle.append(pain_event)
                        current_acute_pain_sum += pain_event.current_intensity # type: ignore
                    else:
                        logger_isum.info(
                            f"PHP_UPKEEP - PainSrc Removed (Decayed): ID='{pain_event.id}', Type='{pain_event.type}', Desc='{pain_event.description[:30]}...', FinalIntensity={pain_event.current_intensity:.3f}" # type: ignore
                        )
                else:
                    logger_isum.debug(f"PHP_UPKEEP - Resolved PainSrc '{pain_event.id}' ({pain_event.description[:30]}) formally removed from active list. Intensity was: {pain_event.current_intensity:.3f}") # type: ignore
        else:
            logger_isum.debug("PHP_UPKEEP - PainSource class is a mock/dummy or unavailable. Skipping active_pain_sources decay loop.")


        agent.active_pain_sources = active_pain_sources_next_cycle
        
        max_pain_shutdown_threshold = float(internal_states_config.get("max_pain_shutdown_threshold", 9.0)) 
        agent.pain_level = min(max_pain_shutdown_threshold, agent.baseline_pain_level + current_acute_pain_sum)
        logger_isum.debug(f"PHP_UPKEEP - TotalPain: {agent.pain_level:.3f} (Base: {agent.baseline_pain_level:.3f}, AcuteSum: {current_acute_pain_sum:.3f}, MaxThreshold: {max_pain_shutdown_threshold})")

        happiness_decay_to_baseline_factor = float(internal_states_config.get("happiness_decay_to_baseline_factor", 0.03)) 
        happiness_baseline_target = float(internal_states_config.get("happiness_baseline_target", 5.0)) 
        old_happiness = agent.happiness_level
        agent.happiness_level += happiness_decay_to_baseline_factor * \
                                (happiness_baseline_target - agent.happiness_level)
        logger_isum.debug(f"PHP_UPKEEP - Happiness (Post-Decay): Old={old_happiness:.3f} -> New={agent.happiness_level:.3f} (BaselineTarget: {happiness_baseline_target}, DecayFactor: {happiness_decay_to_baseline_factor})")
        agent.happiness_level = max(0.0, min(10.0, agent.happiness_level))

        purpose_decay_rate_per_cycle = float(internal_states_config.get("purpose_decay_rate_per_cycle", 0.0005)) 
        old_purpose = agent.purpose_level
        agent.purpose_level -= purpose_decay_rate_per_cycle
        logger_isum.debug(f"PHP_UPKEEP - Purpose (Post-Decay): Old={old_purpose:.3f} -> New={agent.purpose_level:.3f} (DecayRate: {purpose_decay_rate_per_cycle})")
        agent.purpose_level = max(0.0, min(10.0, agent.purpose_level))

    async def perform_upkeep_post_action_learning(self,
                                            goal_just_achieved_this_cycle: bool,
                                            priority_of_achieved_goal: float,
                                            achieved_goal_object: Optional[Any] = None):
        agent = self.agent_controller
        internal_states_config = agent.config.get("internal_states", {}) 

        if goal_just_achieved_this_cycle:
            happiness_from_goal_priority_scale_factor = float(internal_states_config.get("happiness_from_goal_priority_scale_factor", 0.2))
            effective_priority_for_happiness = priority_of_achieved_goal
            happiness_gain = effective_priority_for_happiness * happiness_from_goal_priority_scale_factor
            old_happiness = agent.happiness_level
            agent.happiness_level += happiness_gain
            logger_isum.info(f"PHP_POST_LEARN - Happiness increased by {happiness_gain:.2f} from achieving goal. Old={old_happiness:.3f} -> NewRaw={agent.happiness_level:.3f}")

            is_pain_source_real_class_pupal = agent._PainSource and not (hasattr(agent._PainSource, '_is_dummy') and agent._PainSource._is_dummy()) # type: ignore
            if achieved_goal_object and hasattr(achieved_goal_object, 'id') and is_pain_source_real_class_pupal:
                goal_id_for_resolution = achieved_goal_object.id
                for pain_event in agent.active_pain_sources:
                    if not isinstance(pain_event, agent._PainSource): continue # type: ignore
                    if pain_event.source_goal_id == goal_id_for_resolution and not pain_event.is_resolved: # type: ignore
                        pain_event.is_resolved = True # type: ignore
                        old_pain_intensity = pain_event.current_intensity # type: ignore
                        intensity_reduction = pain_event.current_intensity * 0.90  # type: ignore
                        pain_event.current_intensity -= intensity_reduction # type: ignore
                        
                        happiness_bonus_for_resolution = intensity_reduction * 0.5 
                        old_happiness_before_bonus = agent.happiness_level
                        agent.happiness_level += happiness_bonus_for_resolution
                        logger_isum.info(
                            f"PHP_POST_LEARN - PainSource for goal '{goal_id_for_resolution}' ({pain_event.description[:30]}) resolved. " # type: ignore
                            f"Reduced pain intensity by {intensity_reduction:.2f} (Old: {old_pain_intensity:.3f} -> New: {pain_event.current_intensity:.3f}). " # type: ignore
                            f"Happiness bonus: {happiness_bonus_for_resolution:.2f} (OldHappy: {old_happiness_before_bonus:.3f} -> NewHappy: {agent.happiness_level:.3f})."
                        )
                        break 

            complex_goal_priority_threshold = float(internal_states_config.get("complex_goal_priority_threshold", 4.5))
            purpose_from_high_priority_goal_factor = float(internal_states_config.get("purpose_from_high_priority_goal_factor", 0.2))
            if effective_priority_for_happiness >= complex_goal_priority_threshold:
                old_purpose = agent.purpose_level
                agent.purpose_level += purpose_from_high_priority_goal_factor
                logger_isum.info(
                    f"PHP_POST_LEARN - Purpose increased by {purpose_from_high_priority_goal_factor:.2f} "
                    f"due to complex/high-priority goal achievement (Prio: {effective_priority_for_happiness:.2f}). Old={old_purpose:.3f} -> New={agent.purpose_level:.2f}"
                )
        
        # Purpose from DSM Capability Gain
        dsm_component = getattr(agent, "dynamic_self_model", None)
        if dsm_component and hasattr(dsm_component, "get_status"):
            dsm_status = {}
            try:
                get_status_method_dsm = getattr(dsm_component, "get_status")
                if asyncio.iscoroutinefunction(get_status_method_dsm):
                    dsm_status = await get_status_method_dsm()
                else: dsm_status = get_status_method_dsm() # type: ignore
            except Exception as e_dsm_status_isum:
                logger_isum.warning(f"Could not get DSM status for purpose update: {e_dsm_status_isum}")

            if isinstance(dsm_status, dict):
                current_dsm_caps_count = dsm_status.get("num_capabilities", 0)
                
                if not self._dsm_status_initialized:
                    self._last_dsm_capabilities_count = current_dsm_caps_count
                    self._dsm_status_initialized = True
                    logger_isum.info(f"PHP_POST_LEARN - Initialized _last_dsm_capabilities_count to: {self._last_dsm_capabilities_count}")

                if current_dsm_caps_count > self._last_dsm_capabilities_count:
                    cap_increase_amount = current_dsm_caps_count - self._last_dsm_capabilities_count
                    purpose_from_capability_gain_factor = float(internal_states_config.get("purpose_from_capability_gain_factor", 0.05))
                    purpose_gain_caps = cap_increase_amount * purpose_from_capability_gain_factor
                    old_purpose = agent.purpose_level
                    agent.purpose_level += purpose_gain_caps
                    logger_isum.info(
                        f"PHP_POST_LEARN - Purpose increased by {purpose_gain_caps:.3f} due to {cap_increase_amount} new capability/ies. "
                        f"Old={old_purpose:.3f} -> New={agent.purpose_level:.2f}"
                    )
                self._last_dsm_capabilities_count = current_dsm_caps_count
        
        pain_impact_on_happiness_scale_factor = float(internal_states_config.get("pain_impact_on_happiness_scale_factor", 0.2))
        old_happiness_before_pain_impact = agent.happiness_level
        happiness_reduction_from_pain = agent.pain_level * pain_impact_on_happiness_scale_factor
        if happiness_reduction_from_pain > 0:
            agent.happiness_level -= happiness_reduction_from_pain
            logger_isum.debug(f"PHP_POST_LEARN - Happiness reduced by {happiness_reduction_from_pain:.2f} due to pain level {agent.pain_level:.2f}. Old={old_happiness_before_pain_impact:.3f} -> New={agent.happiness_level:.2f}")
        
        agent.happiness_level = max(0.0, min(10.0, agent.happiness_level))
        agent.purpose_level = max(0.0, min(10.0, agent.purpose_level))

    def generate_pain_from_goal_failure(self, failed_goal_object: Any, failure_type: str = "GenericGoalFailure"):
        """Generates a PainSource object when a goal fails."""
        agent = self.agent_controller
        internal_states_config = agent.config.get("internal_states", {}) 
        
        is_pain_source_real_class_gen = agent._PainSource and not (hasattr(agent._PainSource, '_is_dummy') and agent._PainSource._is_dummy()) # type: ignore

        if not (failed_goal_object and hasattr(failed_goal_object, 'id') and is_pain_source_real_class_gen):
            logger_isum.warning(f"Cannot generate pain for goal failure: Invalid goal object or PainSource class missing/dummy. Goal: {failed_goal_object}")
            return

        goal_id_for_pain = failed_goal_object.id
        goal_desc_for_pain = getattr(failed_goal_object, 'description', "Unknown Goal")
        
        default_prio = 1.0 
        if hasattr(agent, 'DEFAULT_OBSERVE_GOAL_PRIORITY'): # Check agent_controller directly
            default_prio = agent.DEFAULT_OBSERVE_GOAL_PRIORITY 
        goal_priority_for_pain = getattr(failed_goal_object, 'priority', default_prio)


        existing_pain_event = next((ps for ps in agent.active_pain_sources if isinstance(ps, agent._PainSource) and ps.source_goal_id == goal_id_for_pain and not ps.is_resolved), None) # type: ignore
        if existing_pain_event:
            logger_isum.info(f"Pain event for goal '{goal_id_for_pain}' ({failure_type}) already exists. Not creating duplicate. Current intensity: {existing_pain_event.current_intensity:.2f}") # type: ignore
            return

        pain_event_max_initial_intensity = float(internal_states_config.get("pain_event_max_initial_intensity", 2.0))
        
        pain_scale_factor = float(internal_states_config.get("acute_pain_goal_fail_priority_scale_factor", 0.3)) 
        if failure_type == "PersistentPlanningFailure":
            pain_scale_factor = float(internal_states_config.get("pain_from_planning_failure_scale_factor", 0.25))
        elif failure_type == "PersistentExecutionFailure":
            pain_scale_factor = float(internal_states_config.get("pain_from_execution_failure_scale_factor", 0.35))

        initial_pain_intensity = min(
            pain_event_max_initial_intensity, 
            goal_priority_for_pain * pain_scale_factor
        )
        
        pain_event_min_intensity_to_retain_local = float(internal_states_config.get("pain_event_min_intensity_to_retain", 0.01))
        default_pain_event_decay_rate_local = float(internal_states_config.get("default_pain_event_decay_rate_per_cycle", 0.005))

        if initial_pain_intensity > pain_event_min_intensity_to_retain_local:
            new_pain_source = agent._PainSource( # type: ignore
                id=f"PainSource_{failure_type}_{goal_id_for_pain}_{int(time.time())}",
                description=f"Failed ({failure_type}): {goal_desc_for_pain[:70]}",
                initial_intensity=initial_pain_intensity,
                # NO current_intensity=initial_pain_intensity HERE
                timestamp_created=time.time(),
                decay_rate_per_cycle=default_pain_event_decay_rate_local,
                type=failure_type,
                source_goal_id=goal_id_for_pain
                # is_resolved is False by default in PainSource definition
                # resolution_conditions is None by default in PainSource definition
            )
            agent.active_pain_sources.append(new_pain_source)
            logger_isum.warning(
                f"PHP_PAIN_GEN - Created PainSource for {failure_type} of goal '{goal_desc_for_pain}' (ID: {goal_id_for_pain}) "
                f"with initial intensity: {initial_pain_intensity:.2f}"
            )
            
            current_acute_pain_sum_after_new = sum(ps.current_intensity for ps in agent.active_pain_sources if hasattr(ps, 'is_resolved') and not ps.is_resolved and hasattr(ps, 'current_intensity')) # type: ignore
            max_pain_threshold_local = float(internal_states_config.get("max_pain_shutdown_threshold", 9.0))
            agent.pain_level = min(max_pain_threshold_local, agent.baseline_pain_level + current_acute_pain_sum_after_new)
            logger_isum.info(f"PHP_PAIN_GEN - agent.pain_level updated to {agent.pain_level:.3f} immediately after new PainSource creation.")
        else:
             logger_isum.info(f"Calculated pain intensity {initial_pain_intensity:.2f} for {failure_type} of goal '{goal_id_for_pain}' was below retain threshold. No PainSource created.")


    def check_existential_thresholds(self):
        """Checks if P/H/P levels have crossed shutdown thresholds."""
        agent = self.agent_controller
        internal_states_config = agent.config.get("internal_states", {}) 
        agent_config = agent.config.get("agent", {}) 

        max_pain_shutdown_threshold = float(internal_states_config.get("max_pain_shutdown_threshold", 9.0))
        min_purpose_shutdown_threshold = float(internal_states_config.get("min_purpose_shutdown_threshold", 1.0))
        min_cycles_before_purpose_shutdown = int(agent_config.get("min_cycles_before_purpose_shutdown", 100))

        shutdown_reason = None

        if agent.pain_level >= max_pain_shutdown_threshold:
            shutdown_reason = f"MaxPainThreshold: {agent.pain_level:.2f} >= {max_pain_shutdown_threshold:.2f}"
        elif agent.purpose_level <= min_purpose_shutdown_threshold:
            if agent.agent_age_cycles > min_cycles_before_purpose_shutdown:
                shutdown_reason = f"MinPurposeThreshold: {agent.purpose_level:.2f} <= {min_purpose_shutdown_threshold:.2f} (Age: {agent.agent_age_cycles})"
            else:
                logger_isum.debug(f"Purpose level {agent.purpose_level:.2f} is low, but agent age {agent.agent_age_cycles} is below threshold for shutdown.")

        if shutdown_reason:
            logger_isum.critical(f"SHUTDOWN TRIGGERED: {shutdown_reason}")
            agent._log_to_ui("critical", f"Agent Shutdown: {shutdown_reason}")
            
            narrative_constructor_instance = getattr(agent, 'narrative_constructor', None)
            ems_instance = getattr(agent, 'emergent_motivation_system', None)
            if narrative_constructor_instance and hasattr(narrative_constructor_instance, 'process') and agent.CORE_DEPENDENCIES_AVAILABLE: # Check agent_controller directly
                try:
                    shutdown_content_reason = "max_pain" if "MaxPainThreshold" in shutdown_reason else "min_purpose"
                    final_p_state_content = {"shutdown_reason": shutdown_content_reason, "details": shutdown_reason}
                    
                    final_p_state = agent._PhenomenalState( # type: ignore
                        content=final_p_state_content, 
                        intensity=1.0 if "MaxPainThreshold" in shutdown_reason else 0.5, 
                        valence=-1.0 if "MaxPainThreshold" in shutdown_reason else -0.5
                    )
                    final_event_summary = {
                        "shutdown_trigger": shutdown_content_reason, 
                        "final_pain_level": agent.pain_level,
                        "final_purpose_level": agent.purpose_level,
                        "final_happiness_level": agent.happiness_level
                        }
                    drive_values_for_final_narr = {}
                    if ems_instance and hasattr(ems_instance, 'get_drive_values'): # Assuming get_drive_values is sync
                       drive_values_for_final_narr = ems_instance.get_drive_values() 

                    if agent._asyncio_loop and agent._asyncio_loop.is_running():
                        async def _final_narrative_task():
                            await narrative_constructor_instance.process({ # type: ignore
                                "phenomenal_state": final_p_state,
                                "last_action_result": {"type": "INTERNAL_STATE_CHECK", "outcome": "critical_failure"},
                                "loop_info": None, "meta_analysis": {}, "prediction_error": None,
                                "current_drives": drive_values_for_final_narr
                            })
                        asyncio.ensure_future(_final_narrative_task(), loop=agent._asyncio_loop)
                    else:
                        logger_isum.warning("Asyncio loop not running for final narrative entry.")

                except Exception as e_narr:
                    logger_isum.error(f"Failed to create final narrative entry on shutdown: {e_narr}")
            
            agent.stop() 
            return True 
        return False