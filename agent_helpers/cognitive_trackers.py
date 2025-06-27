# consciousness_experiment/agent_helpers/cognitive_trackers.py
import logging
from collections import deque
from typing import Deque, Tuple, Dict, Any

logger_tracker = logging.getLogger(__name__)

class PainEventTracker:
    """
    Tracks pain events in the Global Workspace to detect and signal rumination,
    while allowing for eventual re-evaluation after a period of inactivity.
    """
    def __init__(self, 
                 pain_rumination_threshold: int = 3, 
                 rumination_window_multiplier: int = 3,
                 inactive_reset_cycles: int = 10):
        """
        Initializes the tracker.

        Args:
            pain_rumination_threshold: How many times a pain_id needs to appear in 
                                       the recent GWM history to be considered for suppression.
            rumination_window_multiplier: Determines the size of the recent history deque 
                                          (maxlen = threshold * multiplier).
            inactive_reset_cycles: Number of agent cycles a pain_id must be absent 
                                   from GWM before its history entries are pruned.
        """
        self.rumination_threshold: int = pain_rumination_threshold
        self.max_history: int = pain_rumination_threshold * rumination_window_multiplier
        self.inactive_reset_cycles: int = inactive_reset_cycles
        
        if self.rumination_threshold <= 0:
            logger_tracker.warning(f"PainEventTracker: pain_rumination_threshold ({self.rumination_threshold}) must be positive. Defaulting to 1.")
            self.rumination_threshold = 1
        if self.max_history <= 0:
            logger_tracker.warning(f"PainEventTracker: max_history ({self.max_history}) must be positive based on threshold and multiplier. Defaulting to pain_rumination_threshold ({self.rumination_threshold}).")
            self.max_history = self.rumination_threshold
        if self.inactive_reset_cycles <= 0:
            logger_tracker.warning(f"PainEventTracker: inactive_reset_cycles ({self.inactive_reset_cycles}) must be positive. Defaulting to 1.")
            self.inactive_reset_cycles = 1

        # Stores (pain_id_str, cycle_count_it_was_in_gwm)
        self._pain_gwm_history: Deque[Tuple[str, int]] = deque(maxlen=self.max_history)
        
        # Stores pain_id_str -> cycle_count_it_was_last_recorded_in_gwm
        # This helps _cleanup_inactive_entries to know the *most recent* sighting
        # of a pain_id, even if older sightings of it are still in _pain_gwm_history.
        self._last_recorded_in_gwm: Dict[str, int] = {}

        logger_tracker.info(
            f"PainEventTracker initialized. RuminationThreshold: {self.rumination_threshold}, "
            f"MaxHistory (deque size): {self.max_history}, InactiveResetCycles: {self.inactive_reset_cycles}"
        )

    def record_pain_in_gwm(self, pain_id: str, cycle_count_in_gwm: int):
        """
        Records that a specific pain event (by its ID) was present in the 
        Global Workspace during the given agent cycle.
        """
        if not isinstance(pain_id, str) or not pain_id:
            logger_tracker.warning("PainEventTracker.record_pain_in_gwm: Invalid pain_id provided.")
            return
        if not isinstance(cycle_count_in_gwm, int) or cycle_count_in_gwm < 0:
            logger_tracker.warning(f"PainEventTracker.record_pain_in_gwm: Invalid cycle_count_in_gwm ({cycle_count_in_gwm}) for pain_id '{pain_id}'.")
            return

        self._pain_gwm_history.append((pain_id, cycle_count_in_gwm))
        self._last_recorded_in_gwm[pain_id] = cycle_count_in_gwm
        logger_tracker.debug(f"PET_RECORD: Recorded pain_id '{pain_id}' in GWM at cycle {cycle_count_in_gwm}. History size: {len(self._pain_gwm_history)}")

    def should_suppress_rumination(self, pain_id_to_check: str, current_agent_cycle: int) -> bool:
        """
        Checks if the given pain_id should be suppressed due to rumination,
        based on its frequency in recent GWM history and its last seen time.
        This method also triggers a cleanup of inactive entries.
        """
        if not isinstance(pain_id_to_check, str) or not pain_id_to_check:
            logger_tracker.warning("PainEventTracker.should_suppress_rumination: Invalid pain_id_to_check.")
            return False
        if not isinstance(current_agent_cycle, int) or current_agent_cycle < 0:
            logger_tracker.warning(f"PainEventTracker.should_suppress_rumination: Invalid current_agent_cycle ({current_agent_cycle}) for pain_id '{pain_id_to_check}'.")
            return False 

        logger_tracker.info( # Changed to INFO for the test
            f"PET_SHOULD_SUPPRESS_ENTRY: Checking pain_id='{pain_id_to_check}', current_agent_cycle={current_agent_cycle}. "
            f"RuminationThreshold={self.rumination_threshold}. HistoryBeforeCleanup={list(self._pain_gwm_history)}"
        )

        # --- ADD THIS LOG LINE ---
        logger_tracker.info(f"PET_SHOULD_SUPPRESS: Calling _cleanup_inactive_entries for cycle {current_agent_cycle}.")
        self._cleanup_inactive_entries(current_agent_cycle)
        # --- ADD THIS LOG LINE ---
        logger_tracker.info(f"PET_SHOULD_SUPPRESS: _cleanup_inactive_entries finished for cycle {current_agent_cycle}.")
        
        logger_tracker.info( # Also make this one INFO for the test if the others are
            f"PET_SHOULD_SUPPRESS_POST_CLEANUP: HistoryAfterCleanup={list(self._pain_gwm_history)}"
        )
        
        recent_occurrence_count = sum(1 for pid, _ in self._pain_gwm_history if pid == pain_id_to_check)
        
        is_ruminating = recent_occurrence_count >= self.rumination_threshold
        
        logger_tracker.info(
            f"PET_SHOULD_SUPPRESS_DECISION: For pain_id='{pain_id_to_check}', current_cycle={current_agent_cycle}: "
            f"HistoryContents={list(self._pain_gwm_history)}, "
            f"RecentOccurrenceCount={recent_occurrence_count}, RuminationThreshold={self.rumination_threshold}, "
            f"IsRuminatingDecision={is_ruminating}"
        )
            
        return is_ruminating

    def _cleanup_inactive_entries(self, current_agent_cycle: int):
        """
        Removes entries from _pain_gwm_history and _last_recorded_in_gwm
        if they haven't been recorded in GWM for more than 'inactive_reset_cycles'.
        """
        if self.inactive_reset_cycles <= 0: # Should not happen if __init__ validates
            return

        cutoff_cycle_for_inactivity = current_agent_cycle - self.inactive_reset_cycles
        
        # Rebuild _pain_gwm_history: only keep entries whose recorded cycle_count_in_gwm
        # is greater than the cutoff_cycle_for_inactivity.
        # This means the event was seen *more recently* than the inactivity threshold allows.
        initial_history_len = len(self._pain_gwm_history)
        active_history_entries = [
            (pid, cycle_in_gwm) for pid, cycle_in_gwm in self._pain_gwm_history 
            if cycle_in_gwm > cutoff_cycle_for_inactivity
        ]
        
        pruned_count_history = initial_history_len - len(active_history_entries)
        if pruned_count_history > 0:
            # --- CHANGE TO INFO ---
            logger_tracker.info( 
                f"PET_CLEANUP: Pruned {pruned_count_history} inactive entries from _pain_gwm_history "
                f"for pain_ids not seen after cycle {cutoff_cycle_for_inactivity} "
                f"(current agent cycle for check: {current_agent_cycle})."
            )
            self._pain_gwm_history.clear()
            self._pain_gwm_history.extend(active_history_entries)

        # Clean up _last_recorded_in_gwm similarly
        initial_last_seen_len = len(self._last_recorded_in_gwm)
        active_last_seen_entries = {
            pid: last_cycle for pid, last_cycle in self._last_recorded_in_gwm.items()
            if last_cycle > cutoff_cycle_for_inactivity
        }
        pruned_count_last_seen = initial_last_seen_len - len(active_last_seen_entries)
        if pruned_count_last_seen > 0:
             # --- CHANGE TO INFO ---
             logger_tracker.info( 
                f"PET_CLEANUP: Pruned {pruned_count_last_seen} inactive entries from _last_recorded_in_gwm "
                f"for pain_ids not seen after cycle {cutoff_cycle_for_inactivity}."
            )
        self._last_recorded_in_gwm = active_last_seen_entries

    def get_status_summary(self) -> Dict[str, Any]:
        """Returns a summary of the tracker's state for debugging or status."""
        return {
            "pain_rumination_threshold": self.rumination_threshold,
            "max_history_deque_len": self.max_history,
            "inactive_reset_cycles": self.inactive_reset_cycles,
            "current_pain_gwm_history_size": len(self._pain_gwm_history),
            "current_last_recorded_in_gwm_size": len(self._last_recorded_in_gwm),
            # Optionally, list a few recent items for detailed debugging
            # "recent_history_sample": list(self._pain_gwm_history)[-5:],
            # "last_recorded_sample": dict(list(self._last_recorded_in_gwm.items())[:5])
        }