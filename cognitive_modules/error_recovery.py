# --- START OF CORRECTED error_recovery.py ---

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Deque, Callable, Counter # Added Callable, Counter
from collections import deque, Counter # Added Counter

# --- Use standard relative imports ---
try:
    from ..protocols import CognitiveComponent
    # Import RecoveryMode enum if defined and needed
    from ..models.enums import RecoveryMode
except ImportError:
    # Fallback for different execution context (e.g., combined script)
    logging.warning("ErrorRecoverySystem: Relative imports failed, relying on globally defined types.")
    if 'CognitiveComponent' not in globals(): raise ImportError("CognitiveComponent not found via relative import or globally")
    if 'RecoveryMode' not in globals(): logging.warning("RecoveryMode enum not found globally.") # Allow it to continue for now
    CognitiveComponent = globals().get('CognitiveComponent')
    RecoveryMode = globals().get('RecoveryMode') # Might be None

logger_error_recovery = logging.getLogger(__name__) # Use standard module logger name

# Default config values
DEFAULT_MAX_ERROR_HISTORY = 50
DEFAULT_ERROR_FREQUENCY_WINDOW = 10 # Check frequency over last N errors
DEFAULT_ERROR_FREQUENCY_THRESHOLD = 3 # Trigger stronger recovery if same error type occurs N times in window

# --- Inherit correctly from CognitiveComponent ---
class ErrorRecoverySystem(CognitiveComponent):
    """
    Handles system errors detected during the cognitive cycle or action execution.
    Determines and suggests appropriate recovery actions.
    """

    def __init__(self):
        self._controller: Optional[Any] = None
        self._config: Dict[str, Any] = {}
        # Configuration loaded during initialization
        self.max_history: int = DEFAULT_MAX_ERROR_HISTORY
        self.freq_window: int = DEFAULT_ERROR_FREQUENCY_WINDOW
        self.freq_threshold: int = DEFAULT_ERROR_FREQUENCY_THRESHOLD
        # Track recent errors for frequency analysis
        self.error_history: Deque[Dict[str, Any]] = deque(maxlen=self.max_history)
        # Mapping of error types to potential recovery strategies (can be expanded)
        # Using lambda functions here for simplicity, can be refs to methods too
        self.recovery_strategies: Dict[type, Callable] = {} # Initialize empty, fill in init


    async def initialize(self, config: Dict[str, Any], controller: Any) -> bool:
        """Initialize error recovery system with configuration."""
        self._controller = controller
        err_config = config.get("error_recovery", {})
        self._config = err_config

        self.max_history = err_config.get("max_error_history", DEFAULT_MAX_ERROR_HISTORY)
        self.freq_window = err_config.get("frequency_window", DEFAULT_ERROR_FREQUENCY_WINDOW)
        self.freq_threshold = err_config.get("frequency_threshold", DEFAULT_ERROR_FREQUENCY_THRESHOLD)

        # Recreate deque with configured size
        self.error_history = deque(maxlen=self.max_history)

        # Populate strategies *after* RecoveryMode is potentially defined globally or imported
        _RecoveryMode = globals().get('RecoveryMode')
        if _RecoveryMode:
             self.recovery_strategies = {
                 # Standard Python errors
                 MemoryError: lambda e, ctx: _RecoveryMode.MEDIUM_RESET,
                 TimeoutError: lambda e, ctx: _RecoveryMode.SOFT_RESET,
                 asyncio.CancelledError: lambda e, ctx: None,
                 ConnectionError: lambda e, ctx: _RecoveryMode.SOFT_RESET,
                 TypeError: lambda e, ctx: _RecoveryMode.SOFT_RESET,
                 ValueError: lambda e, ctx: _RecoveryMode.SOFT_RESET,
                 AttributeError: lambda e, ctx: _RecoveryMode.SOFT_RESET,
                 KeyError: lambda e, ctx: _RecoveryMode.SOFT_RESET,
                 ImportError: lambda e, ctx: _RecoveryMode.HARD_RESET,
                 # Fallback for generic exceptions
                 Exception: self._suggest_generic_recovery,
             }
        else:
             logger_error_recovery.error("RecoveryMode enum not defined/imported. Cannot initialize recovery strategies.")
             self.recovery_strategies = {Exception: lambda e, ctx: None} # Minimal fallback


        logger_error_recovery.info(f"ErrorRecoverySystem initialized. History: {self.max_history}, "
                    f"Freq Window: {self.freq_window}, Freq Threshold: {self.freq_threshold}")
        return True

    async def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Optional['RecoveryMode']: # Use quotes for hint
        """
        Analyzes an error and suggests an appropriate recovery mode.
        Logs the error and context. Returns a RecoveryMode enum member or None if no action suggested.
        """
        _RecoveryMode = globals().get('RecoveryMode') # Get ref to enum
        if context is None: context = {}
        current_time = time.time()
        error_type_name = type(error).__name__
        error_msg = str(error)

        logger_error_recovery.error(f"--- Error Handled by ErrorRecoverySystem ---")
        logger_error_recovery.error(f"  Timestamp: {current_time}")
        logger_error_recovery.error(f"  Type: {error_type_name}")
        logger_error_recovery.error(f"  Message: {error_msg}")
        logger_error_recovery.error(f"  Context: {str(context)[:500]}...") # Limit context logging size
        logger_error_recovery.error(f"------------------------------------------")

        # Log concise error to UI if possible
        if self._controller and hasattr(self._controller, '_log_to_ui'):
             log_msg = f"ERR ({error_type_name}): {str(error_msg)[:80]}..."
             try: self._controller._log_to_ui("error", log_msg)
             except Exception as ui_err: logger_error_recovery.warning(f"Failed to log error to UI: {ui_err}")

        # Store error record BEFORE suggesting recovery
        error_record = { "timestamp": current_time, "error_type": error_type_name,
                         "error_msg": error_msg, "context": context, "suggested_recovery": None }
        self.error_history.append(error_record)

        # --- Select recovery strategy ---
        recovery_suggester = self._get_recovery_suggester(error)
        suggested_mode = None
        try:
            if asyncio.iscoroutinefunction(recovery_suggester): suggested_mode = await recovery_suggester(error, context)
            else: suggested_mode = recovery_suggester(error, context)
        except Exception as suggester_err:
            logger_error_recovery.error(f"Error executing recovery suggester for {error_type_name}: {suggester_err}")

        # Validate and log the suggested mode
        recovery_name = "None"
        final_mode = None # This will hold the actual enum member or None
        if _RecoveryMode and isinstance(suggested_mode, _RecoveryMode):
             recovery_name = suggested_mode.name
             final_mode = suggested_mode
        elif suggested_mode is not None: # Handle case where mode is returned but not the enum
             recovery_name = str(suggested_mode) # Log what was returned
             logger_error_recovery.warning(f"Recovery suggester returned non-enum type: {type(suggested_mode)}")

        error_record["suggested_recovery"] = recovery_name # Log the name/string representation
        logger_error_recovery.info(f"Error handled. Suggested recovery mode: {recovery_name}")

        # Return the actual enum member (or None)
        return final_mode

    def _get_recovery_suggester(self, error: Exception) -> Callable:
        """Find the most specific recovery suggestion function for the error type."""
        for error_type, suggester in self.recovery_strategies.items():
            if isinstance(error, error_type):
                logger_error_recovery.debug(f"Found recovery suggester for {type(error).__name__} via {error_type.__name__}")
                return suggester
        logger_error_recovery.warning(f"No specific suggester found for {type(error).__name__}, using generic.")
        return getattr(self, '_suggest_generic_recovery', lambda e, ctx: None)


    # --- Specific Recovery Suggestion Functions ---

    # Renamed to avoid clash with async version, keep it sync
    def _suggest_generic_recovery(self, error: Exception, context: Dict[str, Any]) -> Optional[Any]: # Return optional RecoveryMode
        """Generic fallback recovery suggestion, checks frequency."""
        _RecoveryMode = globals().get('RecoveryMode')
        if not _RecoveryMode: return None

        error_type_name = type(error).__name__
        logger_error_recovery.warning(f"Handling generic Exception: {error_type_name}")
        error_frequency = self._check_error_frequency(error_type_name)
        logger_error_recovery.debug(f"Frequency of '{error_type_name}' in last {self.freq_window} errors: {error_frequency}")

        if error_frequency >= self.freq_threshold:
            logger_error_recovery.error(f"Frequent occurrence ({error_frequency}+) of '{error_type_name}' detected! Escalating recovery.")
            if error_frequency >= self.freq_threshold * 2:
                 return _RecoveryMode.HARD_RESET
            else:
                 return _RecoveryMode.MEDIUM_RESET
        else:
            return _RecoveryMode.SOFT_RESET


    def _check_error_frequency(self, error_type_name: str) -> int:
        """Check frequency of a specific error type in recent history."""
        if not self.error_history: return 0
        window_size = min(len(self.error_history), self.freq_window)
        if window_size == 0: return 0
        try:
            history_list = list(self.error_history)
            relevant_history = history_list[-window_size:]
            recent_error_types = [record.get("error_type", "Unknown") for record in relevant_history if isinstance(record, dict)]
        except Exception as e:
            logger_error_recovery.error(f"Error processing error history for frequency check: {e}")
            return 0
        counts = Counter(recent_error_types)
        return counts.get(error_type_name, 0)


    # --- CognitiveComponent Implementation ---

    async def process(self, input_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """ Error Recovery is typically invoked via handle_error(), not the standard process(). """
        return None

    async def reset(self) -> None:
        """Reset error history."""
        self.error_history.clear()
        logger_error_recovery.info("ErrorRecoverySystem reset (error history cleared).")

    async def get_status(self) -> Dict[str, Any]:
        """Return status, including recent error summary."""
        last_error = self.error_history[-1] if self.error_history else None
        error_count = len(self.error_history)
        recent_types = Counter(rec.get("error_type", "Unknown") for rec in self.error_history if isinstance(rec, dict))

        return {
            "component": "ErrorRecoverySystem",
            "status": "operational",
            "total_errors_in_history": error_count,
            "history_capacity": self.max_history,
            "last_error_type": last_error.get("error_type") if isinstance(last_error, dict) else None,
            "last_error_time": last_error.get("timestamp") if isinstance(last_error, dict) else None,
            "recent_error_type_counts": dict(recent_types.most_common(5)) # Top 5 error types
        }

    async def shutdown(self) -> None:
        """Perform any necessary cleanup."""
        logger_error_recovery.info("ErrorRecoverySystem shutting down.")

# --- END OF CORRECTED error_recovery.py ---