# --- START OF FILE scripts/kill_agent.py ---

import os
import sys
import time
import signal
import logging
from pathlib import Path

try:
    import psutil
except ImportError:
    print("Error: 'psutil' library not found. Please install it using: pip install psutil")
    sys.exit(1)

# --- Configuration ---
PID_FILE_PATH = Path("/tmp/oscar_c.pid") # Match agent_controller.py
AGENT_NAME = "OSCAR-C Agent"
WAIT_TIMEOUT_S = 5 # Seconds to wait for graceful shutdown
LOG_LEVEL = logging.INFO # Set to logging.DEBUG for more detail

# --- Logging Setup ---
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_pid_file(pid_path: Path) -> Optional[int]:
    """Reads the PID from the specified file."""
    try:
        pid_str = pid_path.read_text().strip()
        pid = int(pid_str)
        if pid <= 0:
            logger.error(f"PID file '{pid_path}' contains non-positive value: {pid_str}")
            return None
        return pid
    except FileNotFoundError:
        logger.debug(f"PID file not found at '{pid_path}'.")
        return None
    except ValueError:
        logger.error(f"PID file '{pid_path}' contains non-integer value: {pid_str}")
        return None
    except Exception as e:
        logger.error(f"Error reading PID file '{pid_path}': {e}")
        return None

def attempt_shutdown(pid: int) -> bool:
    """Attempts graceful then forceful shutdown of the process."""
    logger.info(f"Attempting shutdown for PID: {pid}")
    try:
        proc = psutil.Process(pid)
        proc_name = proc.name() # Get process name for logging
        logger.info(f"Found process '{proc_name}' (PID: {pid}). Attempting graceful shutdown (SIGINT/Ctrl+C)...")

        # --- Graceful Shutdown Attempt ---
        # Send SIGINT (like Ctrl+C). On Windows, this might need specific handling
        # or might translate to CTRL_C_EVENT. psutil tries to abstract this.
        # os.kill(pid, signal.SIGINT) # Standard way, might not work well cross-platform
        proc.send_signal(signal.SIGINT) # psutil's cross-platform way

        try:
            # Wait for the process to terminate
            logger.info(f"Waiting up to {WAIT_TIMEOUT_S} seconds for graceful shutdown...")
            proc.wait(timeout=WAIT_TIMEOUT_S)
            logger.info(f"Process {pid} shut down gracefully.")
            return True # Process terminated within timeout
        except psutil.TimeoutExpired:
            logger.warning(f"Process {pid} did not shut down gracefully within {WAIT_TIMEOUT_S} seconds.")
            # --- Forceful Shutdown Attempt ---
            logger.warning("Attempting forceful termination (terminate/kill)...")
            try:
                proc.terminate() # Sends SIGTERM on Unix, TerminateProcess on Windows
                try:
                    proc.wait(timeout=2) # Wait briefly after terminate
                    logger.info(f"Process {pid} terminated successfully after terminate() call.")
                    return True
                except psutil.TimeoutExpired:
                    logger.warning(f"Process {pid} did not terminate after terminate(). Sending kill().")
                    proc.kill() # Sends SIGKILL on Unix, TerminateProcess on Windows (stronger)
                    try:
                        proc.wait(timeout=1)
                        logger.info(f"Process {pid} terminated successfully after kill() call.")
                        return True
                    except psutil.TimeoutExpired:
                         logger.error(f"❌ CRITICAL: Process {pid} could not be terminated even with kill().")
                         return False
            except psutil.NoSuchProcess:
                logger.info(f"Process {pid} already terminated before forceful attempt.")
                return True # Already gone
            except Exception as e_term:
                 logger.error(f"Error during forceful termination of PID {pid}: {e_term}")
                 return False # Failed to terminate

    except psutil.NoSuchProcess:
        logger.warning(f"Process with PID {pid} does not exist (already terminated or PID is stale).")
        return True # Consider it "shutdown" as it's not running
    except psutil.AccessDenied:
         logger.error(f"Permission denied trying to signal/terminate PID {pid}. Try running script with higher privileges (e.g., sudo).")
         return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while handling PID {pid}: {e}")
        return False

def main():
    """Main execution logic."""
    logger.info(f"--- {AGENT_NAME} Emergency Kill Switch (Python) ---")

    pid = read_pid_file(PID_FILE_PATH)

    if pid is None:
        if PID_FILE_PATH.exists():
             logger.warning(f"PID file '{PID_FILE_PATH}' exists but is invalid or unreadable.")
             # Decide whether to remove invalid file? Safer not to by default.
             # try: PID_FILE_PATH.unlink(); logger.info("Removed invalid PID file.")
             # except OSError as e: logger.error(f"Could not remove invalid PID file: {e}")
        else:
             logger.info("No agent seems to be running (PID file not found).")
        sys.exit(0) # Nothing to do

    logger.info(f"Found PID {pid} in '{PID_FILE_PATH}'.")

    shutdown_successful = attempt_shutdown(pid)

    if shutdown_successful:
        # Clean up PID file only if process is confirmed gone or wasn't running
        if PID_FILE_PATH.exists():
             try:
                 # Double check process is gone before deleting PID file
                 if not psutil.pid_exists(pid):
                     PID_FILE_PATH.unlink()
                     logger.info(f"Removed PID file '{PID_FILE_PATH}'.")
                 else:
                      logger.warning(f"Process {pid} might still exist despite shutdown attempt. PID file not removed.")
             except OSError as e:
                 logger.error(f"Error removing PID file '{PID_FILE_PATH}': {e}")
             except psutil.Error as e:
                  logger.error(f"Error checking PID existence before removing PID file: {e}")
        logger.info(f"✅ Shutdown sequence for PID {pid} completed.")
        sys.exit(0)
    else:
        logger.error(f"❌ Failed to shut down process {pid}.")
        sys.exit(1)


if __name__ == "__main__":
    main()

# --- END OF FILE scripts/kill_agent.py ---