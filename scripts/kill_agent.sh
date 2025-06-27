#!/bin/bash
# --- START OF FILE scripts/kill_agent.sh ---

# Emergency kill-switch script for OSCAR-C agent

# Define the path to the PID file (should match agent_controller.py)
PID_FILE="/tmp/oscar_c.pid"
AGENT_NAME="OSCAR-C Agent" # For messages

echo "--- ${AGENT_NAME} Emergency Kill Switch ---"

# Check if the PID file exists
if [ -f "$PID_FILE" ]; then
    # Read the PID from the file
    PID=$(cat "$PID_FILE")

    # Validate if PID is a number and seems valid
    if ! [[ "$PID" =~ ^[0-9]+$ ]]; then
        echo "❌ Error: PID file '$PID_FILE' contains invalid content: '$PID'"
        # Optionally remove the bad PID file
        # rm -f "$PID_FILE"
        exit 1
    fi

    echo "Found PID file for ${AGENT_NAME}. Target PID: $PID"

    # Check if the process with that PID actually exists
    if ps -p "$PID" > /dev/null; then
        echo "Process $PID found. Attempting graceful shutdown (SIGINT)..."
        # Send SIGINT (Ctrl+C equivalent) for graceful shutdown
        kill -SIGINT "$PID"

        # Wait for graceful shutdown (adjust timeout as needed)
        TIMEOUT=5 # seconds
        echo -n "Waiting up to ${TIMEOUT}s for graceful shutdown..."
        for (( i=0; i<TIMEOUT; i++ )); do
            if ! ps -p "$PID" > /dev/null; then
                echo "" # Newline after dots
                echo "✅ Process $PID shut down gracefully."
                # Remove the PID file since the process stopped
                rm -f "$PID_FILE"
                echo "Removed PID file '$PID_FILE'."
                exit 0 # Success
            fi
            echo -n "." # Progress indicator
            sleep 1
        done
        echo "" # Newline after timeout dots

        # If still running after timeout, force kill
        if ps -p "$PID" > /dev/null; then
            echo "⚠️ Process $PID did not shut down gracefully after ${TIMEOUT}s."
            echo "Attempting forceful termination (SIGKILL)..."
            kill -SIGKILL "$PID" # Use SIGKILL (-9) to force termination
            sleep 1 # Give OS a moment

            # Final check
            if ps -p "$PID" > /dev/null; then
                echo "❌ Error: Failed to terminate process $PID even with SIGKILL."
                # PID file is *not* removed here as process might still be stuck
                exit 1
            else
                echo "✅ Process $PID forcefully terminated."
                # Remove the PID file
                rm -f "$PID_FILE"
                echo "Removed PID file '$PID_FILE'."
                exit 0 # Success (though forceful)
            fi
        fi
    else
        echo "⚠️ Warning: Process with PID $PID not found, but PID file exists."
        echo "Removing stale PID file '$PID_FILE'."
        rm -f "$PID_FILE"
        exit 0 # Consider this a success as the target process isn't running
    fi
else
    echo "No agent running (PID file '$PID_FILE' not found)."
    # Optional: Search for processes by name as a fallback?
    # pgrep -f agent_controller.py
    exit 0 # Success (nothing to kill)
fi

# --- END OF FILE scripts/kill_agent.sh ---