# --- START OF FILE agent_config.py ---

import os

# --- Constants & Configuration ---
AGENT_CYCLE_INTERVAL_S = 0.1
OLLAMA_CALL_TIMEOUT_S = 180.0 # Keep increased timeout
REFLECTION_FREQUENCY = 5
MOTIVATION_EVALUATION_CYCLES = 10
GOAL_EVALUATION_CYCLES = 25
DECIDE_ACT_TEMPERATURE = 0.4 # Keep slightly lower temperature

_EXTENSION_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH = os.path.join(os.path.dirname(_EXTENSION_BASE_DIR), "agent_memory")
# --- Revert Prompt Size Optimizations ---
MAX_RECENT_TURNS_IN_PROMPT = 4 # Keep slightly reduced
MEMORY_RETRIEVAL_COUNT = 3
MAX_FILES_LISTED = 20
MAX_FILE_READ_CHARS = 3500

# --- AVAILABLE ACTIONS: Reverted to include placeholders ---
AVAILABLE_ACTIONS = """
Available Actions:
- [ACTION: THINKING content="<reasoning>"] - Output reasoning/thought process (Default if no other action).
- [ACTION: PATH_STRATEGY strategy="current_dir"] - Get the current working directory path.
- [ACTION: LIST_FILES path="<path>"] - List directory contents. Allows relative paths.
- [ACTION: READ_FILE path="<path>"] - Read a text file's content (up to 3k chars). Allows relative paths.
- [ACTION: QUERY_MEMORY query="<search>"] - Search internal memory for reflections, past events, etc.
- [ACTION: SET_GOAL goal="<desc>"] - Set a new objective for yourself.
- [ACTION: EXPLORE path="<path>" depth="<1 or 2>"] - Explore directory structure starting at path (defaults to current dir, depth 1). Allows relative paths.
"""
# NOTE: RESET_STRATEGY is NOT listed, handled via instructions only

# Add any other constants if they appear later and are configuration-like
# --- END OF FILE agent_config.py ---