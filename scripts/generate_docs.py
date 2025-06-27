# --- START OF FILE scripts/generate_docs.py ---

"""Generate API documentation for OSCAR-C using pdoc."""

import subprocess
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Define the modules/packages to document relative to the project root
# Adjust these if your package structure differs
MODULES_TO_DOC = [
    "models",       # Includes enums.py, datatypes.py
    "protocols",
    "cognitive_modules", # Includes submodules like knowledge_base.py etc.
    "agent_controller",
    "migrations",   # Document the migration script(s)
    "external_comms",
    "agent_state",
    # Add other top-level modules/files if needed
    # "agent_config", # Usually not needed unless it contains complex classes/funcs
]

# Output directory for documentation (relative to project root)
DOCS_OUTPUT_DIR = Path("docs/api")

# Project root directory (assuming this script is in project_root/scripts/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def run_command(cmd_list: list, working_dir: Path):
    """Runs a command using subprocess."""
    try:
        logger.info(f"Running command: {' '.join(cmd_list)}")
        # Set PYTHONPATH to ensure imports work correctly relative to project root
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + env.get('PYTHONPATH', '')

        result = subprocess.run(
            cmd_list,
            check=True,
            capture_output=True,
            text=True,
            cwd=working_dir, # Run from project root
            env=env
        )
        logger.debug(f"Command stdout:\n{result.stdout}")
        if result.stderr:
             logger.warning(f"Command stderr:\n{result.stderr}")
        return True
    except FileNotFoundError:
        logger.error(f"Error: Command '{cmd_list[0]}' not found. Is pdoc installed and in PATH?")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}: {' '.join(cmd_list)}")
        logger.error(f"Stderr:\n{e.stderr}")
        logger.error(f"Stdout:\n{e.stdout}")
        return False
    except Exception as e:
        logger.exception(f"An unexpected error occurred while running command: {e}")
        return False

def generate_api_docs():
    """Generates API documentation using pdoc."""
    logger.info("--- Starting API Documentation Generation ---")

    # Ensure output directory exists
    output_dir_abs = PROJECT_ROOT / DOCS_OUTPUT_DIR
    try:
        output_dir_abs.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured documentation output directory exists: {output_dir_abs}")
    except OSError as e:
        logger.error(f"Failed to create documentation directory {output_dir_abs}: {e}")
        return False

    overall_success = True
    pdoc_executable = sys.executable # Use the same python interpreter that runs the script
    base_cmd = [pdoc_executable, "-m", "pdoc"]

    # Generate docs for each specified module/package
    for module_name in MODULES_TO_DOC:
        logger.info(f"Generating docs for module/package: {module_name}")
        # Output directly into the main API docs dir, pdoc handles structure
        module_output_dir = output_dir_abs

        cmd = base_cmd + [
            "--html",       # Generate HTML output
            "--force",      # Overwrite existing files
            "--output-dir", str(module_output_dir),
            module_name     # The module/package name to document
        ]

        if not run_command(cmd, PROJECT_ROOT):
            overall_success = False
            logger.error(f"Failed to generate docs for {module_name}")
            # Optionally stop on first error, or continue to try others
            # break

    # --- Generate Main Index (Optional but helpful) ---
    # Creates a simple landing page linking to the generated modules
    index_path = output_dir_abs / "index.html"
    logger.info(f"Generating main index file at {index_path}...")
    try:
        with open(index_path, "w") as f:
             f.write("<!DOCTYPE html><html><head><title>OSCAR-C API Documentation</title>")
             # Add some basic styling
             f.write("<style>body { font-family: sans-serif; margin: 2em; } "
                     "ul { list-style: none; padding-left: 0; } "
                     "li { margin-bottom: 0.5em; } "
                     "a { text-decoration: none; color: #005fcc; } "
                     "a:hover { text-decoration: underline; } </style>")
             f.write("</head><body>")
             f.write("<h1>OSCAR-C API Documentation</h1>")
             f.write("<p>Generated documentation for core modules:</p>")
             f.write("<ul>")
             # Create links assuming pdoc creates subdirs named after modules
             for module_name in MODULES_TO_DOC:
                  # Handle nested modules if necessary (e.g., cognitive_modules.knowledge_base)
                  link_path = f"{module_name.replace('.', '/')}/index.html"
                  # Check if it's a top-level file like protocols.py -> protocols.html
                  module_file_path = PROJECT_ROOT / f"{module_name.replace('.', '/')}.py"
                  if module_file_path.is_file():
                       link_path = f"{module_name}.html" # pdoc usually creates module.html for single files

                  # A more robust way might check actual output dir structure, but this is simpler
                  f.write(f'<li><a href="{link_path}">{module_name}</a></li>')

             f.write("</ul>")
             f.write("</body></html>")
        logger.info("Main index file generated.")
    except IOError as e:
        logger.error(f"Failed to write main index file {index_path}: {e}")
        overall_success = False


    if overall_success:
        logger.info("--- API Documentation Generation Finished Successfully ---")
        return True
    else:
        logger.error("--- API Documentation Generation Finished with Errors ---")
        return False

if __name__ == "__main__":
    # Ensure pdoc is available
    try:
        import pdoc
        logger.debug(f"pdoc found: {pdoc.__version__}")
    except ImportError:
        logger.error("Error: 'pdoc' library not found.")
        logger.error("Please install it using: pip install pdoc")
        sys.exit(1)

    if not generate_api_docs():
        sys.exit(1) # Exit with error code if generation failed
    sys.exit(0) # Exit successfully


# --- END OF FILE scripts/generate_docs.py ---