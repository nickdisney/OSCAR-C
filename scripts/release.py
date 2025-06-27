# --- START OF FILE scripts/release.py ---

"""Release automation script for OSCAR-C."""

import subprocess
import sys
import json
import toml
import os
from pathlib import Path
from datetime import datetime
import logging
import argparse # For command-line arguments

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Files to update with the new version string
# Path relative to project root
VERSION_FILES = {
    "VERSION.toml": "toml", # Root version file
    "web/package.json": "json", # Example: Dashboard package file
    # Add other files if needed, e.g., pyproject.toml
    # "pyproject.toml": "toml_pyproject",
}

CHANGELOG_FILE = "CHANGELOG.md"
GIT_REMOTE_NAME = "origin" # Default remote name
MAIN_BRANCH_NAME = "main" # Or "master", depending on your convention

# Project root directory (assuming this script is in project_root/scripts/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def run_command(cmd_list: list, check=True, capture_output=False, working_dir=PROJECT_ROOT):
    """Run shell command safely."""
    cmd_str = ' '.join(cmd_list)
    logger.info(f"Running command: {cmd_str} (in {working_dir})")
    try:
        result = subprocess.run(
            cmd_list,
            check=check,
            capture_output=capture_output,
            text=True,
            cwd=working_dir,
            env=os.environ.copy() # Pass environment
        )
        stdout = result.stdout.strip() if result.stdout else ""
        stderr = result.stderr.strip() if result.stderr else ""
        if stdout: logger.debug(f"Stdout:\n{stdout}")
        if stderr: logger.warning(f"Stderr:\n{stderr}")

        if capture_output:
            return stdout
        return True
    except FileNotFoundError:
        logger.error(f"Error: Command '{cmd_list[0]}' not found. Is git installed and in PATH?")
        return None if capture_output else False
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}: {cmd_str}")
        logger.error(f"Stderr:\n{e.stderr}")
        logger.error(f"Stdout:\n{e.stdout}")
        return None if capture_output else False
    except Exception as e:
        logger.exception(f"An unexpected error occurred while running command: {e}")
        return None if capture_output else False

def ensure_clean_repo():
    """Ensure repository is clean before release."""
    logger.info("Checking repository status...")
    status = run_command(["git", "status", "--porcelain"], capture_output=True)
    if status is None: # Error running command
        return False
    if status:
        logger.error("❌ Repository has uncommitted changes or untracked files.")
        logger.error("Please commit or stash changes before creating a release.")
        logger.error(f"git status output:\n{status}")
        return False
    logger.info("Repository is clean.")
    return True

def ensure_main_branch():
    """Ensure the current branch is the main release branch."""
    logger.info(f"Checking current branch (should be '{MAIN_BRANCH_NAME}')...")
    current_branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True)
    if current_branch is None:
         return False # Error running git command
    if current_branch != MAIN_BRANCH_NAME:
        logger.error(f"❌ You must be on the '{MAIN_BRANCH_NAME}' branch to create a release. Current branch: '{current_branch}'.")
        return False
    logger.info(f"Currently on branch '{current_branch}'.")
    return True

def get_latest_tag() -> Optional[str]:
    """Gets the latest git tag using describe."""
    # Fetch tags first to ensure we have the latest from remote
    run_command(["git", "fetch", "--tags", GIT_REMOTE_NAME])
    tag = run_command(["git", "describe", "--tags", "--abbrev=0"], capture_output=True, check=False) # check=False allows no tags yet
    if tag is None or "fatal:" in tag.lower(): # Handle errors or no tags found
        logger.warning("Could not find previous tags. Changelog will include all history.")
        return None
    logger.info(f"Latest tag found: {tag}")
    return tag

def update_version_files(version: str):
    """Update version in configured files."""
    logger.info(f"Updating version to '{version}' in configured files...")
    success = True
    current_time_iso = datetime.now().isoformat()

    for file_rel_path, file_type in VERSION_FILES.items():
        file_path = PROJECT_ROOT / file_rel_path
        if not file_path.exists():
            logger.warning(f"Version file not found: {file_path}. Skipping.")
            continue

        try:
            logger.debug(f"Updating {file_path} ({file_type})...")
            content = file_path.read_text()

            if file_type == "toml":
                data = toml.loads(content)
                data['version'] = version
                # Add release date if key exists (like in VERSION.toml)
                if 'release_date' in data: data['release_date'] = current_time_iso
                new_content = toml.dumps(data)
            elif file_type == "json":
                data = json.loads(content)
                data['version'] = version
                # Add release date if key exists (e.g., custom field)
                # if 'releaseDate' in data: data['releaseDate'] = current_time_iso
                new_content = json.dumps(data, indent=2) + "\n" # Ensure newline at end
            # Add handlers for other types like pyproject.toml if needed
            # elif file_type == "toml_pyproject":
            #     data = toml.loads(content)
            #     # Update under [tool.poetry] or similar section
            #     if 'tool' in data and 'poetry' in data['tool'] and 'version' in data['tool']['poetry']:
            #         data['tool']['poetry']['version'] = version
            #     else: logger.warning(f"Could not find version key in pyproject.toml structure.")
            #     new_content = toml.dumps(data)
            else:
                logger.warning(f"Unsupported file type '{file_type}' for version update: {file_path}")
                continue

            file_path.write_text(new_content)
            logger.info(f"Updated version in {file_path}")

        except Exception as e:
            logger.exception(f"❌ Failed to update version in {file_path}: {e}")
            success = False

    return success


def generate_changelog_entry(version: str, previous_tag: Optional[str]) -> Optional[str]:
    """Generate CHANGELOG entry content."""
    logger.info("Generating changelog entry...")
    try:
        # Get commits since last tag or from beginning if no tag
        commit_range = f"{previous_tag}..HEAD" if previous_tag else "HEAD"
        # Use a format that includes commit hash and subject
        log_format = "%h %s" # Short hash, subject
        commits_str = run_command(
            ["git", "log", commit_range, f"--pretty=format:{log_format}", "--no-merges"],
            capture_output=True, check=False # Allow empty history
        )

        if commits_str is None: # Error running git log
            return None

        commits = commits_str.strip().split('\n') if commits_str.strip() else ["No significant changes."]

        # Prepare changelog entry using Markdown
        entry_lines = []
        entry_lines.append(f"## [v{version}] - {datetime.now().strftime('%Y-%m-%d')}\n")
        # Categorize commits based on keywords (optional, simple example)
        features = [f"- {c}" for c in commits if c.lower().startswith("feat")]
        fixes = [f"- {c}" for c in commits if c.lower().startswith("fix")]
        others = [f"- {c}" for c in commits if not c.lower().startswith(("feat", "fix")) and c != "No significant changes."]

        if features: entry_lines.append("### Features"); entry_lines.extend(features)
        if fixes: entry_lines.append("### Bug Fixes"); entry_lines.extend(fixes)
        if others: entry_lines.append("### Other Changes"); entry_lines.extend(others)
        if not features and not fixes and not others and commits != ["No significant changes."]:
            # Fallback if no categorization works
            entry_lines.extend([f"- {c}" for c in commits])

        logger.info("Changelog entry generated.")
        return "\n".join(entry_lines) + "\n" # Add extra newline at end

    except Exception as e:
        logger.exception(f"❌ Failed to generate changelog entry: {e}")
        return None


def update_changelog_file(entry: str):
    """Prepend the new entry to the CHANGELOG file."""
    changelog_path = PROJECT_ROOT / CHANGELOG_FILE
    logger.info(f"Updating {changelog_path}...")
    try:
        existing_content = changelog_path.read_text() if changelog_path.exists() else ""
        # Ensure there's a newline between the new entry and old content if needed
        separator = "\n" if existing_content and not entry.endswith("\n\n") else ""
        new_content = entry + separator + existing_content
        changelog_path.write_text(new_content)
        logger.info("CHANGELOG.md updated successfully.")
        return True
    except Exception as e:
        logger.exception(f"❌ Failed to update {changelog_path}: {e}")
        return False

def create_release(version: str, push: bool = False):
    """Creates and tags a new release."""
    logger.info(f"--- Starting Release Process for v{version} ---")

    # --- Pre-checks ---
    if not ensure_main_branch(): return False
    if not ensure_clean_repo(): return False

    # --- Version Update ---
    if not update_version_files(version): return False

    # --- Changelog ---
    latest_tag = get_latest_tag()
    new_entry = generate_changelog_entry(version, latest_tag)
    if not new_entry: return False
    if not update_changelog_file(new_entry): return False

    # --- Documentation (Optional - depends on your workflow) ---
    # logger.info("Generating documentation...")
    # if not run_command([sys.executable, str(PROJECT_ROOT / "scripts/generate_docs.py")]):
    #     logger.warning("Documentation generation script failed or was skipped.")
        # Decide if this should block the release

    # --- Git Commit ---
    logger.info("Committing release changes...")
    if not run_command(["git", "add", "."]): return False
    commit_message = f"chore(release): Release v{version}"
    if not run_command(["git", "commit", "-m", commit_message]): return False

    # --- Git Tag ---
    logger.info(f"Tagging release v{version}...")
    tag_message = f"Release v{version}"
    # Use -s for a signed tag if GPG is configured
    # if not run_command(["git", "tag", "-s", f"v{version}", "-m", tag_message]): return False
    if not run_command(["git", "tag", "-a", f"v{version}", "-m", tag_message]): return False


    # --- Git Push (Optional) ---
    if push:
        logger.info(f"Pushing commit and tag to remote '{GIT_REMOTE_NAME}'...")
        # Push the branch first
        if not run_command(["git", "push", GIT_REMOTE_NAME, MAIN_BRANCH_NAME]):
            logger.error(f"Failed to push branch '{MAIN_BRANCH_NAME}'. Tag was not pushed.")
            return False
        # Push the tag
        if not run_command(["git", "push", GIT_REMOTE_NAME, f"v{version}"]):
            logger.error("Failed to push tag.")
            # Technically commit is pushed, but tag failed. Consider overall failure?
            return False
        logger.info("Commit and tag pushed successfully.")
    else:
        logger.warning("Release commit and tag created locally. Remember to push manually:")
        logger.warning(f"  git push {GIT_REMOTE_NAME} {MAIN_BRANCH_NAME}")
        logger.warning(f"  git push {GIT_REMOTE_NAME} v{version}")


    logger.info(f"✅ Release v{version} process completed successfully" + (" (locally)." if not push else "."))
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate OSCAR-C release process.")
    parser.add_argument("version", help="The new version string (e.g., 2.1.0, 2.0.1-beta.1).")
    parser.add_argument("--push", action="store_true", help="Push the commit and tag to the remote repository.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG) # Set root logger level

    if not create_release(args.version, args.push):
        logger.error("Release process failed.")
        sys.exit(1)
    sys.exit(0)

# --- END OF FILE scripts/release.py ---