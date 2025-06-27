# --- START OF FILE migrations/v1_to_v2.py ---

"""Migrate Phase 1 data model (blackboard dict) to Phase 2 KB predicates."""

import json
import time
import logging
import sys
import argparse # Add argparse import
import asyncio # Add asyncio import
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Need access to Phase 2 data structures and KB component
# Adjust import paths based on your final project structure
# Assuming the script might be run from the root or the migrations dir
# Use globals().get() for safety in combined script
_Predicate_mig = globals().get('Predicate')
_Goal_mig = globals().get('Goal')
_create_goal_from_descriptor_mig = globals().get('create_goal_from_descriptor')
_KnowledgeBase_mig = globals().get('KnowledgeBase')

# Setup logger for migration script
logger_migration = logging.getLogger(__name__ + ".migration")
# Basic config if run standalone
if __name__ == "__main__":
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class V1ToV2Migrator:
    """Handles Phase 1 blackboard dictionary to Phase 2 predicate migration logic."""

    def __init__(self, v1_backup_path: str = "data/phase1_backup.json"):
        self.backup_path = Path(v1_backup_path)
        self.migration_log = [] # Stores stats for each migration run

    def migrate_blackboard(self, blackboard: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transforms a Phase 1 blackboard dictionary into a list of Phase 2 Predicate data dictionaries
        and migration statistics. Does NOT interact with the database directly.
        """
        predicates_data = [] # Store as dicts first for easier testing/serialization
        current_time = time.time()
        migration_stats = {
            "goals_migrated": 0,
            "actions_migrated": 0,
            "paths_migrated": 0,
            "narrative_entries": 0,
            "other_items": 0,
            "predicates_generated": 0
        }

        # Check if dependent classes are available
        if not _create_goal_from_descriptor_mig:
             logger_migration.error("create_goal_from_descriptor function not found. Cannot migrate goals.")
        else:
            # --- Migrate Current Goal ---
            if blackboard.get("current_goal") and isinstance(blackboard["current_goal"], str):
                goal_desc = blackboard["current_goal"]
                # We create a goal object just to get its ID, but only store predicate data
                temp_goal = _create_goal_from_descriptor_mig(goal_desc)
                if temp_goal: # Check if goal creation succeeded
                    goal_id = temp_goal.id
                    predicates_data.append({
                        "name": "isCurrentGoal",
                        "args": (goal_id, goal_desc), # Store ID and original description
                        "value": True, "timestamp": blackboard.get("goal_set_time", current_time)
                    })
                    migration_stats["goals_migrated"] += 1
                    migration_stats["predicates_generated"] += 1
                    logger_migration.debug(f"Migrated goal: {goal_desc}")
                else:
                    logger_migration.warning(f"Could not create Goal object for description: {goal_desc}")


        # --- Migrate Path Knowledge ---
        # Ensure nested structure exists safely
        knowledge = blackboard.get("self_model", {}).get("knowledge", {})
        validated_count = 0
        invalid_count = 0
        if isinstance(knowledge, dict):
            validated = knowledge.get("validated_paths", {})
            invalid = knowledge.get("invalid_paths", [])

            if isinstance(validated, dict):
                for path, source_info in validated.items():
                    predicates_data.append({
                        "name": "isValidPath",
                        "args": (path,), # Argument is the path string
                        "value": True, "timestamp": current_time # Use current time, old timestamp lost
                    })
                    migration_stats["paths_migrated"] += 1
                    migration_stats["predicates_generated"] += 1
                    validated_count += 1
            if isinstance(invalid, list):
                 for path in invalid:
                     predicates_data.append({
                         "name": "isInvalidPath",
                         "args": (path,),
                         "value": True, "timestamp": current_time
                     })
                     migration_stats["paths_migrated"] += 1
                     migration_stats["predicates_generated"] += 1
                     invalid_count += 1
            logger_migration.debug(f"Migrated path knowledge: {validated_count} valid, {invalid_count} invalid.")


        # --- Migrate Action History ---
        recent_actions = blackboard.get("recent_actions", [])
        actions_migrated_count = 0
        preds_from_actions = 0
        if isinstance(recent_actions, list):
             # Limit history to avoid overwhelming KB? Or migrate all? Let's limit for now.
             limit = 50
             actions_to_migrate = recent_actions[-limit:]
             for action in actions_to_migrate:
                 if isinstance(action, dict):
                     action_type = action.get("type", "UNKNOWN")
                     outcome = action.get("outcome", "unknown")
                     timestamp = action.get("timestamp", current_time)
                     error = action.get("error")
                     params_str = str(action.get("params", {}))[:100] # Truncate params string

                     # Basic action execution event
                     predicates_data.append({
                         "name": "eventOccurred",
                         "args": ("actionExecution", action_type, outcome),
                         "value": True, "timestamp": timestamp
                     })
                     actions_migrated_count += 1
                     preds_from_actions += 1

                     # Add failure details if applicable
                     if outcome == "failure" and error:
                          predicates_data.append({
                             "name": "actionFailed",
                             "args": (action_type, params_str, str(error)[:100]), # Truncate error
                             "value": True, "timestamp": timestamp
                          })
                          preds_from_actions += 1
             migration_stats["actions_migrated"] = actions_migrated_count
             migration_stats["predicates_generated"] += preds_from_actions
             logger_migration.debug(f"Migrated {migration_stats['actions_migrated']} action events.")


        # --- Migrate Narrative Entries ---
        narrative = blackboard.get("narrative", [])
        narrative_count = 0
        if isinstance(narrative, list):
            limit = 20 # Limit narrative entries
            entries_to_migrate = narrative[-limit:]
            for entry in entries_to_migrate:
                if isinstance(entry, dict) and "content" in entry:
                    predicates_data.append({
                        "name": "narrativeEntryRecorded",
                        "args": (entry["content"][:150],), # Store truncated content as argument
                        "value": True,
                        "timestamp": entry.get("timestamp", current_time)
                    })
                    narrative_count += 1
                    migration_stats["predicates_generated"] += 1
            migration_stats["narrative_entries"] = narrative_count
            logger_migration.debug(f"Migrated {migration_stats['narrative_entries']} narrative entries.")


        # --- Migrate Other Potential Blackboard Items (Example) ---
        other_count = 0
        if "last_reflection" in blackboard and isinstance(blackboard["last_reflection"], dict):
             reflection_content = str(blackboard["last_reflection"].get("immediate",""))[:150] # Example field
             if reflection_content:
                  predicates_data.append({
                     "name": "reflectionPerformed",
                     "args": (reflection_content,),
                     "value": True,
                     "timestamp": blackboard["last_reflection"].get("timestamp", current_time)
                  })
                  other_count += 1
                  migration_stats["predicates_generated"] += 1
                  logger_migration.debug("Migrated last reflection.")
        migration_stats["other_items"] = other_count

        self.migration_log.append(migration_stats)
        logger_migration.info(f"Blackboard migration generated {migration_stats['predicates_generated']} predicates.")
        return {
            "predicates": predicates_data, # Return list of predicate dictionaries
            "statistics": migration_stats
        }

    def save_backup(self, data: Dict[str, Any]):
        """Saves the input data (Phase 1 blackboard) to the backup path."""
        try:
            self.backup_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.backup_path, "w") as f:
                json.dump(data, f, indent=2, default=str) # Use default=str for non-serializables
            logger_migration.info(f"Phase 1 data backed up to {self.backup_path}")
        except Exception as e:
            logger_migration.exception(f"Failed to save Phase 1 backup to {self.backup_path}: {e}")

    def verify_migration_counts(self, original: Dict[str, Any], migrated_stats: Dict[str, Any]) -> bool:
        """Performs basic count verification between original and migrated stats."""
        logger_migration.info("Verifying migration counts...")
        checks_passed = True

        # Goal check
        original_goal_count = 1 if original.get("current_goal") else 0
        if original_goal_count != migrated_stats.get("goals_migrated", -1):
            logger_migration.warning(f"Goal count mismatch: Original={original_goal_count}, Migrated={migrated_stats.get('goals_migrated')}")
            checks_passed = False

        # Action check (approximate due to limit)
        original_actions = len(original.get("recent_actions", []))
        migrated_actions = migrated_stats.get("actions_migrated", -1)
        # Limit used was 50 in migrate_blackboard
        if migrated_actions != min(original_actions, 50):
             logger_migration.warning(f"Action count mismatch (approx): Original={original_actions}, Migrated={migrated_actions}")
             # Don't fail test for this approximate check, just warn.

        # Path check (needs careful counting in original)
        v_paths = len(original.get("self_model", {}).get("knowledge", {}).get("validated_paths", {}))
        i_paths = len(original.get("self_model", {}).get("knowledge", {}).get("invalid_paths", []))
        original_paths = v_paths + i_paths
        if original_paths != migrated_stats.get("paths_migrated", -1):
            logger_migration.warning(f"Path count mismatch: Original={original_paths}, Migrated={migrated_stats.get('paths_migrated')}")
            checks_passed = False # Consider path counts important

        # Narrative check (approximate due to limit)
        original_narrative = len(original.get("narrative", []))
        migrated_narrative = migrated_stats.get("narrative_entries", -1)
        # Limit used was 20
        if migrated_narrative != min(original_narrative, 20):
             logger_migration.warning(f"Narrative count mismatch (approx): Original={original_narrative}, Migrated={migrated_narrative}")
             # Don't fail test for this approximate check, just warn.

        if checks_passed: logger_migration.info("Basic migration count verification passed.")
        else: logger_migration.error("Basic migration count verification failed.")
        return checks_passed


async def run_migration(phase1_data_path_str: str, kb_db_path_str: str, backup: bool = True):
    """
    Loads Phase 1 data, migrates it, verifies, and populates Phase 2 Knowledge Base.
    """
    logger_migration.info("--- Starting Phase 1 to Phase 2 Data Migration ---")

    # --- Dependency Check ---
    if not _Predicate_mig or not _KnowledgeBase_mig:
         logger_migration.critical("Predicate or KnowledgeBase class not found. Cannot run migration.")
         return False

    phase1_data_path = Path(phase1_data_path_str)

    # --- Load Phase 1 Data ---
    if not phase1_data_path.exists():
        logger_migration.error(f"Phase 1 data file not found at '{phase1_data_path_str}'. Aborting.")
        return False
    logger_migration.info(f"Loading Phase 1 data from {phase1_data_path}...")
    try:
        with open(phase1_data_path, 'r') as f:
            blackboard = json.load(f)
        logger_migration.info("Phase 1 data loaded.")
    except Exception as e:
        logger_migration.exception(f"Failed to load Phase 1 data: {e}. Aborting.")
        return False

    # --- Initialize Migrator and Backup ---
    migrator = V1ToV2Migrator()
    if backup:
        migrator.save_backup(blackboard)

    # --- Perform Migration Logic ---
    logger_migration.info("Migrating blackboard data to predicate format...")
    migration_result = migrator.migrate_blackboard(blackboard)
    predicates_data = migration_result["predicates"]
    migration_stats = migration_result["statistics"]

    # --- Verify Migration (Basic Counts) ---
    if not migrator.verify_migration_counts(blackboard, migration_stats):
        logger_migration.error("Migration verification failed. Aborting KB population.")
        return False
    logger_migration.info("Migration verification passed.")

    # --- Populate Knowledge Base ---
    logger_migration.info(f"Populating Knowledge Base at '{kb_db_path_str}'...")
    kb = _KnowledgeBase_mig()
    # Need to pass dummy config or extract actual KB config path
    kb_config = {"knowledge_base": {"db_path": kb_db_path_str}}
    # Use dummy controller reference (None) as KB initialize/assert doesn't need it
    initialized = await kb.initialize(kb_config, None)

    if not initialized:
        logger_migration.error("Failed to initialize Knowledge Base. Aborting population.")
        return False

    populated_count = 0
    failed_count = 0
    for i, pred_data in enumerate(predicates_data):
        try:
            # Recreate Predicate object using the correct class reference
            predicate = _Predicate_mig(
                name=pred_data["name"],
                args=tuple(pred_data["args"]), # Ensure args is tuple
                value=pred_data["value"],
                timestamp=pred_data["timestamp"]
            )
            await kb.assert_fact(predicate)
            populated_count += 1
            if (i + 1) % 100 == 0: logger_migration.info(f"Asserted {i+1}/{len(predicates_data)} predicates...")
        except Exception as e:
            logger_migration.error(f"Failed to assert predicate {pred_data}: {e}")
            failed_count += 1

    logger_migration.info(f"Knowledge Base population complete. Asserted: {populated_count}, Failed: {failed_count}")

    # --- Shutdown KB ---
    await kb.shutdown()

    logger_migration.info("--- Data Migration Finished ---")
    return failed_count == 0 # Return True if successful


if __name__ == "__main__":
    # --- Ensure necessary classes are available in the global scope ---
    # If running standalone, these would need to be defined or imported here.
    # This assumes they are defined in the larger context where this script is embedded.
    if not all([_Predicate_mig, _Goal_mig, _create_goal_from_descriptor_mig, _KnowledgeBase_mig]):
         print("ERROR: Required classes (Predicate, Goal, KnowledgeBase) or function (create_goal_from_descriptor) not found in global scope.")
         sys.exit(1)

    parser = argparse.ArgumentParser(description="Migrate OSCAR-C Phase 1 data to Phase 2 Knowledge Base.")
    parser.add_argument("--phase1-data", default="data/blackboard.json", help="Path to Phase 1 blackboard JSON file.")
    parser.add_argument("--kb-path", default="oscar_c_kb.db", help="Path for the new Phase 2 SQLite Knowledge Base.")
    parser.add_argument("--no-backup", action="store_true", help="Skip backing up Phase 1 data.")

    args = parser.parse_args()

    # Setup asyncio loop to run the async migration function
    async def main():
         success = await run_migration(
             phase1_data_path_str=args.phase1_data,
             kb_db_path_str=args.kb_path,
             backup=not args.no_backup
         )
         return 0 if success else 1

    exit_code = asyncio.run(main())
    sys.exit(exit_code)

# --- END OF FILE migrations/v1_to_v2.py ---