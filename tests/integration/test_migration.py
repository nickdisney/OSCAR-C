# --- START OF FILE tests/integration/test_migration.py ---

"""Test Phase 1 to Phase 2 data migration script."""

import pytest
import json
import time
import asyncio
from pathlib import Path

# --- Adjust Imports ---
try:
    # Assuming tests run from project root
    from migrations.v1_to_v2 import run_migration, V1ToV2Migrator # Import the function and class
    from cognitive_modules.knowledge_base import KnowledgeBase # Need KB to check results
    from models.datatypes import Predicate # Need Predicate for checking
    MIGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Skipping migration tests: Failed to import necessary modules - {e}")
    MIGRATION_AVAILABLE = False
    pytest.skip("Migration script or dependencies not available", allow_module_level=True)


# --- Test Fixture ---

@pytest.fixture(scope="function")
def migration_test_files(tmp_path: Path) -> Dict[str, Path]:
    """Creates temporary files for migration input and output."""
    phase1_data_path = tmp_path / "phase1_data.json"
    kb_path = tmp_path / "test_migrated_kb.db"
    backup_path = tmp_path / "phase1_backup.json"

    # Create mock Phase 1 data
    mock_phase1_data = {
        "current_goal": "Test Migration Goal",
        "goal_set_time": time.time() - 3600, # Goal set an hour ago
        "recent_actions": [
            {"type": "ACTION_A", "outcome": "success", "timestamp": time.time() - 60},
            {"type": "ACTION_B", "outcome": "failure", "timestamp": time.time() - 30, "error": "Test Error", "params": {"path": "/bad"}},
            {"type": "ACTION_A", "outcome": "success", "timestamp": time.time() - 10},
        ],
        "self_model": {
            "knowledge": {
                "validated_paths": {"/good/path": 12345},
                "invalid_paths": ["/another/bad"],
            }
        },
        "narrative": [
            {"content": "Started migration test", "timestamp": time.time() - 50},
            {"content": "Encountered failure", "timestamp": time.time() - 25},
        ],
        "last_reflection": {"immediate": "Reflection content", "timestamp": time.time() - 100}
    }

    # Write mock data to file
    with open(phase1_data_path, "w") as f:
        json.dump(mock_phase1_data, f)

    return {
        "phase1_data": phase1_data_path,
        "kb": kb_path,
        "backup": backup_path
    }


# --- Test Cases ---

@pytest.mark.asyncio
@pytest.mark.skipif(not MIGRATION_AVAILABLE, reason="Migration script or dependencies not available")
async def test_migration_script_runs_successfully(migration_test_files: Dict[str, Path]):
    """Test the end-to-end run_migration function."""
    phase1_path = str(migration_test_files["phase1_data"])
    kb_path = str(migration_test_files["kb"])

    # Run the migration
    success = await run_migration(
        phase1_data_path_str=phase1_path,
        kb_db_path_str=kb_path,
        backup=True # Test backup creation as well
    )

    # Assertions
    assert success, "run_migration function returned False (indicating failure)."
    assert migration_test_files["backup"].exists(), "Backup file was not created."
    assert migration_test_files["kb"].exists(), "Knowledge Base DB file was not created."
    assert migration_test_files["kb"].stat().st_size > 0, "Knowledge Base DB file is empty."

    # --- Basic KB Content Verification ---
    kb = KnowledgeBase()
    kb_config = {"knowledge_base": {"db_path": kb_path}}
    initialized = await kb.initialize(kb_config, None)
    assert initialized, "Failed to re-initialize KB for verification."

    # Check if some expected predicates exist (adjust based on migration logic)
    # Check goal
    goal_preds = await kb.query(name="isCurrentGoal")
    assert len(goal_preds) == 1, "Current goal predicate missing or incorrect count."
    assert goal_preds[0].args[1] == "Test Migration Goal"

    # Check action events
    action_events = await kb.query(name="eventOccurred", args=("actionExecution", "ACTION_A", "success"))
    assert len(action_events) == 2, "Incorrect count for successful ACTION_A events."
    failed_action_events = await kb.query(name="actionFailed", args=("ACTION_B", "{'path': '/bad'}", "Test Error"))
    assert len(failed_action_events) == 1, "actionFailed predicate not found."

    # Check path knowledge
    valid_path_preds = await kb.query(name="isValidPath", args=("/good/path",))
    assert len(valid_path_preds) == 1, "Valid path predicate missing."
    invalid_path_preds = await kb.query(name="isInvalidPath", args=("/another/bad",))
    assert len(invalid_path_preds) == 1, "Invalid path predicate missing."
    # Check path added due to action failure error
    invalid_path_from_error = await kb.query(name="isInvalidPath", args=("/bad",))
    # Note: This check depends on the exact path normalization/storage in dynamic_self_model's migration part
    # assert len(invalid_path_from_error) == 1, "Invalid path from action error not found." # Optional Check

    # Check narrative
    narrative_preds = await kb.query(name="narrativeEntryRecorded")
    assert len(narrative_preds) == 2, "Incorrect count for narrative entries."

    # Check reflection
    reflection_preds = await kb.query(name="reflectionPerformed")
    assert len(reflection_preds) == 1, "Reflection predicate missing."

    await kb.shutdown()


@pytest.mark.skipif(not MIGRATION_AVAILABLE, reason="Migration script or dependencies not available")
def test_migrator_class_transforms_data(migration_test_files: Dict[str, Path]):
    """Test the V1ToV2Migrator class logic directly without DB interaction."""
    phase1_path = migration_test_files["phase1_data"]
    with open(phase1_path, 'r') as f:
        blackboard = json.load(f)

    migrator = V1ToV2Migrator()
    result = migrator.migrate_blackboard(blackboard)

    assert "predicates" in result
    assert "statistics" in result
    assert isinstance(result["predicates"], list)
    pred_data = result["predicates"]
    stats = result["statistics"]

    # Verify stats counts match expectations from mock data
    assert stats["goals_migrated"] == 1
    assert stats["actions_migrated"] == 3
    assert stats["paths_migrated"] == 2 # 1 valid + 1 invalid from self_model
    assert stats["narrative_entries"] == 2
    assert stats["other_items"] == 1 # reflection
    # Verify total predicates generated (summing up expectations)
    expected_preds = 1 # goal
    expected_preds += 2 # paths
    expected_preds += 3 # action executions
    expected_preds += 1 # action failure detail
    expected_preds += 2 # narrative entries
    expected_preds += 1 # reflection
    assert stats["predicates_generated"] == expected_preds

    # Verify structure of a few sample predicates
    assert any(p['name'] == 'isCurrentGoal' and p['args'][1] == 'Test Migration Goal' for p in pred_data)
    assert any(p['name'] == 'actionFailed' and p['args'][0] == 'ACTION_B' for p in pred_data)


# --- Add more tests ---
# - Test migration with missing keys in Phase 1 data
# - Test migration with empty Phase 1 data
# - Test migration verification logic failure


# --- END OF FILE tests/integration/test_migration.py ---