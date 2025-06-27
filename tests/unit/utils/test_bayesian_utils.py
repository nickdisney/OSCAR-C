import pytest
from typing import Set

# Ensure the utils module can be found (adjust path if necessary)
try:
    from consciousness_experiment.utils.bayesian_utils import discretize_variable, extract_features_for_cbn
    from consciousness_experiment.models.datatypes import Predicate # For extract_features test
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    # Fallbacks for testing if imports fail
    def discretize_variable(value, bins): return str(value)
    def extract_features_for_cbn(action, context, pre_state): return {}
    class Predicate:
        def __init__(self, name, args, value=True): self.name=name; self.args=args; self.value=value
        def __hash__(self): return hash((self.name, self.args, self.value))
        def __eq__(self, other): return isinstance(other, Predicate) and self.name == other.name and self.args == other.args and self.value == other.value

@pytest.mark.skipif(not UTILS_AVAILABLE, reason="Bayesian utils or dependencies not available")
class TestBayesianUtils:

    def test_discretize_variable_basic(self):
        bins = [(None, 10.0, "low"), (10.0, 20.0, "medium"), (20.0, None, "high")]
        assert discretize_variable(5.0, bins) == "low"
        assert discretize_variable(10.0, bins) == "medium" # Min is inclusive
        assert discretize_variable(19.9, bins) == "medium"
        assert discretize_variable(20.0, bins) == "high"
        assert discretize_variable(100.0, bins) == "high"
        assert discretize_variable(-5.0, bins) == "low"

    def test_discretize_variable_edge_cases(self):
        bins_only_low = [(None, 10.0, "low")]
        assert discretize_variable(5.0, bins_only_low) == "low"
        assert discretize_variable(15.0, bins_only_low) == "outside_bins" # Above max

        bins_only_high = [(20.0, None, "high")]
        assert discretize_variable(25.0, bins_only_high) == "high"
        assert discretize_variable(15.0, bins_only_high) == "outside_bins" # Below min

        bins_middle_gap = [(None, 10.0, "low"), (20.0, None, "high")]
        assert discretize_variable(15.0, bins_middle_gap) == "outside_bins"

        assert discretize_variable(10, []) == "10" # Empty bins
        assert discretize_variable(10, None) == "10" # type: ignore # None bins

    # Modify existing tests to expect new features or "NotApplicable"
    def test_extract_features_for_cbn_read_file_action(self): # Renamed for clarity
        action = {"type": "READ_FILE", "params": {"path": "/sandbox/data.txt"}}
        context = {
            "php_levels": {"pain": 1.5, "happiness": 6.0, "purpose": 7.1}, # low_pain, medium_happy, high_purpose
            "current_cs_level_name": "CONSCIOUS", # NormalActivity
            "drives": {"curiosity": {"value": 0.8}} # high_drive
        }
        pre_state: Set[Predicate] = { Predicate("isFile", ("/sandbox/data.txt",), True) } # type: ignore
        
        features = extract_features_for_cbn(action, context, pre_state)

        assert features["ActionTypeNode"] == "READ_FILE"
        assert features["ActionParam_PathType"] == "sandboxed"
        assert features["PreState_FileExists_Target"] == "True_File"
        assert features["Context_PainLevel"] == "low_pain"
        assert features["Context_ConsciousState"] == "NormalActivity"
        assert features["Context_HappinessLevel"] == "medium_happy"
        assert features["Context_PurposeLevel"] == "high_purpose"
        assert features["Context_CuriosityDrive"] == "high_drive"
        # Check that features specific to other action types are "NotApplicable"
        assert features["ActionParam_PromptLength"] == "NotApplicable"
        assert features["ActionParam_CommandCategory"] == "NotApplicable"
        assert features["Context_LLM_TaskType"] == "NotApplicable"
        assert features["PreState_FilePermissions_Target"] == "Unknown" # Current placeholder

    def test_extract_features_for_cbn_call_llm_action_summarize(self): # Renamed
        action = {"type": "CALL_LLM", "params": {"prompt": "Summarize this text: ... for goal 'my summary goal'"}}
        context = {
            "php_levels": {"pain": 8.0}, "current_cs_level_name": "META_CONSCIOUS",
            "active_goal_details": {"description": "Goal: summarize the document for review"}
        }
        pre_state: Set[Predicate] = set() # type: ignore
        
        features = extract_features_for_cbn(action, context, pre_state)
        assert features["ActionTypeNode"] == "CALL_LLM"
        assert features["ActionParam_PromptLength"] == "short" # Prompt len 51 is < 100
        assert features["Context_PainLevel"] == "high_pain" # Updated from "high" to "high_pain"
        assert features["Context_ConsciousState"] == "HighActivity"
        assert features["Context_LLM_TaskType"] == "Summarization" # Due to "summarize" in prompt/goal
        assert features["ActionParam_PathType"] == "NotApplicable"
        assert features["PreState_FileExists_Target"] == "NotApplicable"

    # Add new tests for other action types and feature variations
    def test_extract_features_for_cbn_execute_command(self):
        action = {"type": "EXECUTE_COMMAND", "params": {"command": "rm /sandbox/tempfile.txt"}}
        context = {"current_cs_level_name": "PRE_CONSCIOUS"} # This maps to LowActivity
        pre_state: Set[Predicate] = set() # type: ignore

        features = extract_features_for_cbn(action, context, pre_state)
        assert features["ActionTypeNode"] == "EXECUTE_COMMAND"
        assert features["ActionParam_CommandCategory"] == "Deletion"
        assert features["Context_ConsciousState"] == "LowActivity"
        assert features["ActionParam_PathType"] == "NotApplicable" # EXECUTE_COMMAND doesn't use PathType feature
        assert features["PreState_FileExists_Target"] == "NotApplicable"

    def test_extract_features_for_cbn_unknown_action_default_features(self):
        action = {"type": "MY_NEW_CUSTOM_ACTION", "params": {}}
        context = {"current_cs_level_name": "UNKNOWN"}
        pre_state: Set[Predicate] = set() # type: ignore

        features = extract_features_for_cbn(action, context, pre_state)
        assert features["ActionTypeNode"] == "MY_NEW_CUSTOM_ACTION"
        assert features["Context_ConsciousState"] == "UnknownActivity"
        # Assert that action-specific param/prestate features default to "NotApplicable"
        assert features["ActionParam_PathType"] == "NotApplicable"
        assert features["ActionParam_PromptLength"] == "NotApplicable"
        assert features["ActionParam_CommandCategory"] == "NotApplicable"
        assert features["PreState_FileExists_Target"] == "NotApplicable"
        assert features["Context_LLM_TaskType"] == "NotApplicable"

    def test_extract_features_for_cbn_respond_to_user(self):
        action = {"type": "RESPOND_TO_USER", "params": {"text": "Hello there! This is a response."}}
        context = {
            "php_levels": {"pain": 0.5, "happiness": 8.0, "purpose": 6.0},
            "current_cs_level_name": "CONSCIOUS",
            "drives": {"curiosity": {"value": 0.2}}
        }
        pre_state: Set[Predicate] = set() # type: ignore

        features = extract_features_for_cbn(action, context, pre_state)
        assert features["ActionTypeNode"] == "RESPOND_TO_USER"
        assert features["ActionParam_PromptLength"] == "short" # Based on "text" param length
        assert features["Context_LLM_TaskType"] == "General" # Default if no keywords
        assert features["Context_PainLevel"] == "low_pain"
        assert features["Context_HappinessLevel"] == "high_happy"
        assert features["Context_PurposeLevel"] == "medium_purpose"
        assert features["Context_ConsciousState"] == "NormalActivity"
        assert features["Context_CuriosityDrive"] == "low_drive"

    def test_extract_features_for_cbn_write_file_no_pre_state(self):
        action = {"type": "WRITE_FILE", "params": {"path": "new_notes.txt"}}
        context = {} # Minimal context
        pre_state: Set[Predicate] = set() # type: ignore

        features = extract_features_for_cbn(action, context, pre_state)
        assert features["ActionTypeNode"] == "WRITE_FILE"
        assert features["ActionParam_PathType"] == "general_external"
        assert features["PreState_FileExists_Target"] == "False" # File does not exist
        assert features["PreState_FilePermissions_Target"] == "Unknown"


    def test_extract_features_for_cbn_list_files_critical_path(self):
        action = {"type": "LIST_FILES", "params": {"path": "/etc/some_config_dir"}}
        context = {"php_levels": {"pain": 5.0}}
        pre_state: Set[Predicate] = {Predicate("isDirectory", ("/etc/some_config_dir",), True)} # type: ignore

        features = extract_features_for_cbn(action, context, pre_state)
        assert features["ActionTypeNode"] == "LIST_FILES"
        assert features["ActionParam_PathType"] == "critical_system"
        assert features["PreState_FileExists_Target"] == "True_Directory"
        assert features["Context_PainLevel"] == "medium_pain"

    def test_extract_features_for_cbn_default_drive_and_php(self):
        action = {"type": "THINKING", "params": {}}
        context = {"current_cs_level_name": "CONSCIOUS"} # php_levels and drives keys are missing

        features = extract_features_for_cbn(action, context, set()) # type: ignore
        
        # --- CORRECTED ASSERTIONS for missing keys ---
        assert features["Context_PainLevel"] == "unknown_pain" 
        assert features["Context_HappinessLevel"] == "unknown_happy"
        assert features["Context_PurposeLevel"] == "unknown_purpose"
        assert features["Context_CuriosityDrive"] == "unknown_drive" # Assuming 'drives' key also missing
        # --- END CORRECTIONS ---
        
        assert features["Context_ConsciousState"] == "NormalActivity" # This part was fine
        # Assert that action-specific param/prestate features default to "NotApplicable"
        assert features["ActionParam_PathType"] == "NotApplicable"
        assert features["ActionParam_PromptLength"] == "NotApplicable"

    def test_extract_features_for_cbn_llm_task_type_from_goal_desc(self):
        action = {"type": "CALL_LLM", "params": {"prompt": "Do it."}} # Generic prompt
        context = {
            "active_goal_details": {"description": "The goal is to translate this document into French."}
        }
        features = extract_features_for_cbn(action, context, set()) # type: ignore
        assert features["Context_LLM_TaskType"] == "Translation"

    def test_extract_features_unknown_pain_if_unparsable(self):
        action = {"type": "THINKING", "params": {}}
        context = {"php_levels": {"pain": "this_is_not_a_number"}}
        features = extract_features_for_cbn(action, context, set()) # type: ignore
        assert features["Context_PainLevel"] == "unknown_pain"

    def test_extract_features_unknown_pain_if_key_missing(self):
        action = {"type": "THINKING", "params": {}}
        context = {"php_levels": {"happiness": 5.0}} # Pain key is missing
        features = extract_features_for_cbn(action, context, set()) # type: ignore
        assert features["Context_PainLevel"] == "unknown_pain"

    # calculate_cpd_entropy is a placeholder, so no meaningful test for it now.
    # def test_calculate_cpd_entropy(self):
    #     pass