# tests/unit/test_experience_stream.py

import pytest
import asyncio
import time
import logging
import queue
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock # Added AsyncMock
from typing import Optional
import os
import pytest_asyncio


# Attempt to import the component and datatypes
try:
    from consciousness_experiment.cognitive_modules.experience_stream import ExperienceStream, DEFAULT_STOPWORDS
    from consciousness_experiment.models.datatypes import PhenomenalState
    COMPONENT_AVAILABLE = True
except ImportError as e:
    COMPONENT_AVAILABLE = False
    ExperienceStream = None # type: ignore
    PhenomenalState = None # type: ignore
    DEFAULT_STOPWORDS = set() # type: ignore
    logging.warning(f"Skipping test_experience_stream: Failed to import - {e}")

logger = logging.getLogger(__name__)

@pytest_asyncio.fixture 
async def exp_stream_instance():
    """Provides a fresh ExperienceStream instance for each test."""
    if not COMPONENT_AVAILABLE:
        pytest.skip("ExperienceStream component not available.")
    
    mock_controller = MagicMock()
    
    # Configure attributes that ExperienceStream will try to access on the controller
    # For Global Workspace capacity (existing mock setup)
    mock_controller.global_workspace = MagicMock()
    mock_controller.global_workspace.capacity = 7 
    mock_controller.global_workspace.workspace_weights = {} # If ES uses this

    # --- NEW: Configure pain_level and happiness_level on the mock controller ---
    # These will be returned by getattr(self._controller, 'pain_level', 0.0)
    mock_controller.pain_level = 0.0  # Default to no pain for most unit tests
    mock_controller.happiness_level = 5.0 # Default to baseline happiness
    # --- END NEW ---

    # Configuration for ExperienceStream itself
    dummy_config = {
        "experience_stream": {
            "intensity_factor": 0.8,
            "valence_decay": 0.1,
            "custom_stopwords": ["testword", "custom"] # Ensure "custom" is here for test_content_diversity_lexical
        },
        "global_workspace": { 
            "capacity": mock_controller.global_workspace.capacity # Use the one from above
        },
         # Add other sections if ExperienceStream's initialize accesses them (e.g., for happiness_baseline_target)
        "internal_states": { # For happiness_baseline_for_valence_effect if ES reads it
            "happiness_baseline_target": 5.0
        }
    }
    mock_controller.config = dummy_config # Assign the config to the mock controller


    stream = ExperienceStream()
    # Initialize _current_valence to a known state for predictable valence tests
    stream._current_valence = 0.0 
    await stream.initialize(mock_controller.config, mock_controller)
    return stream

@pytest.mark.asyncio
@pytest.mark.skipif(not COMPONENT_AVAILABLE, reason="ExperienceStream not available")
class TestExperienceStreamSubMetrics:

    async def test_distinct_source_count(self, exp_stream_instance: ExperienceStream): # type: ignore
        logger.info("--- Test: ExperienceStream - Distinct Source Count ---")
        stream = exp_stream_instance # type: ignore

        # Scenario 1: Only broadcast content
        bc1 = {"percept_visual_1": "red ball", "goal_active_A": "get ball"}
        p_state1 = await stream.integrate_experience({}, [], {}, bc1)
        assert p_state1.distinct_source_count == 2, \
            f"Expected 2 sources (percept, goal from bc keys), got {p_state1.distinct_source_count}"

        # Scenario 2: Broadcast, percepts, memories, context
        bc2 = {"memory_recent_B": "saw dog"} # Adds "memory" source type from bc key
        percepts2 = {"user_input": "hello", "internal_error": "minor issue", "system_state": {"cpu":50}}
        # Adds "user_input_percept_source", "internal_error_percept_source", "system_state_percept_source"
        memories2 = [{"id": "m1", "summary": "past event"}] # Adds "retrieved_memory_source"
        context2 = {"last_action_type": "READ_FILE", "last_action_outcome": "success"} # Adds "action_context_source"
        p_state2 = await stream.integrate_experience(percepts2, memories2, context2, bc2)
        # Expected: memory (from bc prefix), user_input_percept_source, internal_error_percept_source,
        #           system_state_percept_source, retrieved_memory_source, action_context_source
        assert p_state2.distinct_source_count == 6, \
            f"Expected 6 sources, got {p_state2.distinct_source_count}. Sources found: {p_state2.content.get('_debug_source_types_present', 'Not debugged')}"


        # Scenario 3: Empty inputs
        p_state3 = await stream.integrate_experience({}, [], {}, {})
        assert p_state3.distinct_source_count == 0, \
            f"Expected 0 sources for empty inputs, got {p_state3.distinct_source_count}"

        # Scenario 4: Overlapping source types from different inputs
        bc4 = {"percept_audio_X": "bark"} # Adds "percept" source type
        percepts4 = {"system_state": {"cpu":10}} # Adds "system_state_percept_source"
        p_state4 = await stream.integrate_experience(percepts4, [], {}, bc4)
        assert p_state4.distinct_source_count == 2, \
             f"Expected 2 distinct source types (percept from bc, system_state_percept_source from percepts), got {p_state4.distinct_source_count}"


    async def test_content_diversity_lexical(self, exp_stream_instance: ExperienceStream): # type: ignore
        logger.info("--- Test: ExperienceStream - Content Diversity (Lexical TTR) ---")
        stream = exp_stream_instance # type: ignore
        # stream._stopwords was already updated by fixture via config

        # Scenario 1: Unique words
        bc1 = {"item1": {"content": "apple banana orange"}}
        p_state1 = await stream.integrate_experience({}, [], {}, bc1)
        assert abs(p_state1.content_diversity_lexical - 1.0) < 0.001, \
            f"Expected TTR 1.0, got {p_state1.content_diversity_lexical}"

        # Scenario 2: Repeated words
        bc2 = {"item1": "dog cat dog", "item2": {"content": "cat bird"}} # Mix direct string and nested
        p_state2 = await stream.integrate_experience({}, [], {}, bc2)
        # integrated_content texts: "dog cat dog", "cat bird"
        # all_text_for_diversity: ["dog cat dog", "cat bird"] -> "dog cat dog cat bird"
        # words: dog, cat, dog, cat, bird (5 total, no stopwords)
        # unique: dog, cat, bird (3 unique) -> TTR = 3/5 = 0.6
        assert abs(p_state2.content_diversity_lexical - 0.6) < 0.001, \
            f"Expected TTR 0.6, got {p_state2.content_diversity_lexical}"

        # Scenario 3: With stopwords and custom stopword ("custom" and "testword" from fixture config)
        bc3 = {"item1": "the quick brown fox", "item2": "a custom testword fox"}
        p_state3 = await stream.integrate_experience({}, [], {}, bc3)
        # text1: "the quick brown fox" -> words (no stops): quick, brown, fox
        # text2: "a custom testword fox" -> words (no stops): fox
        # combined filtered words: quick, brown, fox, fox (4 total)
        # unique filtered: quick, brown, fox (3 unique) -> TTR = 3/4 = 0.75
        assert abs(p_state3.content_diversity_lexical - 0.75) < 0.001, \
            f"Expected TTR 0.75, got {p_state3.content_diversity_lexical}"

        # Scenario 4: No textual content
        bc4 = {"item1": {"data": [1,2,3]}, "item2": 123}
        p_state4 = await stream.integrate_experience({}, [], {}, bc4)
        assert abs(p_state4.content_diversity_lexical - 0.0) < 0.001, \
            f"Expected TTR 0.0 for non-text, got {p_state4.content_diversity_lexical}"
            
        # Scenario 5: Single word
        bc5 = {"item1": "word"}
        p_state5 = await stream.integrate_experience({}, [], {}, bc5)
        assert abs(p_state5.content_diversity_lexical - 1.0) < 0.001, \
            f"Expected TTR 1.0 for single word, got {p_state5.content_diversity_lexical}"


    async def test_shared_concept_count_gw(self, exp_stream_instance: ExperienceStream): # type: ignore
        logger.info("--- Test: ExperienceStream - Shared Concept Count (GW) ---")
        stream = exp_stream_instance # type: ignore

        # Scenario 1: High sharing
        bc1 = {"itemA": "red apple green pear", "itemB": "red pear blue orange"}
        p_state1 = await stream.integrate_experience({}, [], {}, bc1)
        # Words (no stopwords from default list):
        # A: {red, apple, green, pear}
        # B: {red, pear, blue, orange}
        # Union (all_words_in_gw): {red, apple, green, pear, blue, orange} (6 unique)
        # Shared (count > 1): red, pear (2 shared)
        # Result: 2/6 = 0.333...
        assert abs(p_state1.shared_concept_count_gw - (2/6)) < 0.001, \
            f"Expected SharedConcepts ~0.333, got {p_state1.shared_concept_count_gw}"

        # Scenario 2: No sharing
        bc2 = {"itemA": "cat dog", "itemB": "fish bird"}
        p_state2 = await stream.integrate_experience({}, [], {}, bc2)
        assert abs(p_state2.shared_concept_count_gw - 0.0) < 0.001, \
            f"Expected SharedConcepts 0.0, got {p_state2.shared_concept_count_gw}"

        # Scenario 3: One item in broadcast
        bc3 = {"itemA": "hello world"}
        p_state3 = await stream.integrate_experience({}, [], {}, bc3)
        assert abs(p_state3.shared_concept_count_gw - 0.0) < 0.001, \
            f"Expected SharedConcepts 0.0 for single item, got {p_state3.shared_concept_count_gw}"

        # Scenario 4: Empty broadcast
        bc4 = {}
        p_state4 = await stream.integrate_experience({}, [], {}, bc4)
        assert abs(p_state4.shared_concept_count_gw - 0.0) < 0.001, \
            f"Expected SharedConcepts 0.0 for empty broadcast, got {p_state4.shared_concept_count_gw}"
            
        # Scenario 5: All items share one word, "a" is a stopword
        bc5 = {"i1": "common_word a b", "i2": "common_word c d", "i3": "common_word e f"}
        # After stopword "a" removal:
        # Text1 words: {common_word, b}
        # Text2 words: {common_word, c, d}
        # Text3 words: {common_word, e, f}
        # All unique words in GW: {common_word, b, c, d, e, f} (6 unique)
        # Shared words (count > 1): common_word (1 shared)
        # Result: 1/6 = 0.166...
        p_state5 = await stream.integrate_experience({}, [], {}, bc5)
        assert abs(p_state5.shared_concept_count_gw - (1/6)) < 0.001, \
            f"Expected SharedConcepts ~0.167 (1/6 due to 'a' as stopword), got {p_state5.shared_concept_count_gw}"

    async def test_integration_level_old_proxy(self, exp_stream_instance: ExperienceStream): # type: ignore
        logger.info("--- Test: ExperienceStream - Old Integration Level Proxy ---")
        stream = exp_stream_instance # type: ignore

        # NOTE: This test verifies the 'integration_level' field, which in the current
        # ExperienceStream.py is calculated as: min(1.0, new_distinct_source_count / 4.0).
        # The new_distinct_source_count itself is calculated based on identified source types.

        # Scenario 1: Only broadcast content
        bc1 = {"percept_visual_1": "red ball"} # Has "percept_" prefix
        p_state1 = await stream.integrate_experience({}, [], {}, bc1)
        # New distinct_source_count: 1 (from "percept_visual_1")
        # Expected old IL = 1/4 = 0.25
        logger.info(
            f"test_integration_level_old_proxy (Scenario 1): "
            f"Observed old IL = {p_state1.integration_level:.3f}. "
            f"New distinct_source_count = {p_state1.distinct_source_count}."
        )
        assert abs(p_state1.integration_level - 0.25) < 0.001, \
            f"Old IL for bc-only: Expected 0.25, got {p_state1.integration_level}"

        # Scenario 2: Multiple distinct new source types
        bc2 = {"item_no_prefix": "data"} # Does not contribute to new distinct_source_count by prefix
        percepts2 = {"user_input": "text"} # Contributes "user_input_percept_source"
        memories2 = ["memory_data"]      # Contributes "retrieved_memory_source"
        context2 = {"key": "value"}        # Contributes "action_context_source"
        # Total new distinct_source_count should be 3 (user_input, memory, context)
        # Expected old IL = 3/4 = 0.75
        p_state2 = await stream.integrate_experience(percepts2, memories2, context2, bc2)
        logger.info(
            f"test_integration_level_old_proxy (Scenario 2): "
            f"Observed old IL = {p_state2.integration_level:.3f}. "
            f"New distinct_source_count = {p_state2.distinct_source_count}."
        )
        assert abs(p_state2.integration_level - 0.75) < 0.001, \
            f"Old IL for scenario 2: Expected 0.75, got {p_state2.integration_level}. " \
            f"New distinct_source_count: {p_state2.distinct_source_count}"

        # Scenario 3: Broadcast (no prefix) and one type of percept
        bc3 = {"item_another_no_prefix": "data"} # Does not contribute to new distinct_source_count by prefix
        percepts3 = {"internal_error": "oops"} # Contributes "internal_error_percept_source"
        # Total new distinct_source_count should be 1 (internal_error)
        # Expected old IL = 1/4 = 0.25
        p_state3 = await stream.integrate_experience(percepts3, [], {}, bc3)
        logger.info(
            f"test_integration_level_old_proxy (Scenario 3): "
            f"Observed old IL = {p_state3.integration_level:.3f}. "
            f"New distinct_source_count = {p_state3.distinct_source_count}."
        )
        assert abs(p_state3.integration_level - 0.25) < 0.001, \
            f"Old IL for bc+percepts (scenario 3): Expected 0.25, got {p_state3.integration_level}"

        # Scenario 4: System state percept and goal in broadcast
        bc4 = {"goal_A": "achieve something"} # Contributes "goal" to new distinct_source_count
        percepts4 = {"system_state": {"cpu": 10}} # Contributes "system_state_percept_source"
        # Total new distinct_source_count should be 2
        # Expected old IL = 2/4 = 0.5
        p_state4 = await stream.integrate_experience(percepts4, [], {}, bc4)
        logger.info(
            f"test_integration_level_old_proxy (Scenario 4): "
            f"Observed old IL = {p_state4.integration_level:.3f}. "
            f"New distinct_source_count = {p_state4.distinct_source_count}."
        )
        assert abs(p_state4.integration_level - 0.5) < 0.001, \
            f"Old IL for scenario 4: Expected 0.5, got {p_state4.integration_level}"