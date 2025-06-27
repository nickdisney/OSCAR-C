# tests/unit/test_consciousness_level_assessor.py

import pytest
import asyncio
import time
import re # For CLA's _normalize_and_tokenize
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import deque
from unittest.mock import MagicMock, patch
import pytest_asyncio

# Attempt to import from the project structure
try:
    from consciousness_experiment.cognitive_modules.consciousness_level_assessor import (
        ConsciousnessLevelAssessor, 
        DEFAULT_THRESH_META, DEFAULT_THRESH_CONSCIOUS, 
        DEFAULT_THRESH_PRECONSCIOUS, DEFAULT_UNCONSCIOUS_THRESHOLD,
        CLA_DEFAULT_STOPWORDS # For testing _normalize_and_tokenize
    )
    from consciousness_experiment.models.enums import ConsciousState
    from consciousness_experiment.models.datatypes import PhenomenalState
    from consciousness_experiment.protocols import CognitiveComponent, ConsciousnessAssessor
    
    # Try to import networkx for tests that depend on it
    import networkx as nx
    NETWORKX_AVAILABLE_TEST = True
    CLA_MODELS_AVAILABLE = True
except ImportError:
    CLA_MODELS_AVAILABLE = False
    NETWORKX_AVAILABLE_TEST = False # Assume false if other imports fail too

    # Minimal fallbacks for the test structure to be parseable
    class CognitiveComponent: pass
    class ConsciousnessAssessor(CognitiveComponent): pass # type: ignore
    class ConsciousState(str): # type: ignore
        UNCONSCIOUS = "UNCONSCIOUS"; PRE_CONSCIOUS = "PRE_CONSCIOUS"
        CONSCIOUS = "CONSCIOUS"; META_CONSCIOUS = "META_CONSCIOUS"; REFLECTIVE = "REFLECTIVE"
        @classmethod
        def __members__(cls): return {k: getattr(cls,k) for k in dir(cls) if not k.startswith('_') and isinstance(getattr(cls,k), str)}
    
    class PhenomenalState:
        def __init__(self, content=None, **kwargs):
            self.content = content if content is not None else {}
            # Add other attributes if tests directly access them on mock_pstate
            self.distinct_source_count = kwargs.get('distinct_source_count', 0)
            self.content_diversity_lexical = kwargs.get('content_diversity_lexical', 0.0)
            self.shared_concept_count_gw = kwargs.get('shared_concept_count_gw', 0.0)


    class ConsciousnessLevelAssessor(ConsciousnessAssessor): # type: ignore
        def __init__(self):
            self._config = {}; self._controller = None; self._PhenomenalStateClass = PhenomenalState
            self._ConsciousStateEnum = ConsciousState; self._stopwords_cla = set(); self._goal_related_keywords_cla = set()
            self.unconscious_threshold = 0.1; self.pre_conscious_threshold = 0.25
            self.conscious_threshold = 0.5; self.meta_conscious_threshold = 0.75
            self.last_phi_proxy_score = 0.0; self.last_assessed_level_name = ""
        async def initialize(self, config, controller):
            self._config = config.get("consciousness_assessor", {})
            self._controller = controller
            self.unconscious_threshold = self._config.get("unconscious_threshold", 0.1)
            self.pre_conscious_threshold = self._config.get("pre_conscious_threshold", 0.25)
            self.conscious_threshold = self._config.get("conscious_threshold", 0.5)
            self.meta_conscious_threshold = self._config.get("meta_conscious_threshold", 0.75)
            return True
        def _has_self_reference(self, ec, wc): return False
        def _normalize_and_tokenize(self, text): return set(re.findall(r'\b\w+\b', text.lower()))
        def _calculate_nonstring_relationship(self,c1,c2,d=0): return 0.0
        def _calculate_relationship_strength(self,c1,c2,d=0): return 0.0
        def measure_workspace_coherence(self, wc): return 0.5 # Mocked default
        def calculate_differentiation_integration(self, ps, wc): return 0.5 # Mocked default
        async def assess_consciousness_level(self, exp, wc): return ConsciousState.PRE_CONSCIOUS # type: ignore

    DEFAULT_THRESH_META = 0.75; DEFAULT_THRESH_CONSCIOUS = 0.50
    DEFAULT_THRESH_PRECONSCIOUS = 0.25; DEFAULT_UNCONSCIOUS_THRESHOLD = 0.10
    CLA_DEFAULT_STOPWORDS = set(["a", "the"])


# --- Fixture for CLA Instance ---
@pytest_asyncio.fixture
async def cla_instance():
    if not CLA_MODELS_AVAILABLE:
        pytest.skip("CLA or its core dependencies not available.")
    
    cla = ConsciousnessLevelAssessor()
    mock_controller = MagicMock()
    # Basic config for thresholds and new params
    test_config = {
        "consciousness_assessor": {
            "unconscious_threshold": 0.10,
            "pre_conscious_threshold": 0.25,
            "conscious_threshold": 0.50,
            "meta_conscious_threshold": 0.75,
            "differentiation_norm_factor": 20.0,
            "cla_stopwords": ["is", "of", "and"], # Custom stopwords for test
            "cla_goal_keywords": ["objective", "mission"] # Custom goal words
        },
        "global_workspace": {"capacity": 7} # For gw_capacity_for_norm fallback
    }
    await cla.initialize(test_config, mock_controller)
    return cla

# --- Tests for Helper Methods (C.2.2 additions) ---
@pytest.mark.skipif(not CLA_MODELS_AVAILABLE, reason="CLA models not available.")
class TestCLAHelpers:
    def test_normalize_and_tokenize(self, cla_instance: ConsciousnessLevelAssessor):
        text = "This IS a Test sentence with The word Objective."
        cla_instance._stopwords_cla = {"is", "a", "the"} # Override for specific test
        tokens = cla_instance._normalize_and_tokenize(text)
        assert tokens == {"this", "test", "sentence", "with", "word", "objective"}
        assert "is" not in tokens
        assert "objective" in tokens # Not a stopword here

    def test_calculate_relationship_strength_strings(self, cla_instance: ConsciousnessLevelAssessor):
        cla_instance._stopwords_cla = {"a", "the", "is"}
        cla_instance._goal_related_keywords_cla = {"goal", "task"}

        # High overlap, one goal keyword
        s1 = cla_instance._calculate_relationship_strength("This is a test task", "A test of the task")
        # tokens1 = {this, test, task}, tokens2 = {test, of, task} -> intersect=2, union=4, base=0.5
        # goal_keywords_found = {task}, bonus = 0.05. Total = 0.55
        assert s1 == pytest.approx(0.5 + 0.05)

        # No overlap
        s2 = cla_instance._calculate_relationship_strength("Apple banana kiwi", "Orange pear grape")
        assert s2 == 0.0

        # Partial overlap, no goal keywords
        s3 = cla_instance._calculate_relationship_strength("Quick brown fox", "Lazy brown dog")
        # tokens1 = {quick, brown, fox}, tokens2 = {lazy, brown, dog} -> intersect=1, union=5, base=0.2
        assert s3 == pytest.approx(0.2)
        
        # Empty strings
        assert cla_instance._calculate_relationship_strength("", "test") == 0.0
        assert cla_instance._calculate_relationship_strength("test", "") == 0.0
        assert cla_instance._calculate_relationship_strength("", "") == 0.0

    def test_calculate_nonstring_relationship_dicts(self, cla_instance: ConsciousnessLevelAssessor):
        # Mock _calculate_relationship_strength to control recursive calls for string values
        with patch.object(cla_instance, '_calculate_relationship_strength', side_effect=lambda c1,c2,d: 1.0 if c1==c2 else 0.0) as mock_str_rel:
            dict1 = {"a": "apple", "b": "ball", "c": "cat"}
            dict2 = {"b": "ball", "c": "cat", "d": "dog"} # Common: b, c. Values match for b,c.
            # common_keys = {b,c}, len=2. keys1=3, keys2=3. union_keys=4. key_jaccard=2/4 = 0.5
            # val_sim(b,b)=1.0, val_sim(c,c)=1.0. avg_val_sim = (1+1)/2 = 1.0
            # expected = 0.5 * 0.4 + 1.0 * 0.6 = 0.2 + 0.6 = 0.8
            sim_dicts = cla_instance._calculate_nonstring_relationship(dict1, dict2)
            assert sim_dicts == pytest.approx(0.8)

            dict3 = {"x": "xenon"}
            assert cla_instance._calculate_nonstring_relationship(dict1, dict3) == 0.0 # No common keys

    def test_calculate_nonstring_relationship_lists(self, cla_instance: ConsciousnessLevelAssessor):
        with patch.object(cla_instance, '_calculate_relationship_strength', side_effect=lambda c1,c2,d: 1.0 if c1==c2 else (0.5 if c1=="a" and c2=="b" else 0.0) ) as mock_str_rel:
            list1 = ["a", "x", "y"]
            list2 = ["b", "x", "z", "w"] # Common: "x" at index 1
            # len_sim = 1 - abs(3-4)/4 = 0.75
            # samples (a,b), (x,x), (y,z) -> sim = 0.5, 1.0, 0.0. avg_sample_sim = (0.5+1.0+0.0)/3 = 0.5
            # expected = 0.75 * 0.3 + 0.5 * 0.7 = 0.225 + 0.35 = 0.575
            sim_lists = cla_instance._calculate_nonstring_relationship(list1, list2)
            assert sim_lists == pytest.approx(0.575)
            
            assert cla_instance._calculate_nonstring_relationship([], []) == 1.0 
            assert cla_instance._calculate_nonstring_relationship(["a"], []) == 0.0 * 0.3 # len_sim = 0

# --- Tests for measure_workspace_coherence (C.2.2) ---
@pytest.mark.skipif(not NETWORKX_AVAILABLE_TEST or not CLA_MODELS_AVAILABLE, reason="NetworkX or CLA models not available.")
class TestWorkspaceCoherence:

    def test_measure_workspace_coherence_empty_or_single(self, cla_instance: ConsciousnessLevelAssessor):
        assert cla_instance.measure_workspace_coherence({}) == 0.0
        assert cla_instance.measure_workspace_coherence({"item1": "content"}) == 0.0

    def test_measure_workspace_coherence_unrelated_items(self, cla_instance: ConsciousnessLevelAssessor):
        workspace = {"item1": "apple banana", "item2": "orange grape", "item3": "kiwi pear"}
        # Mock relationship strength to always be low (no edges formed)
        with patch.object(cla_instance, '_calculate_relationship_strength', return_value=0.1):
            coherence = cla_instance.measure_workspace_coherence(workspace)
            assert coherence == 0.0 # No edges -> 0 clustering, 0 path connectivity

    def test_measure_workspace_coherence_fully_connected(self, cla_instance: ConsciousnessLevelAssessor):
        workspace = {"itemA": "text a", "itemB": "text b", "itemC": "text c"}
        # Mock relationship strength to always be high (all items related)
        with patch.object(cla_instance, '_calculate_relationship_strength', return_value=0.8):
            coherence = cla_instance.measure_workspace_coherence(workspace)
            # For a fully connected graph (K3):
            # Clustering for each node is 1.0. Average clustering = 1.0.
            # Average shortest path length, using edge weights (0.8)
            # In networkx, if weight is provided to average_shortest_path_length, it's used as "distance".
            # For similarity as weight, we usually want 1/weight as distance or use unweighted for path count.
            # Assuming the implementation uses the weight as-is for shortest_path (which might be inverse of desired if weight=similarity)
            # If shortest_path uses 'weight' attribute as distance: path_length = 0.8 (each edge)
            # If it calculates unweighted path length: path_length = 1.0
            # The code's `nx.average_shortest_path_length(graph, weight="weight")` will use 0.8 as distance.
            expected_norm_path_k3 = 1.0 / (1.0 + 0.8) 
            expected_coherence_k3 = (1.0 + expected_norm_path_k3) / 2.0
            assert coherence == pytest.approx(expected_coherence_k3)


    def test_measure_workspace_coherence_line_graph(self, cla_instance: ConsciousnessLevelAssessor):
        workspace = {"itemA": "a", "itemB": "b", "itemC": "c"}
        
        def mock_rel_strength_line(c1, c2, depth=0):
            pairs = {("a","b"), ("b","a"), ("b","c"), ("c","b")}
            return 0.8 if (str(c1), str(c2)) in pairs else 0.0

        with patch.object(cla_instance, '_calculate_relationship_strength', side_effect=mock_rel_strength_line):
            coherence = cla_instance.measure_workspace_coherence(workspace)
            # Graph: A-B-C, edge weights 0.8
            # Clustering: A=0, B=0, C=0. Avg Clustering = 0
            # Weighted paths: A-B (0.8), B-C (0.8), A-C (0.8+0.8=1.6).
            # Avg shortest path = (0.8 + 0.8 + 1.6)/3 = 3.2/3 
            avg_weighted_path_line = 3.2 / 3.0
            expected_norm_path_line = 1.0 / (1.0 + avg_weighted_path_line)
            expected_coherence_line = (0.0 + expected_norm_path_line) / 2.0
            assert coherence == pytest.approx(expected_coherence_line)


# --- Tests for Differentiation-Integration Metric (C.2.3) ---
@pytest.mark.skipif(not CLA_MODELS_AVAILABLE, reason="CLA models not available.")
class TestDifferentiationIntegration:

    def test_calculate_differentiation_integration_basic(self, cla_instance: ConsciousnessLevelAssessor):
        mock_pstate = MagicMock(spec=PhenomenalState)
        # Test with 10 items in pstate.content. differentiation_norm_factor = 20 (default or from fixture config)
        # Differentiation = 10 / 20 = 0.5
        mock_pstate.content = {f"ps_item_{i}": f"data {i}" for i in range(10)}
        
        mock_workspace = {"ws1": "content a", "ws2": "content b"} # For coherence calculation

        with patch.object(cla_instance, 'measure_workspace_coherence', return_value=0.7) as mock_mwc:
            di_value = cla_instance.calculate_differentiation_integration(mock_pstate, mock_workspace)
            # Expected DI = differentiation (0.5) * integration_mocked (0.7) = 0.35
            assert di_value == pytest.approx(0.35)
            mock_mwc.assert_called_once_with(mock_workspace)

    def test_calculate_differentiation_integration_zero_diff(self, cla_instance: ConsciousnessLevelAssessor):
        mock_pstate = MagicMock(spec=PhenomenalState); mock_pstate.content = {} # Empty pstate content
        mock_workspace = {"ws1": "a"}
        with patch.object(cla_instance, 'measure_workspace_coherence', return_value=0.8):
            di_value = cla_instance.calculate_differentiation_integration(mock_pstate, mock_workspace)
            assert di_value == 0.0 # Differentiation is 0

    def test_calculate_differentiation_integration_zero_integ(self, cla_instance: ConsciousnessLevelAssessor):
        mock_pstate = MagicMock(spec=PhenomenalState); mock_pstate.content = {f"i{x}":"v{x}" for x in range(10)}
        mock_workspace = {"ws1": "a", "ws2": "b"}
        with patch.object(cla_instance, 'measure_workspace_coherence', return_value=0.0): # No coherence
            di_value = cla_instance.calculate_differentiation_integration(mock_pstate, mock_workspace)
            assert di_value == 0.0 # Integration is 0

# --- Tests for Updated assess_consciousness_level (C.2.4) ---
@pytest.mark.skipif(not CLA_MODELS_AVAILABLE, reason="CLA models not available.")
class TestAssessConsciousnessLevelUpdated:

    @pytest.mark.asyncio
    async def test_assess_cs_level_uses_di_value_and_thresholds(self, cla_instance: ConsciousnessLevelAssessor):
        mock_pstate = MagicMock(spec=PhenomenalState)
        mock_pstate.content = {"item1": "data"}
        mock_pstate.distinct_source_count = 1
        mock_pstate.content_diversity_lexical = 0.5
        mock_pstate.shared_concept_count_gw = 0.2
        mock_workspace = {"ws_item1": "ws_data"}

        cla_instance.unconscious_threshold = 0.10
        cla_instance.pre_conscious_threshold = 0.25
        cla_instance.conscious_threshold = 0.50
        cla_instance.meta_conscious_threshold = 0.75

        di_scores_to_test = {
            0.05: ConsciousState.UNCONSCIOUS,
            0.15: ConsciousState.UNCONSCIOUS,
            0.30: ConsciousState.PRE_CONSCIOUS,
            0.60: ConsciousState.CONSCIOUS,
            0.80: ConsciousState.CONSCIOUS, # Still CONSCIOUS if no self-ref
        }
        
        # Patch _has_self_reference for the whole test or control it per loop
        with patch.object(cla_instance, '_has_self_reference', return_value=False) as mock_has_self_ref_method:
            for di_score, expected_cs_level_enum in di_scores_to_test.items():
                with patch.object(cla_instance, 'calculate_differentiation_integration', return_value=di_score):
                    cs_level_obj = await cla_instance.assess_consciousness_level(mock_pstate, mock_workspace)
                    
                    # Compare enum objects directly if ConsciousState is the real enum
                    if CLA_MODELS_AVAILABLE and isinstance(expected_cs_level_enum, ConsciousState):
                        assert cs_level_obj == expected_cs_level_enum, f"For DI score {di_score} (no self-ref)"
                    else: # Fallback if ConsciousState is the mock string enum
                        assert cs_level_obj.name == expected_cs_level_enum.name, f"For DI score {di_score} (no self-ref)" # type: ignore
                    
                    assert cla_instance.last_phi_proxy_score == pytest.approx(di_score)

            # Now test META_CONSCIOUS: _has_self_reference should return True
            mock_has_self_ref_method.return_value = True # Change return value of the existing mock
            di_score_for_meta = 0.80 # A score that would trigger META if self-ref is true
            with patch.object(cla_instance, 'calculate_differentiation_integration', return_value=di_score_for_meta):
                cs_level_meta_obj = await cla_instance.assess_consciousness_level(mock_pstate, mock_workspace)

                expected_meta_enum = ConsciousState.META_CONSCIOUS if CLA_MODELS_AVAILABLE else ConsciousState("META_CONSCIOUS") # type: ignore
                if CLA_MODELS_AVAILABLE and isinstance(expected_meta_enum, ConsciousState):
                    assert cs_level_meta_obj == expected_meta_enum, f"For DI score {di_score_for_meta} (with self-ref)"
                else:
                    assert cs_level_meta_obj.name == expected_meta_enum.name, f"For DI score {di_score_for_meta} (with self-ref)" # type: ignore

    @pytest.mark.asyncio
    async def test_assess_cs_level_no_experience(self, cla_instance: ConsciousnessLevelAssessor):
        cs_level = await cla_instance.assess_consciousness_level(None, {})
        if isinstance(ConsciousState.UNCONSCIOUS, ConsciousState):
             assert cs_level == ConsciousState.UNCONSCIOUS
        else: # Fallback
            assert cs_level.name == ConsciousState.UNCONSCIOUS.name # type: ignore
        assert cla_instance.last_phi_proxy_score == 0.0

    