# consciousness_experiment/cognitive_modules/consciousness_level_assessor.py

import asyncio
import logging
import time
import re # For _has_self_reference and new _normalize_and_tokenize
from typing import Dict, Any, Optional, Set, List, Tuple # Added Set, List, Tuple for new methods

# --- NEW: Import networkx ---
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.getLogger(__name__).warning("ConsciousnessLevelAssessor: networkx library not found. Workspace coherence metric will be disabled.")
# --- END NEW ---

# --- Use standard relative imports ---
try:
    from ..protocols import ConsciousnessAssessor
    from ..protocols import CognitiveComponent
    from ..models.enums import ConsciousState
    from ..models.datatypes import PhenomenalState
except ImportError:
    logging.warning("ConsciousnessLevelAssessor: Relative imports failed, relying on globally defined types.")
    if 'ConsciousnessAssessor' not in globals(): raise ImportError("ConsciousnessAssessor not found via relative import or globally")
    if 'CognitiveComponent' not in globals(): raise ImportError("CognitiveComponent not found via relative import or globally")
    if 'ConsciousState' not in globals(): raise ImportError("ConsciousState not found via relative import or globally")
    if 'PhenomenalState' not in globals(): raise ImportError("PhenomenalState not found via relative import or globally")
    ConsciousnessAssessor = globals().get('ConsciousnessAssessor')
    ConsciousState = globals().get('ConsciousState')
    PhenomenalState = globals().get('PhenomenalState')

logger_consciousness_assessor = logging.getLogger(__name__)

# Default thresholds for phi_proxy_score (can be overridden by config)
DEFAULT_THRESH_META = 0.75 # Adjusted as phi_proxy might have a different scale
DEFAULT_THRESH_CONSCIOUS = 0.50
DEFAULT_THRESH_PRECONSCIOUS = 0.25
DEFAULT_UNCONSCIOUS_THRESHOLD = 0.05 # Threshold below which it's UNCONSCIOUS

# Default weights for Φ-Proxy sub-metrics (configurable)
DEFAULT_DIFF_WEIGHT_SOURCES = 0.4
DEFAULT_DIFF_WEIGHT_LEXICAL = 0.6
DEFAULT_INT_WEIGHT_SHARED_CONCEPTS = 1.0
DEFAULT_PHI_CONTRIB_DIFF = 0.5 # Contribution of differentiation to final phi_proxy
DEFAULT_PHI_CONTRIB_INT = 0.5  # Contribution of integration to final phi_proxy
DEFAULT_GW_CAPACITY_FOR_NORM = 7 # For normalizing distinct_source_count

# --- NEW: Default stopwords if not available from ExperienceStream ---
# This is a simplified list, ideally use a more comprehensive one or share with ES.
CLA_DEFAULT_STOPWORDS = set([
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "and", "in", "on", "it", "this", "that"
])
DEFAULT_CLA_COHERENCE_EDGE_THRESHOLD = 0.05


class ConsciousnessLevelAssessor(ConsciousnessAssessor): # type: ignore
    """
    Assesses the agent's current level of consciousness based on IIT-inspired
    Φ-Proxy metrics calculated from the PhenomenalState and workspace content.
    """

    def __init__(self):
        self._controller: Optional[Any] = None
        self._config: Dict[str, Any] = {}
        # Thresholds for phi_proxy_score
        self.meta_conscious_threshold: float = DEFAULT_THRESH_META
        self.conscious_threshold: float = DEFAULT_THRESH_CONSCIOUS
        self.pre_conscious_threshold: float = DEFAULT_THRESH_PRECONSCIOUS
        self.unconscious_threshold: float = DEFAULT_UNCONSCIOUS_THRESHOLD
        # Weights for OLD sub-metrics
        self.diff_weight_sources: float = DEFAULT_DIFF_WEIGHT_SOURCES
        self.diff_weight_lexical: float = DEFAULT_DIFF_WEIGHT_LEXICAL
        self.int_weight_shared_concepts: float = DEFAULT_INT_WEIGHT_SHARED_CONCEPTS
        self.phi_contrib_diff: float = DEFAULT_PHI_CONTRIB_DIFF
        self.phi_contrib_int: float = DEFAULT_PHI_CONTRIB_INT
        self.gw_capacity_for_norm: int = DEFAULT_GW_CAPACITY_FOR_NORM
        
        self.last_assessed_level_name: str = "UNINITIALIZED"
        self.last_phi_proxy_score: float = 0.0 # This will now be the combined score
        self._PhenomenalStateClass = globals().get('PhenomenalState')
        self._ConsciousStateEnum = globals().get('ConsciousState')

        # --- NEW: For _calculate_relationship_strength ---
        # Ideally, stopwords would be configurable or passed from ExperienceStream's config
        self._stopwords_cla: Set[str] = CLA_DEFAULT_STOPWORDS.copy() 
        # Could add goal_keywords from config or dynamically from active_goal
        self._goal_related_keywords_cla: Set[str] = {'goal', 'objective', 'target', 'achieve', 'complete', 'task'}
        # --- END NEW ---
        self.cla_coherence_edge_threshold: float = DEFAULT_CLA_COHERENCE_EDGE_THRESHOLD

    async def initialize(self, config: Dict[str, Any], controller: Any) -> bool:
        self._controller = controller
        cs_config = config.get("consciousness_assessor", {}) 
        if not cs_config: cs_config = config.get("consciousness_thresholds", {})
        self._config = cs_config

        self.meta_conscious_threshold = cs_config.get("meta_conscious_threshold", DEFAULT_THRESH_META)
        self.conscious_threshold = cs_config.get("conscious_threshold", DEFAULT_THRESH_CONSCIOUS)
        self.pre_conscious_threshold = cs_config.get("pre_conscious_threshold", DEFAULT_THRESH_PRECONSCIOUS)
        self.unconscious_threshold = cs_config.get("unconscious_threshold", DEFAULT_UNCONSCIOUS_THRESHOLD)

        self.diff_weight_sources = cs_config.get("diff_weight_sources", DEFAULT_DIFF_WEIGHT_SOURCES)
        self.diff_weight_lexical = cs_config.get("diff_weight_lexical", DEFAULT_DIFF_WEIGHT_LEXICAL)
        self.int_weight_shared_concepts = cs_config.get("int_weight_shared_concepts", DEFAULT_INT_WEIGHT_SHARED_CONCEPTS)
        self.phi_contrib_diff = cs_config.get("phi_contrib_diff", DEFAULT_PHI_CONTRIB_DIFF)
        self.phi_contrib_int = cs_config.get("phi_contrib_int", DEFAULT_PHI_CONTRIB_INT)
        
        self.gw_capacity_for_norm = cs_config.get(
            "global_workspace_capacity_for_norm",
            config.get("global_workspace", {}).get("capacity", DEFAULT_GW_CAPACITY_FOR_NORM)
        )
        if self.gw_capacity_for_norm <= 0: self.gw_capacity_for_norm = DEFAULT_GW_CAPACITY_FOR_NORM

        self.cla_coherence_edge_threshold = float(
            cs_config.get("cla_coherence_edge_threshold", DEFAULT_CLA_COHERENCE_EDGE_THRESHOLD)
        )
        if not (0.0 < self.cla_coherence_edge_threshold <= 1.0):
            logger_consciousness_assessor.warning(
                f"Invalid cla_coherence_edge_threshold ({self.cla_coherence_edge_threshold}). Using default {DEFAULT_CLA_COHERENCE_EDGE_THRESHOLD}."
            )
            self.cla_coherence_edge_threshold = DEFAULT_CLA_COHERENCE_EDGE_THRESHOLD

        # --- NEW: Load stopwords and goal keywords for CLA specific use ---
        cla_stopwords_config = cs_config.get("cla_stopwords", [])
        if isinstance(cla_stopwords_config, list):
            self._stopwords_cla.update(word.lower() for word in cla_stopwords_config if isinstance(word, str))
        
        cla_goal_keywords_config = cs_config.get("cla_goal_keywords", [])
        if isinstance(cla_goal_keywords_config, list):
            self._goal_related_keywords_cla.update(word.lower() for word in cla_goal_keywords_config if isinstance(word, str))
        # --- END NEW ---

        # Validate thresholds order
        if not (0 <= self.unconscious_threshold < self.pre_conscious_threshold <
                self.conscious_threshold < self.meta_conscious_threshold <= 1.0):
            logger_consciousness_assessor.error(
                "Invalid consciousness thresholds order or range. Using defaults."
            )
            # Reset all to defaults
            self.meta_conscious_threshold = DEFAULT_THRESH_META
            self.conscious_threshold = DEFAULT_THRESH_CONSCIOUS
            self.pre_conscious_threshold = DEFAULT_THRESH_PRECONSCIOUS
            self.unconscious_threshold = DEFAULT_UNCONSCIOUS_THRESHOLD

        logger_consciousness_assessor.info(
            f"CLA initialized. Thresholds (U/P/C/M): "
            f"{self.unconscious_threshold:.2f}/{self.pre_conscious_threshold:.2f}/"
            f"{self.conscious_threshold:.2f}/{self.meta_conscious_threshold:.2f}. "
            f"Φ-Proxy Weights (DiffSrc/Lex/SharedConcept/ContribDiff/ContribInt): "
            f"{self.diff_weight_sources:.2f}/{self.diff_weight_lexical:.2f}/"
            f"{self.int_weight_shared_concepts:.2f}/{self.phi_contrib_diff:.2f}/{self.phi_contrib_int:.2f}. "
            f"CLA CoherenceEdgeThr: {self.cla_coherence_edge_threshold:.2f}. " 
            f"CLA Stopwords: {len(self._stopwords_cla)}, CLA GoalKeywords: {len(self._goal_related_keywords_cla)}"
        )
        if not self._PhenomenalStateClass:
             logger_consciousness_assessor.warning("PhenomenalState class not loaded during CLA init.")
        if not self._ConsciousStateEnum:
             logger_consciousness_assessor.error("ConsciousState enum not loaded during CLA init. Assessment will be problematic.")

        return True

    def _has_self_reference(self, experience_content: Dict[str, Any], workspace_content: Dict[str, Any]) -> bool:
        """Check for self-referential content in experience or workspace."""
        # Simple keyword check for now. Can be expanded.
        self_ref_terms = ['self', 'my ', ' i ', ' me ', 'agent', 'consciousness', 'oscar', 'model', 'state']
        
        # Check experience content
        for key, value in experience_content.items():
            if isinstance(value, str):
                value_lower = value.lower()
                if any(term in value_lower for term in self_ref_terms):
                    logger_consciousness_assessor.debug(f"Self-reference found in experience content (key: {key})")
                    return True
        
        # Check workspace content
        for key, value_ws in workspace_content.items():
            item_text_content = ""
            if isinstance(value_ws, str): item_text_content = value_ws
            elif isinstance(value_ws, dict):
                if 'content' in value_ws and isinstance(value_ws['content'], str): item_text_content = value_ws['content']
                elif 'description' in value_ws and isinstance(value_ws['description'], str): item_text_content = value_ws['description']

            if item_text_content:
                content_lower = item_text_content.lower()
                if any(term in content_lower for term in self_ref_terms):
                    logger_consciousness_assessor.debug(f"Self-reference found in workspace content (key: {key})")
                    return True
            # Check if key itself is self-referential (e.g. 'self_model_status')
            if any(term in key.lower() for term in self_ref_terms):
                    logger_consciousness_assessor.debug(f"Self-reference found in workspace key: {key}")
                    return True
        return False

    def _normalize_and_tokenize(self, text: str) -> Set[str]:
        """Normalizes text to lowercase and tokenizes into a set of words, removing stopwords."""
        if not isinstance(text, str):
            return set()
        # Simple tokenization: find words, lowercase, remove stopwords
        words = re.findall(r'\b\w+\b', text.lower())
        return {word for word in words if word not in self._stopwords_cla}

    def _calculate_nonstring_relationship(self, content1: Any, content2: Any, 
                                          active_goal_tokens: Optional[Set[str]] = None,
                                          depth: int = 0) -> float:
        """
        Calculate relationship for non-string content types.
        (From User Proposal for C.2.2)
        """
        MAX_RECURSION_DEPTH_REL = 2 # Prevent deep recursion for complex objects
        if depth > MAX_RECURSION_DEPTH_REL:
            return 0.0

        # Type-based similarity
        if type(content1) != type(content2): # If types are different, low similarity
            return 0.1 # Small chance they are related via string form

        if isinstance(content1, dict) and isinstance(content2, dict):
            # For dictionaries, check key overlap and recursive value similarity
            keys1 = set(content1.keys())
            keys2 = set(content2.keys())
            common_keys = keys1.intersection(keys2)
            if not common_keys: return 0.0
            
            similarity_sum = 0.0
            for key in common_keys:
                # Recursively call relationship strength for values
                similarity_sum += self._calculate_relationship_strength(content1[key], content2[key], active_goal_tokens, depth + 1)
            
            # Jaccard for keys + average value similarity
            key_jaccard = len(common_keys) / (len(keys1) + len(keys2) - len(common_keys)) if (len(keys1) + len(keys2) - len(common_keys)) > 0 else 0.0
            avg_value_similarity = similarity_sum / len(common_keys) if common_keys else 0.0
            return (key_jaccard * 0.4 + avg_value_similarity * 0.6)

        elif isinstance(content1, (list, tuple)) and isinstance(content2, (list, tuple)):
            # For lists/tuples, simplistic: compare lengths and sample a few items
            if not content1 or not content2: return 0.0 if content1 != content2 else 1.0
            len_similarity = 1.0 - abs(len(content1) - len(content2)) / max(len(content1), len(content2))
            
            sample_similarity_sum = 0.0
            num_samples = min(3, len(content1), len(content2))
            if num_samples > 0:
                for i in range(num_samples): # Compare first few items
                    sample_similarity_sum += self._calculate_relationship_strength(content1[i], content2[i], active_goal_tokens, depth + 1)
                avg_sample_similarity = sample_similarity_sum / num_samples
                return (len_similarity * 0.3 + avg_sample_similarity * 0.7)
            return len_similarity * 0.3 # If no samples to compare (e.g. one list empty but not both)


        elif hasattr(content1, "__dict__") and hasattr(content2, "__dict__"):
            return self._calculate_nonstring_relationship(content1.__dict__, content2.__dict__, active_goal_tokens, depth + 1)
        
        return self._calculate_relationship_strength(str(content1), str(content2), active_goal_tokens, depth + 1) * 0.7

    def _calculate_relationship_strength(self, 
                                         content1: Any, 
                                         content2: Any, 
                                         active_goal_tokens: Optional[Set[str]] = None, 
                                         depth: int = 0 
                                        ) -> float:
        logger_consciousness_assessor.debug(
            f"CLA_REL_STRENGTH (Depth {depth}): Calculating relationship between: "
            f"C1_Type={type(content1)}, C1_Content='{str(content1)[:70]}...' AND "
            f"C2_Type={type(content2)}, C2_Content='{str(content2)[:70]}...' "
            f"GoalTokens: {str(list(active_goal_tokens))[:70] if active_goal_tokens else 'None'}"
        )
        MAX_RECURSION_DEPTH_STR = 3
        if depth > MAX_RECURSION_DEPTH_STR:
            return 0.0

        if not (isinstance(content1, str) and isinstance(content2, str)):
            logger_consciousness_assessor.debug(
                f"CLA_REL_STRENGTH (Depth {depth}): Not both strings. Delegating to _calculate_nonstring_relationship."
            )
            return self._calculate_nonstring_relationship(content1, content2, active_goal_tokens, depth)

        str1_lower = content1.lower() 
        str2_lower = content2.lower() 

        tokens1 = self._normalize_and_tokenize(str1_lower) 
        tokens2 = self._normalize_and_tokenize(str2_lower) 
        logger_consciousness_assessor.debug(
            f"CLA_REL_STRENGTH (Depth {depth}): String Tokens1 (count {len(tokens1)}): {str(list(tokens1))[:100]}..."
        )
        logger_consciousness_assessor.debug(
            f"CLA_REL_STRENGTH (Depth {depth}): String Tokens2 (count {len(tokens2)}): {str(list(tokens2))[:100]}..."
        )
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection_count = len(tokens1.intersection(tokens2))
        union_count = len(tokens1.union(tokens2))
        base_similarity = intersection_count / union_count if union_count > 0 else 0.0
        logger_consciousness_assessor.debug(
            f"CLA_REL_STRENGTH (Depth {depth}): Intersection={intersection_count}, Union={union_count}, BaseSimilarity (Jaccard)={base_similarity:.3f}"
        )
        
        keyword_bonus = 0.0
        combined_generic_goal_keywords_found = (tokens1.intersection(self._goal_related_keywords_cla)).union(
                                               tokens2.intersection(self._goal_related_keywords_cla))
        keyword_bonus += min(0.1, len(combined_generic_goal_keywords_found) * 0.05) 

        if active_goal_tokens and isinstance(active_goal_tokens, set) and active_goal_tokens:
            active_goal_overlap1 = tokens1.intersection(active_goal_tokens)
            active_goal_overlap2 = tokens2.intersection(active_goal_tokens)
            combined_active_goal_keywords_found = active_goal_overlap1.union(active_goal_overlap2)
            active_goal_bonus = min(0.2, len(combined_active_goal_keywords_found) * 0.1) 
            keyword_bonus += active_goal_bonus
        
        logger_consciousness_assessor.debug(
            f"CLA_REL_STRENGTH (Depth {depth}): KeywordBonus (generic goal + active goal)={keyword_bonus:.3f}"
        )
        final_similarity = min(1.0, base_similarity + keyword_bonus)
        logger_consciousness_assessor.debug(
            f"CLA_REL_STRENGTH (Depth {depth}): Final Relationship Strength = {final_similarity:.3f}"
        )
        return final_similarity

    def measure_workspace_coherence(self, workspace_content: Dict[str, Any]) -> float:
        """
        Measures how coherently the elements in the global workspace relate to each other.
        (From User Proposal for C.2.2)
        """
        if not NETWORKX_AVAILABLE:
            logger_consciousness_assessor.warning("NetworkX not available, cannot measure workspace coherence. Returning 0.0.")
            return 0.0
        if not workspace_content or len(workspace_content) < 2:
            return 0.0 
        logger_consciousness_assessor.debug(
            f"CLA_COHERENCE: Starting coherence calculation for {len(workspace_content)} workspace items. "
            f"Keys: {list(workspace_content.keys())}"
        )

        active_goal_tokens_for_coherence: Optional[Set[str]] = None
        if self._controller and hasattr(self._controller, '_oscar_get_active_goal'):
            active_goal_obj = self._controller._oscar_get_active_goal() # type: ignore
            if active_goal_obj and hasattr(active_goal_obj, 'description') and isinstance(active_goal_obj.description, str):
                active_goal_tokens_for_coherence = self._normalize_and_tokenize(active_goal_obj.description)
                logger_consciousness_assessor.debug(f"Workspace Coherence: Using active goal tokens: {list(active_goal_tokens_for_coherence)[:5]}")

        graph = nx.Graph()
        item_keys = list(workspace_content.keys())

        for key in item_keys:
            graph.add_node(key) 
        
        for i in range(len(item_keys)):
            for j in range(i + 1, len(item_keys)):
                key1 = item_keys[i]
                key2 = item_keys[j]
                content1 = workspace_content[key1]
                content2 = workspace_content[key2]
                
                relationship_strength = self._calculate_relationship_strength(content1, content2, active_goal_tokens_for_coherence)
                logger_consciousness_assessor.debug(
                    f"CLA_COHERENCE: Pair ('{str(key1)[:20]}...', '{str(key2)[:20]}...'): Strength = {relationship_strength:.3f}"
                )
                # USE CONFIGURABLE INSTANCE ATTRIBUTE
                if relationship_strength > self.cla_coherence_edge_threshold: 
                # END CHANGE
                    graph.add_edge(key1, key2, weight=relationship_strength)
                    logger_consciousness_assessor.info(
                        f"CLA_COHERENCE_GRAPH: Added Edge! ('{str(key1)[:20]}...' --({relationship_strength:.2f})-- '{str(key2)[:20]}...')"
                    )
        
        nodes_list = list(graph.nodes())
        edges_list = list(graph.edges(data=True)) # Get edges with weights
        logger_consciousness_assessor.info(
            f"CLA_COHERENCE_GRAPH: Graph built. Nodes ({len(nodes_list)}): {str(nodes_list)[:150]}... "
            f"Edges ({len(edges_list)}): {str([(u,v,d.get('weight','N/A')) for u,v,d in edges_list])[:250]}..."
        )
        if not graph.edges(): 
            logger_consciousness_assessor.debug("Workspace Coherence: No edges formed in graph (all relationships below threshold). Coherence=0.0")
            return 0.0
            
        num_nodes = len(graph.nodes)
        if num_nodes < 2: 
            return 0.0
            
        try:
            if num_nodes > 0 :
                 clustering = nx.average_clustering(graph, weight="weight") if graph.edges() else 0.0
            else: clustering = 0.0
        except Exception as e_clust: 
            logger_consciousness_assessor.warning(f"Error calculating average_clustering: {e_clust}. Defaulting to 0.")
            clustering = 0.0
        
        normalized_path = 0.0
        if nx.is_connected(graph):
            try:
                path_length = nx.average_shortest_path_length(graph, weight="weight")
                normalized_path = 1.0 / (1.0 + path_length) if path_length > 0 else 1.0
            except nx.NetworkXError: 
                normalized_path = 0.0 
        else: 
            if num_nodes > 0 and graph.edges(): 
                try:
                    largest_cc = max(nx.connected_components(graph), key=len, default=None)
                    if largest_cc and len(largest_cc) > 1:
                        subgraph = graph.subgraph(largest_cc)
                        path_length_cc = nx.average_shortest_path_length(subgraph, weight="weight")
                        normalized_path_cc = 1.0 / (1.0 + path_length_cc) if path_length_cc > 0 else 1.0
                        normalized_path = normalized_path_cc * (len(largest_cc) / num_nodes)
                    else: 
                        normalized_path = 0.0
                except Exception as e_path: 
                    logger_consciousness_assessor.warning(f"Error calculating avg_shortest_path on largest_cc: {e_path}. Defaulting normalized_path to 0.")
                    normalized_path = 0.0
            else: 
                 normalized_path = 0.0


        coherence = (clustering + normalized_path) / 2.0
        coherence = min(1.0, max(0.0, coherence)) 
        
        logger_consciousness_assessor.debug(
            f"Workspace Coherence: Nodes={num_nodes}, Edges={len(graph.edges())}, "
            f"Clustering={clustering:.3f}, NormPath={normalized_path:.3f} -> Coherence={coherence:.3f}"
        )
        return coherence

    def calculate_differentiation_integration(self, 
                                            phenomenal_state: Optional['PhenomenalState'], # type: ignore
                                            workspace_content: Dict[str, Any]
                                           ) -> float:
        """
        Calculate a combined metric for differentiation (complexity of phenomenal state) 
        and integration (coherence of workspace content).
        (Based on User Proposal for C.2.3)
        
        Returns:
        - DI value between 0.0 and 1.0
        """
        _PState_di = globals().get('PhenomenalState')
        if not _PState_di or not isinstance(phenomenal_state, _PState_di): # type: ignore
            logger_consciousness_assessor.debug("DI_Calc: Invalid or missing PhenomenalState. Differentiation defaults to 0.")
            differentiation = 0.0
        elif not hasattr(phenomenal_state, 'content') or not isinstance(phenomenal_state.content, dict): # type: ignore
            logger_consciousness_assessor.debug("DI_Calc: PhenomenalState.content missing or not a dict. Differentiation defaults to 0.")
            differentiation = 0.0
        else:
            diff_normalization_factor = float(self._config.get("differentiation_norm_factor", 20.0))
            if diff_normalization_factor <= 0: diff_normalization_factor = 20.0
            
            num_distinct_items_in_pstate = len(phenomenal_state.content.keys()) # type: ignore
            differentiation = min(1.0, num_distinct_items_in_pstate / diff_normalization_factor)
            logger_consciousness_assessor.debug(f"DI_Calc: Differentiation = {differentiation:.3f} (Items: {num_distinct_items_in_pstate}, NormFactor: {diff_normalization_factor})")

        integration = self.measure_workspace_coherence(workspace_content)
        logger_consciousness_assessor.debug(f"DI_Calc: Integration (Workspace Coherence) = {integration:.3f}")
        
        effective_information_proxy = differentiation * integration
        
        logger_consciousness_assessor.debug(
            f"DI_Calc: Differentiation={differentiation:.3f} * Integration={integration:.3f} -> DI_Value={effective_information_proxy:.3f}"
        )
        return round(effective_information_proxy, 3) 

    async def assess_consciousness_level(self,
                                         experience: Optional['PhenomenalState'], # type: ignore
                                         workspace_content: Dict[str, Any]
                                        ) -> 'ConsciousState': # type: ignore
        if not self._ConsciousStateEnum:
            logger_consciousness_assessor.error("ConsciousState enum not defined. Cannot assess level.")
            class MockState: value = 0; name = "UNKNOWN_ENUM_MISSING"; # type: ignore
            return MockState() # type: ignore

        current_cs_level = self._ConsciousStateEnum.UNCONSCIOUS # Default
        final_score_for_thresholding = 0.0 # This will now be based on DI_Value

        if experience is None or not (isinstance(experience, self._PhenomenalStateClass if self._PhenomenalStateClass else dict) or isinstance(experience, dict)): # type: ignore
            logger_consciousness_assessor.debug("Assessing: No valid phenomenal state, defaulting to UNCONSCIOUS.")
            self.last_phi_proxy_score = final_score_for_thresholding # Update what we call "phi_proxy_score" for status
            self.last_assessed_level_name = current_cs_level.name
            return current_cs_level

        di_value = self.calculate_differentiation_integration(experience, workspace_content)
        final_score_for_thresholding = di_value
        
        distinct_sources_old = getattr(experience, 'distinct_source_count', 0) if self._PhenomenalStateClass and isinstance(experience, self._PhenomenalStateClass) else experience.get('distinct_source_count',0) # type: ignore
        lexical_diversity_old = getattr(experience, 'content_diversity_lexical', 0.0) if self._PhenomenalStateClass and isinstance(experience, self._PhenomenalStateClass) else experience.get('content_diversity_lexical',0.0) # type: ignore
        shared_concepts_old = getattr(experience, 'shared_concept_count_gw', 0.0) if self._PhenomenalStateClass and isinstance(experience, self._PhenomenalStateClass) else experience.get('shared_concept_count_gw',0.0) # type: ignore

        logger_consciousness_assessor.debug(
            f"New DI_Value for CS assessment: {final_score_for_thresholding:.3f}. "
            f"(Old sub-metrics for ref: DSrc={distinct_sources_old}, LexDiv={lexical_diversity_old:.3f}, ShCon={shared_concepts_old:.3f})"
        )
        self.last_phi_proxy_score = final_score_for_thresholding 

        experience_content_for_self_ref = getattr(experience, 'content', {}) if self._PhenomenalStateClass and isinstance(experience, self._PhenomenalStateClass) else experience.get('content', {}) # type: ignore

        if final_score_for_thresholding >= self.meta_conscious_threshold:
            if self._has_self_reference(experience_content_for_self_ref, workspace_content): # type: ignore
                current_cs_level = self._ConsciousStateEnum.META_CONSCIOUS
            else:
                current_cs_level = self._ConsciousStateEnum.CONSCIOUS
                logger_consciousness_assessor.debug("High DI_Value but no self-reference; assessed as CONSCIOUS.")
        elif final_score_for_thresholding >= self.conscious_threshold:
            current_cs_level = self._ConsciousStateEnum.CONSCIOUS
        elif final_score_for_thresholding >= self.pre_conscious_threshold:
            current_cs_level = self._ConsciousStateEnum.PRE_CONSCIOUS
        elif final_score_for_thresholding >= self.unconscious_threshold:
            current_cs_level = self._ConsciousStateEnum.UNCONSCIOUS
        else: 
            current_cs_level = self._ConsciousStateEnum.UNCONSCIOUS 
            logger_consciousness_assessor.debug(f"DI_Value score {final_score_for_thresholding:.3f} is below unconscious_threshold {self.unconscious_threshold:.3f}.")

        if current_cs_level.name != self.last_assessed_level_name: # type: ignore
            logger_consciousness_assessor.info(
                f"Assessed consciousness level changed to: {current_cs_level.name} " # type: ignore
                f"(New DI_Value Score: {final_score_for_thresholding:.3f})"
            )
        self.last_assessed_level_name = current_cs_level.name # type: ignore
        
        return current_cs_level


    async def process(self, input_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not input_state:
            logger_consciousness_assessor.warning("ConsciousnessAssessor process: Missing input state.")
            # Return default state if no input
            return {"conscious_state": self._ConsciousStateEnum.UNCONSCIOUS if self._ConsciousStateEnum else "UNAVAILABLE",
                    "phi_proxy_score": 0.0}

        experience = input_state.get("experience") # Can be PState object or dict
        workspace_content = input_state.get("workspace_content", {})

        if self._PhenomenalStateClass: # Only validate if PState class is loaded
            if experience is not None and not isinstance(experience, (self._PhenomenalStateClass, dict)):
                logger_consciousness_assessor.error(
                    f"CLA process: Invalid type for 'experience', expected {self._PhenomenalStateClass} or dict."
                )
                return {"conscious_state": self._ConsciousStateEnum.UNCONSCIOUS if self._ConsciousStateEnum else "UNAVAILABLE",
                        "phi_proxy_score": 0.0}
        elif experience is not None and not isinstance(experience, dict): # Fallback if PState class missing
             logger_consciousness_assessor.error("CLA process: Invalid type for 'experience' (expected dict).")
             return {"conscious_state": self._ConsciousStateEnum.UNCONSCIOUS if self._ConsciousStateEnum else "UNAVAILABLE",
                        "phi_proxy_score": 0.0}


        if not isinstance(workspace_content, dict):
            logger_consciousness_assessor.error("ConsciousnessAssessor process: Invalid type for 'workspace_content'.")
            return {"conscious_state": self._ConsciousStateEnum.UNCONSCIOUS if self._ConsciousStateEnum else "UNAVAILABLE",
                        "phi_proxy_score": 0.0}

        conscious_state = await self.assess_consciousness_level(experience, workspace_content)
        
        # Return the result including the phi_proxy_score
        return {"conscious_state": conscious_state, "phi_proxy_score": self.last_phi_proxy_score}

    async def reset(self) -> None:
        self.last_assessed_level_name = "RESET"
        self.last_phi_proxy_score = 0.0
        logger_consciousness_assessor.info("ConsciousnessLevelAssessor reset.")

    async def get_status(self) -> Dict[str, Any]:
        return {
            "component": "ConsciousnessLevelAssessor",
            "status": "operational",
            "last_assessed_level": self.last_assessed_level_name,
            "last_main_score (DI_Value)": round(self.last_phi_proxy_score, 3), # Clarify this is the DI_Value
            "thresholds": { # These thresholds now apply to the DI_Value
                "meta_conscious": self.meta_conscious_threshold,
                "conscious": self.conscious_threshold,
                "pre_conscious": self.pre_conscious_threshold,
                "unconscious": self.unconscious_threshold
            },
            "cla_coherence_edge_threshold": self.cla_coherence_edge_threshold, # Add to status
            # Old phi_proxy_weights are less relevant now, could be removed or kept for reference
            # "phi_proxy_weights_OLD": {
            #     "diff_sources": self.diff_weight_sources, 
            #     # ...
            # }
        }

    async def shutdown(self) -> None:
        logger_consciousness_assessor.info("ConsciousnessLevelAssessor shutting down.")