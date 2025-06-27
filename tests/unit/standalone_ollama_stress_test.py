import asyncio
import logging
import time
import random
import json # For formatting the messages if needed, though call_ollama expects dicts
from typing import Any, Dict, List, Tuple, Optional

# --- BEGIN COPIED/ADAPTED external_comms.py ---
# This section should be a direct copy of your project's
# consciousness_experiment/external_comms.py content,
# or at least the call_ollama function and its direct dependencies.
# Make sure OLLAMA_AVAILABLE is correctly set.

# consciousness_experiment/cognitive_modules/experience_stream.py

import asyncio
import logging
import time
import re # For lexical diversity tokenization
from collections import Counter # For shared concept counting
from typing import Dict, Any, Optional, List, Set

# --- Use standard relative imports ---
try:
    from ..protocols import ExperienceIntegrator # Import the specific protocol
    from ..protocols import CognitiveComponent # For type checking if needed
    from ..models.datatypes import PhenomenalState # Import PhenomenalState datatype
except ImportError:
    # Fallback for different execution context (e.g., combined script)
    logging.warning("ExperienceStream: Relative imports failed, relying on globally defined types.")
    if 'ExperienceIntegrator' not in globals(): raise ImportError("ExperienceIntegrator not found via relative import or globally")
    if 'CognitiveComponent' not in globals(): raise ImportError("CognitiveComponent not found via relative import or globally")
    if 'PhenomenalState' not in globals(): raise ImportError("PhenomenalState not found via relative import or globally")
    ExperienceIntegrator = globals().get('ExperienceIntegrator')
    PhenomenalState = globals().get('PhenomenalState')
    # CognitiveComponent is implicitly included via ExperienceIntegrator inheritance


logger_experience_stream = logging.getLogger(__name__)

# Default config values
DEFAULT_INTENSITY_FACTOR = 0.8
DEFAULT_VALENCE_DECAY = 0.1
# For lexical diversity, a common set of English stopwords (can be expanded or configured)
DEFAULT_STOPWORDS = set([
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "should", "can",
    "could", "may", "might", "must", "and", "but", "or", "nor", "for", "so", "yet",
    "in", "on", "at", "to", "from", "by", "with", "about", "above", "after", "again",
    "against", "all", "am", "as", "because", "before", "below", "between", "both",
    "down", "during", "each", "few", "further", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "into", "it", "its", "itself", "me",
    "more", "most", "my", "myself", "no", "not", "now", "of", "off", "once", "only",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "some",
    "such", "than", "that", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "too", "under", "until", "up", "very",
    "we", "what", "when", "where", "which", "while", "who", "whom", "why", "you", "your",
    "yours", "yourself", "yourselves"
])


class ExperienceStream(ExperienceIntegrator):
    """
    Integrates perceptions, memories, and context into a unified PhenomenalState,
    including calculation of sub-metrics for Φ-Proxy.
    """

    def __init__(self):
        self._controller: Optional[Any] = None
        self._config: Dict[str, Any] = {}
        self._current_valence: float = 0.0
        self._intensity_factor: float = DEFAULT_INTENSITY_FACTOR
        self._valence_decay: float = DEFAULT_VALENCE_DECAY
        self._stopwords: Set[str] = DEFAULT_STOPWORDS.copy() # Use a copy to allow modification
        self._PhenomenalStateClass = globals().get('PhenomenalState')

    async def initialize(self, config: Dict[str, Any], controller: Any) -> bool:
        self._controller = controller
        exp_config = config.get("experience_stream", {})
        self._config = exp_config

        self._intensity_factor = exp_config.get("intensity_factor", DEFAULT_INTENSITY_FACTOR)
        self._valence_decay = exp_config.get("valence_decay", DEFAULT_VALENCE_DECAY)
        
        # Load custom stopwords from config if provided, appending to defaults
        custom_stopwords = exp_config.get("custom_stopwords", [])
        if isinstance(custom_stopwords, list):
            self._stopwords.update(word.lower() for word in custom_stopwords if isinstance(word, str))

        logger_experience_stream.info(
            f"ExperienceStream initialized. IntensityFactor: {self._intensity_factor:.2f}, "
            f"ValenceDecay: {self._valence_decay:.2f}, Stopwords: {len(self._stopwords)}"
        )
        return True

    def _extract_textual_content_from_item(self, item_data_val: Any, depth: int = 0, max_depth: int = 2) -> List[str]:
        """
        Recursively extracts textual content from various data structures.
        """
        logger_experience_stream.debug(
            f"ES_EXTRACT_TEXT (Depth {depth}): Processing item_data_val of type: {type(item_data_val)}. "
            f"Content (first 100 chars): {str(item_data_val)[:100]}"
        )
        texts: List[str] = []
        if depth > max_depth:
            return texts

        if isinstance(item_data_val, str):
            logger_experience_stream.debug(
                f"ES_EXTRACT_TEXT (Depth {depth}): Found string: '{item_data_val[:100]}...'"
            )
            texts.append(item_data_val.lower())
        # NEW REPLACEMENT for the dict handling block:
        elif isinstance(item_data_val, dict):
            for key, value in item_data_val.items():
                logger_experience_stream.debug(
                    f"ES_EXTRACT_TEXT (Depth {depth}): Dict iteration - Key: '{key}', Value Type: {type(value)}"
                )
                # 1. Add key as text (converted to words)
                # Example: "cpu_percent" -> "cpu", "percent"
                key_words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|[\d\.]+', key) # Split camelCase/snake_case, keep numbers
                key_text_cleaned = " ".join(k.lower() for k in key_words if k.lower() not in self._stopwords and len(k) > 1) # Min length for key parts
                if key_text_cleaned:
                    texts.append(key_text_cleaned)
                    logger_experience_stream.debug(
                        f"ES_EXTRACT_TEXT (Depth {depth}): Dict added cleaned key text: '{key_text_cleaned}' from original key '{key}'"
                    )

                # 2. Process value based on its type
                if isinstance(value, str):
                    logger_experience_stream.debug(
                        f"ES_EXTRACT_TEXT (Depth {depth}): Dict value for key '{key}' is string: '{value[:70]}...'"
                    )
                    texts.append(value.lower())
                elif isinstance(value, (dict, list)): # Recurse for nested dicts/lists
                    logger_experience_stream.debug(
                        f"ES_EXTRACT_TEXT (Depth {depth}): Dict recursing for key '{key}' with value of type {type(value)}."
                    )
                    texts.extend(self._extract_textual_content_from_item(value, depth + 1, max_depth))
                elif isinstance(value, (int, float, bool)): # Convert simple numerics/bools to string
                    str_value = str(value).lower()
                    texts.append(str_value)
                    logger_experience_stream.debug(
                        f"ES_EXTRACT_TEXT (Depth {depth}): Dict value for key '{key}' is numeric/bool, added as string: '{str_value}'"
                    )
                # Add other specific type handling if needed, e.g., for Predicate objects if they appear as values
                elif self._PhenomenalStateClass and isinstance(value, self._PhenomenalStateClass): # Check if it's a PState object
                     logger_experience_stream.debug(
                        f"ES_EXTRACT_TEXT (Depth {depth}): Dict recursing for key '{key}' with PhenomenalState value."
                    )
                     if hasattr(value, 'content') and isinstance(getattr(value, 'content'), dict):
                         texts.extend(self._extract_textual_content_from_item(getattr(value, 'content'), depth + 1, max_depth))
        # END NEW REPLACEMENT
        # NEW REPLACEMENT for the list processing block:
        elif isinstance(item_data_val, list):
            for sub_item in item_data_val:
                logger_experience_stream.debug(
                    f"ES_EXTRACT_TEXT (Depth {depth}): List processing sub_item of type {type(sub_item)}."
                )
                if isinstance(sub_item, str): # If sub_item is a string, add it directly
                    texts.append(sub_item.lower())
                    logger_experience_stream.debug(
                        f"ES_EXTRACT_TEXT (Depth {depth}): List directly added string sub_item: '{sub_item[:70]}...'"
                    )
                elif depth < max_depth: # Only recurse if not yet at max_depth for non-strings
                    logger_experience_stream.debug(
                        f"ES_EXTRACT_TEXT (Depth {depth}): List recursing for sub_item of type {type(sub_item)} (next depth {depth + 1})."
                    )
                    texts.extend(self._extract_textual_content_from_item(sub_item, depth + 1, max_depth))
                # If depth == max_depth and sub_item is not a string, it won't be processed further by this logic,
                # which is intended to stop deep recursion for complex objects at max_depth.
                # Basic numerics/bools at max_depth inside lists would also be missed by this specific block.
                # The standalone numeric/bool handler later in the function might catch them if they appear as top-level items.
        # END NEW REPLACEMENT
        elif isinstance(item_data_val, (float, int, bool)): # Check for int as well
            str_val = str(item_data_val).lower()
            texts.append(str_val)
            logger_experience_stream.debug(
                f"ES_EXTRACT_TEXT (Depth {depth}): Item is numeric/bool, added as string: '{str_val}'"
            )
        # Add handling for other types if necessary, e.g., PhenomenalState objects directly
        elif self._PhenomenalStateClass and isinstance(item_data_val, self._PhenomenalStateClass): # Check if class is available
            if hasattr(item_data_val, 'content') and isinstance(getattr(item_data_val, 'content'), dict):
                 texts.extend(self._extract_textual_content_from_item(getattr(item_data_val, 'content'), depth + 1, max_depth))
            # Could also extract from its valence, intensity if desired as text, e.g. "valence_high"
        
        logger_experience_stream.debug(
            f"ES_EXTRACT_TEXT (Depth {depth}): Returning {len(texts)} text strings: {str(texts)[:200]}..."
        )
        return texts

    async def integrate_experience(self,
                                   percepts: Dict[str, Any],
                                   memories: List[Any],
                                   context: Dict[str, Any],
                                   broadcast_content: Dict[str, Any]
                                  ) -> 'PhenomenalState':
        current_time = time.time()
        logger_experience_stream.debug("Integrating experience...")
        _PhenomenalStateClass = globals().get('PhenomenalState')

        # --- 1. Combine Content & Identify Sources ---
        integrated_content = broadcast_content.copy()
        source_types_present: Set[str] = set() # For distinct_source_count

        if broadcast_content:
            # Assuming broadcast_content keys might indicate source (e.g., "percept_...", "goal_...")
            for key in broadcast_content:
                if key.startswith("percept_"): source_types_present.add("percept")
                elif key.startswith("goal_"): source_types_present.add("goal")
                elif key.startswith("memory_") or key == "last_experience": source_types_present.add("memory")
                elif key.startswith("prediction_"): source_types_present.add("prediction")
                # Add other identifiable prefixes as needed

        if isinstance(percepts, dict):
            if percepts.get("user_input"):
                integrated_content["user_input_percept"] = percepts["user_input"]
                source_types_present.add("user_input_percept_source") # Distinct source
            if percepts.get("internal_error"):
                integrated_content["internal_error_percept"] = percepts["internal_error"]
                source_types_present.add("internal_error_percept_source") # Distinct source
            if percepts.get("system_state"): # Consider system state a distinct source
                 integrated_content["system_state_percept"] = percepts["system_state"]
                 source_types_present.add("system_state_percept_source")


        if memories and isinstance(memories, list):
            integrated_content["relevant_memories"] = memories[:3] # Limit display
            if memories: source_types_present.add("retrieved_memory_source")

        if isinstance(context, dict) and context: # Ensure context is not empty
            integrated_content["action_context"] = context
            source_types_present.add("action_context_source")

        distinct_source_count = len(source_types_present)
        logger_experience_stream.debug(f"Distinct source count: {distinct_source_count} from sources: {source_types_present}")

        # --- 2. Calculate Intensity ---
        # (Logic remains similar, potentially refined based on actual workspace_weights format)
        focus_intensity = 0.0
        workspace_capacity = 7 # Default
        if self._controller and hasattr(self._controller, 'global_workspace'):
            workspace_ref = self._controller.global_workspace
            # Assuming workspace_weights is available from GWM if needed, or use broadcast_content size
            if hasattr(workspace_ref, 'workspace_weights') and isinstance(workspace_ref.workspace_weights, dict):
                focus_intensity = sum(workspace_ref.workspace_weights.values())
            else: # Fallback if workspace_weights not directly accessible
                focus_intensity = len(broadcast_content)
            if hasattr(workspace_ref, 'capacity'):
                workspace_capacity = max(1, workspace_ref.capacity)
        else: # Fallback if no controller or workspace
            focus_intensity = len(broadcast_content)

        intensity = min(1.0, focus_intensity / workspace_capacity if workspace_capacity > 0 else 0.0) * self._intensity_factor
        logger_experience_stream.debug(f"Calculated Intensity: {intensity:.3f} (Focus: {focus_intensity:.2f}, Capacity: {workspace_capacity})")


        # --- 3. Calculate Valence ---
        # Get current pain and happiness levels from the controller
        current_total_pain_level_es = 0.0
        current_happiness_level_es = 5.0 # Default to baseline happiness
        if self._controller:
            current_total_pain_level_es = getattr(self._controller, 'pain_level', 0.0)
            current_happiness_level_es = getattr(self._controller, 'happiness_level', 5.0)
        
        # Start with valence based on last action outcome and internal errors (existing logic)
        self._current_valence *= (1.0 - self._valence_decay) # Apply decay to internal valence accumulator
        
        last_outcome = context.get("last_action_outcome", "unknown") if isinstance(context, dict) else "unknown"
        valence_shift_from_events = 0.0
        if last_outcome == "success": 
            valence_shift_from_events += 0.1
        elif last_outcome == "failure": 
            valence_shift_from_events -= 0.2
        if isinstance(percepts, dict) and percepts.get("internal_error"): 
            valence_shift_from_events -= 0.3
        
        self._current_valence = max(-1.0, min(1.0, self._current_valence + valence_shift_from_events))
        
        # Now, modulate this event-driven valence by overall pain and happiness
        # These factors can be made configurable later
        pain_effect_on_valence = -current_total_pain_level_es * 0.05  # e.g., each point of pain reduces valence by 0.05
        
        # Happiness effect: if happiness is above baseline, positive effect; if below, negative.
        happiness_baseline_for_valence_effect = 5.0 # Could be config: self._controller.config.get("internal_states",{}).get("happiness_baseline_target", 5.0)
        happiness_deviation = current_happiness_level_es - happiness_baseline_for_valence_effect
        happiness_effect_on_valence = happiness_deviation * 0.02 # e.g., each point of happiness deviation from baseline shifts valence by 0.02

        final_valence = self._current_valence + pain_effect_on_valence + happiness_effect_on_valence
        valence = max(-1.0, min(1.0, final_valence)) # Clamp final valence

        logger_experience_stream.debug(
            f"ES_PHP_VALENCE - FinalValence: {valence:.3f} (EventValenceAccum: {self._current_valence:.2f}, "
            f"PainEffect: {pain_effect_on_valence:.2f} from ControllerPainLvl {current_total_pain_level_es:.2f}, "
            f"HappyEffect: {happiness_effect_on_valence:.2f} from ControllerHappyLvl {current_happiness_level_es:.2f})"
        )

        # --- 4. Calculate Integration Level (Old Placeholder - will be less important) ---
        integration_level_old_proxy = min(1.0, distinct_source_count / 4.0) # Max 4 types considered before
        logger_experience_stream.debug(f"Old Integration Level Proxy: {integration_level_old_proxy:.3f} (Sources: {distinct_source_count})")

        # --- 5. Calculate Attention Weight (Placeholder - remains similar) ---
        attention_weight = intensity
        logger_experience_stream.debug(f"Calculated Attention Weight: {attention_weight:.3f}")

        # --- 6. Calculate New Φ-Proxy Sub-Metrics ---
        # A. Content Diversity (Lexical - TTR)
        all_text_for_diversity_flat: List[str] = []
        for item_id, item_data_val_div in integrated_content.items():
            # Use the new helper to extract all strings
            all_text_for_diversity_flat.extend(self._extract_textual_content_from_item(item_data_val_div))
        
        logger_experience_stream.debug(f"ES LexDiv: Texts collected: '{('; '.join(all_text_for_diversity_flat))[:200]}...'") # Log collected text

        content_diversity_lexical = 0.0
        if all_text_for_diversity_flat:
            full_text = " ".join(all_text_for_diversity_flat)
            logger_experience_stream.debug(f"ES_LEXDIV_DEBUG - Full text for TTR (len {len(full_text)}): '{full_text[:300]}...'")
            # NEW REPLACEMENT BLOCK:
            raw_tokens = re.findall(r'\b\w+\b', full_text) # Tokenize (already lowercased from _extract_textual_content_from_item)
            logger_experience_stream.debug(f"ES_LEXDIV_DEBUG - Raw tokens before stopwords (count {len(raw_tokens)}): {str(raw_tokens)[:200]}...")

            words_after_stopwords = [word for word in raw_tokens if word not in self._stopwords]
            logger_experience_stream.debug(
                f"ES_LEXDIV_DEBUG - Tokens after stopwords (count {len(words_after_stopwords)}): {str(words_after_stopwords)[:200]}..."
            )

            if words_after_stopwords: # Use words_after_stopwords for TTR
                unique_words = set(words_after_stopwords)
                content_diversity_lexical = len(unique_words) / len(words_after_stopwords) if len(words_after_stopwords) > 0 else 0.0
                logger_experience_stream.debug(
                    f"ES_LEXDIV_DEBUG - Final for TTR: Unique words={len(unique_words)}, Total words (after stop)={len(words_after_stopwords)}, TTR={content_diversity_lexical:.3f}"
                )
            else:
                logger_experience_stream.debug(f"ES_LEXDIV_DEBUG - No words left after stopword removal for TTR calculation.")
            # END NEW REPLACEMENT BLOCK

        logger_experience_stream.debug(f"Calculated Content Diversity (Lexical TTR): {content_diversity_lexical:.3f}")

        # B. Shared Concept Count (GW)
        item_contents_tokenized: List[Set[str]] = []
        for item_id_bc, item_data_bc in broadcast_content.items():
            # Use the new helper to extract all strings from this broadcast item
            texts_for_this_item = self._extract_textual_content_from_item(item_data_bc)
            logger_experience_stream.debug(f"ES_SHARED_CONCEPTS_DEBUG - Item '{item_id_bc}': Extracted texts for tokenization: {str(texts_for_this_item)[:200]}...")
            
            if texts_for_this_item:
                full_text_for_item = " ".join(texts_for_this_item)
                item_words_bc = set(word for word in re.findall(r'\b\w+\b', full_text_for_item) if word not in self._stopwords)
                logger_experience_stream.debug(f"ES_SHARED_CONCEPTS_DEBUG - Item '{item_id_bc}': Final token set (count {len(item_words_bc)}): {str(list(item_words_bc))[:150]}...")
                if item_words_bc:
                    item_contents_tokenized.append(item_words_bc)
                    logger_experience_stream.debug(f"ES SharedConcepts: Item '{item_id_bc}' tokenized words: {list(item_words_bc)[:10]}")

        shared_concept_count_gw = 0.0
        if len(item_contents_tokenized) > 1:
            all_words_in_gw = set.union(*item_contents_tokenized) if item_contents_tokenized else set()
            logger_experience_stream.debug(f"ES_SHARED_CONCEPTS_DEBUG - All unique words in GW (count {len(all_words_in_gw)}): {str(list(all_words_in_gw))[:200]}...")
            word_occurrence_counter = Counter()
            for word_set_gw in item_contents_tokenized:
                word_occurrence_counter.update(list(word_set_gw))
            logger_experience_stream.debug(f"ES_SHARED_CONCEPTS_DEBUG - Word occurrence counter (top 5): {word_occurrence_counter.most_common(5)}")
            
            shared_words_count = 0
            for word_gw, count_gw in word_occurrence_counter.items():
                if count_gw > 1: 
                    shared_words_count += 1
            logger_experience_stream.debug(f"ES_SHARED_CONCEPTS_DEBUG - Total shared words count (appeared in >1 item): {shared_words_count}")
            
            shared_concept_count_gw = shared_words_count / len(all_words_in_gw) if all_words_in_gw else 0.0
            logger_experience_stream.debug(f"ES SharedConcepts: Total unique words in GW: {len(all_words_in_gw)}, Shared words: {shared_words_count}")
        elif item_contents_tokenized: # Only one item with text in GW
             logger_experience_stream.debug(f"ES SharedConcepts: Only one item with text in GW, shared concepts is 0.")
        else: # No items with text in GW
            logger_experience_stream.debug(f"ES SharedConcepts: No textual content found in GW items for shared concept calculation.")


        logger_experience_stream.debug(f"Calculated Shared Concept Count (GW): {shared_concept_count_gw:.3f}")


        # --- Create PhenomenalState Object with new fields ---
        phenomenal_state_obj = None
        if _PhenomenalStateClass:
            phenomenal_state_obj = _PhenomenalStateClass(
                content=integrated_content,
                intensity=intensity,
                valence=valence,
                integration_level=integration_level_old_proxy, # Keep old proxy for now
                attention_weight=attention_weight,
                timestamp=current_time,
                # New Φ-Proxy sub-metrics
                distinct_source_count=distinct_source_count,
                content_diversity_lexical=round(content_diversity_lexical, 3),
                shared_concept_count_gw=round(shared_concept_count_gw, 3)
            )
        else:
            logger_experience_stream.error("PhenomenalState class not defined. Cannot create state object.")
            # Fallback dictionary if class is missing
            phenomenal_state_obj = {
                "content": integrated_content, "intensity": intensity, "valence": valence,
                "integration_level": integration_level_old_proxy, "attention_weight": attention_weight,
                "timestamp": current_time, "error": "PhenomenalState class missing",
                "distinct_source_count": distinct_source_count,
                "content_diversity_lexical": round(content_diversity_lexical, 3),
                "shared_concept_count_gw": round(shared_concept_count_gw, 3)
            }

        logger_experience_stream.info(
            f"Experience integrated. I:{intensity:.2f}, V:{valence:.2f}, IL(old):{integration_level_old_proxy:.2f}, "
            f"Sources:{distinct_source_count}, LexDiv:{content_diversity_lexical:.2f}, SharedGWConcepts:{shared_concept_count_gw:.2f}"
        )
        return phenomenal_state_obj


    async def process(self, input_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not input_state:
            logger_experience_stream.warning("ExperienceStream process: Missing input state.")
            return None
        
        percepts = input_state.get("percepts", {})
        memories = input_state.get("memories", [])
        context = input_state.get("context", {})
        broadcast_content = input_state.get("broadcast_content", {}) # Crucial input

        # Basic type validation
        if not all(isinstance(arg, dict) for arg in [percepts, context, broadcast_content]) or \
           not isinstance(memories, list):
            logger_experience_stream.error("ExperienceStream process: Invalid type for one or more input state components.")
            return None
            
        phenomenal_state_result = await self.integrate_experience(percepts, memories, context, broadcast_content)
        return {"phenomenal_state": phenomenal_state_result}

    async def reset(self) -> None:
        self._current_valence = 0.0
        logger_experience_stream.info("ExperienceStream reset.")

    async def get_status(self) -> Dict[str, Any]:
        return {
            "component": "ExperienceStream",
            "status": "operational",
            "current_valence": round(self._current_valence, 3),
            "config_intensity_factor": self._intensity_factor,
            "config_valence_decay": self._valence_decay
        }

    async def shutdown(self) -> None:
        logger_experience_stream.info("ExperienceStream shutting down.")

# --- END COPIED/ADAPTED external_comms.py ---


# --- Test Configuration ---
OLLAMA_MODEL_FOR_TEST = "granite3.3:2b" # Same model as in the freeze logs
NARRATIVE_TEMPERATURE = 0.75
CALL_TIMEOUT_S = 10.0  # Timeout for each individual call
NUM_ITERATIONS = 50    # How many times to call the LLM
DELAY_BETWEEN_CALLS_S = 0.1 # Small delay to mimic agent cycle time roughly

# Configure logging for the test script
logging.basicConfig(
    level=logging.DEBUG, # Set to DEBUG to see httpcore logs if they appear
    format="[%(asctime)s] [%(levelname)-7s] [%(name)-30s] [%(funcName)-20s] %(message)s",
    datefmt="%H:%M:%S"
)
# Silence noisy loggers if necessary, but for debugging httpcore might be useful
# logging.getLogger("httpcore").setLevel(logging.INFO)
# logging.getLogger("httpx").setLevel(logging.INFO)

def generate_narrative_style_prompt(cycle_num: int) -> List[Dict[str, str]]:
    """Generates a prompt similar to what NarrativeConstructor might create."""
    system_prompt = (
        "You are the AI agent, OSCAR-C, writing a brief, reflective entry in your first-person "
        "autobiographical narrative ('I felt...', 'I decided...', 'This made me observe...'). "
        "Focus on the *immediate impact* or *feeling* of the triggering event in light of your "
        "current state. Keep entries concise (1-3 sentences). "
        "Do NOT mention you are a language model."
    )
    
    # Simulate some varying context
    events = [
        "a significant change in my internal drives",
        "a surprising prediction error about file access",
        "my consciousness level shifting to PRE_CONSCIOUS",
        "a moment of high computational load",
        "achieving a minor sub-goal for 'data analysis'",
        "noticing a repetitive pattern in my recent actions",
        "a feeling of moderate pain due to a persistent error",
        "a sudden influx of new information into my workspace"
    ]
    trigger_reason = random.choice(events)
    pain = round(random.uniform(0, 3), 1)
    happiness = round(random.uniform(4, 7), 1)
    purpose = round(random.uniform(3, 6), 1)
    cs_level = random.choice(["PRE_CONSCIOUS", "UNCONSCIOUS", "CONSCIOUS"])
    drives_str = f"Curiosity:{random.uniform(0.2,0.8):.2f}, Satisfaction:{random.uniform(0.3,0.7):.2f}"

    user_prompt = (
        f"My Current Internal State (Cycle {cycle_num}):\n"
        f"- Consciousness Level: {cs_level}\n"
        f"- Drives: {drives_str}\n"
        f"- Pain Level: {pain:.1f}/10\n"
        f"- Happiness Level: {happiness:.1f}/10\n"
        f"- Sense of Purpose: {purpose:.1f}/10\n\n"
        f"Triggering Event Summary:\n- Reason for this entry: {trigger_reason}\n\n"
        f"Write the next narrative entry from my perspective (I...):"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

async def main_test_loop():
    if not OLLAMA_AVAILABLE:
        logger_external_comms_test.critical("Ollama library is not available. Cannot run test.")
        return

    logger_external_comms_test.info(f"--- Starting Ollama Stress Test ---")
    logger_external_comms_test.info(f"Model: {OLLAMA_MODEL_FOR_TEST}, Iterations: {NUM_ITERATIONS}, Timeout per call: {CALL_TIMEOUT_S}s")
    
    loop = asyncio.get_running_loop()
    successful_calls = 0
    failed_calls = 0

    for i in range(NUM_ITERATIONS):
        logger_external_comms_test.info(f"--- Iteration {i + 1}/{NUM_ITERATIONS} ---")
        messages = generate_narrative_style_prompt(i + 1)
        
        # Log the prompt for debugging if needed, especially if freezes occur
        # logger_external_comms_test.debug(f"Generated Prompt (User Role): {messages[-1]['content']}")

        start_time = time.monotonic()
        try:
            response_text, error_text = await call_ollama(
                selected_ollama_model=OLLAMA_MODEL_FOR_TEST,
                messages=messages,
                temperature=NARRATIVE_TEMPERATURE,
                loop=loop,
                timeout=CALL_TIMEOUT_S 
            )
            duration = time.monotonic() - start_time

            if error_text:
                logger_external_comms_test.error(f"Call {i+1} FAILED after {duration:.2f}s. Error: {error_text}")
                failed_calls += 1
            elif response_text is not None:
                logger_external_comms_test.info(f"Call {i+1} SUCCEEDED in {duration:.2f}s. Response length: {len(response_text)}. Snippet: '{response_text[:100]}...'")
                successful_calls += 1
            else: # Should ideally be covered by error_text
                logger_external_comms_test.error(f"Call {i+1} FAILED after {duration:.2f}s. Returned None response and None error.")
                failed_calls += 1
        
        except Exception as e: # Catch any unexpected errors from call_ollama itself
            duration = time.monotonic() - start_time
            logger_external_comms_test.exception(f"Call {i+1} THREW UNEXPECTED EXCEPTION after {duration:.2f}s: {e}")
            failed_calls += 1
            # If a critical error occurs, we might want to stop, but for a stress test, let's try to continue.

        if i < NUM_ITERATIONS - 1:
            await asyncio.sleep(DELAY_BETWEEN_CALLS_S)
            
    logger_external_comms_test.info(f"--- Ollama Stress Test Finished ---")
    logger_external_comms_test.info(f"Successful calls: {successful_calls}")
    logger_external_comms_test.info(f"Failed calls: {failed_calls}")

if __name__ == "__main__":
    try:
        asyncio.run(main_test_loop())
    except KeyboardInterrupt:
        logger_external_comms_test.info("Stress test interrupted by user.")
    except Exception as e:
        logger_external_comms_test.exception(f"Critical error in test runner: {e}")