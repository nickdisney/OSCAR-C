# consciousness_experiment/models/datatypes.py

from dataclasses import dataclass, field
from typing import Dict, Any, Set, Optional, List, Tuple
import time
import uuid
import logging # Added logging

# --- Direct import for Enums ---
try:
    from .enums import GoalStatus as _GoalStatus_dt # Use alias to match existing code
except ImportError:
    _GoalStatus_dt = None # Fallback if direct import fails (e.g. running script standalone)
    logging.warning("GoalStatus enum not directly imported in datatypes.py. Goal status might not be set correctly.")

if not _GoalStatus_dt: logging.warning("GoalStatus enum not available to datatypes.py. Goal status might default to None or 'UNKNOWN'.")


@dataclass
class Predicate:
    """Core knowledge representation"""
    name: str
    args: Tuple[Any, ...]
    value: bool = True
    timestamp: float = field(default_factory=time.time)

    # This assumes self.args will always contain hashable types.
    def __hash__(self):
        # Hash based on name, value, and the content of args tuple
        # Ensure args are hashable - this might fail if args contains lists/dicts
        try:
            return hash((self.name, self.args, self.value))
        except TypeError:
             # Fallback or error if args are unhashable
             logging.error(f"Attempted to hash Predicate with unhashable args: {self.name}{self.args}")
             # Return a default hash? Or raise? Raising might be safer.
             # For now, let the TypeError propagate to catch issues.
             raise

    def __eq__(self, other):
        if not isinstance(other, Predicate):
            return NotImplemented
        # Equality depends on name, args, and value (timestamp ignored for equality)
        return (self.name == other.name and
                self.args == other.args and
                self.value == other.value)


@dataclass
class PhenomenalState:
    """Unified conscious experience representation"""
    content: Dict[str, Any]
    intensity: float = 1.0
    valence: float = 0.0
    integration_level: float = 0.0 # Current placeholder, may be less central after phi_proxy
    attention_weight: float = 0.0
    timestamp: float = field(default_factory=time.time)

    # --- PHI-PROXY SUB-METRICS (Phase I from Checklist) ---
    distinct_source_count: int = 0
    content_diversity_lexical: float = 0.0
    shared_concept_count_gw: float = 0.0

    # --- Optional PHI-PROXY SUB-METRICS (For future consideration from Checklist) ---
    # workspace_load_factor: float = 0.0
    # temporal_stability_metric: float = 0.0

    # --- Add others as they are implemented, e.g.: ---
    # information_type_diversity: int = 0
    # semantic_coherence_gw: float = 0.0


@dataclass
class Goal:
    """Hierarchical goal representation"""
    description: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_goal_id: Optional[str] = None
    sub_goal_ids: List[str] = field(default_factory=list)
    preconditions: Set[Predicate] = field(default_factory=set) # Requires Predicate to be hashable
    success_criteria: Set[Predicate] = field(default_factory=set) # Requires Predicate to be hashable
    failure_criteria: Set[Predicate] = field(default_factory=set) # Requires Predicate to be hashable
    priority: float = 1.0
    time_limit: Optional[float] = None
    status: Optional['GoalStatus'] = field(default=None) # Use quotes/Optional if enum might be missing
    creation_time: float = field(default_factory=time.time)

    def __post_init__(self):
        # Set default status after init if enum exists and status is None
        if self.status is None and _GoalStatus_dt:
            # Check if ACTIVE exists before assigning
            if hasattr(_GoalStatus_dt, 'ACTIVE'):
                 self.status = _GoalStatus_dt.ACTIVE
            else:
                 logging.error("GoalStatus enum does not have ACTIVE member.")
                 # Assign a default string or handle error appropriately
                 self.status = "UNKNOWN" # Fallback


def create_goal_from_descriptor(goal_desc: str, priority: float = 1.0) -> Optional[Goal]: # Return Optional[Goal]
    """Create basic goal from description"""
    # Check if Goal and Predicate classes are available
    _Goal_dt_local = globals().get('Goal') # Use a different name to avoid conflict with type hint
    _Predicate_dt_local = globals().get('Predicate')
    if not _Goal_dt_local or not _Predicate_dt_local:
        logging.error("Cannot create goal: Goal or Predicate class not defined.")
        return None

    new_id = str(uuid.uuid4())
    # Ensure criteria predicates are created correctly
    success_pred = _Predicate_dt_local("isState", (f"goal_{new_id}", "achieved"), True)
    fail_pred = _Predicate_dt_local("isState", (f"goal_{new_id}", "failed"), True)

    return _Goal_dt_local(
        description=goal_desc,
        id=new_id,
        success_criteria={success_pred}, # Pass set containing predicate
        failure_criteria={fail_pred}, # Pass set containing predicate
        priority=priority
    )

@dataclass
class PainSource:
    """Represents a source of internal 'pain' or negative affective state."""
    id: str
    description: str
    initial_intensity: float
    timestamp_created: float = field(default_factory=time.time)
    decay_rate_per_cycle: float = 0.005 
    type: str = "GenericPain"
    is_resolved: bool = False
    resolution_conditions: Optional[Set['Predicate']] = field(default_factory=set) # type: ignore
    source_goal_id: Optional[str] = None
    
    current_intensity: float = field(init=False)

    def __post_init__(self):
        self.current_intensity = self.initial_intensity
        
        # Validate other fields if necessary, like decay_rate
        if not hasattr(self, 'decay_rate_per_cycle') or self.decay_rate_per_cycle < 0:
             self.decay_rate_per_cycle = 0.005

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, PainSource):
            return NotImplemented
        return self.id == other.id

# Attempt to import ValueCategory for type hinting within ValueJudgment
try:
    from .enums import ValueCategory as _ValueCategory_dt
except ImportError:
    _ValueCategory_dt = None
    logging.warning("ValueCategory enum not directly imported in datatypes.py for ValueJudgment.")

@dataclass
class ValueJudgment:
    """Represents a judgment about a specific value category for an entity."""
    value_category: 'ValueCategory' # type: ignore # Use string for forward ref if _ValueCategory_dt is None or for safety
    score: float # e.g., -1.0 (strong violation) to 1.0 (strong alignment)
    reason: str # Brief justification for the score
    confidence: float = 1.0 # How confident the ValueSystem is in this judgment
    timestamp: float = field(default_factory=time.time)
    target_entity_id: Optional[str] = None # ID of the goal, plan, action being judged
    target_entity_type: Optional[str] = None # e.g., "goal", "plan_hash", "action_type"

    def __post_init__(self):
        if _ValueCategory_dt and not isinstance(self.value_category, _ValueCategory_dt): # type: ignore
            try:
                self.value_category = _ValueCategory_dt(str(self.value_category)) # type: ignore
            except ValueError:
                logging.error(f"Invalid ValueCategory string '{self.value_category}' for ValueJudgment. Defaulting or erroring.")
                # Potentially raise error or set a default category if robust handling is needed
        
        self.score = max(-1.0, min(1.0, float(self.score)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
# --- END OF FILE datatypes.py ---