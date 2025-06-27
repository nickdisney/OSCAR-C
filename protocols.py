# --- START OF CORRECTED protocols.py ---

from typing import Protocol, Dict, Any, Optional, List, Set, Tuple, runtime_checkable # Added runtime_checkable

# Forward references using strings for internal types
Predicate = 'Predicate'
PhenomenalState = 'PhenomenalState'
Goal = 'Goal'
ConsciousState = 'ConsciousState'
# Add RecoveryMode if used in any protocol signature (not currently)
# RecoveryMode = 'RecoveryMode'

# --- Add @runtime_checkable to allow isinstance() and issubclass() ---
@runtime_checkable
class CognitiveComponent(Protocol):
    """Standard interface for all OSCAR-C components"""

    async def initialize(self, config: Dict[str, Any], controller: Any) -> bool:
        """Initialize component with configuration and controller reference."""
        ...

    async def process(self, input_state: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Process input state and return output. Can be a no-op for some components."""
        ...

    async def reset(self) -> None:
        """Reset component to its initial state."""
        ...

    async def get_status(self) -> Dict[str, Any]:
        """Get the component's current status and key metrics."""
        ...

    async def shutdown(self) -> None:
        """Perform any necessary cleanup before the agent stops."""
        ...

# --- Specific Component Protocols (also make runtime_checkable for consistency) ---

@runtime_checkable
class AttentionMechanism(CognitiveComponent, Protocol):
    """Interface for components allocating attention."""
    # Keep original method definitions
    async def allocate_attention(self, candidates: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate attention weights for candidates."""
        ...

@runtime_checkable
class WorkspaceManager(CognitiveComponent, Protocol):
    """Interface for components managing the global workspace."""
    # Keep original method definitions
    async def update_workspace(self, attention_weights: Dict[str, float], all_candidates_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Update workspace content based on attention."""
        ...
    async def broadcast(self) -> Dict[str, Any]:
        """Return the current content of the workspace."""
        ...

@runtime_checkable
class ExperienceIntegrator(CognitiveComponent, Protocol):
    """Interface for components integrating information into phenomenal experience."""
    # Keep original method definitions
    async def integrate_experience(self,
                                   percepts: Dict[str, Any],
                                   memories: List[Any],
                                   context: Dict[str, Any],
                                   broadcast_content: Dict[str, Any] # Added explicit broadcast
                                  ) -> PhenomenalState: # Use forward reference
        """Combine inputs into a PhenomenalState."""
        ...

@runtime_checkable
class ConsciousnessAssessor(CognitiveComponent, Protocol):
    """Interface for components assessing the level of consciousness."""
    # Keep original method definitions
    async def assess_consciousness_level(self,
                                         experience: Optional[PhenomenalState], # Use forward reference
                                         workspace_content: Dict[str, Any]
                                        ) -> ConsciousState: # Use forward reference
        """Assess the current level of consciousness."""
        ...

@runtime_checkable
class Planner(CognitiveComponent, Protocol):
    """Interface for planning components."""
    # Keep original method definitions
    async def plan(self, goal: Goal, current_state: Set[Predicate]) -> Optional[List[Dict[str, Any]]]: # Use forward references
        """Generate a plan (list of action dictionaries) to achieve the goal."""
        ...

@runtime_checkable
class StateQueryable(CognitiveComponent, Protocol):
    """Interface for components that maintain queryable state (like KB)."""
    # Keep original method definitions
    async def query_state(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Handle structured queries about the component's internal state."""
        ...

# --- END OF CORRECTED protocols.py ---