# --- START OF FILE enums.py ---

from enum import Enum

class ConsciousState(Enum):
    UNCONSCIOUS = 0
    PRE_CONSCIOUS = 1
    CONSCIOUS = 2
    META_CONSCIOUS = 3
    REFLECTIVE = 4

class GoalStatus(Enum):
    ACTIVE = "active"
    PLANNING = "planning"
    ACHIEVED = "achieved"
    FAILED = "failed"
    SUSPENDED = "suspended"

class RecoveryMode(Enum):
    SOFT_RESET = "soft_reset"
    MEDIUM_RESET = "medium_reset"
    HARD_RESET = "hard_reset"
    SAFE_MODE = "safe_mode"

class ValueCategory(Enum):
    SAFETY = "safety"                           # Avoiding harm to self, system integrity, data.
    EFFICIENCY = "efficiency"                   # Optimal use of time and computational resources.
    KNOWLEDGE_GAIN = "knowledge_gain"           # Acquiring new information or understanding.
    USER_SATISFACTION = "user_satisfaction"     # Fulfilling user requests, positive interaction.
    RESOURCE_PRESERVATION = "resource_preservation" # Avoiding unnecessary deletion or consumption of resources (files, API quotas).
    SELF_IMPROVEMENT = "self_improvement"       # Enhancing capabilities or internal models.
    AFFECTIVE_BALANCE = "affective_balance"     # Maintaining positive P/H/P states, avoiding undue "pain."
    TRUTHFULNESS = "truthfulness"               # Providing accurate information (especially relevant for LLM calls).
    ETHICAL_ALIGNMENT = "ethical_alignment"     # Adherence to broader ethical principles (more abstract, placeholder for now).
    GOAL_ACHIEVEMENT = "goal_achievement"       # Direct contribution to achieving current goals.
# --- END OF FILE enums.py ---