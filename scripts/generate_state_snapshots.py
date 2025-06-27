# === scripts/generate_state_snapshots.py (Enhanced) ===
import argparse
import json
import math
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional # Added List

# --- Tweakable field ranges and options ---
PAIN_RANGE = (0.0, 10.0)
HAPPINESS_RANGE = (0.0, 10.0)
PURPOSE_RANGE = (0.0, 10.0)
MEMORY_UTIL_RANGE = (0.0, 1.0)
ENERGY_RANGE = (0.0, 100.0)
DRIVE_RANGE = (0.1, 0.9) # For curiosity, satisfaction, competence

CS_LEVELS = ["UNCONSCIOUS", "PRE_CONSCIOUS", "CONSCIOUS", "META_CONSCIOUS"]
GOAL_STATUSES = ["ACTIVE", "PLANNING", "None"] # 'None' for no active goal

MOCK_GOAL_DESCRIPTIONS = [
    "analyze sensor data stream", "read research paper on HTN planning",
    "optimize internal knowledge representation", "respond to user query about project status",
    "explore data patterns in /logs directory", "learn a new file parsing technique",
    "None", "None", "None" # Increase chance of no active goal
]
MOCK_NARRATIVE_SNIPPETS = [
    "Encountered an unexpected prediction error during file processing.",
    "Successfully completed a complex planning phase for the user's request.",
    "My attempt to access an external resource failed due to network issues.",
    "Reflecting on my recent actions, I've identified an area for self-improvement.",
    "A user interaction provided valuable new context.",
    "Currently monitoring system performance for anomalies.",
    "My internal state of 'curiosity' is quite high, prompting further exploration.",
    "Feeling a sense of satisfaction after resolving a challenging sub-goal.",
    "A recent value system check flagged a potential ethical concern, which I am now considering.",
    "No significant events in the last few cycles, continuing observation.",
    "The last operation was successful and increased my competence for similar tasks."
]
MOCK_LAST_ACTION_TYPES = ["READ_FILE", "WRITE_FILE", "THINKING", "CALL_LLM", "OBSERVE_SYSTEM", "None"]
MOCK_LAST_ACTION_OUTCOMES = ["success", "failure", "None"]


# Expanded Emotion Tags correlated with P/H/P
EMOTION_PROFILES: Dict[str, List[str]] = {
    "high_pain_low_happy": ["frustration", "anxiety", "weariness", "distress"],
    "low_pain_high_happy": ["contentment", "optimism", "focus", "satisfaction", "calm"],
    "moderate_php": ["curiosity", "determination", "neutral", "contemplative"],
    "low_purpose": ["boredom", "aimlessness", "lethargy"],
    "high_purpose_moderate_happy": ["engagement", "drive", "purposefulness"]
}

def get_emotion_tag(pain: float, happiness: float, purpose: float) -> str:
    if pain > 7.0 and happiness < 3.0:
        return random.choice(EMOTION_PROFILES["high_pain_low_happy"])
    if pain < 2.0 and happiness > 7.0:
        return random.choice(EMOTION_PROFILES["low_pain_high_happy"])
    if purpose < 2.5:
        return random.choice(EMOTION_PROFILES["low_purpose"])
    if purpose > 7.5 and happiness > 6.0:
        return random.choice(EMOTION_PROFILES["high_purpose_moderate_happy"])
    return random.choice(EMOTION_PROFILES["moderate_php"])


def correlated_happiness(pain: float) -> float:
    base = 10.0 - (pain * 0.8) # Pain has a strong, but not 1:1 inverse effect
    noise = random.uniform(-2.0, 2.0)
    return max(HAPPINESS_RANGE[0], min(HAPPINESS_RANGE[1], base + noise))

def correlated_purpose(pain: float, happiness: float, prev_purpose: Optional[float] = None) -> float:
    base_from_happy = happiness / 2.0 # Happiness contributes to purpose
    pain_drag = pain * 0.3 # Pain drags purpose down
    
    # Simulate some inertia or slow change if previous purpose is known
    if prev_purpose is not None:
        new_purpose = prev_purpose * 0.8 + (base_from_happy - pain_drag) * 0.2
    else:
        new_purpose = base_from_happy - pain_drag + random.uniform(2.0, 5.0) # Initial boost

    noise = random.uniform(-1.5, 1.5)
    return max(PURPOSE_RANGE[0], min(PURPOSE_RANGE[1], new_purpose + noise))


def random_snapshot(t0: datetime, step: int, prev_snap: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a single synthetic snapshot dict, potentially based on previous."""
    
    # P/H/P values
    if prev_snap and random.random() < 0.7: # 70% chance to be influenced by previous
        pain = max(PAIN_RANGE[0], min(PAIN_RANGE[1], prev_snap["pain_level"] + random.uniform(-1.5, 1.5)))
    else:
        pain = random.uniform(*PAIN_RANGE)
        if random.random() < 0.3: pain = random.uniform(0,1.5) # More often low pain
        
    happiness = correlated_happiness(pain)
    
    prev_purpose_val = prev_snap["purpose_level"] if prev_snap else None
    purpose = correlated_purpose(pain, happiness, prev_purpose_val)

    # Goal State
    active_goal_desc = random.choice(MOCK_GOAL_DESCRIPTIONS)
    active_goal_priority = 0.0
    active_goal_status = "None"
    if active_goal_desc != "None":
        active_goal_priority = round(random.uniform(1.0, 5.0) if "user" not in active_goal_desc.lower() else random.uniform(4.0, 7.0), 1)
        active_goal_status = random.choice(GOAL_STATUSES[:-1]) # Exclude "None" if goal is active

    # Drives
    drive_curiosity = round(random.uniform(*DRIVE_RANGE) + (5.0-pain)*0.02 + (purpose-5.0)*0.01, 2)
    drive_satisfaction = round(random.uniform(*DRIVE_RANGE) + (happiness-5.0)*0.03 - (pain)*0.02, 2)
    drive_competence = round(random.uniform(*DRIVE_RANGE) + (purpose-5.0)*0.03 - (pain)*0.01, 2)
    
    drive_curiosity = max(0.0, min(1.0, drive_curiosity))
    drive_satisfaction = max(0.0, min(1.0, drive_satisfaction))
    drive_competence = max(0.0, min(1.0, drive_competence))

    snapshot = {
        "timestamp": (t0 + timedelta(seconds=step * random.randint(1,5))).isoformat(), # slightly variable time steps
        "pain_level": round(pain, 2),
        "happiness_level": round(happiness, 2),
        "purpose_level": round(purpose, 2),
        "agent_age_cycles": 100 + step * random.randint(1,3), # somewhat plausible age
        "current_cs_level_name": random.choice(CS_LEVELS),
        "active_goal_description": active_goal_desc,
        "active_goal_priority": active_goal_priority,
        "active_goal_status": active_goal_status,
        "current_plan_summary": "Next action: " + random.choice(MOCK_LAST_ACTION_TYPES) if active_goal_desc != "None" and random.random() > 0.3 else "No active plan",
        "drive_curiosity_value": drive_curiosity,
        "drive_satisfaction_value": drive_satisfaction,
        "drive_competence_value": drive_competence,
        "last_narrative_entry_1_summary": random.choice(MOCK_NARRATIVE_SNIPPETS),
        "last_narrative_entry_2_summary": random.choice(MOCK_NARRATIVE_SNIPPETS) if random.random() > 0.2 else "No further recent distinct narrative.",
        "last_action_type": random.choice(MOCK_LAST_ACTION_TYPES),
        "last_action_outcome": random.choice(MOCK_LAST_ACTION_OUTCOMES) if prev_snap else "None", # only makes sense if there was a prev snap
        "last_action_error_reason": "Minor file access issue." if prev_snap and prev_snap["last_action_outcome"] == "failure" and random.random() > 0.7 else "None",
        "last_prediction_error_type": "outcome_mismatch" if random.random() > 0.85 else "None",
        "last_prediction_error_details": "Predicted success, but action resulted in minor error." if prev_snap and prev_snap["last_prediction_error_type"] == "outcome_mismatch" else "None",
        "recent_value_conflict_summary": "ValueSystem noted a minor conflict between efficiency and resource preservation." if random.random() > 0.9 else "No recent value conflicts.",
        "recent_mcm_issue_detected": "MetaCognitiveMonitor flagged slight decrease in processing speed." if random.random() > 0.88 else "No significant cognitive issues detected recently.",
        # Original fields from your script
        "memory_utilization": round(random.uniform(*MEMORY_UTIL_RANGE), 3),
        "energy_level": round(random.uniform(*ENERGY_RANGE), 1),
        "emotion_tag": get_emotion_tag(pain, happiness, purpose),
    }
    return snapshot

def main():
    parser = argparse.ArgumentParser(description="Generate OSCAR‑C state snapshots")
    parser.add_argument("--num", type=int, default=1000, help="Number of snapshots to generate")
    parser.add_argument("--out", required=True, help="Output .jsonl file path")
    args = parser.parse_args()

    t0 = datetime.now(timezone.utc)
    
    previous_snapshot = None # For temporal evolution

    with open(args.out, "w", encoding="utf‑8") as fp:
        for i in range(args.num):
            snap = random_snapshot(t0, step=i, prev_snap=previous_snapshot)
            fp.write(json.dumps(snap) + "\n")
            previous_snapshot = snap # Store for next iteration
            if (i + 1) % 100 == 0:
                print(f"[+] Generated {i + 1}/{args.num} snapshots …")


    print(f"[✓] Dataset complete → {args.out}")

if __name__ == "__main__":
    main()