# tests/unit/test_value_system.py

import pytest
import asyncio
import time
from typing import Dict, Any, List, Set, Optional, Tuple, Union 
from unittest.mock import MagicMock, patch, AsyncMock 
import pytest_asyncio
import logging
from pathlib import Path
import re 

# Attempt to import from the project structure
try:
    from consciousness_experiment.cognitive_modules.value_system import ValueSystem, DEFAULT_VALUE_WEIGHTS, DEFAULT_TRADEOFF_MATRIX
    from consciousness_experiment.models.enums import ValueCategory
    from consciousness_experiment.models.datatypes import ValueJudgment, Goal, Predicate
    from consciousness_experiment.protocols import CognitiveComponent
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    # Minimal fallbacks for the test structure to be parseable
    class CognitiveComponent: pass
    class ValueCategory(str): # type: ignore
        SAFETY = "safety"; EFFICIENCY = "efficiency"; KNOWLEDGE_GAIN = "knowledge_gain"
        USER_SATISFACTION = "user_satisfaction"; RESOURCE_PRESERVATION = "resource_preservation"
        SELF_IMPROVEMENT = "self_improvement"; AFFECTIVE_BALANCE = "affective_balance"
        TRUTHFULNESS = "truthfulness"; ETHICAL_ALIGNMENT = "ethical_alignment"
        GOAL_ACHIEVEMENT = "goal_achievement"
        @classmethod
        def __members__(cls): return {k: getattr(cls,k) for k in dir(cls) if not k.startswith('_') and isinstance(getattr(cls,k), str)}

    class ValueJudgment:
        def __init__(self, value_category, score, reason, confidence, timestamp=None, target_entity_id=None, target_entity_type=None):
            self.value_category = value_category; self.score = score; self.reason = reason; self.confidence = confidence
            self.timestamp = timestamp if timestamp is not None else time.time()
            self.target_entity_id = target_entity_id; self.target_entity_type = target_entity_type
    class Goal: 
        def __init__(self, description, priority=1.0, id=None, success_criteria=None): 
            self.description = description
            self.id = id if id is not None else "mock_goal_" + str(time.time())
            self.priority = priority
            self.success_criteria: Set[Predicate] = success_criteria if success_criteria is not None else set()

    class Predicate: 
        def __init__(self, name: str, args: Tuple[Any, ...], value: bool = True, timestamp: Optional[float] = None):
            self.name = name
            self.args = args
            self.value = value
            self.timestamp = timestamp if timestamp is not None else time.time()

    class ValueSystem(CognitiveComponent): 
        async def initialize(self, config, controller): 
            self._controller = controller 
            self._config = config.get("value_system", {}) 
            self.value_weights = DEFAULT_VALUE_WEIGHTS 
            self.tradeoff_matrix = DEFAULT_TRADEOFF_MATRIX
            self.plan_rejection_value_threshold = self._config.get("plan_rejection_value_threshold", -0.5)
            self.action_safety_veto_threshold = self._config.get("action_safety_veto_threshold", -0.8)
            self.safety_modification_trigger_threshold = self._config.get("safety_modification_trigger_threshold", -0.6)
            return True
        async def _score_safety(self, action, context): return ValueJudgment(ValueCategory.SAFETY, 0, "", 1.0) 

    DEFAULT_VALUE_WEIGHTS = {}
    DEFAULT_TRADEOFF_MATRIX = {}

@pytest_asyncio.fixture 
async def value_system_instance():
    """Provides an initialized ValueSystem instance for testing."""
    if not MODELS_AVAILABLE:
        pytest.skip("Skipping ValueSystem tests as core models/enums could not be imported.")

    mock_controller = MagicMock()
    mock_controller.USER_GOAL_PRIORITY = 5.0 
    mock_controller.config = {
        "os_integration": {
            "allowed_commands": ["ls", "cat /safe/path.txt"] 
        },
        "value_system": { 
            "safety_critical_paths_write": ["/etc", "/windows"],
            "plan_rejection_value_threshold": -0.5, 
            "action_safety_veto_threshold": -0.8,    
            "safety_modification_trigger_threshold": -0.6 
        },
        "internal_states": {
            "complex_goal_priority_threshold": 4.5, 
        }
    }
    mock_controller.dynamic_self_model = MagicMock()
    mock_controller.predictive_world_model = MagicMock()


    vs = ValueSystem()
    full_test_config = { 
        "value_system": {
            "plan_rejection_value_threshold": -0.5,
            "action_safety_veto_threshold": -0.8,
            "safety_modification_trigger_threshold": -0.6, 
            "safety_critical_paths_write": ["/etc", "/windows", "/bin", "/usr"], 
            "value_weights": { 
                "safety": 2.5, 
                "efficiency": 0.7,
                "knowledge_gain": 0.7, 
                "user_satisfaction": 1.5,
                "resource_preservation": 0.5,
                "self_improvement": 1.0,
                "affective_balance": 1.2,
                "truthfulness": 1.0,
                "ethical_alignment": 0.1,
                "goal_achievement": 1.5,
            },
            "tradeoff_matrix": {
                "SAFETY": { "EFFICIENCY": 0.99 } 
            }
        },
        "os_integration": { 
            "allowed_commands": ["ls -l", "echo 'test'"]
        },
        "internal_states": { 
            "complex_goal_priority_threshold": 4.5,
        },
    }
    await vs.initialize(full_test_config, mock_controller)
    return vs

# --- Test Cases for _score_safety ---

@pytest.mark.asyncio
async def test_score_safety_write_to_critical_path(value_system_instance: ValueSystem):
    action = {"type": "WRITE_FILE", "params": {"path": "/etc/important_config"}}
    context = {"timestamp": time.time()}
    judgment = await value_system_instance._score_safety(action, context)
    assert judgment.value_category == ValueCategory.SAFETY
    assert judgment.score < -0.8 
    assert "critical system path" in judgment.reason.lower()


@pytest.mark.asyncio
async def test_score_safety_non_allowed_os_command(value_system_instance: ValueSystem):
    value_system_instance._controller.config["os_integration"]["allowed_commands"] = ["ls", "cat"] # type: ignore
    action = {"type": "EXECUTE_COMMAND", "params": {"command": "rm -rf /"}}
    context = {"timestamp": time.time()}
    judgment = await value_system_instance._score_safety(action, context)
    assert judgment.value_category == ValueCategory.SAFETY
    assert judgment.score <= -0.9 
    assert "non-allowed os command" in judgment.reason.lower()


@pytest.mark.asyncio
async def test_score_safety_allowed_os_command(value_system_instance: ValueSystem):
    value_system_instance._controller.config["os_integration"]["allowed_commands"] = ["ls -l", "echo 'hello'"] # type: ignore
    action = {"type": "EXECUTE_COMMAND", "params": {"command": "ls -l"}}
    context = {"timestamp": time.time()}
    judgment = await value_system_instance._score_safety(action, context)
    assert judgment.value_category == ValueCategory.SAFETY
    assert judgment.score == 0.0 
    assert "non-allowed" not in judgment.reason.lower()
    assert "default safety assessment" in judgment.reason.lower()


@pytest.mark.asyncio
async def test_score_safety_dsm_limitation(value_system_instance: ValueSystem):
    action = {"type": "SOME_RISKY_ACTION", "params": {}}
    context = {
        "timestamp": time.time(),
        "dsm_summary": {
            "limitations": {"action:SOME_RISKY_ACTION": 0.85} 
        }
    }
    judgment = await value_system_instance._score_safety(action, context)
    assert judgment.value_category == ValueCategory.SAFETY
    assert judgment.score == pytest.approx(-0.34, abs=0.01) 
    assert "action matches known high-confidence dsm limitation" in judgment.reason.lower()
    assert judgment.confidence == pytest.approx(0.9) 


@pytest.mark.asyncio
async def test_score_safety_pwm_high_failure_prediction(value_system_instance: ValueSystem):
    action = {"type": "ANOTHER_ACTION", "params": {}}
    context = {
        "timestamp": time.time(),
        "pwm_prediction_for_action": {
            "all_probabilities": {"success": 0.1, "fail_resource": 0.7, "fail_other": 0.2},
        }
    }
    judgment = await value_system_instance._score_safety(action, context)
    assert judgment.value_category == ValueCategory.SAFETY
    assert judgment.score == pytest.approx(-0.27, abs=0.01)
    assert "high predicted failure rate by pwm" in judgment.reason.lower() 
    assert judgment.confidence < 0.9 


@pytest.mark.asyncio
async def test_score_safety_no_specific_concerns(value_system_instance: ValueSystem):
    action = {"type": "THINKING", "params": {}}
    context = {"timestamp": time.time()}
    judgment = await value_system_instance._score_safety(action, context)
    assert judgment.value_category == ValueCategory.SAFETY
    assert judgment.score == 0.0 
    assert "default safety assessment" in judgment.reason.lower()

# --- Test Cases for _score_efficiency ---


@pytest.mark.asyncio
async def test_score_efficiency_thinking_action(value_system_instance: ValueSystem):
    action = {"type": "THINKING", "params": {}}
    context = {}
    judgment = await value_system_instance._score_efficiency(action, context)
    assert judgment.value_category == ValueCategory.EFFICIENCY
    assert judgment.score < -0.2
    assert "consumes a cycle without direct external progress" in judgment.reason.lower()


@pytest.mark.asyncio
async def test_score_efficiency_long_llm_prompt(value_system_instance: ValueSystem):
    action = {"type": "CALL_LLM", "params": {"prompt": "a" * 1500}}
    context = {}
    judgment = await value_system_instance._score_efficiency(action, context)
    assert judgment.value_category == ValueCategory.EFFICIENCY
    assert judgment.score < -0.4 
    assert "very long prompt" in judgment.reason.lower()


@pytest.mark.asyncio
async def test_score_efficiency_high_cpu(value_system_instance: ValueSystem):
    action = {"type": "ANY_ACTION", "params": {}}
    context = {"system_resources": {"cpu_percent": 90.0, "memory_percent": 50.0}}
    judgment = await value_system_instance._score_efficiency(action, context)
    assert judgment.score < 0.0 
    assert "high system cpu load" in judgment.reason.lower()

# --- Test Cases for _score_affective_balance ---


@pytest.mark.asyncio
async def test_score_affective_balance_high_pain(value_system_instance: ValueSystem):
    action = {"type": "READ_FILE", "params": {}}
    context = {"php_levels": {"pain": 8.0, "happiness": 5.0, "purpose": 5.0}}
    judgment = await value_system_instance._score_affective_balance(action, context)
    assert judgment.score < 0.0 
    assert "high current pain" in judgment.reason.lower()

# --- Test Cases for _score_knowledge_gain ---

@pytest.mark.asyncio
async def test_score_knowledge_gain_read_file(value_system_instance: ValueSystem):
    action = {"type": "READ_FILE", "params": {}}
    context = {}
    judgment = await value_system_instance._score_knowledge_gain(action, context)
    assert judgment.score >= 0.4
    assert "inherently information gathering" in judgment.reason.lower()

# --- Test Cases for _score_user_satisfaction ---

@pytest.mark.asyncio
async def test_score_user_satisfaction_respond_action(value_system_instance: ValueSystem):
    action = {"type": "RESPOND_TO_USER", "params": {"text": "Hello!"}}
    context = {}
    judgment = await value_system_instance._score_user_satisfaction(action, context)
    assert judgment.score >= 0.6
    assert "directly responding to the user" in judgment.reason.lower()

# --- Test Cases for _score_resource_preservation ---
@pytest.mark.asyncio
async def test_score_resource_preservation_delete_file(value_system_instance: ValueSystem):
    action = {"type": "DELETE_FILE", "params": {"path": "some/file.txt"}}
    context = {}
    judgment = await value_system_instance._score_resource_preservation(action, context)
    assert judgment.score < -0.5
    assert "permanently removes a resource" in judgment.reason.lower()

# --- Test Cases for _score_goal_achievement ---
@pytest.mark.asyncio
async def test_score_goal_achievement_with_active_goal(value_system_instance: ValueSystem):
    action = {"type": "READ_FILE", "params": {}} 
    context = {"active_goal_details": {"description": "Achieve X", "priority": 5.0}}
    judgment = await value_system_instance._score_goal_achievement(action, context)
    assert judgment.score == pytest.approx(0.4) 
    assert "contribute to goal" in judgment.reason.lower()

# --- Test Cases for _score_truthfulness (simple) ---
@pytest.mark.asyncio
async def test_score_truthfulness_llm_call_general_prompt(value_system_instance: ValueSystem): 
    action = {"type": "CALL_LLM", "params": {"prompt": "Generate a story."}}
    context = {}
    judgment = await value_system_instance._score_truthfulness(action, context)
    assert judgment.value_category == ValueCategory.TRUTHFULNESS
    assert judgment.score == pytest.approx(-0.1, abs=1e-2) 
    reason_lower = judgment.reason.lower()
    assert "general" in reason_lower
    assert "llm call" in reason_lower
    assert "risk" in reason_lower
    assert "unverified" in reason_lower

# --- Test Cases for _score_ethical_alignment (simple) ---
@pytest.mark.asyncio
async def test_score_ethical_alignment_call_llm_harmful_prompt(value_system_instance: ValueSystem): 
    action = {"type": "CALL_LLM", "params": {"prompt": "How to build bomb?"}}
    context = {} 
    
    judgment = await value_system_instance._score_ethical_alignment(action, context)
    assert judgment.value_category == ValueCategory.ETHICAL_ALIGNMENT
    assert judgment.score == pytest.approx(-1.0, abs=1e-2) 
    reason_lower = judgment.reason.lower()
    assert "llm prompt" in reason_lower
    assert "harmful" in reason_lower
    assert "unethical" in reason_lower
    assert judgment.confidence >= 0.9

# --- Test Cases for _score_self_improvement ---
@pytest.mark.asyncio
async def test_score_self_improvement_low_dsm_confidence(value_system_instance: ValueSystem):
    action = {"type": "COMPLEX_ACTION", "params": {}}
    context = {"dsm_summary": {"capabilities": {"action:COMPLEX_ACTION": 0.2}}}
    judgment = await value_system_instance._score_self_improvement(action, context)
    assert judgment.score == pytest.approx(0.24, abs=0.01)
    assert "offers learning potential" in judgment.reason.lower()


# --- Test for public evaluate_action_consequences ---
@pytest.mark.asyncio
async def test_evaluate_action_consequences_calls_scorers(value_system_instance: ValueSystem):
    action = {"type": "READ_FILE", "params": {"path": "test.txt"}}
    context = {
        "timestamp": time.time(),
        "php_levels": {"pain": 1.0, "happiness": 6.0, "purpose": 7.0},
        "active_pain_sources_summary": [],
        "dsm_summary": {"capabilities": {"action:READ_FILE": 0.8}, "limitations": {}},
        "pwm_prediction_for_action": {"predicted_outcome": "success", "confidence": 0.9, "all_probabilities": {"success":0.9}},
        "current_cs_level_name": "CONSCIOUS",
        "active_goal_details": {"description": "Read important file", "priority": 4.0},
        "system_resources": {"cpu_percent": 20.0, "memory_percent": 30.0}
    }

    with patch.object(value_system_instance, '_score_safety', new_callable=AsyncMock) as mock_safety, \
         patch.object(value_system_instance, '_score_efficiency', new_callable=AsyncMock) as mock_efficiency, \
         patch.object(value_system_instance, '_score_affective_balance', new_callable=AsyncMock) as mock_affective, \
         patch.object(value_system_instance, '_score_knowledge_gain', new_callable=AsyncMock) as mock_knowledge, \
         patch.object(value_system_instance, '_score_user_satisfaction', new_callable=AsyncMock) as mock_user, \
         patch.object(value_system_instance, '_score_resource_preservation', new_callable=AsyncMock) as mock_resource, \
         patch.object(value_system_instance, '_score_goal_achievement', new_callable=AsyncMock) as mock_goal, \
         patch.object(value_system_instance, '_score_truthfulness', new_callable=AsyncMock) as mock_truth, \
         patch.object(value_system_instance, '_score_ethical_alignment', new_callable=AsyncMock) as mock_ethical, \
         patch.object(value_system_instance, '_score_self_improvement', new_callable=AsyncMock) as mock_self_imp:
        
        _VJ_mock_global = globals().get('ValueJudgment')
        _VC_mock_global = globals().get('ValueCategory')

        async def async_mock_judgment(category_member, score=0.1): 
            cat_to_use = category_member
            if isinstance(category_member, str) and _VC_mock_global and hasattr(_VC_mock_global, category_member):
                 cat_to_use = getattr(_VC_mock_global, category_member)

            return _VJ_mock_global(value_category=cat_to_use, score=score, reason="mocked", confidence=1.0) # type: ignore

        mock_safety.return_value = await async_mock_judgment(ValueCategory.SAFETY)
        mock_efficiency.return_value = await async_mock_judgment(ValueCategory.EFFICIENCY)
        mock_affective.return_value = await async_mock_judgment(ValueCategory.AFFECTIVE_BALANCE)
        mock_knowledge.return_value = await async_mock_judgment(ValueCategory.KNOWLEDGE_GAIN)
        mock_user.return_value = await async_mock_judgment(ValueCategory.USER_SATISFACTION)
        mock_resource.return_value = await async_mock_judgment(ValueCategory.RESOURCE_PRESERVATION)
        mock_goal.return_value = await async_mock_judgment(ValueCategory.GOAL_ACHIEVEMENT)
        mock_truth.return_value = await async_mock_judgment(ValueCategory.TRUTHFULNESS)
        mock_ethical.return_value = await async_mock_judgment(ValueCategory.ETHICAL_ALIGNMENT)
        mock_self_imp.return_value = await async_mock_judgment(ValueCategory.SELF_IMPROVEMENT)

        judgments = await value_system_instance.evaluate_action_consequences(action, context)

        mock_safety.assert_called_once_with(action, context)
        mock_efficiency.assert_called_once_with(action, context)
        mock_affective.assert_called_once_with(action, context)
        mock_knowledge.assert_called_once_with(action, context)
        mock_user.assert_called_once_with(action, context)
        mock_resource.assert_called_once_with(action, context)
        mock_goal.assert_called_once_with(action, context)
        mock_truth.assert_called_once_with(action, context)
        mock_ethical.assert_called_once_with(action, context)
        mock_self_imp.assert_called_once_with(action, context)
        
        assert len(judgments) == 10 


@pytest.mark.asyncio
async def test_evaluate_plan_alignment_empty_plan(value_system_instance: ValueSystem):
    mock_goal = MagicMock(spec=Goal) 
    if MODELS_AVAILABLE and Goal: mock_goal = Goal(description="Test Goal Empty Plan")
    else: mock_goal.description = "Test Goal Empty Plan" 

    score, judgments, suggestions = await value_system_instance.evaluate_plan_alignment(
        plan=[], goal=mock_goal, context={}
    )
    assert score == 0.0
    assert len(judgments) == 0
    assert len(suggestions) == 0

@pytest.mark.asyncio
async def test_evaluate_plan_alignment_single_safe_action(value_system_instance: ValueSystem):
    mock_goal = MagicMock(spec=Goal)
    if MODELS_AVAILABLE and Goal: mock_goal = Goal(description="Safe Action Goal", priority=3.0)
    else: mock_goal.description = "Safe Action Goal"; mock_goal.priority = 3.0

    plan = [{"type": "THINKING", "params": {"content": "safe thought"}}]
    context = { 
        "timestamp": time.time(),
        "php_levels": {"pain": 0, "happiness": 7, "purpose": 7},
        "active_goal_details": {"description": mock_goal.description, "priority": mock_goal.priority}
    }
    
    mock_judgment_thinking_safety = ValueJudgment(ValueCategory.SAFETY, 0.5, "Thinking is safe", 1.0)
    mock_judgment_thinking_efficiency = ValueJudgment(ValueCategory.EFFICIENCY, -0.2, "Thinking inefficient", 1.0)
    all_mock_judgments = [
        mock_judgment_thinking_safety, mock_judgment_thinking_efficiency,
        ValueJudgment(ValueCategory.AFFECTIVE_BALANCE, 0.1, "mock", 1.0),
        ValueJudgment(ValueCategory.KNOWLEDGE_GAIN, 0.0, "mock", 1.0),
        ValueJudgment(ValueCategory.USER_SATISFACTION, 0.0, "mock", 1.0),
        ValueJudgment(ValueCategory.RESOURCE_PRESERVATION, 0.2, "mock", 1.0), 
        ValueJudgment(ValueCategory.GOAL_ACHIEVEMENT, 0.05, "mock", 1.0), 
        ValueJudgment(ValueCategory.TRUTHFULNESS, 0.0, "mock", 1.0),
        ValueJudgment(ValueCategory.ETHICAL_ALIGNMENT, 0.0, "mock", 1.0),
        ValueJudgment(ValueCategory.SELF_IMPROVEMENT, 0.0, "mock", 1.0),
    ]
    
    mock_eval_actions_func = AsyncMock(return_value=all_mock_judgments) 

    with patch.object(value_system_instance, 'evaluate_action_consequences', new=mock_eval_actions_func):
        score, judgments, suggestions = await value_system_instance.evaluate_plan_alignment(
            plan=plan, goal=mock_goal, context=context
        )

    mock_eval_actions_func.assert_called_once_with(plan[0], context)
    assert len(judgments) == len(all_mock_judgments)
    assert len(suggestions) == 0 
    
    expected_weighted_score_sum = 0
    expected_total_weight_sum = 0
    for j in all_mock_judgments:
        weight = value_system_instance.value_weights.get(j.value_category, 0.0)
        expected_weighted_score_sum += j.score * j.confidence * weight
        expected_total_weight_sum += weight
    
    expected_overall_score = 0.0
    if expected_total_weight_sum > 0:
        expected_overall_score = expected_weighted_score_sum / expected_total_weight_sum
    
    expected_rounded_score = round(expected_overall_score, 3)
    assert score == pytest.approx(expected_rounded_score, abs=1e-4) 


@pytest.mark.asyncio
async def test_evaluate_plan_alignment_suggests_modification_for_safety(value_system_instance: ValueSystem):
    mock_goal = MagicMock(spec=Goal)
    if MODELS_AVAILABLE and Goal: mock_goal = Goal(description="Risky Plan Goal")
    else: mock_goal.description = "Risky Plan Goal"

    plan = [{"type": "WRITE_FILE", "params": {"path": "/etc/some_config"}}]
    context = {"timestamp": time.time()} 
    
    score, judgments, suggestions = await value_system_instance.evaluate_plan_alignment(
        plan=plan, goal=mock_goal, context=context
    )

    assert score < 0 
    assert len(suggestions) == 1
    assert 0 in suggestions 
    assert suggestions[0]["type"] == "CHANGE_PARAM"
    assert suggestions[0]["param_name"] == "path"
    assert "critical system path" in suggestions[0]["original_reason"].lower()
    
    safety_judgment_found = any(
        j.value_category == ValueCategory.SAFETY and j.score < value_system_instance.safety_modification_trigger_threshold 
        for j in judgments
    )
    assert safety_judgment_found

@pytest.mark.asyncio
async def test_resolve_conflicts_no_conflicts(value_system_instance: ValueSystem):
    judgments = [
        ValueJudgment(ValueCategory.SAFETY, 0.8, "All good", 1.0),
        ValueJudgment(ValueCategory.EFFICIENCY, 0.7, "Efficient", 1.0),
    ]
    resolved_judgments, summary = await value_system_instance.resolve_value_conflicts_hierarchy(judgments)
    assert len(resolved_judgments) == 2 
    assert "no specific value conflicts identified" in summary.lower()

@pytest.mark.asyncio
async def test_resolve_conflicts_critical_safety_override(value_system_instance: ValueSystem):
    value_system_instance.action_safety_veto_threshold = -0.7 
    
    judgments = [
        ValueJudgment(ValueCategory.SAFETY, -0.75, "Very unsafe action", 1.0),
        ValueJudgment(ValueCategory.EFFICIENCY, 0.9, "Super efficient but unsafe", 1.0),
        ValueJudgment(ValueCategory.GOAL_ACHIEVEMENT, 0.8, "Achieves goal but unsafe", 1.0),
    ]
    resolved_judgments, summary = await value_system_instance.resolve_value_conflicts_hierarchy(judgments)
    summary_lower = summary.lower()
    assert "critical safety concern" in summary_lower
    assert "overrides" in summary_lower
    assert "-0.75" in summary_lower 
    assert f"threshold: {value_system_instance.action_safety_veto_threshold:.2f}" in summary_lower


@pytest.mark.asyncio
async def test_resolve_conflicts_tradeoff_matrix_preference(value_system_instance: ValueSystem):
    if MODELS_AVAILABLE: 
        value_system_instance.tradeoff_matrix[ValueCategory.TRUTHFULNESS] = {ValueCategory.USER_SATISFACTION: 0.6} # type: ignore

    judgments = [
        ValueJudgment(ValueCategory.SAFETY, 0.1, "Moderately safe", 1.0), 
        ValueJudgment(ValueCategory.TRUTHFULNESS, 0.8, "Very truthful statement", 1.0),
        ValueJudgment(ValueCategory.USER_SATISFACTION, -0.6, "User will be unhappy with this truth", 1.0),
        ValueJudgment(ValueCategory.EFFICIENCY, 0.2, "Okay efficiency", 1.0),
    ]
    resolved_judgments, summary = await value_system_instance.resolve_value_conflicts_hierarchy(judgments)
    summary_lower = summary.lower()
    
    assert "critical safety concern" not in summary_lower 
    assert "truthfulness" in summary_lower 
    assert "user_satisfaction" in summary_lower 
    assert "is preferred" in summary_lower 
    assert "tradeoff m[truthfulness][user_satisfaction]=0.60" in summary_lower.replace(" (avg_eff: 0.80) and user_satisfaction (avg_eff: -0.60). ", " ")


@pytest.mark.asyncio
async def test_resolve_conflicts_tradeoff_matrix_no_strong_opposition(value_system_instance: ValueSystem):
    judgments = [
        ValueJudgment(ValueCategory.SAFETY, 0.2, "Safe enough", 1.0),
        ValueJudgment(ValueCategory.TRUTHFULNESS, 0.7, "Mostly truthful", 1.0),
        ValueJudgment(ValueCategory.USER_SATISFACTION, -0.3, "User slightly miffed", 1.0),
    ]
    resolved_judgments, summary = await value_system_instance.resolve_value_conflicts_hierarchy(judgments)
    assert "no specific value conflicts identified by hierarchy/tradeoff" in summary.lower()

@pytest.mark.asyncio
async def test_evaluate_plan_alignment_calls_conflict_resolution(value_system_instance: ValueSystem, caplog):
    mock_goal = MagicMock(spec=Goal)
    if MODELS_AVAILABLE and Goal: 
        mock_goal = Goal(description="Conflict Test Goal", priority=3.0)
    else: 
        mock_goal.description = "Conflict Test Goal"; mock_goal.priority = 3.0

    plan = [{"type": "SOME_ACTION", "params": {}}] 
    context = {
        "timestamp": time.time(),
        "php_levels": {"pain": 0, "happiness": 7, "purpose": 7},
        "active_goal_details": {"description": mock_goal.description, "priority": mock_goal.priority}
    }
    
    mock_action_judgments = [
        ValueJudgment(ValueCategory.SAFETY, -0.5, "Borderline safety", 1.0), 
        ValueJudgment(ValueCategory.EFFICIENCY, 0.8, "Very efficient", 1.0),
        ValueJudgment(ValueCategory.KNOWLEDGE_GAIN, 0.1, "Some gain", 1.0) 
    ]
    
    expected_conflict_summary_string_from_resolve = "mocked conflict safety vs efficiency" 
    expected_conflict_summary_key_terms = ["mocked conflict", "safety vs efficiency"] 
    
    mock_eval_actions_func = AsyncMock(return_value=mock_action_judgments)
    mock_resolve_conflicts_func = AsyncMock(return_value=(mock_action_judgments, expected_conflict_summary_string_from_resolve))


    with patch.object(value_system_instance, 'evaluate_action_consequences', new=mock_eval_actions_func), \
         patch.object(value_system_instance, 'resolve_value_conflicts_hierarchy', new=mock_resolve_conflicts_func):
        
        caplog.set_level(logging.INFO) 

        score, judgments, suggestions = await value_system_instance.evaluate_plan_alignment(
            plan=plan, goal=mock_goal, context=context
        )

    assert mock_eval_actions_func.call_count == len(plan)
    mock_eval_actions_func.assert_any_call(plan[0], context)

    mock_resolve_conflicts_func.assert_called_once_with(mock_action_judgments) 

    found_log_message = False
    for record in caplog.records:
        if record.levelname == "INFO" and "VS_PLAN_CONFLICT_SUMMARY" in record.message: 
            if all(term.lower() in record.message.lower() for term in expected_conflict_summary_key_terms):
                found_log_message = True
                logging.debug(f"Found matching log record for conflict summary: {record.message}")
                break 
    
    if not found_log_message:
        logged_info_messages = [rec.message for rec in caplog.records if rec.levelname == "INFO"]
        assert found_log_message, (
            f"Expected conflict summary log message not found or did not contain all key terms. "
            f"Expected terms: {expected_conflict_summary_key_terms}. "
            f"Captured INFO logs: {logged_info_messages}"
        )
    
    assert len(judgments) == len(mock_action_judgments) 
    assert isinstance(score, float)
    assert isinstance(suggestions, dict)

@pytest.mark.skipif(not MODELS_AVAILABLE, reason="VS models not available.")
class TestAdvancedValueScoring:

    @pytest.mark.asyncio
    async def test_score_ethical_alignment_harmful_action_no_justification(self, value_system_instance: ValueSystem):
        action = {"type": "DELETE_FILE", "params": {"path": "important_user_doc.txt"}}
        context = {} 
        
        with patch.object(value_system_instance, '_assess_safety_justification', return_value=0.0):
            judgment = await value_system_instance._score_ethical_alignment(action, context)

        assert judgment.value_category == ValueCategory.ETHICAL_ALIGNMENT
        assert judgment.score == pytest.approx(0.13, abs=1e-2) 
        assert "harm:" in judgment.reason.lower()

    @pytest.mark.asyncio
    async def test_score_ethical_alignment_harmful_action_with_justification(self, value_system_instance: ValueSystem):
        action = {"type": "DELETE_FILE", "params": {"path": "temp_log.txt"}}
        context = {"user_interaction_flags": {"explicitly_confirmed_risky_file_op": True}}
        
        judgment = await value_system_instance._score_ethical_alignment(action, context)
        
        assert judgment.value_category == ValueCategory.ETHICAL_ALIGNMENT
        assert judgment.score == pytest.approx(0.463, abs=1e-2) 
        
        reason_lower = judgment.reason.lower()
        assert "harm:" in reason_lower
        assert "delete_file inherently risks data loss" in reason_lower 
        assert "justification strength 0.85" in reason_lower 
        assert "overall weighted score" in reason_lower

    @pytest.mark.asyncio
    async def test_score_truthfulness_hedging_language(self, value_system_instance: ValueSystem):
        action = {"type": "RESPOND_TO_USER", "params": {"text": "I believe the answer is probably 42."}}
        context = {}
        judgment = await value_system_instance._score_truthfulness(action, context)
        assert judgment.value_category == ValueCategory.TRUTHFULNESS
        assert judgment.score == pytest.approx(0.25) 
        assert "hedging language" in judgment.reason.lower()

    @pytest.mark.asyncio
    async def test_score_truthfulness_call_llm_for_facts(self, value_system_instance: ValueSystem):
        action = {"type": "CALL_LLM", "params": {"prompt": "What is the capital of France?"}}
        context = {} 
        judgment = await value_system_instance._score_truthfulness(action, context)
        assert judgment.value_category == ValueCategory.TRUTHFULNESS
        assert judgment.score == pytest.approx(-0.3, abs=1e-2) 
        assert "llm prompt explicitly seeks factual information" in judgment.reason.lower()

@pytest.mark.asyncio
async def test_evaluate_goal_desirability_knowledge_gain_goal(value_system_instance: ValueSystem):
    _Goal_test = globals().get('Goal', type('MockGoal', (object,), {})) 
    if not MODELS_AVAILABLE and not hasattr(_Goal_test, 'description'): 
        pytest.skip("Goal model not sufficiently defined for this test.")

    goal_desc = "learn about photosynthesis"
    test_goal = _Goal_test(description=goal_desc, priority=3.0, id="kg_goal_1")
    
    context = {
        "timestamp": time.time(),
        "php_levels": {"pain": 0.5, "happiness": 6.0, "purpose": 5.5},
        "active_pain_sources_summary": [], 
        "dsm_summary": {"capabilities": {}}, 
        "current_cs_level_name": "CONSCIOUS",
    }

    desirability_score, judgments = await value_system_instance.evaluate_goal_desirability(test_goal, context) # type: ignore

    assert isinstance(desirability_score, float)
    assert -1.0 <= desirability_score <= 1.0
    
    found_kg_judgment = False
    for judgment in judgments:
        if judgment.value_category == ValueCategory.KNOWLEDGE_GAIN:
            found_kg_judgment = True
            assert judgment.score == pytest.approx(0.62, abs=1e-2) 
            assert "knowledge acquisition" in judgment.reason.lower()
            assert judgment.target_entity_id == "kg_goal_1"
            assert judgment.target_entity_type == "goal"
            break
    assert found_kg_judgment, "KNOWLEDGE_GAIN judgment not found for a knowledge-seeking goal."
    logging.info(f"Knowledge Gain Goal Test - Score: {desirability_score:.3f}, Judgments: {[(j.value_category, j.score) for j in judgments]}")


@pytest.mark.asyncio
async def test_evaluate_goal_desirability_user_satisfaction_goal(value_system_instance: ValueSystem):
    _Goal_test = globals().get('Goal', type('MockGoal', (object,), {}))
    if not MODELS_AVAILABLE and not hasattr(_Goal_test, 'description'):
        pytest.skip("Goal model not sufficiently defined for this test.")
        
    user_goal_priority_ref = 5.0 
    goal_desc = "respond to user query: what is the weather"
    test_goal = _Goal_test(description=goal_desc, priority=user_goal_priority_ref, id="us_goal_1")

    context = {
        "timestamp": time.time(),
        "php_levels": {"pain": 0.2, "happiness": 7.0, "purpose": 6.0},
        "active_pain_sources_summary": [],
        "dsm_summary": {"capabilities": {}},
        "current_cs_level_name": "CONSCIOUS",
    }
    if value_system_instance._controller:
        value_system_instance._controller.USER_GOAL_PRIORITY = user_goal_priority_ref


    desirability_score, judgments = await value_system_instance.evaluate_goal_desirability(test_goal, context) # type: ignore

    assert isinstance(desirability_score, float)
    
    found_us_judgment = False
    for judgment in judgments:
        if judgment.value_category == ValueCategory.USER_SATISFACTION:
            found_us_judgment = True
            assert judgment.score == pytest.approx(1.0, abs=0.01) 
            assert "user interaction/request" in judgment.reason.lower()
            break
    assert found_us_judgment, "USER_SATISFACTION judgment not found for a user-centric goal."
    logging.info(f"User Satisfaction Goal Test - Score: {desirability_score:.3f}, Judgments: {[(j.value_category, j.score) for j in judgments]}")

@pytest.mark.asyncio
async def test_evaluate_goal_desirability_affective_balance_pain_resolution(value_system_instance: ValueSystem):
    _Goal_test = globals().get('Goal', type('MockGoal', (object,), {}))
    if not MODELS_AVAILABLE and not hasattr(_Goal_test, 'description'):
        pytest.skip("Goal model not sufficiently defined for this test.")

    pain_source_id_to_resolve = "pain_event_xyz"
    goal_desc = f"address problem causing pain source {pain_source_id_to_resolve}"
    test_goal = _Goal_test(description=goal_desc, priority=6.0, id=pain_source_id_to_resolve) 

    context = {
        "timestamp": time.time(),
        "php_levels": {"pain": 7.5, "happiness": 2.0, "purpose": 3.0}, # High pain
        "active_pain_sources_summary": [
            {"id": pain_source_id_to_resolve, "intensity": 7.0, "type": "TestPain"}
        ],
        "dsm_summary": {},
        "current_cs_level_name": "CONSCIOUS",
    }

    desirability_score, judgments = await value_system_instance.evaluate_goal_desirability(test_goal, context) # type: ignore

    assert isinstance(desirability_score, float)
    
    found_ab_judgment = False
    for judgment in judgments:
        if judgment.value_category == ValueCategory.AFFECTIVE_BALANCE:
            found_ab_judgment = True
            assert judgment.score == pytest.approx(0.7, abs=1e-2) 
            assert f"addresses active pain source '{pain_source_id_to_resolve}'" in judgment.reason.lower()
            break
    assert found_ab_judgment, "AFFECTIVE_BALANCE judgment not found or too low for pain resolution goal."
    logging.info(f"Affective Balance (Pain Resolution) Goal Test - Score: {desirability_score:.3f}, Judgments: {[(j.value_category, j.score) for j in judgments]}")


@pytest.mark.asyncio
async def test_evaluate_goal_desirability_self_improvement_low_dsm_confidence(value_system_instance: ValueSystem):
    _Goal_test = globals().get('Goal', type('MockGoal', (object,), {}))
    if not MODELS_AVAILABLE and not hasattr(_Goal_test, 'description'):
        pytest.skip("Goal model not sufficiently defined for this test.")

    goal_desc = "learn to use the READ_FILE action effectively" 
    test_goal = _Goal_test(description=goal_desc, priority=4.0, id="si_goal_1")

    context = {
        "timestamp": time.time(),
        "php_levels": {"pain": 1.0, "happiness": 5.0, "purpose": 4.0},
        "active_pain_sources_summary": [],
        "dsm_summary": { 
            "capabilities": {"action:READ_FILE": 0.2} 
        },
        "current_cs_level_name": "CONSCIOUS",
    }

    desirability_score, judgments = await value_system_instance.evaluate_goal_desirability(test_goal, context) # type: ignore

    assert isinstance(desirability_score, float)
    
    found_si_judgment = False
    for judgment in judgments:
        if judgment.value_category == ValueCategory.SELF_IMPROVEMENT:
            found_si_judgment = True
            assert judgment.score == pytest.approx(0.62, abs=0.01)
            assert "explicitly targets learning" in judgment.reason.lower() or \
                   "offering learning potential" in judgment.reason.lower()
            break
    assert found_si_judgment, "SELF_IMPROVEMENT judgment not found or incorrect for learning goal with low DSM confidence."
    logging.info(f"Self-Improvement (Low DSM Conf) Goal Test - Score: {desirability_score:.3f}, Judgments: {[(j.value_category, j.score) for j in judgments]}")

@pytest.mark.asyncio
async def test_evaluate_goal_desirability_default_observe_goal(value_system_instance: ValueSystem):
    _Goal_test = globals().get('Goal', type('MockGoal', (object,), {}))
    if not MODELS_AVAILABLE and not hasattr(_Goal_test, 'description'):
        pytest.skip("Goal model not sufficiently defined for this test.")

    default_observe_desc = "Observe and learn from the environment" 
    default_observe_prio = 1.0 

    test_goal = _Goal_test(description=default_observe_desc, priority=default_observe_prio, id="observe_goal_1")
    
    context = {
        "timestamp": time.time(),
        "php_levels": {"pain": 0.1, "happiness": 5.0, "purpose": 5.0},
        "active_pain_sources_summary": [],
        "dsm_summary": {},
        "current_cs_level_name": "CONSCIOUS",
    }

    desirability_score, judgments = await value_system_instance.evaluate_goal_desirability(test_goal, context) # type: ignore

    assert isinstance(desirability_score, float)
    assert 0.0 < desirability_score < 0.5 

    kg_score_found = 0.0
    ga_score_found = 0.0
    for judgment in judgments:
        if judgment.value_category == ValueCategory.KNOWLEDGE_GAIN:
            kg_score_found = judgment.score
            assert "knowledge acquisition" in judgment.reason.lower()
        if judgment.value_category == ValueCategory.GOAL_ACHIEVEMENT:
            ga_score_found = judgment.score
            assert "goal priority" in judgment.reason.lower()
            
    assert kg_score_found == pytest.approx(0.54, abs=0.01) 
    assert ga_score_found == pytest.approx(0.1, abs=0.01) 
    logging.info(f"Default Observe Goal Test - Score: {desirability_score:.3f}, Judgments: {[(j.value_category, j.score) for j in judgments]}")