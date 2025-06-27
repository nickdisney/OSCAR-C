# tests/unit/htn_test_mocks.py
import asyncio # Ensure asyncio is imported
import time # Added for Predicate/Goal fallbacks
from typing import Dict, Any, Optional, List, Tuple, Union
import logging # For logger_test_htn

logger_test_htn = logging.getLogger(__name__)


# --- Mock Predicate and Goal for tests if not importable ---
# (Assuming MODELS_AVAILABLE might be False in some test environments)
try:
    from consciousness_experiment.models.datatypes import Predicate as RealPredicate
    from consciousness_experiment.models.datatypes import Goal as RealGoal
    MODELS_AVAILABLE_HTN_MOCK = True
except ImportError:
    MODELS_AVAILABLE_HTN_MOCK = False
    class RealPredicate:
        def __init__(self, name: str, args: Tuple[Any, ...], value: bool = True, timestamp: Optional[float] = None):
            self.name = name
            self.args = args
            self.value = value
            self.timestamp = timestamp if timestamp is not None else time.time()
        def __eq__(self, other):
            if not isinstance(other, RealPredicate): return NotImplemented
            return self.name == other.name and self.args == other.args and self.value == other.value
        def __hash__(self): return hash((self.name, self.args, self.value))

    class RealGoal:
        def __init__(self, description: str, id: str = "mock_goal_id", priority: float = 1.0,
                     success_criteria: Optional[set] = None, failure_criteria: Optional[set] = None,
                     creation_time: Optional[float] = None):
            self.description = description
            self.id = id
            self.priority = priority
            self.success_criteria = success_criteria if success_criteria is not None else set()
            self.failure_criteria = failure_criteria if failure_criteria is not None else set()
            self.creation_time = creation_time if creation_time is not None else time.time()

Predicate = RealPredicate
Goal = RealGoal

# --- Mock LLM Responses ---

MOCK_LLM_SKETCH_SUCCESS_READ_FILE = {
    "response_str": """
    ```json
    [
        ["READ_FILE", "data/file_to_read.txt"]
    ]
    ```
    Precondition Hints: The file 'data/file_to_read.txt' must exist. Agent should have read permissions.
    """,
    "error": None
}

MOCK_LLM_SKETCH_MULTI_STEP_WRITE_THEN_READ = {
    "response_str": """
    Okay, here's a plan:
    ```json
    [
        ["WRITE_FILE", "data/output_for_llm.txt", "Hello from LLM sketch!"],
        ["READ_FILE", "data/output_for_llm.txt"]
    ]
    ```
    This plan first writes content and then reads it back.
    Precondition Hints: Agent must have write access to 'data/output_for_llm.txt'.
    """,
    "error": None
}

MOCK_LLM_SKETCH_FAILURE_NULL_RESPONSE = {
    "response_str": "null", # LLM indicates no plan found
    "error": None
}

MOCK_LLM_SKETCH_FAILURE_INVALID_JSON = {
    "response_str": "I'm sorry, I can't create a plan for that. ```json [[\"INVALID\"```",
    "error": None
}

MOCK_LLM_SKETCH_FAILURE_UNKNOWN_OPERATOR = {
    "response_str": """
    ```json
    [
        ["NON_EXISTENT_OPERATOR", "param1"]
    ]
    ```
    """,
    "error": None
}

MOCK_LLM_SKETCH_FAILURE_PARAM_MISMATCH = {
    "response_str": """
    ```json
    [
        ["READ_FILE", "path_param_1", "extra_param_not_expected"]
    ]
    ```
    """,
    "error": None
}

MOCK_LLM_SKETCH_API_ERROR = {
    "response_str": None,
    "error": "Ollama API Error: Connection refused"
}

MOCK_LLM_SKETCH_SUCCESS_NO_PARAMS_OP = {
    "response_str": """
    ```json
    [
        ["OBSERVE_SYSTEM"]
    ]
    ```
    """,
    "error": None
}

MOCK_LLM_SKETCH_SUCCESS_QUERY_KB_CORRECT_PARAMS = {
    "response_str": """
    ```json
    [
        ["QUERY_KB", "isFile", ["data/some_file.txt"], true]
    ]
    ```
    Precondition Hints: None needed for this specific query.
    """,
    "error": None
}


# --- Mock AgentController for HTNPlanner Tests ---
# This mock only needs specific attributes/methods used by HTNPlanner's LLM sketching path

class MockAgentControllerForHTN:
    def __init__(self, mock_llm_response_queue: Optional[asyncio.Queue] = None):
        self.model_name = "test_llm_model"
        try:
            self._asyncio_loop = asyncio.get_running_loop()
        except RuntimeError: # Fallback if no loop is running (e.g. module import time)
            self._asyncio_loop = asyncio.new_event_loop() 
            asyncio.set_event_loop(self._asyncio_loop)

        self.mock_llm_response_queue = mock_llm_response_queue if mock_llm_response_queue else asyncio.Queue()
        
        self.mock_completion_events: Dict[asyncio.Future, asyncio.Event] = {}
        self._background_tasks: Set[asyncio.Task] = set() # To keep track of created tasks

        self.knowledge_base = None 
        self.config = { 
            "htn_planner": {
                "llm_sketch_timeout_s": 10.0,
                "llm_sketch_temperature": 0.1
            }
        }
        self.agent_root_path = "." 

    def schedule_offline_task(
        self,
        coro_func: Any, 
        *args: Any,
        callback_on_done: Optional[Any] = None, 
        **kwargs: Any
    ) -> asyncio.Future:
        
        task_future: asyncio.Future = self._asyncio_loop.create_future()
        
        completion_event = asyncio.Event()
        self.mock_completion_events[task_future] = completion_event

        async def _complete_future_with_mock(future_to_complete: asyncio.Future, event_to_set: asyncio.Event):
            item_retrieved_successfully = False
            try:
                if self._asyncio_loop.is_closed():
                    if not future_to_complete.done():
                        future_to_complete.set_exception(RuntimeError("Event loop closed before mock task could run"))
                    return

                mock_response_tuple = await asyncio.wait_for(self.mock_llm_response_queue.get(), timeout=1.0)
                item_retrieved_successfully = True

                if not future_to_complete.done():
                    future_to_complete.set_result(mock_response_tuple)

            except asyncio.TimeoutError:
                if not future_to_complete.done():
                    future_to_complete.set_result((None, "Mock LLM response queue timed out"))
            except asyncio.CancelledError: # Handle task cancellation
                if not future_to_complete.done():
                    future_to_complete.set_exception(asyncio.CancelledError("Mock task was cancelled"))
                raise # Re-raise to ensure the task is properly marked as cancelled
            except Exception as e:
                if not future_to_complete.done():
                    future_to_complete.set_exception(e)
            finally:
                if not self._asyncio_loop.is_closed(): # Check loop state before task_done
                    if item_retrieved_successfully and hasattr(self.mock_llm_response_queue, 'task_done'):
                        try:
                            self.mock_llm_response_queue.task_done()
                        except ValueError as ve:
                            logger_test_htn.error(f"MockAgentController: Error in task_done during mock completion: {ve}")
                        except RuntimeError as rterr: # E.g. if loop closed during task_done call
                            logger_test_htn.error(f"MockAgentController: RuntimeError in task_done: {rterr}")


                    event_to_set.set()
                    if future_to_complete in self.mock_completion_events:
                        del self.mock_completion_events[future_to_complete]

        # Create task and add it to a set to manage its lifecycle if needed
        task = asyncio.create_task(_complete_future_with_mock(task_future, completion_event))
        self._background_tasks.add(task)
        # Ensure tasks are cleaned up when they are done to prevent resource leaks
        task.add_done_callback(self._background_tasks.discard)
        
        return task_future

    async def wait_for_mock_completion(self, future_obj: asyncio.Future, timeout: float = 2.0):
        if future_obj in self.mock_completion_events:
            try:
                await asyncio.wait_for(self.mock_completion_events[future_obj].wait(), timeout=timeout)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Mock completion event for future {future_obj} timed out after {timeout}s.")

    # Helper to cancel any remaining background tasks, useful in test teardown
    async def cleanup_background_tasks(self):
        if self._background_tasks:
            for task in list(self._background_tasks): # Iterate over a copy
                if not task.done():
                    task.cancel()
                    try:
                        await task # Allow task to process cancellation
                    except asyncio.CancelledError:
                        pass # Expected
                    except Exception as e:
                        logger_test_htn.error(f"Error during background task cleanup: {e}")
            self._background_tasks.clear()


    def _get_active_goal_type(self) -> str:
        return "mock_test_goal_type"