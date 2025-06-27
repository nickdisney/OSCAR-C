import enum

class AgentState(enum.Enum):
    STOPPED = 0
    STARTING = 1
    RUNNING = 2
    STOPPING = 3
    ERROR = 4
    PAUSED = 5