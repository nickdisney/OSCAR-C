# consciousness_experiment/models/exceptions.py

class SecurityException(Exception):
    """Custom exception for security-related issues, e.g., path traversal."""
    pass

class AgentControllerException(Exception):
    """Base exception for AgentController specific errors."""
    pass

class ComponentInitializationError(AgentControllerException):
    """Raised when a cognitive component fails to initialize."""
    pass

# Add other custom exceptions as needed.