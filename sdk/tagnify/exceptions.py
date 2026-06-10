#Tagnify error hierarchy

class TagnifyError(Exception):
    """Base class for all Tagnify errors."""
    pass

class SchemaValidationError(TagnifyError):
    """Raised when a schema is defined incorrectly."""
    pass

class OutputParserError(TagnifyError):
    """Raised when the LLM output cannot be parsed."""
    pass

class ValidationError(TagnifyError):
    """Raised when the parsed output is invalid."""
    pass

class BackendError(TagnifyError):
    """Raised when the LLM backend fails at API level."""
    pass

class MaxRetriesExceededError(TagnifyError):
    """Raised when the number of retries is exhausted."""
    pass
