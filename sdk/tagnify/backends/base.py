from abc import ABC, abstractmethod


class BaseBackend(ABC):
    """Base class every LLM must implement.

    Contract: one method, complete().
    Input:  a prompt string.
    Output: raw LLM response text.
    """
    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the raw response text."""
        
        raise NotImplementedError
