from abc import ABC, abstractmethod


class LLM(ABC):
    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> str:
        """Return the model's text response."""
