import httpx

from tagnify.backends.base import BaseBackend
from tagnify.exceptions import BackendError


class OllamaBackend(BaseBackend):
    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_TIMEOUT = 120.0

    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        timeout: float = DEFAULT_TIMEOUT,
        temperature: float = 0.1,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    def complete(self, prompt: str) -> str:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
        except httpx.ConnectError as e:
            raise BackendError(
                f"Could not connect to Ollama at {self.host}. "
                f"Is Ollama running? Start it with: ollama serve"
            ) from e
        except httpx.TimeoutException as e:
            raise BackendError(
                f"Ollama did not respond within {self.timeout}s. "
                f"The model may be loading. Try again, or increase timeout."
            ) from e
        except httpx.HTTPStatusError as e:
            raise BackendError(
                f"Ollama returned HTTP {e.response.status_code}. "
                f"Response: {e.response.text}"
            ) from e

        data = response.json()
        return data["message"]["content"]