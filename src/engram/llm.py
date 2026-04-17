"""LLM client — thin wrapper for OpenAI-compatible API endpoints.

Supports Ollama, llama.cpp, LM Studio, and any OpenAI-compatible API.
Handles streaming, health checks, and connection error wrapping.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Iterator, Optional

import httpx

logger = logging.getLogger(__name__)


class EngramConnectionError(Exception):
    """Raised when the LLM endpoint is unreachable."""
    pass


class LLMClient:
    """OpenAI-compatible LLM API client.
    
    Features:
    - Auto-detection of Ollama at localhost:11434
    - Streaming chat completions
    - Structured extraction calls
    - Health check with helpful error messages
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)
        self._is_ollama: Optional[bool] = None

    # ---------- Health check ----------

    def check_health(self) -> bool:
        """Check if the LLM endpoint is reachable.
        
        Returns:
            True if the endpoint responds.
            
        Raises:
            EngramConnectionError with helpful message if unreachable.
        """
        try:
            # Try Ollama-style health check first
            resp = self._client.get(f"{self.base_url}/api/tags")
            if resp.status_code == 200:
                self._is_ollama = True
                return True
        except httpx.ConnectError:
            pass

        try:
            # Try OpenAI-style health check
            resp = self._client.get(f"{self.base_url}/v1/models")
            if resp.status_code == 200:
                self._is_ollama = False
                return True
        except httpx.ConnectError:
            pass

        raise EngramConnectionError(
            f"Cannot connect to LLM at {self.base_url}.\n"
            f"If using Ollama, make sure it's running: ollama serve\n"
            f"Or set a custom endpoint: ENGRAM_LLM_BASE_URL=http://your-llm:port"
        )

    def is_ollama(self) -> bool:
        """Check if the connected endpoint is Ollama."""
        if self._is_ollama is None:
            try:
                self.check_health()
            except EngramConnectionError:
                return False
        return self._is_ollama or False

    # ---------- Chat completions ----------

    def chat(
        self,
        model: str,
        messages: list[dict],
        stream: bool = True,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Send a chat completion request and yield response chunks.
        
        Args:
            model: Model name (e.g., "llama3", "mistral").
            messages: Chat messages in OpenAI format.
            stream: If True, yield chunks as they arrive.
            temperature: Sampling temperature.
            
        Yields:
            Response text chunks (if streaming) or full response (if not).
        """
        # Determine the right endpoint
        if self.is_ollama():
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "options": {"temperature": temperature},
            }
        else:
            url = f"{self.base_url}/v1/chat/completions"
            payload = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "temperature": temperature,
            }

        try:
            if stream:
                yield from self._stream_response(url, payload)
            else:
                yield self._blocking_response(url, payload)
        except httpx.ConnectError:
            raise EngramConnectionError(
                f"Lost connection to LLM at {self.base_url}. "
                "Is the model still running?"
            )

    def chat_full(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> str:
        """Send a chat completion and return the full response as a string.
        
        Non-streaming convenience method for extraction/contradiction checks.
        """
        chunks = list(self.chat(model, messages, stream=False, temperature=temperature))
        return "".join(chunks)

    # ---------- Model info ----------

    def get_model_info(self, model: str) -> Optional[dict]:
        """Get model metadata from Ollama (context window, etc.).
        
        Returns None if not using Ollama or request fails.
        """
        if not self.is_ollama():
            return None
        try:
            resp = self._client.post(
                f"{self.base_url}/api/show",
                json={"name": model},
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None

    def get_context_window(self, model: str) -> int:
        """Get the model's context window size.
        
        Falls back to 4096 if the info can't be retrieved.
        """
        info = self.get_model_info(model)
        if info:
            # Ollama returns model parameters including context length
            params = info.get("parameters", "")
            if isinstance(params, str):
                for line in params.split("\n"):
                    if "num_ctx" in line:
                        try:
                            return int(line.split()[-1])
                        except (ValueError, IndexError):
                            pass
            # Try modelfile info
            model_info = info.get("model_info", {})
            for key, value in model_info.items():
                if "context_length" in key:
                    try:
                        return int(value)
                    except (ValueError, TypeError):
                        pass
        return 4096  # Conservative fallback

    def list_models(self) -> list[str]:
        """List available models."""
        try:
            if self.is_ollama():
                resp = self._client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                return [m["name"] for m in resp.json().get("models", [])]
            else:
                resp = self._client.get(f"{self.base_url}/v1/models")
                resp.raise_for_status()
                return [m["id"] for m in resp.json().get("data", [])]
        except Exception:
            return []

    # ---------- Internal streaming ----------

    def _stream_response(self, url: str, payload: dict) -> Iterator[str]:
        """Handle streaming responses from both Ollama and OpenAI-style APIs."""
        with self._client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            
            if self._is_ollama:
                # Ollama streams JSON objects, one per line
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if data.get("done", False):
                            return
                    except json.JSONDecodeError:
                        continue
            else:
                # OpenAI-style SSE streaming
                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]  # Strip "data: " prefix
                    if data_str == "[DONE]":
                        return
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    def _blocking_response(self, url: str, payload: dict) -> str:
        """Handle non-streaming response."""
        payload["stream"] = False
        resp = self._client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        if self._is_ollama:
            return data.get("message", {}).get("content", "")
        else:
            return data["choices"][0]["message"]["content"]

    # ---------- Cleanup ----------

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
