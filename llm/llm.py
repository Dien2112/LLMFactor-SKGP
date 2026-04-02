"""
llm.py - Generic LLM Interface
Routes to GeminiLLMManager (geminillm.py).
When switching to OpenAI/Anthropic, just modify this file.
"""

from __future__ import annotations
import logging
from .geminillm import GeminiLLMManager

logger = logging.getLogger(__name__)


class llm:
    """
    Generic LLM Interface - delegates to GeminiLLMManager.
    Does not manage specific models or rate constraints - Manager handles that.
    """

    def __init__(
        self,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
    ):
        self.manager = GeminiLLMManager(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    @property
    def cached_calls(self) -> int:
        return self.manager.cached_calls

    @cached_calls.setter
    def cached_calls(self, value: int) -> None:
        self.manager.cached_calls = value

    def generate(self, prompt: str, use_strong_model: bool = False) -> str:
        """Call LLM, internally handling round-robin/retry."""
        return self.manager.generate(prompt, use_strong_model=use_strong_model)

