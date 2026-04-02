"""
geminillm.py - Gemini LLM Manager
Manages the model pool, round-robin when quota errors (429/409) occur.
llm.py (and other modules) only import this without needing to know specific models.
"""

import logging
import os
import time
from typing import Optional

from google import genai
from google.genai import types
from google.genai import errors
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ==============================================================================
# GEMINI MODEL POOL CONFIG
# -- Select 4 Text-out models suitable for SKGP pipeline ----------
# ==============================================================================

GEMINI_MODELS_CONFIG: list[dict] = [
    
    {
        "id":       "gemma-3-1b-it",
        "rpm":      30,
        "tpm":      15_000,
        "rpd":      14_400,
    },
    {
        "id":       "gemma-3-4b-it",
        "rpm":      30,
        "tpm":      15_000,
        "rpd":      14_400,
    },
    {
        "id":       "gemma-3-12b-it",
        "rpm":      30,
        "tpm":      15_000,
        "rpd":      14_400,
    },
    {
        "id":       "gemini-2.5-flash",
        "rpm":      10,
        "tpm":      250_000,
        "rpd":      500,
    },
]

STRONG_GEMINI_MODELS_CONFIG: list[dict] = [
    {
        "id":       "gemma-3-27b-it",
        "rpm":      30,
        "tpm":      15_000,
        "rpd":      14_400,
    },
    {
        "id":       "gemini-2.5-pro",
        "rpm":      10,
        "tpm":      100_000,
        "rpd":      500,
    },
]


class GeminiLLMManager:
    """
    Manages the Gemini model pool.

    Responsibilities:
      - Initialize all models in the pool once.
      - Auto rate-limit: wait for min_interval between calls on the same model.
      - Round-robin: skip to the next model upon ResourceExhausted (429) or Conflict (409)
        instead of crashing.
      - Track usage stats (calls, tokens).
    """

    def __init__(self, temperature: float = 0.0, max_output_tokens: int = 512):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not found. "
                "Please create a .env file with GEMINI_API_KEY=<your_key>"
            )

        self.client = genai.Client(api_key=api_key)

        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        

        self.models_pool: list[dict] = []
        for conf in GEMINI_MODELS_CONFIG:
            rpm = conf["rpm"]
            self.models_pool.append({
                "config":       conf,
                "last_call_ts": 0.0,
                # min seconds between 2 calls on the same model
                "min_interval": (60.0 / rpm) if rpm and rpm > 0 else 0.0,
            })
            
        self.strong_models_pool: list[dict] = []
        for conf in STRONG_GEMINI_MODELS_CONFIG:
            rpm = conf["rpm"]
            self.strong_models_pool.append({
                "config":       conf,
                "last_call_ts": 0.0,
                "min_interval": (60.0 / rpm) if rpm and rpm > 0 else 0.0,
            })

        self.current_idx: int = 0
        self.current_strong_idx: int = 0

        # -- Usage stats --------------------------------------------------------
        self.total_calls:        int = 0
        self.cached_calls:       int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    # -- Public API -------------------------------------------------------------

    @property
    def current_model_id(self) -> str:
        """ID of the active model in the pool."""
        return self.models_pool[self.current_idx]["config"]["id"]

    def generate(self, prompt: str, use_strong_model: bool = False) -> str:
        """
        Call Gemini API.
        - Auto rate-limits.
        - Round-robins to the next model for 429 or 409 errors.
        - Raises RuntimeError if all fail after 2 full iterations.
        """
        pool = self.strong_models_pool if use_strong_model else self.models_pool
        num_models  = len(pool)
        max_attempts = num_models * 2

        for attempt in range(max_attempts):
            pool_item = pool[self.current_strong_idx if use_strong_model else self.current_idx]
            conf      = pool_item["config"]

            elapsed = time.monotonic() - pool_item["last_call_ts"]
            if elapsed < pool_item["min_interval"]:
                wait = pool_item["min_interval"] - elapsed
                logger.debug(
                    f"[GeminiLLM] {conf['id']} rate-limit: waiting {wait:.2f}s"
                )
                time.sleep(wait)

            try:
                logger.debug(
                    f"[GeminiLLM] attempt={attempt + 1}/{max_attempts} "
                    f"model={conf['id']}"
                )
                response = self.client.models.generate_content(
                    model=conf['id'],
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_output_tokens,
                        candidate_count=1,
                    )
                )

                pool_item["last_call_ts"] = time.monotonic()
                self.total_calls += 1

                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    self.total_input_tokens  += getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                    self.total_output_tokens += getattr(response.usage_metadata, "candidates_token_count", 0) or 0

                return response.text.strip()

            except errors.APIError as e:
                logger.warning(
                    f"[GeminiLLM] {conf['id']} -> API error: {e}. "
                    f"Rotating to the next model..."
                )
                if use_strong_model:
                    self._rotate_strong_model()
                else:
                    self._rotate_model()
                time.sleep(2)

            except Exception as e:
                logger.error(
                    f"[GeminiLLM] {conf['id']} -> Unexpected error: {e}. "
                    f"Rotating to the next model..."
                )
                if use_strong_model:
                    self._rotate_strong_model()
                else:
                    self._rotate_model()
                time.sleep(3)

        raise RuntimeError(
            f"[GeminiLLM] All {num_models} models failed "
            f"after {max_attempts} attempts."
        )

    def usage_summary(self) -> dict:
        return {
            "current_active_model": self.current_model_id,
            "api_calls":            self.total_calls,
            "cached_calls":         self.cached_calls,
            "input_tokens":         self.total_input_tokens,
            "output_tokens":        self.total_output_tokens,
            "total_tokens":         self.total_input_tokens + self.total_output_tokens,
        }


    def _rotate_model(self) -> None:
        """Rotate to the next model in the pool."""
        old_idx = self.current_idx
        self.current_idx = (self.current_idx + 1) % len(self.models_pool)
        logger.info(
            f"[GeminiLLM] Rotate: {self.models_pool[old_idx]['config']['id']} -> "
            f"{self.current_model_id}"
        )

    def _rotate_strong_model(self) -> None:
        """Rotate to the next strong model in the pool."""
        old_idx = self.current_strong_idx
        self.current_strong_idx = (self.current_strong_idx + 1) % len(self.strong_models_pool)
        new_model_id = self.strong_models_pool[self.current_strong_idx]['config']['id']
        logger.info(
            f"[GeminiLLM] Rotate Strong: {self.strong_models_pool[old_idx]['config']['id']} -> "
            f"{new_model_id}"
        )
