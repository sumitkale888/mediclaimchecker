"""Groq LLM client for structured verification responses."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.core.config import settings
from app.core.exceptions import LLMServiceError
from app.models.schemas import LLMVerificationResult
from app.prompts.verification_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class LLMService:
    """Call Groq and parse a structured verification payload."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key_override = api_key
        self._model_override = model
        self.api_key = (api_key if api_key is not None else settings.groq_api_key).strip()
        self.model = (model or settings.groq_model).strip()
        self._client = None
        self._client_key: str | None = None

    def _refresh_config(self) -> None:
        if self._api_key_override is None:
            self.api_key = settings.groq_api_key.strip()
        if self._model_override is None:
            self.model = settings.groq_model.strip()

    def _get_client(self):
        self._refresh_config()
        if not self.api_key:
            raise LLMServiceError(
                "GROQ_API_KEY is not configured. Set it in the .env file before verifying claims."
            )
        if self._client is None or self._client_key != self.api_key:
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
            self._client_key = self.api_key
        return self._client

    def _complete(self, messages: list[dict[str, str]], json_mode: bool) -> str:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1200,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        completion = client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content or ""

    def generate_verification(self, user_prompt: str) -> LLMVerificationResult:
        """Send the RAG prompt to Groq and validate JSON output."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            content = self._complete(messages, json_mode=True)
        except LLMServiceError:
            raise
        except Exception as exc:
            lowered = str(exc).lower()
            if "json_validate_failed" in lowered or "failed to validate json" in lowered:
                logger.warning("Groq JSON mode failed; retrying without response_format")
                try:
                    content = self._complete(messages, json_mode=False)
                except LLMServiceError:
                    raise
                except Exception as retry_exc:
                    logger.exception("Groq request failed after JSON-mode retry")
                    raise LLMServiceError(self._public_error(retry_exc)) from retry_exc
            else:
                logger.exception("Groq request failed")
                raise LLMServiceError(self._public_error(exc)) from exc

        payload = self._parse_json(content)
        try:
            return LLMVerificationResult.model_validate(payload)
        except Exception as exc:
            logger.warning("LLM returned JSON that did not match the schema")
            raise LLMServiceError("The language model returned an invalid verification result.") from exc

    def _public_error(self, exc: Exception) -> str:
        text = str(exc)
        lowered = text.lower()
        if "model_not_found" in lowered or "does not exist" in lowered:
            return (
                f"Groq model '{self.model}' is not available on this account. "
                "Update GROQ_MODEL in .env, stop every process on port 8000, and restart the server."
            )
        if "invalid api key" in lowered or "authentication" in lowered or "unauthorized" in lowered:
            return "Groq rejected the API key. Check GROQ_API_KEY in .env and restart the server."
        if "rate limit" in lowered:
            return "Groq rate limit reached. Wait a minute and try again."
        return f"The language model is currently unavailable. ({type(exc).__name__})"

    def _parse_json(self, content: str) -> dict[str, Any]:
        text = (content or "").strip()
        if not text:
            raise LLMServiceError("The language model returned an empty response.")

        candidates = [text]
        fenced = _JSON_FENCE_RE.search(text)
        if fenced:
            candidates.insert(0, fenced.group(1).strip())

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError as exc:
                last_error = exc
        logger.warning("Failed to parse LLM JSON output")
        raise LLMServiceError("The language model returned unreadable output.") from last_error
