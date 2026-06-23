from __future__ import annotations

import time

from config import settings
from models.base_runner import BaseModelRunner, ModelOutput


class LlamaRunner(BaseModelRunner):
    """Runner for Meta Llama models via Groq's OpenAI-compatible API."""

    GROQ_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, runtime_mode: str = "mock", model_name: str = "llama-3.1-8b-instant") -> None:
        super().__init__(model_name=model_name, runtime_mode=runtime_mode)

    def generate(self, prompt: str, category: str = "coding") -> ModelOutput:
        if self.runtime_mode == "mock":
            return self.make_mock_output(prompt, category)

        api_key = getattr(settings, "groq_api_key", None)
        if not api_key:
            return ModelOutput(
                model_name=self.model_name,
                response_text="",
                latency_ms=0,
                input_tokens=0,
                output_tokens=0,
                error="GROQ_API_KEY is not configured (required for Llama inference)",
            )

        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=self.GROQ_BASE_URL)
            start = time.perf_counter()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            text = response.choices[0].message.content or ""
            in_tokens = response.usage.prompt_tokens if response.usage else self.estimate_tokens(prompt)
            out_tokens = response.usage.completion_tokens if response.usage else self.estimate_tokens(text)
            return ModelOutput(
                model_name=self.model_name,
                response_text=text,
                latency_ms=latency_ms,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
            )
        except Exception as exc:
            return ModelOutput(
                model_name=self.model_name,
                response_text="",
                latency_ms=0,
                input_tokens=0,
                output_tokens=0,
                error=str(exc),
            )
