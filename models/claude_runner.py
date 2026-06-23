from __future__ import annotations

import time

from config import settings
from models.base_runner import BaseModelRunner, ModelOutput


class ClaudeRunner(BaseModelRunner):
    def __init__(self, runtime_mode: str = "mock", model_name: str = "claude-3-5-haiku-latest") -> None:
        super().__init__(model_name=model_name, runtime_mode=runtime_mode)

    def generate(self, prompt: str, category: str = "coding") -> ModelOutput:
        if self.runtime_mode == "mock":
            return self.make_mock_output(prompt, category)

        if not settings.anthropic_api_key:
            return ModelOutput(
                model_name=self.model_name,
                response_text="",
                latency_ms=0,
                input_tokens=0,
                output_tokens=0,
                error="ANTHROPIC_API_KEY is not configured",
            )

        try:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            start = time.perf_counter()
            response = client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            latency_ms = (time.perf_counter() - start) * 1000
            text = ""
            if response.content and len(response.content) > 0:
                text = response.content[0].text
            return ModelOutput(
                model_name=self.model_name,
                response_text=text,
                latency_ms=latency_ms,
                input_tokens=self.estimate_tokens(prompt),
                output_tokens=self.estimate_tokens(text),
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
