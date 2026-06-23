from __future__ import annotations

import time

from config import settings
from models.base_runner import BaseModelRunner, ModelOutput


class GeminiRunner(BaseModelRunner):
    def __init__(self, runtime_mode: str = "mock", model_name: str = "gemini-2.5-flash") -> None:
        super().__init__(model_name=model_name, runtime_mode=runtime_mode)

    def generate(self, prompt: str, category: str = "coding") -> ModelOutput:
        if self.runtime_mode == "mock":
            return self.make_mock_output(prompt, category)

        if not settings.google_api_key:
            return ModelOutput(
                model_name=self.model_name,
                response_text="",
                latency_ms=0,
                input_tokens=0,
                output_tokens=0,
                error="GOOGLE_API_KEY is not configured",
            )

        try:
            import google.generativeai as genai

            genai.configure(api_key=settings.google_api_key)
            model = genai.GenerativeModel(self.model_name)
            start = time.perf_counter()
            response = model.generate_content(prompt)
            latency_ms = (time.perf_counter() - start) * 1000
            text = response.text or ""
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
