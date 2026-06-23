from __future__ import annotations

import time

from models.base_runner import BaseModelRunner, ModelOutput


class DynamicRunner(BaseModelRunner):
    """Runner for any OpenAI-compatible API endpoint (user-added models)."""

    def __init__(
        self,
        model_key: str,
        model_id: str,
        base_url: str,
        api_key: str,
        runtime_mode: str = "real",
    ) -> None:
        super().__init__(model_name=model_id, runtime_mode=runtime_mode)
        self.model_key = model_key
        self.base_url = base_url
        self.api_key = api_key

    def generate(self, prompt: str, category: str = "coding") -> ModelOutput:
        if self.runtime_mode == "mock":
            return self.make_mock_output(prompt, category)

        if not self.api_key:
            return ModelOutput(
                model_name=self.model_name,
                response_text="",
                latency_ms=0,
                input_tokens=0,
                output_tokens=0,
                error=f"No API key configured for custom model '{self.model_key}'",
            )

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
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
