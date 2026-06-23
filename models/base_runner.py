from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelOutput:
    model_name: str
    response_text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    error: str | None = None


class BaseModelRunner(ABC):
    def __init__(self, model_name: str, runtime_mode: str = "mock") -> None:
        self.model_name = model_name
        self.runtime_mode = runtime_mode

    @abstractmethod
    def generate(self, prompt: str, category: str = "coding") -> ModelOutput:
        raise NotImplementedError

    @staticmethod
    def estimate_tokens(text: str) -> int:
        # Approximation used consistently across providers for fallback accounting.
        return max(1, len(text) // 4)

    def mock_response(self, prompt: str, category: str) -> str:
        p = prompt.lower()
        if category == "coding":
            if "factorial" in p:
                return "def factorial(n):\n    if n < 2:\n        return 1\n    return n * factorial(n - 1)"
            if "fibonacci" in p:
                return "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"
            if "reverse_string" in p or "reversed string" in p:
                return "def reverse_string(s):\n    return s[::-1]"
            if "two_sum" in p or "two numbers" in p:
                return (
                    "def two_sum(nums, target):\n"
                    "    seen = {}\n"
                    "    for i, n in enumerate(nums):\n"
                    "        if target - n in seen:\n"
                    "            return [seen[target - n], i]\n"
                    "        seen[n] = i\n"
                    "    return []"
                )
            if "binary_search" in p:
                return (
                    "def binary_search(arr, target):\n"
                    "    lo, hi = 0, len(arr) - 1\n"
                    "    while lo <= hi:\n"
                    "        mid = (lo + hi) // 2\n"
                    "        if arr[mid] == target:\n"
                    "            return mid\n"
                    "        if arr[mid] < target:\n"
                    "            lo = mid + 1\n"
                    "        else:\n"
                    "            hi = mid - 1\n"
                    "    return -1"
                )
            if "merge_sorted_arrays" in p or "merges two sorted lists" in p:
                return (
                    "def merge_sorted_arrays(a, b):\n"
                    "    i = j = 0\n"
                    "    out = []\n"
                    "    while i < len(a) and j < len(b):\n"
                    "        if a[i] <= b[j]:\n"
                    "            out.append(a[i]); i += 1\n"
                    "        else:\n"
                    "            out.append(b[j]); j += 1\n"
                    "    return out + a[i:] + b[j:]"
                )
            return "def solution():\n    return None"

        if category == "sql":
            if "active users" in p:
                return "SELECT * FROM users WHERE status = 'active';"
            if "order id" in p and "customer" in p:
                return "SELECT o.id, c.name FROM orders o JOIN customers c ON o.customer_id = c.id;"
            return "SELECT 1;"

        if category == "support":
            if "refund" in p or "charged" in p:
                return "refund"
            if "cancel" in p:
                return "cancellation"
            if "payment" in p:
                return "payment_failure"
            return "technical_issue"

        return ""

    def make_mock_output(self, prompt: str, category: str) -> ModelOutput:
        start = time.perf_counter()
        response = self.mock_response(prompt, category)
        latency_ms = (time.perf_counter() - start) * 1000
        return ModelOutput(
            model_name=self.model_name,
            response_text=response,
            latency_ms=latency_ms,
            input_tokens=self.estimate_tokens(prompt),
            output_tokens=self.estimate_tokens(response),
        )
