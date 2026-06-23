from __future__ import annotations

import ast
import multiprocessing
import queue
import re
import textwrap
from typing import Any

# Banned built-ins and modules that could execute system commands or exfiltrate data.
_BANNED_NAMES: frozenset[str] = frozenset(
    [
        "eval", "exec", "compile", "__import__", "open", "input",
        "breakpoint", "globals", "locals", "vars", "dir",
        "getattr", "setattr", "delattr", "hasattr",
    ]
)
_BANNED_IMPORTS: frozenset[str] = frozenset(
    [
        "os", "sys", "subprocess", "socket", "http", "urllib", "requests",
        "shutil", "pathlib", "tempfile", "importlib", "ctypes", "pickle",
        "shelve", "marshal", "pty", "signal", "threading", "multiprocessing",
        "concurrent", "asyncio", "builtins",
    ]
)

_EXECUTION_TIMEOUT_SECONDS: int = 5
_MAX_CODE_LENGTH: int = 8_000


def _static_security_check(code: str) -> str | None:
    """Return an error message if the code fails static analysis, else None."""
    if len(code) > _MAX_CODE_LENGTH:
        return f"Code exceeds maximum allowed length ({_MAX_CODE_LENGTH} chars)"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"

    for node in ast.walk(tree):
        # Block dangerous function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _BANNED_NAMES:
                return f"Blocked call: {node.func.id}()"
            if isinstance(node.func, ast.Attribute) and node.func.attr in _BANNED_NAMES:
                return f"Blocked attribute call: {node.func.attr}()"
        # Block dangerous imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else ([node.module] if node.module else [])
            )
            for name in names:
                root = name.split(".")[0] if name else ""
                if root in _BANNED_IMPORTS:
                    return f"Blocked import: {name}"
    return None


def _run_code_in_worker(
    code: str,
    tests: list[str],
    result_queue: "multiprocessing.Queue[dict[str, Any]]",
) -> None:
    """Isolated worker — runs code + tests in a subprocess with restricted builtins."""
    # Restrict built-ins available to user code
    safe_builtins = {
        k: v
        for k, v in __builtins__.items()  # type: ignore[union-attr]
        if k not in _BANNED_NAMES
    } if isinstance(__builtins__, dict) else {
        k: getattr(__builtins__, k)
        for k in dir(__builtins__)
        if k not in _BANNED_NAMES and not k.startswith("__")
    }

    namespace: dict[str, Any] = {"__builtins__": safe_builtins}
    passed = 0
    first_error = ""
    try:
        exec(code, namespace)  # noqa: S102 — runs inside isolated subprocess
    except Exception as exc:
        result_queue.put({
            "passed": 0,
            "total": len(tests),
            "first_error": f"Compile/runtime error: {exc}",
        })
        return

    for test in tests:
        try:
            exec(test, namespace)  # noqa: S102
            passed += 1
        except Exception as test_exc:
            if not first_error:
                msg = str(test_exc)
                # Bare AssertionError has an empty message — generate a descriptive one
                first_error = msg if msg else f"Assertion failed: {test!r}"

    result_queue.put({"passed": passed, "total": len(tests), "first_error": first_error})


class CodingEvaluator:
    """
    Evaluates LLM-generated Python code.

    Security model:
    - Static AST analysis blocks dangerous imports and built-in calls.
    - Code execution runs in an isolated subprocess with a hard timeout.
    - Restricted built-ins are injected to limit the execution surface.
    """

    EXECUTION_TIMEOUT: int = _EXECUTION_TIMEOUT_SECONDS

    def evaluate(
        self,
        generated_code: str,
        tests: list[str],
        hidden_tests: list[str] | None = None,
    ) -> dict[str, Any]:
        hidden_tests = hidden_tests or []
        all_tests = tests + hidden_tests

        # 1. Static security analysis — fast, no subprocess needed
        security_error = _static_security_check(generated_code)
        if security_error:
            return self._build_result(
                passed=0,
                total=len(all_tests),
                visible=len(tests),
                hidden=len(hidden_tests),
                error=security_error,
                status="Fail",
                reliability_score=0.0,
                confidence_score=0.0,
            )

        # 2. Execute in isolated subprocess with timeout
        ctx = multiprocessing.get_context("spawn")
        result_q: "multiprocessing.Queue[dict[str, Any]]" = ctx.Queue()
        proc = ctx.Process(
            target=_run_code_in_worker,
            args=(generated_code, all_tests, result_q),
            daemon=True,
        )
        proc.start()
        proc.join(timeout=self.EXECUTION_TIMEOUT)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)
            return self._build_result(
                passed=0,
                total=len(all_tests),
                visible=len(tests),
                hidden=len(hidden_tests),
                error=f"Execution timed out after {self.EXECUTION_TIMEOUT}s",
                status="Fail",
                reliability_score=0.0,
                confidence_score=0.0,
            )

        try:
            worker_result = result_q.get_nowait()
        except queue.Empty:
            return self._build_result(
                passed=0,
                total=len(all_tests),
                visible=len(tests),
                hidden=len(hidden_tests),
                error="Worker process produced no result",
                status="Fail",
                reliability_score=0.0,
                confidence_score=0.0,
            )

        passed = worker_result["passed"]
        first_error = worker_result.get("first_error", "")
        total = max(1, len(all_tests))
        pass_pct = (passed / total) * 100
        status = "Pass" if passed == len(all_tests) else "Fail"

        # Reliability: penalise if even a single hidden test failed
        hidden_passed = max(0, passed - len([t for t in tests]))
        reliability_score = round(pass_pct / 100, 4)
        confidence_score = round(
            1.0 if passed == len(all_tests) else max(0.0, reliability_score - 0.1), 4
        )

        return self._build_result(
            passed=passed,
            total=len(all_tests),
            visible=len(tests),
            hidden=len(hidden_tests),
            error=first_error,
            status=status,
            reliability_score=reliability_score,
            confidence_score=confidence_score,
            pass_pct=pass_pct,
        )

    @staticmethod
    def _build_result(
        passed: int,
        total: int,
        visible: int,
        hidden: int,
        error: str,
        status: str,
        reliability_score: float,
        confidence_score: float,
        pass_pct: float | None = None,
    ) -> dict[str, Any]:
        pct = pass_pct if pass_pct is not None else 0.0
        return {
            "accuracy": round(pct, 2),
            "pass_percentage": round(pct, 2),
            "passed_tests": passed,
            "total_tests": total,
            "visible_tests": visible,
            "hidden_tests": hidden,
            "validation_status": status,
            "failure_reason": error,
            "reliability_score": reliability_score,
            "confidence_score": confidence_score,
        }

