"""
Unit tests for the CodingEvaluator.

Tests cover:
- Correct code that passes all tests
- Code with runtime errors
- Security: blocked imports
- Security: blocked built-in calls
- Hidden test handling
- Partial pass / partial fail
"""
from __future__ import annotations

import pytest

from evaluators.coding_evaluator import CodingEvaluator, _static_security_check


@pytest.fixture
def evaluator() -> CodingEvaluator:
    return CodingEvaluator()


# ---------------------------------------------------------------------------
# Static security checks
# ---------------------------------------------------------------------------

class TestStaticSecurityCheck:
    def test_safe_code_passes(self):
        code = "def add(a, b):\n    return a + b"
        assert _static_security_check(code) is None

    def test_os_import_blocked(self):
        code = "import os\nos.system('ls')"
        result = _static_security_check(code)
        assert result is not None
        assert "os" in result

    def test_sys_import_blocked(self):
        code = "import sys\nsys.exit()"
        assert _static_security_check(code) is not None

    def test_subprocess_blocked(self):
        code = "import subprocess\nsubprocess.run(['ls'])"
        assert _static_security_check(code) is not None

    def test_open_blocked(self):
        code = "f = open('/etc/passwd')"
        assert _static_security_check(code) is not None

    def test_exec_call_blocked(self):
        code = "exec('print(1)')"
        assert _static_security_check(code) is not None

    def test_eval_blocked(self):
        code = "x = eval('1+1')"
        assert _static_security_check(code) is not None

    def test_syntax_error_caught(self):
        code = "def foo(:\n    pass"
        result = _static_security_check(code)
        assert result is not None
        assert "SyntaxError" in result

    def test_oversized_code_blocked(self):
        code = "x = 1\n" * 2000  # well over _MAX_CODE_LENGTH
        result = _static_security_check(code)
        assert result is not None
        assert "maximum" in result.lower()


# ---------------------------------------------------------------------------
# Correct implementations
# ---------------------------------------------------------------------------

class TestCodingEvaluatorCorrect:
    def test_factorial_all_pass(self, evaluator: CodingEvaluator):
        code = "def factorial(n):\n    if n < 2: return 1\n    return n * factorial(n-1)"
        tests = ["assert factorial(5) == 120", "assert factorial(0) == 1"]
        result = evaluator.evaluate(code, tests)
        assert result["validation_status"] == "Pass"
        assert result["accuracy"] == 100.0
        assert result["passed_tests"] == 2
        assert result["reliability_score"] == 1.0

    def test_fibonacci_all_pass(self, evaluator: CodingEvaluator):
        code = (
            "def fibonacci(n):\n"
            "    if n <= 1: return n\n"
            "    a, b = 0, 1\n"
            "    for _ in range(2, n+1): a, b = b, a+b\n"
            "    return b"
        )
        tests = ["assert fibonacci(0) == 0", "assert fibonacci(7) == 13"]
        result = evaluator.evaluate(code, tests)
        assert result["validation_status"] == "Pass"
        assert result["accuracy"] == 100.0

    def test_hidden_tests_counted(self, evaluator: CodingEvaluator):
        code = "def add(a, b): return a + b"
        visible = ["assert add(1, 2) == 3"]
        hidden = ["assert add(-1, 1) == 0", "assert add(0, 0) == 0"]
        result = evaluator.evaluate(code, visible, hidden)
        assert result["total_tests"] == 3
        assert result["visible_tests"] == 1
        assert result["hidden_tests"] == 2
        assert result["passed_tests"] == 3


# ---------------------------------------------------------------------------
# Failing code
# ---------------------------------------------------------------------------

class TestCodingEvaluatorFailing:
    def test_wrong_implementation_fails(self, evaluator: CodingEvaluator):
        code = "def factorial(n): return n"  # wrong
        tests = ["assert factorial(5) == 120"]
        result = evaluator.evaluate(code, tests)
        assert result["validation_status"] == "Fail"
        assert result["passed_tests"] == 0
        assert result["accuracy"] == 0.0
        assert result["failure_reason"] != ""

    def test_runtime_error_caught(self, evaluator: CodingEvaluator):
        code = "def divide(a, b): return a / b"
        tests = ["assert divide(10, 0) == 5"]  # ZeroDivisionError
        result = evaluator.evaluate(code, tests)
        assert result["validation_status"] == "Fail"

    def test_malicious_import_blocked(self, evaluator: CodingEvaluator):
        code = "import os\ndef factorial(n): return os.getpid()"
        tests = ["assert factorial(5) == 120"]
        result = evaluator.evaluate(code, tests)
        assert result["validation_status"] == "Fail"
        assert result["failure_reason"] != ""

    def test_empty_code_fails(self, evaluator: CodingEvaluator):
        result = evaluator.evaluate("", ["assert factorial(5) == 120"])
        assert result["validation_status"] == "Fail"
from evaluators.coding_evaluator import CodingEvaluator


def test_coding_evaluator_passes_factorial() -> None:
    evaluator = CodingEvaluator()
    code = """
def factorial(n):
    if n < 2:
        return 1
    return n * factorial(n - 1)
"""
    tests = ["assert factorial(5) == 120", "assert factorial(0) == 1"]

    result = evaluator.evaluate(code, tests)

    assert result["validation_status"] == "Pass"
    assert result["pass_percentage"] == 100.0
