from __future__ import annotations

import sqlite3
import textwrap


class SQLEvaluator:
    """
    Evaluates LLM-generated SQL by executing both predicted and expected queries
    against a shared in-memory SQLite database seeded with sample data, then
    comparing result sets.

    Falls back to normalised string comparison if execution fails for both.
    """

    # Schema and seed data shared across all test evaluations.
    _SCHEMA_SQL = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            status TEXT DEFAULT 'active',
            created_at TEXT DEFAULT '2024-01-01'
        );
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES users(id),
            amount REAL,
            status TEXT DEFAULT 'completed',
            created_at TEXT DEFAULT '2024-01-01'
        );
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            price REAL
        );
        INSERT OR IGNORE INTO users VALUES
            (1,'Alice','alice@example.com','active','2024-01-01'),
            (2,'Bob','bob@example.com','inactive','2024-02-01'),
            (3,'Carol','carol@example.com','active','2024-03-01');
        INSERT OR IGNORE INTO orders VALUES
            (1,1,250.0,'completed','2024-01-10'),
            (2,1,80.0,'completed','2024-02-10'),
            (3,2,320.0,'pending','2024-03-10'),
            (4,3,45.0,'completed','2024-04-10');
        INSERT OR IGNORE INTO products VALUES
            (1,'Widget A','electronics',29.99),
            (2,'Gadget B','electronics',99.99),
            (3,'Thing C','apparel',14.99);
    """)

    @staticmethod
    def _normalize(query: str) -> str:
        return " ".join(query.strip().lower().split()).rstrip(";")

    def _make_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.executescript(self._SCHEMA_SQL)
        return conn

    def _execute_query(self, conn: sqlite3.Connection, query: str) -> tuple[list, str | None]:
        """Returns (rows, error_message). rows is [] on error."""
        try:
            cur = conn.execute(query)
            return cur.fetchall(), None
        except sqlite3.Error as exc:
            return [], str(exc)

    def evaluate(self, predicted_query: str, expected_query: str) -> dict:
        # Attempt execution-based comparison first
        try:
            conn = self._make_connection()
            expected_rows, expected_err = self._execute_query(conn, expected_query)
            predicted_rows, predicted_err = self._execute_query(conn, predicted_query)
            conn.close()

            if expected_err is None and predicted_err is None:
                # Both executed — compare result sets
                matched = set(map(tuple, predicted_rows)) == set(map(tuple, expected_rows))
                accuracy = 100.0 if matched else 0.0
                failure = "" if matched else (
                    f"Result mismatch: expected {len(expected_rows)} row(s), got {len(predicted_rows)} row(s)"
                )
                return {
                    "accuracy": accuracy,
                    "pass_percentage": accuracy,
                    "validation_status": "Pass" if matched else "Fail",
                    "failure_reason": failure,
                    "validation_method": "execution",
                    "reliability_score": 1.0 if matched else 0.0,
                    "confidence_score": 1.0 if matched else 0.0,
                }

            if predicted_err:
                return {
                    "accuracy": 0.0,
                    "pass_percentage": 0.0,
                    "validation_status": "Fail",
                    "failure_reason": f"Query execution error: {predicted_err}",
                    "validation_method": "execution",
                    "reliability_score": 0.0,
                    "confidence_score": 0.0,
                }
        except Exception:
            pass  # Fall through to string comparison

        # Fallback: normalised string comparison
        predicted_norm = self._normalize(predicted_query)
        expected_norm = self._normalize(expected_query)
        exact = predicted_norm == expected_norm
        return {
            "accuracy": 100.0 if exact else 0.0,
            "pass_percentage": 100.0 if exact else 0.0,
            "validation_status": "Pass" if exact else "Fail",
            "failure_reason": "" if exact else "SQL mismatch (string comparison fallback)",
            "validation_method": "string_comparison",
            "reliability_score": 0.85 if exact else 0.0,
            "confidence_score": 0.7 if exact else 0.0,
        }
