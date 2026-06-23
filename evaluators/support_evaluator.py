from __future__ import annotations


class SupportEvaluator:
    """
    Evaluates support ticket classification.
    Compares predicted label to expected label with partial-match scoring.
    """

    # Map of canonical label to common aliases from LLM output
    _LABEL_ALIASES: dict[str, list[str]] = {
        "refund": ["refund", "money back", "reimbursement", "charge back", "chargeback"],
        "cancellation": ["cancellation", "cancel", "subscription cancel", "unsubscribe"],
        "payment_failure": ["payment_failure", "payment failure", "payment error", "billing issue",
                            "failed payment", "billing_error"],
        "technical_issue": ["technical_issue", "technical issue", "bug", "error", "broken",
                            "not working", "outage", "crash"],
        "account_access": ["account_access", "account access", "login", "password", "locked out",
                           "cannot login"],
        "shipping": ["shipping", "delivery", "not delivered", "lost package", "tracking"],
    }

    def _normalise_label(self, label: str) -> str:
        label = label.strip().lower().replace(" ", "_")
        for canonical, aliases in self._LABEL_ALIASES.items():
            if label in aliases or label == canonical:
                return canonical
        return label

    def evaluate(self, predicted_label: str, expected_label: str) -> dict:
        pred = self._normalise_label(predicted_label)
        exp = self._normalise_label(expected_label)
        matched = pred == exp

        # Partial credit: same root category
        partial = not matched and (pred.split("_")[0] == exp.split("_")[0])
        accuracy = 100.0 if matched else (40.0 if partial else 0.0)

        return {
            "accuracy": accuracy,
            "pass_percentage": accuracy,
            "validation_status": "Pass" if matched else "Fail",
            "failure_reason": "" if matched else f"Expected '{exp}', got '{pred}'",
            "reliability_score": 1.0 if matched else (0.4 if partial else 0.0),
            "confidence_score": 1.0 if matched else (0.35 if partial else 0.0),
        }
