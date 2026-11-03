import pytest

from recoatlas_benchmark.coherence import validate_judgment
from recoatlas_benchmark.query_generation import validate_consistency, validate_query


def test_query_gate():
    good = "I need a practical setup for recording acoustic guitar and keeping the instrument protected during travel"
    assert validate_query(good, ["B001", "B002", "B003"]) == (True, "OK")
    assert (
        validate_query("Please buy B001 and two other products for me today", ["B001"])[1]
        == "WORD_COUNT"
    )
    leaked = "I need a practical setup using B001 for recording music and keeping my instrument protected during travel"
    assert validate_query(leaked, ["B001"])[1] == "PRODUCT_ID_LEAKAGE"


def test_coherence_vote_policy():
    payload = {
        "verdict": "PASS",
        "confidence": "MEDIUM",
        "common_goal": "record a guitar",
        "item_roles": [
            {"label": "P1", "role": "instrument", "fit": "FIT"},
            {"label": "P2", "role": "microphone", "fit": "FIT"},
            {"label": "P3", "role": "stand", "fit": "UNCERTAIN"},
        ],
        "failure_reasons": [],
        "rationale": "Coherent roles.",
    }
    assert validate_judgment(payload, 3)["vote_pass"] is True
    payload["item_roles"][2]["fit"] = "CONFLICT"
    assert validate_judgment(payload, 3)["vote_pass"] is False


def test_consistency_requires_known_schema_values():
    row = validate_consistency(
        {"verdict": "PASS", "confidence": "HIGH", "reason_code": "OK", "reason": "supported"}
    )
    assert row["consistent"] is True
    with pytest.raises(ValueError):
        validate_consistency(
            {"verdict": "MAYBE", "confidence": "HIGH", "reason_code": "OK", "reason": ""}
        )
