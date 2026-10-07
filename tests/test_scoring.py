from recoatlas.scoring import score_report, sethit

CATALOG = {"a", "b", "c", "d", "e"}
REFS = ["a", "b", "c"]


def report(*ids):
    return [{"product_id": i, "reasoning": "r"} for i in ids]


def test_sethit_scores_only_the_first_k_positions():
    # v1 rule: positions are cut to K before duplicates are dropped, so "c" in position 4 earns nothing.
    assert sethit(["a", "a", "b", "c"], REFS, CATALOG) == 2 / 3


def test_valid_report_is_rewarded_with_sethit():
    assert score_report(report("a", "d", "e"), REFS, CATALOG, seen={"a", "d", "e"}) == {
        "sethit": 1 / 3,
        "valid": 1,
        "reward_direct": 1 / 3,
    }


def test_invalid_reports_get_minus_one():
    seen = {"a", "b", "c", "d"}
    invalid = {
        "wrong count": report("a", "b"),
        "duplicate": report("a", "a", "b"),
        "not in catalog": report("a", "b", "z"),
        "never returned by a tool": report("a", "b", "e"),
        "wrong key": [{"item_id": i, "reasoning": "r"} for i in "abc"],
    }
    for reason, results in invalid.items():
        assert score_report(results, REFS, CATALOG, seen)["reward_direct"] == -1.0, reason
    assert score_report(None, REFS, CATALOG, seen) == {"sethit": 0.0, "valid": 0, "reward_direct": -1.0}
