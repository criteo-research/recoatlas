import torch

from tools_builder.retrieval import (
    MultiPositiveQueryDataset,
    build_query_groups,
    compute_multi_positive_retrieval_metrics,
    mine_positive_aware_hard_negatives,
    multi_positive_info_nce,
    query_group_fingerprint,
)


def test_build_query_groups_deduplicates_rows_and_collects_positives():
    queries = {
        ("u1", "a"): "  Guitar   Strings ",
        ("u2", "a"): "guitar strings",
        ("u3", "b"): "GUITAR STRINGS",
        ("u4", "c"): "piano bench",
    }

    groups = build_query_groups(queries)

    assert [(group.query, group.positive_item_ids, group.occurrences) for group in groups] == [
        ("guitar strings", ("a", "b"), 3),
        ("piano bench", ("c",), 1),
    ]
    assert query_group_fingerprint(groups) == query_group_fingerprint(groups)
    assert query_group_fingerprint(groups) != query_group_fingerprint(groups[:1])


def test_dataset_masks_all_positives_and_is_deterministic_for_validation():
    groups = build_query_groups({("u1", "a"): "guitar strings", ("u2", "b"): "guitar strings"})
    dataset = MultiPositiveQueryDataset(
        groups,
        {"a": 0, "b": 1, "c": 2, "d": 3},
        ["a", "b", "c", "d"],
        hard_negatives={"guitar strings": ["b", "c", "c"]},
        hard_negatives_per_query=2,
        random_negatives_per_query=1,
        dynamic=False,
    )

    first = dataset[0]
    second = dataset[0]

    assert first == second
    assert first["positive_indices"] == [0, 1]
    assert 0 not in first["negative_indices"]
    assert 1 not in first["negative_indices"]
    assert len(first["negative_indices"]) == len(set(first["negative_indices"]))


def test_multi_positive_info_nce_rewards_relevant_items_and_masks_collisions():
    items = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    items = torch.nn.functional.normalize(items, dim=-1)
    good_queries = torch.tensor([[1.0, 0.0]])
    bad_queries = torch.tensor([[0.0, 1.0]])
    positives = [[0, 1]]
    negatives = [[2]]

    good_loss, good_metrics = multi_positive_info_nce(
        good_queries, items, positives, negatives, temperature=0.1
    )
    bad_loss, bad_metrics = multi_positive_info_nce(
        bad_queries, items, positives, negatives, temperature=0.1
    )

    assert good_loss < bad_loss
    assert good_metrics["candidate_top1_accuracy"] == 1.0
    assert bad_metrics["candidate_top1_accuracy"] == 0.0


def test_multi_positive_metrics_credit_any_relevant_product():
    items = torch.eye(3)
    queries = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.bfloat16
    )
    metrics = compute_multi_positive_retrieval_metrics(
        queries,
        items,
        ["a", "b", "c"],
        [["a", "b"], ["c"]],
    )

    assert metrics["hit_at_1"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["recall_at_1"] == 0.75
    assert metrics["ndcg_at_1"] == 1.0


def test_positive_aware_mining_removes_too_close_and_known_positive_items():
    item_embeddings = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.999, 0.01], [0.8, 0.6], [0.0, 1.0]]),
        dim=-1,
    )
    groups = build_query_groups({("u", "positive"): "query"})

    mined = mine_positive_aware_hard_negatives(
        lambda texts: torch.tensor([[1.0, 0.0]] * len(texts)),
        groups,
        item_embeddings,
        ["positive", "ambiguous", "hard", "easy"],
        ["positive", "ambiguous", "hard", "easy"],
        top_k=4,
        negatives_per_query=2,
        positive_ratio=0.95,
        batch_size=1,
    )

    assert mined["query"] == ["hard", "easy"]
