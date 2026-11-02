import pandas as pd
import pytest
import torch

from tools_builder.stage4_complementarity import (
    BalancedComplementarityDataset,
    build_temporal_complementarity_split,
    build_temporal_validation_relevance,
    compute_complementarity_ranking_metrics,
    multi_positive_complementarity_loss,
    train_complementarity,
    ComplementaritySplit,
)


def _interactions(rows):
    return pd.DataFrame(
        [
            {
                "user_id": user,
                "parent_asin": item,
                "rating": 5.0,
                "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(days=day),
            }
            for user, item, day in rows
        ]
    )


def test_temporal_split_removes_both_directions_and_filters_same_category():
    interactions = _interactions(
        [
            ("train-and-val", "a", 0),
            ("train-and-val", "b", 1),
            ("train-and-val", "c", 2),
            # b -> a is validation, so the a-b training edge above must disappear.
            ("reverse-leak", "b", 0),
            ("reverse-leak", "a", 1),
            # d-e remains as a genuine training edge.
            ("kept", "d", 0),
            ("kept", "e", 1),
            ("kept", "f", 2),
            # Same-category validation pair must be excluded consistently.
            ("same-category", "g", 0),
            ("same-category", "h", 1),
        ]
    )
    categories = {
        "a": "a-cat", "b": "b-cat", "c": "c-cat",
        "d": "d-cat", "e": "e-cat", "f": "f-cat",
        "g": "shared", "h": "shared",
    }

    split = build_temporal_complementarity_split(
        interactions,
        categories,
        categories,
        window_days=7,
        min_train_count=1,
        cross_subcategory=True,
        validation_user_ids={"reverse-leak", "same-category"},
    )

    assert "b" not in split.train_graph.get("a", {})
    assert "a" not in split.train_graph.get("b", {})
    assert split.train_graph["d"]["e"] > 0
    assert split.train_graph["e"]["d"] > 0
    assert "h" not in split.validation.get("g", [])
    assert split.metadata["post_split_edge_overlap"] == 0
    assert split.metadata["heldout_edges_seen_before_strict_removal"] >= 1


def test_temporal_audit_relevance_uses_only_requested_users_and_last_target():
    interactions = _interactions(
        [
            ("dev", "a", 0),
            ("dev", "b", 2),
            ("dev", "c", 3),
            ("hidden", "d", 0),
            ("hidden", "e", 1),
        ]
    )
    categories = {item: f"cat-{item}" for item in "abcde"}
    relevance, metadata = build_temporal_validation_relevance(
        interactions,
        categories,
        categories,
        user_ids={"dev"},
        window_days=7,
    )

    assert relevance == {"a": ["c"], "b": ["c"]}
    assert "d" not in relevance
    assert metadata["users_with_targets"] == 1
    assert metadata["directed_edges"] == 2


def test_balanced_dataset_is_anchor_level_deterministic_and_positive_aware():
    graph = {
        "a": {"b": 2.0, "c": 1.0},
        "b": {"a": 2.0},
    }
    items = {item: idx for idx, item in enumerate(["a", "b", "c", "d", "e"])}
    categories = {item: f"cat-{item}" for item in items}
    dataset = BalancedComplementarityDataset(
        graph,
        items,
        categories,
        positives_per_anchor=2,
        hard_negatives_per_anchor=1,
        random_negatives_per_anchor=1,
        seed=7,
    )
    dataset.set_hard_negatives({"a": ["b", "d", "d"]})

    first = dataset[dataset.anchors.index("a")]
    second = dataset[dataset.anchors.index("a")]

    assert len(dataset) == 2
    assert first == second
    assert set(first["positive_indices"]) == {items["b"], items["c"]}
    assert not set(first["negative_indices"]) & {items["a"], items["b"], items["c"]}
    assert len(first["negative_indices"]) == len(set(first["negative_indices"]))


class _IdentityProjection:
    temperature = 0.1

    @staticmethod
    def project_anchor(values):
        return torch.nn.functional.normalize(values, dim=-1)

    @staticmethod
    def project_complement(values):
        return torch.nn.functional.normalize(values, dim=-1)


def test_multi_positive_loss_rewards_any_positive_and_masks_known_collisions():
    batch = {
        "anchor_indices": torch.tensor([0]),
        "positive_indices": [[1, 2]],
        "positive_weights": [[2.0, 1.0]],
        "known_positive_indices": [[1, 2]],
        "negative_indices": [[3]],
    }
    good = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    bad = good.clone()
    bad[0] = torch.tensor([0.0, 1.0])

    good_loss, good_metrics = multi_positive_complementarity_loss(_IdentityProjection(), good, batch)
    bad_loss, bad_metrics = multi_positive_complementarity_loss(_IdentityProjection(), bad, batch)

    assert good_loss < bad_loss
    assert good_metrics["candidate_top1_accuracy"] == 1.0
    assert bad_metrics["candidate_top1_accuracy"] == 0.0


def test_ranking_metrics_use_standard_recall_precision_mrr_and_ndcg():
    item_ids = ["a", "b", "c", "d"]
    categories = {item: f"cat-{item}" for item in item_ids}
    anchor_embs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-1.0, 0.0]])
    complement_embs = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.8, 0.2], [-1.0, 0.0]])

    metrics = compute_complementarity_ranking_metrics(
        anchor_embs,
        complement_embs,
        item_ids,
        {"a": ["b", "c"]},
        categories,
        ks=(1, 2),
    )

    assert metrics["hit_at_1"] == 1.0
    assert metrics["recall_at_1"] == 0.5
    assert metrics["precision_at_1"] == 1.0
    assert metrics["recall_at_2"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg_at_2"] == 1.0


def test_checkpoint_interval_cannot_be_negative():
    with pytest.raises(ValueError, match="checkpoint_every"):
        train_complementarity(
            None,
            [],
            {},
            torch.empty(0, 2),
            checkpoint_every=-1,
        )


def test_training_retains_numbered_checkpoint_each_epoch(tmp_path):
    item_ids = ["a", "b", "c", "d"]
    categories = {item: f"cat-{item}" for item in item_ids}
    split = ComplementaritySplit(
        train_graph={"a": {"b": 1.0}, "b": {"a": 1.0}},
        train_counts={"a": {"b": 1}, "b": {"a": 1}},
        validation={"a": ["c"]},
        metadata={"fingerprint": "unit-test"},
    )
    base_embeddings = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]
    )

    train_complementarity(
        split,
        item_ids,
        categories,
        base_embeddings,
        epochs=2,
        batch_size=2,
        proj_dim=2,
        patience=3,
        positives_per_anchor=1,
        random_negatives_per_anchor=1,
        save_dir=str(tmp_path),
        checkpoint_every=1,
    )

    checkpoint_dir = tmp_path / "models/comp_checkpoints_qwen"
    first = torch.load(checkpoint_dir / "epoch_0001.pt", weights_only=False)
    second = torch.load(checkpoint_dir / "epoch_0002.pt", weights_only=False)
    assert first["epoch"] == 1
    assert second["epoch"] == 2
    assert second["training_epochs"] == 2


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_resume_restores_rng_on_cpu_and_does_not_repeat_completed_epochs(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    item_ids = ['a', 'b', 'c', 'd']
    categories = {item: f'cat-{item}' for item in item_ids}
    split = ComplementaritySplit(
        train_graph={'a': {'b': 1.0}, 'b': {'a': 1.0}},
        train_counts={'a': {'b': 1}, 'b': {'a': 1}},
        validation={'a': ['c']}, metadata={'fingerprint': 'resume-test'},
    )
    embeddings = torch.tensor([[1., 0.], [.9, .1], [0., 1.], [-1., 0.]], device=device)
    kwargs = dict(epochs=2, batch_size=2, proj_dim=2, patience=3, positives_per_anchor=1,
                  random_negatives_per_anchor=1, save_dir=str(tmp_path), checkpoint_every=1)
    def interrupt_after_checkpoint(paths, epoch, is_best):
        if epoch == 1:
            raise RuntimeError('simulated interruption after checkpoint')
    with pytest.raises(RuntimeError, match='simulated interruption'):
        train_complementarity(split, item_ids, categories, embeddings,
                              checkpoint_callback=interrupt_after_checkpoint, **kwargs)
    original_load = torch.load
    latest_load_devices = []
    def checked_load(path, *args, **load_kwargs):
        if str(path).endswith('latest.pt'):
            latest_load_devices.append(load_kwargs.get('map_location'))
        return original_load(path, *args, **load_kwargs)
    monkeypatch.setattr(torch, 'load', checked_load)
    completed = []
    model, history, result = train_complementarity(
        split, item_ids, categories, embeddings, resume_from_checkpoint=True,
        checkpoint_callback=lambda paths, epoch, is_best: completed.append(epoch), **kwargs)
    assert completed == [2]
    assert latest_load_devices == ['cpu']
    assert len(history) == 2
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    completed.clear()
    resumed, resumed_history, resumed_result = train_complementarity(
        split, item_ids, categories, embeddings, resume_from_checkpoint=True,
        checkpoint_callback=lambda paths, epoch, is_best: completed.append(epoch), **kwargs)
    assert completed == []
    assert resumed_history == history
    assert resumed_result == result
    for k, v in resumed.state_dict().items():
        torch.testing.assert_close(v, weights[k], rtol=0, atol=0)
