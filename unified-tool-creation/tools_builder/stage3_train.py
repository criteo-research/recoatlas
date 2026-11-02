"""Stage 3: model training."""

import json
import os

import torch

from tools_builder.benchmark_split import load_user_split_manifest
from tools_builder.hf_upload import (
    hf_category,
    hf_run,
    prune_verified_files,
    restore_files,
    upload_files,
)
from tools_builder.utils import save_config
from tools_builder.recsys_model import train_query_retrieval


def publish_stage3(args, save_dir: str):
    if not args.hf_repo:
        return
    remote_category = hf_category(args)
    backbone = getattr(args, "backbone", "qwen")
    files = [
        f"models/recsys_model_{backbone}.pt",
        f"metrics/retrieval_baseline_{backbone}.json",
        f"metrics/retrieval_history_{backbone}.json",
    ]
    upload_files(args.hf_repo, remote_category, save_dir, files)


def _load_queries(path: str):
    with open(path) as f:
        raw = json.load(f)
    parsed = {}
    for key, value in raw.items():
        user_id, item_id = key.split("||", 1)
        parsed[(user_id, item_id)] = value
    return parsed


def _load_product_queries(path: str):
    """Load every generated product variant once, without user-popularity replication."""
    with open(path) as handle:
        raw = json.load(handle)
    parsed = {}
    for item_id, variants in raw.items():
        for index, query in enumerate(variants):
            parsed[(f"variant-{index}", item_id)] = query
    return parsed


def run_stage3(args, save_dir: str):
    data_dir = os.path.join(save_dir, "data")
    queries_dir = os.path.join(save_dir, "queries")
    backbone = getattr(args, "backbone", "qwen")

    split_manifest_path = getattr(args, "benchmark_split_manifest", None)
    required_paths = [
        f"{data_dir}/product_texts.json",
        f"{data_dir}/subcategories.json",
        f"{queries_dir}/queries_train.json",
        f"{queries_dir}/queries_test.json",
        f"{save_dir}/embeddings/item_embeddings_{backbone}.pt",
    ]
    if split_manifest_path:
        required_paths.append(f"{queries_dir}/benchmark_split_metadata.json")
    restore_files(
        args.hf_repo,
        hf_category(args),
        save_dir,
        [
            "data/product_texts.json",
            "data/subcategories.json",
            f"embeddings/item_embeddings_{backbone}.pt",
        ],
    )
    query_artifacts = [
        "queries/queries_train.json",
        "queries/queries_test.json",
        "queries/product_queries_train.json",
        "queries/product_queries_test.json",
    ]
    if split_manifest_path:
        query_artifacts.append("queries/benchmark_split_metadata.json")
    restore_files(
        args.hf_repo,
        hf_run(args) if split_manifest_path else hf_category(args),
        save_dir,
        query_artifacts,
    )
    if getattr(args, "resume", False):
        restore_files(
            args.hf_repo,
            hf_run(args),
            save_dir,
            [
                f"models/recsys_checkpoints_{backbone}/latest.pt",
                f"models/recsys_model_{backbone}.pt",
                f"metrics/retrieval_history_{backbone}.json",
                f"mining/hard_negatives_{backbone}.json",
            ],
        )
    mining_run = getattr(args, "retrieval_mining_run", None)
    if mining_run and not os.path.exists(
        f"{save_dir}/mining/hard_negatives_{backbone}.json"
    ):
        safe_mining_run = mining_run.replace("/", "_").replace("..", "_")
        restore_files(
            args.hf_repo,
            f"{hf_category(args)}/runs/{safe_mining_run}",
            save_dir,
            [f"mining/hard_negatives_{backbone}.json"],
        )
    missing = [p for p in required_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required artifacts:\n- " + "\n- ".join(missing) + "\n"
            f"Please run the required stages first:\n"
            f"  Stage 1: python -m tools_builder.build --stage 1 --category {args.category}\n"
            f"  Stage 2: python -m tools_builder.build --stage 2 --category {args.category}"
        )

    print("Loading stage artifacts...")
    if split_manifest_path:
        manifest = load_user_split_manifest(split_manifest_path)
        with open(f"{queries_dir}/benchmark_split_metadata.json") as handle:
            query_split_metadata = json.load(handle)
        if query_split_metadata.get("fingerprint") != manifest["fingerprint"]:
            raise ValueError(
                "Retrieval queries were generated from a different benchmark split"
            )
        if query_split_metadata.get("source_partition") != "shared_train":
            raise ValueError("Retrieval queries are not restricted to shared_train")
    with open(f"{data_dir}/product_texts.json") as f:
        product_texts = json.load(f)
    with open(f"{data_dir}/subcategories.json") as f:
        subcategories = json.load(f)
    item_embedding_data = torch.load(
        f"{save_dir}/embeddings/item_embeddings_{backbone}.pt",
        map_location="cpu",
        weights_only=False,
    )
    if item_embedding_data.get("item_ids") != list(product_texts.keys()):
        raise ValueError(
            "Stage-1 item embeddings are not aligned with product_texts.json"
        )

    balanced_train_path = f"{queries_dir}/product_queries_train.json"
    balanced_val_path = f"{queries_dir}/product_queries_test.json"
    if os.path.exists(balanced_train_path) and os.path.exists(balanced_val_path):
        train_queries = _load_product_queries(balanced_train_path)
        val_queries = _load_product_queries(balanced_val_path)
        print("Using balanced product-query variants (one row per generated variant).")
    else:
        train_queries = _load_queries(f"{queries_dir}/queries_train.json")
        val_queries = _load_queries(f"{queries_dir}/queries_test.json")
        print("Balanced product variants unavailable; deduplicating legacy user-query rows.")

    def publish_checkpoint(paths, epoch, is_best):
        if not args.hf_repo:
            return
        relative = [os.path.relpath(path, save_dir) for path in paths]
        label = " + best" if is_best else ""
        print(f"Publishing resumable epoch {epoch}{label} checkpoint...")
        upload_files(args.hf_repo, hf_run(args), save_dir, relative)

    def publish_baseline(path):
        if not args.hf_repo:
            return
        print("Publishing untrained Qwen retrieval baseline before optimization...")
        upload_files(
            args.hf_repo,
            hf_run(args),
            save_dir,
            [os.path.relpath(path, save_dir)],
        )

    def publish_mining(path):
        if not args.hf_repo:
            return
        print("Publishing positive-aware hard-negative IDs...")
        upload_files(
            args.hf_repo,
            hf_run(args),
            save_dir,
            [os.path.relpath(path, save_dir)],
        )

    print("Training model...")
    hard_negatives_path = f"{save_dir}/mining/hard_negatives_{backbone}.json"
    model, history = train_query_retrieval(
        train_queries,
        val_queries,
        product_texts,
        item_to_subcategory=subcategories["item_to_subcategory"],
        epochs=args.epochs,
        batch_size=args.train_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=getattr(args, "retrieval_lr", 7.406576281631978e-5),
        save_dir=save_dir,
        patience=getattr(args, "retrieval_patience", 2),
        backbone=getattr(args, "backbone", "qwen"),
        query_instruction=args.query_instruction,
        query_max_length=args.query_max_length,
        item_max_length=args.item_max_length,
        lora_r=getattr(args, "lora_r", 32),
        lora_alpha=getattr(args, "lora_alpha", 32),
        lora_dropout=getattr(args, "lora_dropout", 0.1),
        lora_target_modules=getattr(args, "lora_target_modules", None),
        lr_schedule=getattr(args, "retrieval_lr_schedule", "constant"),
        warmup_ratio=getattr(args, "retrieval_warmup_ratio", 0.0),
        temperature=getattr(args, "contrastive_temperature", 0.020090056606207288),
        hard_negatives_per_query=getattr(args, "hard_negatives_per_query", 4),
        random_negatives_per_query=getattr(args, "random_negatives_per_query", 4),
        hard_negative_candidates=getattr(args, "hard_negative_candidates", 100),
        hard_negative_positive_ratio=getattr(args, "hard_negative_positive_ratio", 0.95),
        early_stop_metric=getattr(args, "early_stop_metric", "val_multi_positive_mrr"),
        checkpoint_callback=publish_checkpoint,
        baseline_callback=publish_baseline,
        mining_callback=publish_mining,
        resume_from_checkpoint=getattr(args, "resume", False),
        catalog_item_embeddings=item_embedding_data["item_embs"],
        hard_negatives_path=hard_negatives_path,
    )

    winner_path = f"{save_dir}/models/recsys_model_{backbone}.pt"
    winner = torch.load(winner_path, map_location="cpu", weights_only=False)
    winner_epoch = int(winner.get("epoch", 0))
    if winner_epoch == 0:
        print(
            "Training complete. No trained epoch beat the untrained baseline; "
            "the baseline adapter was selected for publication."
        )
    else:
        selected = next(row for row in history if row["epoch"] == winner_epoch)
        print(
            f"Training complete. Selected epoch {winner_epoch}: "
            f"exact MRR {selected['baseline_retrieval_mrr']:.4f} -> "
            f"{selected['val_retrieval_mrr']:.4f}; "
            f"multi-positive MRR {selected['baseline_multi_positive_mrr']:.4f} -> "
            f"{selected['val_multi_positive_mrr']:.4f}."
        )

    save_config(args, os.path.join(save_dir, "data"), getattr(args, "backbone", "qwen"))
    stable_publish_allowed = not split_manifest_path or getattr(
        args, "allow_benchmark_stable_publish", False
    )
    if getattr(args, "publish_stable", True) and stable_publish_allowed:
        publish_stage3(args, save_dir)
    else:
        print(
            "Stable category publication disabled; the selected model and metrics "
            f"remain durable under {hf_run(args)}."
        )
    if args.hf_repo:
        prunable = [
            "queries/queries.json",
            "queries/query_types.json",
            "queries/queries_train.json",
            "queries/queries_test.json",
            "queries/product_queries_train.json",
            "queries/product_queries_test.json",
            "queries/product_queries.json",
            "queries/benchmark_split_metadata.json",
            "data/interactions.parquet",
            f"mining/hard_negatives_{backbone}.json",
            f"models/recsys_checkpoints_{backbone}/latest.pt",
            f"metrics/retrieval_baseline_{backbone}.json",
            f"metrics/retrieval_history_{backbone}.json",
        ]
        if not getattr(args, "retain_release_artifacts", False):
            prunable.append(f"models/recsys_model_{backbone}.pt")
        prune_verified_files(save_dir, prunable)
