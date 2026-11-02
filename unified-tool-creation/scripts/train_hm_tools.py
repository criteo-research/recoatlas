#!/usr/bin/env python3
"""Build the H&M catalog, retrieval tool, and residual complementarity tool."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


RETRIEVAL = {
    "epochs": 5,
    "train_batch_size": 32,
    "grad_accum_steps": 2,
    "retrieval_lr": 7.406576281631978e-5,
    "retrieval_lr_schedule": "constant",
    "retrieval_warmup_ratio": 0.0,
    "contrastive_temperature": 0.020090056606207288,
    "hard_negatives_per_query": 4,
    "random_negatives_per_query": 4,
    "hard_negative_candidates": 100,
    "hard_negative_positive_ratio": 0.95,
    "retrieval_patience": 2,
    "lora_r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "early_stop_metric": "val_multi_positive_mrr",
}


def stage_command(args, stage: int) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "tools_builder.build",
        "--stage",
        str(stage),
        "--category",
        "H_and_M",
        "--backbone",
        "qwen",
        "--run_id",
        args.run_id,
        "--hf_repo",
        "",
        "--benchmark_split_manifest",
        str(args.split_manifest),
        "--resume",
        "--no-cleanup",
        "--query_variants_per_product",
        "20",
    ]
    if stage == 2:
        if args.query_seed:
            command.extend(["--product-query-seed-path", str(args.query_seed)])
        if args.query_cache_only:
            command.append("--query-cache-only")
    if stage == 3:
        for key, value in RETRIEVAL.items():
            command.extend(["--" + key, str(value)])
    return command


def configuration(args) -> dict:
    return {
        "profile": "hm-clothing",
        "category": "H_and_M",
        "run_id": args.run_id,
        "articles": str(args.articles) if args.articles else None,
        "transactions": str(args.transactions) if args.transactions else None,
        "split_manifest": str(args.split_manifest) if args.split_manifest else None,
        "retrieval": RETRIEVAL,
        "complementarity": {
            "architecture": "embedding-space low-rank residual",
            "rank": 32,
            "alpha": 32,
            "dropout": 0.1,
            "epochs": 20,
            "selected_epoch": 2,
            "learning_rate": 0.00040792119058488896,
            "temperature": 0.081621255477171,
            "batch_size": 1024,
            "positive_neighbors": 8,
            "random_negatives": 8,
        },
        "positive_policy": "every purchase is positive; rating=5 is adapter-only",
        "source_partition": "shared_train",
    }


def require_inputs(args) -> None:
    missing = [
        name
        for name in ("articles", "transactions", "split_manifest")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit("Missing required inputs: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["prepare", "embed", "queries", "retrieval", "complementarity", "all"],
        default="all",
    )
    parser.add_argument("--articles", type=Path)
    parser.add_argument("--transactions", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--query-seed", type=Path)
    parser.add_argument("--query-cache-only", action="store_true")
    parser.add_argument("--run-id", default="shared-train-residual-v1")
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp/reco-atlas"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(configuration(args), indent=2))
        return
    require_inputs(args)
    from tools_builder.hm_data import encode_hm_catalog, prepare_hm_inputs
    from tools_builder.residual_complementarity import train_residual_complementarity
    from tools_builder.storage import ensure_scratch_layout, run_dir

    os.environ["RECO_SCRATCH_ROOT"] = str(args.scratch_root.resolve())
    ensure_scratch_layout()
    output = run_dir("H_and_M", args.run_id)
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_recipe.json").write_text(
        json.dumps(configuration(args), indent=2) + "\n"
    )

    stages = [args.stage] if args.stage != "all" else [
        "prepare",
        "embed",
        "queries",
        "complementarity",
        "retrieval",
    ]
    for stage in stages:
        print(f"H&M stage: {stage}", flush=True)
        if stage == "prepare":
            summary = prepare_hm_inputs(
                args.articles, args.transactions, args.split_manifest, output
            )
            print(json.dumps(summary, indent=2), flush=True)
        elif stage == "embed":
            print(json.dumps(encode_hm_catalog(output), indent=2), flush=True)
        elif stage == "queries":
            subprocess.run(stage_command(args, 2), cwd=ROOT, check=True)
        elif stage == "retrieval":
            subprocess.run(stage_command(args, 3), cwd=ROOT, check=True)
        else:
            print(
                json.dumps(
                    train_residual_complementarity(
                        output, epochs=20, selected_epoch=2
                    ),
                    indent=2,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
