#!/usr/bin/env python3
"""Build one Amazon domain's catalog, retrieval, and complementarity tools."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools_builder.profiles import load_profile


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
    "retrieval_patience": 2,
    "lora_r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "early_stop_metric": "val_multi_positive_mrr",
}

COMPLEMENTARITY = {
    "comp_epochs": 20,
    "comp_batch_size": 1024,
    "comp_proj_dim": 256,
    "comp_lr": 0.00040792119058488896,
    "comp_temperature": 0.081621255477171,
    "comp_dropout": 0.1,
    "comp_patience": 20,
    "comp_min_pair_count": 1,
    "comp_validation_user_ratio": 0.1,
    "comp_positives_per_anchor": 8,
    "comp_hard_negatives": 0,
    "comp_random_negatives": 8,
    "copurchase_window_days": 7,
    "min_pmi": 0.0,
}


def stage_command(args: argparse.Namespace, stage: int, category: str) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "tools_builder.build",
        "--stage",
        str(stage),
        "--category",
        category,
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
        "--retain-release-artifacts",
        "--comp-cross-subcategory",
        "--llm_model",
        args.llm_model,
        "--query_variants_per_product",
        "20",
    ]
    if args.allow_cpu:
        command.append("--allow_cpu")
    if category == "Electronics":
        command.extend(["--min_year", "2020"])
    if stage == 2:
        if args.query_seed:
            command.extend(["--product-query-seed-path", str(args.query_seed)])
        if args.query_cache_only:
            command.append("--query-cache-only")
    if stage == 3:
        settings = RETRIEVAL
    elif stage == 4:
        settings = COMPLEMENTARITY
    else:
        settings = {}
    for key, value in settings.items():
        command.extend(["--" + key, str(value)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        required=True,
        choices=[
            "amazon-musical-instruments",
            "amazon-video-games",
            "amazon-electronics",
        ],
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "queries", "retrieval", "complementarity", "all"],
        default="all",
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--query-seed", type=Path)
    parser.add_argument("--query-cache-only", action="store_true")
    parser.add_argument("--llm-model", default=os.getenv("RECOATLAS_QUERY_MODEL", "gpt-5.6-luna"))
    parser.add_argument("--run-id", default="shared-train-tools-v1")
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp/reco-atlas"))
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    if profile["domain"] != "amazon":
        raise ValueError("train_amazon_tools.py accepts only Amazon profiles")
    if args.query_cache_only and not args.query_seed and args.stage in ("queries", "all"):
        parser.error("--query-cache-only requires --query-seed")
    if not args.dry_run and not args.split_manifest.is_file():
        parser.error("--split-manifest must be an existing file")
    if args.query_seed and not args.dry_run and not args.query_seed.is_file():
        parser.error("--query-seed must be an existing file")

    os.environ["RECO_SCRATCH_ROOT"] = str(args.scratch_root.resolve())
    names = {
        "prepare": 1,
        "queries": 2,
        "retrieval": 3,
        "complementarity": 4,
    }
    stages = [1, 2, 4, 3] if args.stage == "all" else [names[args.stage]]
    commands = [stage_command(args, stage, profile["category"]) for stage in stages]
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
