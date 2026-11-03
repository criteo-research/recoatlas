"""Command-line driver for benchmark construction."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .pipeline import export_stage, generate_stage, judge_stage, mine_stage


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("stage", choices=("mine", "judge", "generate", "export", "all"))
    result.add_argument("--profile", required=True)
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--split-manifest", type=Path)
    result.add_argument("--amazon-data-dir", type=Path)
    result.add_argument("--hm-articles", type=Path)
    result.add_argument("--hm-transactions", type=Path)
    result.add_argument("--model", default=None)
    result.add_argument("--judge-concurrency", type=int, default=20)
    result.add_argument("--query-concurrency", type=int, default=16)
    return result


def main() -> None:
    load_dotenv()
    args = parser().parse_args()
    model = args.model or os.getenv("RECOATLAS_BENCHMARK_MODEL", "gpt-5.6-luna")
    summaries = []
    if args.stage in {"mine", "all"}:
        if args.split_manifest is None:
            raise ValueError("mine/all requires --split-manifest")
        summaries.append(
            mine_stage(
                profile_name=args.profile,
                split_manifest=args.split_manifest,
                run_dir=args.run_dir,
                model=model,
                amazon_data_dir=args.amazon_data_dir,
                hm_articles=args.hm_articles,
                hm_transactions=args.hm_transactions,
            )
        )
    if args.stage in {"judge", "all"}:
        summaries.append(
            asyncio.run(
                judge_stage(
                    profile_name=args.profile,
                    run_dir=args.run_dir,
                    model=model,
                    concurrency=args.judge_concurrency,
                )
            )
        )
    if args.stage in {"generate", "all"}:
        summaries.append(
            asyncio.run(
                generate_stage(
                    profile_name=args.profile,
                    run_dir=args.run_dir,
                    model=model,
                    concurrency=args.query_concurrency,
                )
            )
        )
    if args.stage in {"export", "all"}:
        summaries.append(export_stage(profile_name=args.profile, run_dir=args.run_dir))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
