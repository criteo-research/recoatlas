#!/usr/bin/env python3
"""Create the protected 70/15/15 benchmark user manifest once."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools_builder.benchmark_split import (  # noqa: E402
    DEFAULT_RATIOS,
    build_user_split_manifest,
    load_user_split_manifest,
)


def _write_new_manifest(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = load_user_split_manifest(output)
        if existing["fingerprint"] == payload["fingerprint"]:
            print(f"Split already exists with the same fingerprint: {output}")
            return
        raise FileExistsError(
            f"Refusing to replace a different protected split: {output}. "
            "Use a new versioned filename."
        )
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.suffix.lower() in {".csv", ".gz", ".parquet"}
    )
    if not files:
        raise FileNotFoundError(f"No CSV or Parquet inputs found below {path}")
    return files


def _amazon_users(path: Path, positive_rating: float) -> set[str]:
    users: set[str] = set()
    for source in _input_files(path):
        if source.suffix.lower() == ".parquet":
            parquet = pq.ParquetFile(source)
            for batch in parquet.iter_batches(columns=["user_id", "rating"]):
                frame = batch.to_pandas()
                selected = frame.loc[
                    pd.to_numeric(frame["rating"], errors="coerce") >= positive_rating,
                    "user_id",
                ]
                users.update(selected.dropna().astype(str))
        else:
            for frame in pd.read_csv(
                source, usecols=["user_id", "rating"], chunksize=500_000
            ):
                selected = frame.loc[
                    pd.to_numeric(frame["rating"], errors="coerce") >= positive_rating,
                    "user_id",
                ]
                users.update(selected.dropna().astype(str))
    return users


def _hm_users(path: Path) -> set[str]:
    users: set[str] = set()
    for source in _input_files(path):
        if source.suffix.lower() == ".parquet":
            parquet = pq.ParquetFile(source)
            for batch in parquet.iter_batches(columns=["customer_id"]):
                column = batch.schema.get_field_index("customer_id")
                users.update(batch.column(column).to_pylist())
        else:
            for frame in pd.read_csv(
                source, usecols=["customer_id"], chunksize=500_000
            ):
                users.update(frame["customer_id"].dropna().astype(str))
    return {str(user) for user in users if user is not None and str(user)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--domain", choices=["amazon", "hm"], default="amazon")
    parser.add_argument("--category", default="Musical_Instruments")
    parser.add_argument("--positive-rating", type=float, default=4.0)
    parser.add_argument(
        "--salt",
        default=None,
        help=(
            "Secret split key. Omit to generate a new 256-bit key. The key is "
            "stored only in the protected manifest and must not be published."
        ),
    )
    parser.add_argument("--ratios", nargs=3, type=float, default=DEFAULT_RATIOS)
    args = parser.parse_args()

    if args.domain == "amazon":
        positive_users = _amazon_users(args.interactions, args.positive_rating)
        positive_policy = f"review rating >= {args.positive_rating:g}"
    else:
        positive_users = _hm_users(args.interactions)
        positive_policy = "every purchase is an implicit positive"
        if args.category == "Musical_Instruments":
            args.category = "H_and_M"
    split_key = args.salt or secrets.token_hex(32)
    manifest = build_user_split_manifest(
        positive_users,
        ratios=args.ratios,
        salt=split_key,
        category=args.category,
        positive_rating_threshold=(
            args.positive_rating if args.domain == "amazon" else None
        ),
    )
    manifest["dataset"] = args.domain
    manifest["positive_policy"] = positive_policy
    _write_new_manifest(manifest, args.output)
    print(
        "Protected user split: "
        + ", ".join(
            f"{name}={count:,}" for name, count in manifest["counts"].items()
        )
    )
    print(f"fingerprint={manifest['fingerprint']}")
    print("Do not publish this file: it contains agent_hidden user IDs.")


if __name__ == "__main__":
    main()
