"""Bounded, revision-pinned input adapter for the Electronics catalog."""

import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import hf_hub_url

from tools_builder.catalog_text import clean_description, first_available_year
from tools_builder.storage import check_ephemeral_budget


REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
MIN_YEAR = 2020


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stream(filename):
    url = hf_hub_url(
        "McAuley-Lab/Amazon-Reviews-2023",
        filename,
        repo_type="dataset",
        revision=REVISION,
    )
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1024 * 1024):
            if line:
                yield json.loads(line)


def _write_filtered(path, filename, transform):
    marker = path.with_suffix(".complete.json")
    policy = {"revision": REVISION, "min_year": MIN_YEAR, "filename": filename}
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["policy"] != policy or saved["sha256"] != _sha(path):
            raise ValueError("Filtered cache provenance changed")
        return

    temporary = path.with_suffix(".incomplete")
    kept = scanned = 0
    last_guard = 0.0
    with temporary.open("w") as target:
        for row in _stream(filename):
            scanned += 1
            record = transform(row)
            if record is not None:
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            if time.monotonic() - last_guard > 30:
                check_ephemeral_budget(
                    256 * 1024 * 1024, reason="streamed Electronics inputs"
                )
                print(
                    f"{filename}: scanned={scanned:,}, retained={kept:,}",
                    flush=True,
                )
                last_guard = time.monotonic()
    os.replace(temporary, path)
    marker.write_text(
        json.dumps(
            {
                "policy": policy,
                "scanned": scanned,
                "retained": kept,
                "sha256": _sha(path),
            }
        )
    )
    print(f"{filename}: complete, {kept:,}/{scanned:,} retained", flush=True)


def load_filtered_data():
    """Stream and retain Electronics products first available in 2020 or later."""
    scratch = os.environ.get("RECO_SCRATCH_ROOT")
    if not scratch:
        raise RuntimeError("RECO_SCRATCH_ROOT must point to ephemeral storage")
    root = Path(scratch) / "electronics-2020-inputs-v1"
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "metadata.jsonl"
    reviews_path = root / "reviews.jsonl"

    def metadata(row):
        details = row.get("details")
        details = json.dumps(details) if isinstance(details, dict) else details
        year = first_available_year(details)
        if year is None or year < MIN_YEAR:
            return None
        try:
            price = float(row.get("price"))
        except (TypeError, ValueError):
            return None
        if not 0.99 <= price <= 10000 or not clean_description(row.get("description") or []):
            return None
        fields = [
            "parent_asin",
            "main_category",
            "title",
            "average_rating",
            "rating_number",
            "description",
            "price",
            "categories",
        ]
        return {**{key: row.get(key) for key in fields}, "details": details}

    _write_filtered(
        metadata_path, "raw/meta_categories/meta_Electronics.jsonl", metadata
    )
    metadata_frame = pd.read_json(metadata_path, lines=True, convert_dates=False)
    allowed = set(metadata_frame.parent_asin)
    if not 0 < len(allowed) <= 100000:
        raise ValueError(f"Unexpected filtered Electronics catalog size: {len(allowed)}")

    def review(row):
        if row["parent_asin"] not in allowed:
            return None
        return {
            key: row.get(key)
            for key in ["user_id", "parent_asin", "timestamp", "rating", "title", "text"]
        }

    _write_filtered(
        reviews_path, "raw/review_categories/Electronics.jsonl", review
    )
    reviews_frame = pd.read_json(reviews_path, lines=True, convert_dates=False)
    return reviews_frame, metadata_frame
