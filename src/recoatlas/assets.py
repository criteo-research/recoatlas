"""Pinned dataset files: where to download them and how to check them.

Values are copied from recoatlas-v1's artifact_sources.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

QUERY_ENCODER = {"repo_id": "Qwen/Qwen3-Embedding-0.6B", "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"}

SOURCES = {
    "hm-data": {
        "repo_type": "dataset",
        "repo_id": "iaouali/hm-benchmark",
        "revision": "72f567a005a17822eed9e16709e36054df4ee01b",
    },
    "hm-tools": {
        "repo_type": "model",
        "repo_id": "iaouali/hm-tools",
        "revision": "56e8a4a53df668d0d332c184665016d448362a3f",
    },
}

DATASETS = {
    "hm-clothing": {
        "catalog_items": 104993,
        "splits": {"shared_train": 9019, "agent_dev": 2028, "agent_hidden": 1960},
        "files": {
            "product_texts": {
                "source": "hm-tools",
                "path": "H_and_M/data/product_texts.json",
                "size": 32847643,
                "sha256": "1c248afcf181c2a593f560fa820fdecdce3c894f1d53ce485f37bff75a1e9290",
            },
            "subcategories": {
                "source": "hm-tools",
                "path": "H_and_M/data/subcategories.json",
                "size": 5203714,
                "git_blob_sha1": "1f47cad854a0556b23bdb8d1b88df14dc599e535",
            },
            "item_embeddings": {
                "source": "hm-tools",
                "path": "H_and_M/embeddings/item_embeddings_qwen.pt",
                "size": 432152712,
                "sha256": "f17d1ea911f9a3590c596c17ce4d59ff31d5d7a31f68c116252301c0fcce58c1",
            },
            "comp_embeddings": {
                "source": "hm-tools",
                "path": "H_and_M/runs/shared-train-best-recipes-v1/embeddings/comp_embeddings_qwen.pt",
                "size": 862204045,
                "sha256": "75fbbcd329df7cba5aa842f93aeb888f73202ae7efadec7d46616c0884beec55",
            },
            "retrieval_model": {
                "source": "hm-tools",
                "path": "H_and_M/runs/shared-train-best-recipes-v1/models/recsys_model_qwen.pt",
                "size": 242748315,
                "sha256": "78615ce6995d01cbf4947fdde4944408cd1889db1f799245c717417107bcf645",
            },
            "split/shared_train": {
                "source": "hm-data",
                "path": "H_and_M/shared_train.jsonl",
                "size": 8509165,
                "git_blob_sha1": "391b8b6bfd6751a6baf94a517ec0e2d1b427ec9a",
            },
            "split/agent_dev": {
                "source": "hm-data",
                "path": "H_and_M/agent_dev.jsonl",
                "size": 1895051,
                "git_blob_sha1": "e7953d4e3744529576fdb9092f53f162fcd7f15b",
            },
            "split/agent_hidden": {
                "source": "hm-data",
                "path": "H_and_M/agent_hidden.jsonl",
                "size": 1849884,
                "git_blob_sha1": "d2a2d213401be881b2afe104c780da58a35c485a",
            },
        },
    },
}

_VERIFIED_CACHE = Path.home() / ".cache" / "recoatlas" / "verified.json"


def _digest(path: Path, spec: dict) -> str:
    # git blob SHA-1 = SHA-1 over "blob <size>\0" followed by the content
    digest = hashlib.sha256() if "sha256" in spec else hashlib.sha1(f"blob {spec['size']}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, spec: dict) -> None:
    """Check size and checksum. A file that passed once is only re-hashed if it changed."""
    path = Path(path).resolve()
    stat = path.stat()
    if stat.st_size != spec["size"]:
        raise ValueError(f"Wrong size for {path}: {stat.st_size} != {spec['size']}")
    key, expected = f"{path}:{stat.st_size}:{stat.st_mtime_ns}", spec.get("sha256") or spec["git_blob_sha1"]
    try:
        verified = json.loads(_VERIFIED_CACHE.read_text())
    except (OSError, ValueError):
        verified = {}
    if verified.get(key) == expected:
        return
    if _digest(path, spec) != expected:
        raise ValueError(f"Checksum mismatch for {path}")
    verified[key] = expected
    _VERIFIED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _VERIFIED_CACHE.write_text(json.dumps(verified, indent=1))


def fetch(dataset: str, name: str) -> Path:
    """Download one pinned file into the Hugging Face cache (if needed), verify it, return its path."""
    from huggingface_hub import hf_hub_download

    spec = DATASETS[dataset]["files"][name]
    path = Path(hf_hub_download(**SOURCES[spec["source"]], filename=spec["path"]))
    verify_file(path, spec)
    return path


def fetch_query_encoder() -> None:
    """Download the pinned query encoder so transformers can load it from the cache."""
    from huggingface_hub import snapshot_download

    snapshot_download(**QUERY_ENCODER)


def load_split(dataset: str, split: str) -> list[dict]:
    """The requests of one split: dicts with candidate_id, query, positive_items, ..."""
    rows = [json.loads(line) for line in fetch(dataset, f"split/{split}").read_text().splitlines() if line.strip()]
    expected = DATASETS[dataset]["splits"][split]
    if len(rows) != expected:
        raise ValueError(f"{dataset}/{split}: expected {expected} rows, found {len(rows)}")
    return rows
