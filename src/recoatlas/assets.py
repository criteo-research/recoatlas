"""Pinned RecoAtlas release files: where to download and how to check them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

QUERY_ENCODER = {"repo_id": "Qwen/Qwen3-Embedding-0.6B", "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"}

SOURCES = {
    "recoatlas": {
        "repo_type": "dataset",
        "repo_id": "iaouali/recoatlas",
        "revision": "5c9cb379ecaecb16161fc9947fb68d70fdc8fd25",
    },
}

DATASETS = {
    "amazon-musical-instruments": {
        "catalog_items": 62961,
        "splits": {"shared_train": 8937, "agent_dev": 2028, "agent_hidden": 1960},
        "files": {
            "product_texts": {
                "source": "recoatlas",
                "path": "amazon-musical-instruments/tools/catalog/product_texts.json",
                "size": 29216932,
                "sha256": "e875e343c29996c8b1d1120d033ad77fbfaca76d7cf2a6023573c90c2275c2fe",
            },
            "subcategories": {
                "source": "recoatlas",
                "path": "amazon-musical-instruments/tools/catalog/subcategories.json",
                "size": 2805895,
                "sha256": "cef77990a943053eb8695603847cf5250faa1fabeac0201d12ee292f0121498f",
            },
            "item_embeddings": {
                "source": "recoatlas",
                "path": "amazon-musical-instruments/tools/embeddings/item_embeddings.pt",
                "size": 259148886,
                "sha256": "f46aed409e66b59d8ddfdde2eaeb4cf276a7e6e71c7329033599d1f375241c97",
            },
            "comp_embeddings": {
                "source": "recoatlas",
                "path": "amazon-musical-instruments/tools/embeddings/complementarity_embeddings.pt",
                "size": 130206013,
                "sha256": "848ec9f284ad9bfb0c236cfec90e1104e7b5e0c0e80a1b96b74ef315dfdba3eb",
            },
            "retrieval_model": {
                "source": "recoatlas",
                "path": "amazon-musical-instruments/tools/retrieval/retrieval_model.pt",
                "size": 242751975,
                "sha256": "bce56bdce6684796832f1f0c20cf7b2acea2c1a6b7264798b704ff300d84c203",
            },
            "split/shared_train": {
                "source": "recoatlas",
                "path": "amazon-musical-instruments/data/shared_train.jsonl",
                "size": 3241535,
                "sha256": "0af30d87a34f2d5c28df9fd36c4f4d2276a049fa27d9d149ab3b32e98c005958",
            },
            "split/agent_dev": {
                "source": "recoatlas",
                "path": "amazon-musical-instruments/data/agent_dev.jsonl",
                "size": 719667,
                "sha256": "c71198f3b846e28a1c8a2dd7c558d41453ae6a921780de4fd49416d66c03cd37",
            },
            "split/agent_hidden": {
                "source": "recoatlas",
                "path": "amazon-musical-instruments/data/agent_hidden.jsonl",
                "size": 716751,
                "sha256": "66b85f6450a9777fba54451a2d53df9b1dbf3da4ae0b35ebb01f8585a1723437",
            },
        },
    },
    "hm-clothing": {
        "catalog_items": 104993,
        "splits": {"shared_train": 9019, "agent_dev": 2028, "agent_hidden": 1960},
        "files": {
            "product_texts": {
                "source": "recoatlas",
                "path": "hm-clothing/tools/catalog/product_texts.json",
                "size": 32847643,
                "sha256": "1c248afcf181c2a593f560fa820fdecdce3c894f1d53ce485f37bff75a1e9290",
            },
            "subcategories": {
                "source": "recoatlas",
                "path": "hm-clothing/tools/catalog/subcategories.json",
                "size": 5203714,
                "sha256": "4deb7132bf7cd7d27acb8a4d7e1db47f0c85e05085f7027f36f36f5e510f1b63",
            },
            "item_embeddings": {
                "source": "recoatlas",
                "path": "hm-clothing/tools/embeddings/item_embeddings.pt",
                "size": 432152712,
                "sha256": "f17d1ea911f9a3590c596c17ce4d59ff31d5d7a31f68c116252301c0fcce58c1",
            },
            "comp_embeddings": {
                "source": "recoatlas",
                "path": "hm-clothing/tools/embeddings/complementarity_embeddings.pt",
                "size": 862204045,
                "sha256": "75fbbcd329df7cba5aa842f93aeb888f73202ae7efadec7d46616c0884beec55",
            },
            "retrieval_model": {
                "source": "recoatlas",
                "path": "hm-clothing/tools/retrieval/retrieval_model.pt",
                "size": 242748315,
                "sha256": "78615ce6995d01cbf4947fdde4944408cd1889db1f799245c717417107bcf645",
            },
            "split/shared_train": {
                "source": "recoatlas",
                "path": "hm-clothing/data/shared_train.jsonl",
                "size": 3107569,
                "sha256": "1618811255fa38b85600e5858db33e5550771bf05ed4b4a5c4cb26eb35eae934",
            },
            "split/agent_dev": {
                "source": "recoatlas",
                "path": "hm-clothing/data/agent_dev.jsonl",
                "size": 684225,
                "sha256": "4f1274e93675aa5696f52e96b0119bade44719686f5bd26f88823677ee01ed33",
            },
            "split/agent_hidden": {
                "source": "recoatlas",
                "path": "hm-clothing/data/agent_hidden.jsonl",
                "size": 677037,
                "sha256": "7c6d8a365c359e5546d8c6bbdef379526ef951795567a8e9d50b43d8e4816f29",
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
