"""Validate and publish the five inference artifacts used by RecoAtlas.

Training checkpoints stay in their resumable model repository.  This module
publishes only the stable inference surface consumed by ``recoatlas-env`` and
never writes manifests, metrics, recipes, caches, or run directories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


LOCAL_TO_RELEASE = {
    "data/product_texts.json": "catalog/product_texts.json",
    "data/subcategories.json": "catalog/subcategories.json",
    "embeddings/item_embeddings_qwen.pt": "embeddings/item_embeddings.pt",
    "embeddings/comp_embeddings_qwen.pt": "embeddings/complementarity_embeddings.pt",
    "models/recsys_model_qwen.pt": "retrieval/retrieval_model.pt",
}


@dataclass(frozen=True)
class ReleaseArtifact:
    local_path: Path
    remote_path: str
    size: int
    sha256: str
    git_blob_sha1: str

    def public_record(self) -> dict[str, Any]:
        value = asdict(self)
        value["local_path"] = str(self.local_path)
        return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path, size: int) -> str:
    digest = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_artifacts(profile: dict, run_dir: Path) -> list[ReleaseArtifact]:
    """Resolve the exact clean release paths and hash every local payload."""
    run_dir = Path(run_dir).resolve()
    prefix = profile["release_prefix"].strip("/")
    if not prefix.endswith("/tools"):
        raise ValueError("release_prefix must end in '/tools'")

    missing = [relative for relative in LOCAL_TO_RELEASE if not (run_dir / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "The completed tool run is missing release artifacts:\n- "
            + "\n- ".join(missing)
        )

    artifacts = []
    for local_name, release_name in LOCAL_TO_RELEASE.items():
        path = run_dir / local_name
        size = path.stat().st_size
        artifacts.append(
            ReleaseArtifact(
                local_path=path,
                remote_path=f"{prefix}/{release_name}",
                size=size,
                sha256=_sha256(path),
                git_blob_sha1=_git_blob_sha1(path, size),
            )
        )
    return artifacts


def _require_string_mapping(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a nonempty JSON object")
    if any(not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()):
        raise ValueError(f"{name} must map strings to strings")
    return value


def validate_release(profile: dict, run_dir: Path) -> list[ReleaseArtifact]:
    """Check every cross-file invariant expected by ``recoatlas-env``."""
    import torch

    artifacts = release_artifacts(profile, run_dir)
    by_release_name = {Path(item.remote_path).name: item.local_path for item in artifacts}

    product_texts = _require_string_mapping(
        json.loads(by_release_name["product_texts.json"].read_text()),
        "product_texts",
    )
    item_ids = list(product_texts)
    if any(not text.strip() for text in product_texts.values()):
        raise ValueError("product_texts contains an empty catalog representation")

    subcategories = json.loads(by_release_name["subcategories.json"].read_text())
    if not isinstance(subcategories, dict):
        raise ValueError("subcategories must be a JSON object")
    item_to_subcategory = _require_string_mapping(
        subcategories.get("item_to_subcategory"), "item_to_subcategory"
    )
    if set(item_to_subcategory) != set(item_ids):
        raise ValueError("subcategories are not aligned with the catalog")
    subcategory_to_items = subcategories.get("subcategory_to_items")
    if not isinstance(subcategory_to_items, dict):
        raise ValueError("subcategories is missing subcategory_to_items")
    inverse = {
        category: sorted(item for item, value in item_to_subcategory.items() if value == category)
        for category in set(item_to_subcategory.values())
    }
    actual_inverse = {
        category: sorted(items)
        for category, items in subcategory_to_items.items()
        if items
    }
    if inverse != actual_inverse:
        raise ValueError("subcategory_to_items is not the inverse of item_to_subcategory")

    item_data = torch.load(
        by_release_name["item_embeddings.pt"], map_location="cpu", weights_only=True
    )
    if list(item_data.get("item_ids", ())) != item_ids:
        raise ValueError("item embeddings are not aligned with product_texts")
    item_embeddings = item_data.get("item_embs")
    if not isinstance(item_embeddings, torch.Tensor) or item_embeddings.ndim != 2:
        raise ValueError("item_embeddings.pt must contain a two-dimensional item_embs tensor")
    if item_embeddings.shape[0] != len(item_ids):
        raise ValueError("item embedding row count differs from the catalog size")
    del item_data, item_embeddings

    complementarity = torch.load(
        by_release_name["complementarity_embeddings.pt"],
        map_location="cpu",
        weights_only=True,
    )
    if list(complementarity.get("item_ids", ())) != item_ids:
        raise ValueError("complementarity embeddings are not aligned with product_texts")
    anchor = complementarity.get("anchor_embs")
    complement = complementarity.get("complement_embs")
    if not isinstance(anchor, torch.Tensor) or not isinstance(complement, torch.Tensor):
        raise ValueError("complementarity artifact is missing anchor/complement tensors")
    if anchor.ndim != 2 or anchor.shape != complement.shape:
        raise ValueError("anchor and complement embeddings must have the same 2-D shape")
    if anchor.shape[0] != len(item_ids):
        raise ValueError("complementarity row count differs from the catalog size")
    del complementarity, anchor, complement

    retrieval = torch.load(
        by_release_name["retrieval_model.pt"], map_location="cpu", weights_only=True
    )
    adapter = retrieval.get("adapter_state_dict")
    if not isinstance(adapter, dict) or not adapter:
        raise ValueError("retrieval checkpoint is missing adapter_state_dict")
    if any(not isinstance(value, torch.Tensor) for value in adapter.values()):
        raise ValueError("retrieval adapter_state_dict contains a non-tensor value")
    config = retrieval.get("model_config", {})
    if not isinstance(config, dict):
        raise ValueError("retrieval model_config must be a mapping")
    if config.get("backbone", "qwen") != "qwen":
        raise ValueError("the release environment expects the Qwen retrieval backbone")
    if config.get("model_name", "Qwen/Qwen3-Embedding-0.6B") != "Qwen/Qwen3-Embedding-0.6B":
        raise ValueError("retrieval checkpoint uses the wrong query encoder")
    revision = config.get("revision")
    if revision not in (None, "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"):
        raise ValueError("retrieval checkpoint uses the wrong query-encoder revision")

    return artifacts


def _lfs_sha256(info: Any) -> str | None:
    lfs = getattr(info, "lfs", None)
    if isinstance(lfs, dict):
        return lfs.get("sha256") or lfs.get("oid")
    return getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)


def _matches_remote(info: Any, artifact: ReleaseArtifact) -> bool:
    if getattr(info, "size", None) != artifact.size:
        return False
    if remote_sha := _lfs_sha256(info):
        return remote_sha == artifact.sha256
    if remote_blob := getattr(info, "blob_id", None):
        return remote_blob == artifact.git_blob_sha1
    # A matching size is insufficient evidence for a non-LFS file.
    return False


def _remote_info(api: Any, repo_id: str, path: str, revision: str) -> Any | None:
    rows = api.get_paths_info(
        repo_id,
        path,
        repo_type="dataset",
        revision=revision,
    )
    return rows[0] if rows else None


def publish_release(
    profile: dict,
    run_dir: Path,
    *,
    repo_id: str | None = None,
    revision: str = "main",
    api: Any | None = None,
    clean: bool = True,
) -> dict[str, Any]:
    """Atomically synchronize the clean inference folder in an HF dataset.

    When ``clean`` is true, files below this profile's ``release_prefix`` that
    are not one of the five inference artifacts are removed in the same commit.
    Sibling ``data`` and ``benchmark`` trees are outside that prefix and cannot
    be affected.
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    if profile.get("release_repo_type") != "dataset":
        raise ValueError("clean RecoAtlas releases must target an HF dataset repository")
    repo_id = repo_id or profile["release_repo"]
    artifacts = validate_release(profile, run_dir)
    api = api or HfApi()
    head = api.repo_info(repo_id, repo_type="dataset", revision=revision).sha

    expected_paths = {item.remote_path for item in artifacts}
    stale_paths: list[str] = []
    if clean:
        prefix = profile["release_prefix"].strip("/") + "/"
        stale_paths = sorted(
            path
            for path in api.list_repo_files(
                repo_id, repo_type="dataset", revision=head
            )
            if path.startswith(prefix) and path not in expected_paths
        )

    changed = []
    for artifact in artifacts:
        remote = _remote_info(api, repo_id, artifact.remote_path, head)
        if remote is None or not _matches_remote(remote, artifact):
            changed.append(artifact)

    if not changed and not stale_paths:
        return {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "revision": head,
            "uploaded": [],
            "removed": [],
            "verified": [item.public_record() for item in artifacts],
        }

    result = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        parent_commit=head,
        operations=[
            CommitOperationAdd(
                path_in_repo=item.remote_path,
                path_or_fileobj=str(item.local_path),
            )
            for item in changed
        ] + [CommitOperationDelete(path_in_repo=path) for path in stale_paths],
        commit_message=f"Publish {profile['profile']} inference tools",
    )
    committed_revision = getattr(result, "oid", None) or getattr(result, "commit_id", None)
    if not committed_revision:
        committed_revision = api.repo_info(
            repo_id, repo_type="dataset", revision=revision
        ).sha

    for artifact in artifacts:
        remote = _remote_info(api, repo_id, artifact.remote_path, committed_revision)
        if remote is None or not _matches_remote(remote, artifact):
            raise IOError(f"Remote verification failed for {artifact.remote_path}")

    return {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": committed_revision,
        "uploaded": [item.remote_path for item in changed],
        "removed": stale_paths,
        "verified": [item.public_record() for item in artifacts],
    }
