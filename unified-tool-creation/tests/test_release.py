"""The public publisher exposes exactly the five environment tool files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools_builder import release as release_module
from tools_builder.release import LOCAL_TO_RELEASE, publish_release, release_artifacts, validate_release


def _write_run(root: Path) -> list[str]:
    torch = pytest.importorskip("torch")
    item_ids = ["a", "b", "c"]
    product_texts = {item: f"description {item}" for item in item_ids}
    subcategories = {
        "item_to_subcategory": {"a": "one", "b": "two", "c": "two"},
        "subcategory_to_items": {"one": ["a"], "two": ["b", "c"]},
        "subcategories": ["one", "two"],
    }
    payloads = {
        "data/product_texts.json": json.dumps(product_texts).encode(),
        "data/subcategories.json": json.dumps(subcategories).encode(),
    }
    for name, payload in payloads.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    item_path = root / "embeddings/item_embeddings_qwen.pt"
    item_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"item_ids": item_ids, "item_embs": torch.eye(3)}, item_path)
    torch.save(
        {
            "item_ids": item_ids,
            "anchor_embs": torch.eye(3),
            "complement_embs": torch.eye(3),
        },
        root / "embeddings/comp_embeddings_qwen.pt",
    )
    model_path = root / "models/recsys_model_qwen.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state_dict": {"query.weight": torch.ones(1)},
            "model_config": {"backbone": "qwen"},
        },
        model_path,
    )
    return item_ids


def _profile() -> dict:
    return {
        "profile": "test-domain",
        "release_repo": "iaouali/recoatlas",
        "release_repo_type": "dataset",
        "release_prefix": "test-domain/tools",
    }


class FakeApi:
    def __init__(self):
        self.files = {
            "test-domain/tools/debug/checkpoint.pt": b"old",
            "test-domain/data/agent_hidden.jsonl": b"keep",
        }
        self.commits = []
        self.sha = "before"

    def repo_info(self, repo_id, **kwargs):
        assert repo_id == "iaouali/recoatlas"
        assert kwargs["repo_type"] == "dataset"
        return SimpleNamespace(sha=self.sha)

    def list_repo_files(self, repo_id, **kwargs):
        assert kwargs["repo_type"] == "dataset"
        return sorted(self.files)

    def get_paths_info(self, repo_id, path, **kwargs):
        assert kwargs["repo_type"] == "dataset"
        if path not in self.files:
            return []
        payload = self.files[path]
        return [
            SimpleNamespace(
                size=len(payload),
                lfs={"sha256": hashlib.sha256(payload).hexdigest()},
            )
        ]

    def create_commit(self, **kwargs):
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["parent_commit"] == self.sha
        changed = {"add": [], "delete": []}
        for operation in kwargs["operations"]:
            if hasattr(operation, "path_or_fileobj"):
                payload = Path(operation.path_or_fileobj).read_bytes()
                self.files[operation.path_in_repo] = payload
                changed["add"].append(operation.path_in_repo)
            else:
                del self.files[operation.path_in_repo]
                changed["delete"].append(operation.path_in_repo)
        self.commits.append(changed)
        self.sha = "after"
        return SimpleNamespace(oid=self.sha)


def test_release_mapping_and_validation(tmp_path):
    _write_run(tmp_path)
    artifacts = validate_release(_profile(), tmp_path)
    assert {item.remote_path for item in artifacts} == {
        f"test-domain/tools/{name}" for name in LOCAL_TO_RELEASE.values()
    }


def test_clean_publication_has_five_files_and_no_release_noise(tmp_path, monkeypatch):
    for local_name in LOCAL_TO_RELEASE:
        path = tmp_path / local_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"payload:{local_name}".encode())
    artifacts = release_artifacts(_profile(), tmp_path)
    monkeypatch.setattr(
        release_module,
        "validate_release",
        lambda profile, run_dir: artifacts,
    )
    api = FakeApi()
    result = publish_release(_profile(), tmp_path, api=api)

    expected = {f"test-domain/tools/{name}" for name in LOCAL_TO_RELEASE.values()}
    assert set(api.commits[0]["add"]) == expected
    assert api.commits[0]["delete"] == ["test-domain/tools/debug/checkpoint.pt"]
    assert {path for path in api.files if path.startswith("test-domain/tools/")} == expected
    assert api.files["test-domain/data/agent_hidden.jsonl"] == b"keep"
    assert not any("manifest" in path or "/runs/" in path for path in api.files)
    assert result["removed"] == ["test-domain/tools/debug/checkpoint.pt"]

    second = publish_release(_profile(), tmp_path, api=api)
    assert len(api.commits) == 1
    assert second["uploaded"] == []
    assert second["removed"] == []
