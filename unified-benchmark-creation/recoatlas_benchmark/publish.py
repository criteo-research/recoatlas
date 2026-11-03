"""Publish only clean split JSONL files under one environment's data prefix."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

from .io import read_jsonl
from .profiles import load_profile

FIELDS = ("candidate_id", "partition", "query", "positive_items")


def validate_release(profile: dict, release_dir: Path) -> dict[str, Path]:
    files = {}
    all_queries: set[str] = set()
    all_sets: set[tuple[str, ...]] = set()
    all_candidates: set[str] = set()
    for partition in profile["partitions"]:
        path = release_dir / f"{partition}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"Empty split: {partition}")
        for row in rows:
            if tuple(row) != FIELDS or row["partition"] != partition:
                raise ValueError(f"Unexpected public schema in {path}")
            candidate = str(row["candidate_id"])
            query = " ".join(str(row["query"]).casefold().split())
            items = tuple(map(str, row["positive_items"]))
            item_set = tuple(sorted(items))
            if candidate in all_candidates or query in all_queries or item_set in all_sets:
                raise ValueError("Candidate, exact-query, or reference-set overlap")
            if not 3 <= len(items) <= 7 or len(set(items)) != len(items):
                raise ValueError("Invalid positive_items")
            all_candidates.add(candidate)
            all_queries.add(query)
            all_sets.add(item_set)
        files[partition] = path
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    profile = load_profile(args.profile)
    files = validate_release(profile, args.release_dir)
    repo = args.repo or profile["release_repo"]
    prefix = profile["release_prefix"].strip("/")
    expected = {f"{prefix}/{partition}.jsonl": path for partition, path in files.items()}
    if args.dry_run:
        for remote, local in expected.items():
            print(f"{local} -> datasets/{repo}/{remote}")
        return
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    head = api.repo_info(repo, repo_type="dataset").sha
    existing = {
        path
        for path in api.list_repo_files(repo, repo_type="dataset", revision=head)
        if path.startswith(prefix + "/")
    }
    operations = [
        CommitOperationDelete(path_in_repo=path) for path in sorted(existing - set(expected))
    ]
    operations.extend(
        CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
        for remote, local in sorted(expected.items())
    )
    result = api.create_commit(
        repo,
        repo_type="dataset",
        parent_commit=head,
        commit_message=f"Publish clean {profile['profile']} benchmark splits",
        operations=operations,
    )
    print(result.oid)


if __name__ == "__main__":
    main()
