"""Local training-state hooks.

The public tool builder deliberately keeps intermediate state local. Stable
serving artifacts are published only through :mod:`tools_builder.release`,
which enforces the clean five-file dataset layout.
"""

from __future__ import annotations

from pathlib import Path


def hf_category(args) -> str:
    return (getattr(args, "hf_prefix", None) or args.category).strip("/")


def hf_run(args) -> str:
    run_id = getattr(args, "run_id", None)
    if not run_id:
        raise ValueError("A run_id is required")
    return f"{hf_category(args)}/runs/{run_id}"


def _require_local(hf_repo: str) -> None:
    if hf_repo:
        raise ValueError(
            "Intermediate checkpoint uploads are not part of the clean public "
            "pipeline. Use --hf_repo '' and publish the selected tools with "
            "scripts/publish_tools.py."
        )


def preflight_hf_repo(repo_id: str) -> None:
    _require_local(repo_id)


def restore_files(
    hf_repo: str,
    hf_category: str,
    save_dir: str,
    files: list[str],
    *,
    required: bool = False,
) -> list[str]:
    _require_local(hf_repo)
    if required:
        missing = [name for name in files if not (Path(save_dir) / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "Required local training artifacts are missing:\n- "
                + "\n- ".join(missing)
            )
    return []


def upload_files(
    hf_repo: str,
    hf_category: str,
    save_dir: str,
    files: list[str],
    *,
    delete_after: bool = False,
) -> list[str]:
    _require_local(hf_repo)
    return []


def prune_verified_files(save_dir: str, files: list[str]) -> None:
    # Without a durable checkpoint upload, retaining local files is the safe
    # behavior. The caller removes scratch after the final clean publication.
    return None
