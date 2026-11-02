"""Bounded local storage helpers for the artifact build pipeline."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


GIB = 1024 ** 3
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCRATCH_ROOT = Path("/tmp/reco-atlas")
_SIZE_CACHE_LOCK = threading.Lock()
_SIZE_CACHE_ROOT: Path | None = None
_SIZE_CACHE_BYTES: int | None = None


class DiskBudgetExceeded(RuntimeError):
    """Raised before the build would exceed a configured disk budget."""


def _env_gib(name: str, default: float) -> int:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a number of GiB") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value * GIB)


def scratch_root() -> Path:
    return Path(os.getenv("RECO_SCRATCH_ROOT", str(DEFAULT_SCRATCH_ROOT))).resolve()


def configure_storage_environment() -> Path:
    """Route library caches and temporary files to the bounded scratch tree.

    This must run before importing datasets/transformers/torch. Existing user
    overrides are honored, but the build validates that they are not on the
    persistent repository filesystem.
    """
    root = scratch_root()
    defaults = {
        "RECO_SCRATCH_ROOT": str(root),
        "HF_HOME": str(root / "huggingface"),
        "HF_DATASETS_CACHE": str(root / "huggingface" / "datasets"),
        "HUGGINGFACE_HUB_CACHE": str(root / "huggingface" / "hub"),
        "FTLANG_CACHE": str(root / "fasttext-langdetect"),
        "TMPDIR": str(root / "tmp"),
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    return root


def ensure_scratch_layout() -> Path:
    root = configure_storage_environment()
    for path in (
        root,
        Path(os.environ["HF_DATASETS_CACHE"]),
        Path(os.environ["HUGGINGFACE_HUB_CACHE"]),
        Path(os.environ["FTLANG_CACHE"]),
        Path(os.environ["TMPDIR"]),
        root / "runs",
    ):
        path.mkdir(parents=True, exist_ok=True)
    return root


def directory_size(path: Path) -> int:
    """Return allocated bytes below path without crossing filesystem links."""
    if not path.exists():
        return 0
    total = 0
    seen_inodes = set()
    for dirpath, _, filenames in os.walk(path, followlinks=False):
        for filename in filenames:
            candidate = Path(dirpath) / filename
            try:
                stat = os.stat(candidate, follow_symlinks=False)
            except FileNotFoundError:
                continue
            inode = (stat.st_dev, stat.st_ino)
            if inode in seen_inodes:
                continue
            seen_inodes.add(inode)
            total += stat.st_blocks * 512
    return total


def _scratch_size(root: Path, *, force_scan: bool) -> int:
    global _SIZE_CACHE_ROOT, _SIZE_CACHE_BYTES
    with _SIZE_CACHE_LOCK:
        if force_scan or _SIZE_CACHE_ROOT != root or _SIZE_CACHE_BYTES is None:
            _SIZE_CACHE_ROOT = root
            _SIZE_CACHE_BYTES = directory_size(root)
        return _SIZE_CACHE_BYTES


def _adjust_scratch_size(root: Path, delta: int) -> None:
    global _SIZE_CACHE_BYTES
    with _SIZE_CACHE_LOCK:
        if _SIZE_CACHE_ROOT == root and _SIZE_CACHE_BYTES is not None:
            _SIZE_CACHE_BYTES = max(0, _SIZE_CACHE_BYTES + delta)


def _allocated_size(path: Path) -> int:
    try:
        return os.stat(path, follow_symlinks=False).st_blocks * 512
    except FileNotFoundError:
        return 0


def _fmt_gib(value: int) -> str:
    return f"{value / GIB:.2f} GiB"


def check_ephemeral_budget(
    expected_bytes: int = 0,
    *,
    reason: str = "operation",
    force_scan: bool = True,
) -> int:
    """Ensure the project scratch tree remains below its operational ceiling."""
    root = ensure_scratch_layout()
    used = _scratch_size(root, force_scan=force_scan)
    soft = _env_gib("RECO_EPHEMERAL_SOFT_GIB", 50)
    hard = _env_gib("RECO_EPHEMERAL_HARD_GIB", 55)
    reserve = _env_gib("RECO_EPHEMERAL_FREE_RESERVE_GIB", 5)
    projected = used + max(0, expected_bytes)
    free = shutil.disk_usage(root).free

    if projected > hard:
        raise DiskBudgetExceeded(
            f"Refusing {reason}: scratch would reach {_fmt_gib(projected)}, "
            f"above the {_fmt_gib(hard)} hard limit."
        )
    if free - max(0, expected_bytes) < reserve:
        raise DiskBudgetExceeded(
            f"Refusing {reason}: only {_fmt_gib(free)} is free and "
            f"{_fmt_gib(reserve)} must remain reserved."
        )
    if projected > soft:
        print(
            f"WARNING: {reason} projects scratch usage at {_fmt_gib(projected)} "
            f"(soft limit: {_fmt_gib(soft)})."
        )
    return used


def check_persistent_budget(*, reason: str = "build startup") -> int:
    """Report repository-filesystem use without imposing a host-specific quota."""
    usage = shutil.disk_usage(REPO_ROOT)
    return usage.used


def validate_bulk_paths_are_ephemeral() -> None:
    """Require every generated cache path to remain below the scratch root."""
    root = scratch_root()
    for name in (
        "RECO_SCRATCH_ROOT",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "FTLANG_CACHE",
        "TMPDIR",
    ):
        path = Path(os.environ[name]).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        if path != root and root not in path.parents:
            raise DiskBudgetExceeded(
                f"{name}={path} is outside RECO_SCRATCH_ROOT={root}."
            )


def run_dir(category: str, run_id: str | None = None) -> Path:
    safe_category = category.replace("/", "_").replace("..", "_")
    resolved_run_id = run_id or os.getenv("RECO_RUN_ID")
    if not resolved_run_id:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        resolved_run_id = f"{timestamp}-{os.getpid()}"
    safe_run_id = resolved_run_id.replace("/", "_").replace("..", "_")
    return ensure_scratch_layout() / "runs" / safe_category / safe_run_id


@contextmanager
def reserved_write(path: str | Path, expected_bytes: int = 0) -> Iterator[Path]:
    """Budget-check a write and verify the budget again after it completes."""
    target = Path(path)
    check_ephemeral_budget(
        expected_bytes, reason=f"write {target.name}", force_scan=False
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    before = _allocated_size(target)
    yield target
    after = _allocated_size(target)
    _adjust_scratch_size(scratch_root(), after - before)
    check_ephemeral_budget(
        reason=f"completed write {target.name}", force_scan=False
    )


def atomic_json_dump(payload, path: str | Path, *, indent: int | None = None) -> None:
    target = Path(path)
    # JSON can expand compared with in-memory data. This is only a preflight
    # estimate; the post-write check remains authoritative.
    estimate = max(1024, len(payload) * 256) if hasattr(payload, "__len__") else 1024
    with reserved_write(target, estimate):
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(payload, handle, indent=indent)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def atomic_torch_save(payload, path: str | Path, *, expected_bytes: int = 0) -> None:
    """Atomically save a torch payload without importing torch at module load."""
    import torch

    target = Path(path)
    with reserved_write(target, expected_bytes):
        temporary = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            torch.save(payload, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def cleanup_run(path: str | Path) -> None:
    """Remove one validated run directory, never the scratch root itself."""
    target = Path(path).resolve()
    root = scratch_root()
    if target == root or root not in target.parents:
        raise ValueError(f"Refusing to clean unsafe path: {target}")
    shutil.rmtree(target)
    _scratch_size(root, force_scan=True)
