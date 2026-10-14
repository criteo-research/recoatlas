"""Durable per-query checkpoints; resuming never selects or reruns by score."""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def code_fingerprint() -> str:
    root = Path(__file__).parent
    return fingerprint(
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.suffix in {".py", ".json"}
        }
    )


@contextmanager
def atomic_file(path: Path):
    """Replace a runner-owned file only after its complete contents reach disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    with atomic_file(path) as stream:
        stream.write((json.dumps(value, indent=2) + "\n").encode())


class RunStore:
    """One writer per arm. A crash loses at most in-flight, not checkpointed, queries.

    Both complete and failed evaluations are final observations. Resume skips both;
    it does not turn infrastructure/model failures into repeated chances at recovery.
    """

    def __init__(self, output: Path, manifest: dict, rows: list[dict], *, resume=False):
        self.output = output
        self.summary = output.with_suffix(".summary.json")
        self.directory = output.with_suffix(output.suffix + ".state")
        self.manifest = manifest
        self.ids = [row["candidate_id"] for row in rows]
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Duplicate candidate IDs in the selected split")
        if output == self.summary:
            raise ValueError("Output must differ from its .summary.json path")
        self.resume = resume
        self.lock = None
        self.done: dict[str, dict] = {}

    def path(self, candidate_id):
        return self.directory / (fingerprint(candidate_id) + ".json.gz")

    def __enter__(self):
        exists = self.directory.exists() or self.output.exists() or self.summary.exists()
        if exists and not self.resume:
            raise ValueError(f"Output already exists: {self.output}; use --resume or a new path")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / ".lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest_path = self.directory / "manifest.json"
            if manifest_path.exists():
                if json.loads(manifest_path.read_text()) != self.manifest:
                    raise ValueError("Resume mismatch: code, settings, runtime or dataset changed; use a new output")
            else:
                if self.output.exists() or self.summary.exists() or list(self.directory.glob("*.json.gz")):
                    raise ValueError("Existing results without a matching resume manifest")
                write_json(manifest_path, self.manifest)
            allowed = set(self.ids)
            for path in self.directory.glob("*.json.gz"):
                with gzip.open(path, "rt") as stream:
                    record = json.load(stream)
                cid = record["candidate_id"]
                if cid not in allowed or path != self.path(cid) or cid in self.done:
                    raise ValueError(f"Unexpected checkpoint: {path.name}")
                if record.get("status") not in {"complete", "error"}:
                    raise ValueError(f"Invalid checkpoint status: {path.name}")
                self.done[cid] = self._score(record)
            return self
        except BaseException:
            self.lock.close()
            raise

    @staticmethod
    def _score(record):
        # Keep only small summaries in RAM: model/tool traces remain on disk.
        return {"status": record["status"], "sethit_at_k": record["feedback"]["sethit_at_k"]}

    def save(self, record):
        cid = record["candidate_id"]
        if cid not in self.ids or cid in self.done:
            raise ValueError("Unexpected or duplicate query result")
        with atomic_file(self.path(cid)) as stream:
            stream.write(gzip.compress(json.dumps(record).encode(), mtime=0))
        self.done[cid] = self._score(record)

    def progress(self):
        completed = sum(r["status"] == "complete" for r in self.done.values())
        return {
            "evaluated_queries": len(self.done),
            "completed": completed,
            "failed": len(self.done) - completed,
            "pending": len(self.ids) - len(self.done),
            # Do not label a partial run as a full-split metric.
            "mean_sethit_at_k": (
                sum(r["sethit_at_k"] for r in self.done.values()) / len(self.ids)
                if self.ids and len(self.done) == len(self.ids)
                else None
            ),
        }

    def export(self):
        """Deterministic dataset order, compatible with recoatlas-judge."""
        with atomic_file(self.output) as stream:
            for cid in self.ids:
                if cid in self.done:
                    with gzip.open(self.path(cid), "rb") as record:
                        stream.write(record.read() + b"\n")

    def __exit__(self, *_):
        self.lock.close()
