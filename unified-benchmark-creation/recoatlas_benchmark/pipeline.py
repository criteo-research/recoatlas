"""Four durable stages: mine, judge, generate, and export."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .coherence import presentation, validate_judgment
from .inputs import load_amazon_data, mine_hm_source
from .io import atomic_json, clean_text, normalize_query, read_jsonl, sha256, write_jsonl
from .llm import JsonCaller
from .mining import Candidate, deduplicate_reference_sets, mine_amazon_candidates
from .profiles import load_profile
from .prompts import CONSISTENCY_SYSTEM, GENERATOR_GUIDANCE, JUDGE_INSTRUCTIONS, QUERY_SYSTEM
from .query_generation import evidence_block, validate_consistency, validate_query
from .splits import load_manifest


def _candidates(path: Path) -> list[Candidate]:
    output = []
    for row in read_jsonl(path):
        output.append(
            Candidate(
                candidate_id=str(row["candidate_id"]),
                partition=str(row["partition"]),
                item_ids=tuple(map(str, row["item_ids"])),
            )
        )
    return output


def _load_catalog(run_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    product_texts = json.loads((run_dir / "data/product_texts.json").read_text(encoding="utf-8"))
    subcategories = json.loads((run_dir / "data/subcategories.json").read_text(encoding="utf-8"))
    return product_texts, subcategories


def _prompt_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256(value.encode()).hexdigest()
        for name, value in {
            "coherence": JUDGE_INSTRUCTIONS,
            "query": QUERY_SYSTEM,
            "consistency": CONSISTENCY_SYSTEM,
            "guidance": GENERATOR_GUIDANCE,
        }.items()
    }


def _freeze_policy(run_dir: Path, profile: dict, manifest: dict, model: str) -> None:
    policy = {
        "profile": profile,
        "split_fingerprint": manifest["fingerprint"],
        "model": model,
        "prompt_sha256": _prompt_hashes(),
        "judge_votes": 3,
        "minimum_pass_votes": 2,
        "query_attempts": 8,
        "selection_uses_agent_results": False,
    }
    path = run_dir / "generation_policy.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != policy:
        raise ValueError("Run policy changed; use a new run directory")
    if not path.exists():
        atomic_json(path, policy)


def _assert_policy(run_dir: Path, profile: dict, model: str) -> dict:
    policy = json.loads((run_dir / "generation_policy.json").read_text(encoding="utf-8"))
    if (
        policy.get("profile") != profile
        or policy.get("model") != model
        or policy.get("prompt_sha256") != _prompt_hashes()
    ):
        raise ValueError("Profile, model, or frozen prompts differ from the run policy")
    return policy


def mine_stage(
    *,
    profile_name: str,
    split_manifest: str | Path,
    run_dir: str | Path,
    model: str,
    amazon_data_dir: str | Path | None = None,
    hm_articles: str | Path | None = None,
    hm_transactions: str | Path | None = None,
) -> dict:
    profile = load_profile(profile_name)
    manifest = load_manifest(split_manifest)
    if manifest.get("category") not in (None, profile["category"]):
        raise ValueError("The protected split belongs to another category")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _freeze_policy(run_dir, profile, manifest, model)

    if profile["domain"] == "amazon":
        if not amazon_data_dir:
            raise ValueError("Amazon profiles require --amazon-data-dir")
        interactions, product_texts, subcategories = load_amazon_data(amazon_data_dir)
        candidates = mine_amazon_candidates(
            interactions,
            manifest=manifest,
            subcategories=subcategories,
            partitions=profile["partitions"],
            positive_rating=float(profile.get("positive_rating", 4)),
            window_days=int(profile["window_days"]),
            max_items=int(profile["max_items"]),
        )
        source = {
            "amazon_data_dir": str(Path(amazon_data_dir).resolve()),
            "interactions_sha256": sha256(Path(amazon_data_dir) / "interactions.parquet"),
        }
    else:
        if not hm_articles or not hm_transactions:
            raise ValueError("H&M requires --hm-articles and --hm-transactions")
        candidates, product_texts, subcategories = mine_hm_source(
            articles_path=hm_articles,
            transactions_path=hm_transactions,
            manifest=manifest,
            partitions=profile["partitions"],
            window_days=int(profile["window_days"]),
            max_items=int(profile["max_items"]),
        )
        source = {
            "hm_articles": str(Path(hm_articles).resolve()),
            "hm_transactions": str(Path(hm_transactions).resolve()),
        }

    candidates, removed = deduplicate_reference_sets(
        candidates, profile["deduplicate_sets_in_order"]
    )
    data_dir = run_dir / "data"
    data_dir.mkdir(exist_ok=True)
    atomic_json(data_dir / "product_texts.json", product_texts)
    atomic_json(data_dir / "subcategories.json", subcategories)
    write_jsonl(run_dir / "candidates.jsonl", (row.as_json() for row in candidates))
    write_jsonl(run_dir / "candidate_exclusions.jsonl", removed)
    summary = {
        "stage": "mined",
        "profile": profile["profile"],
        "candidate_count": len(candidates),
        "candidate_counts": dict(Counter(row.partition for row in candidates)),
        "candidate_sizes": {
            partition: dict(
                sorted(
                    Counter(
                        row.bundle_size for row in candidates if row.partition == partition
                    ).items()
                )
            )
            for partition in profile["partitions"]
        },
        "identical_sets_removed": len(removed),
        "source": source,
    }
    atomic_json(run_dir / "mine_summary.json", summary)
    return summary


def _chunks(rows: list, size: int = 256) -> Iterable[list]:
    for offset in range(0, len(rows), size):
        yield rows[offset : offset + size]


async def judge_stage(
    *, profile_name: str, run_dir: str | Path, model: str, concurrency: int = 20
) -> dict:
    profile = load_profile(profile_name)
    run_dir = Path(run_dir)
    _assert_policy(run_dir, profile, model)
    candidates = _candidates(run_dir / "candidates.jsonl")
    product_texts, subcategories = _load_catalog(run_dir)
    caller = JsonCaller(
        model=model,
        calls_path=run_dir / "calls/coherence.jsonl",
        errors_path=run_dir / "calls/errors.jsonl",
        concurrency=concurrency,
    )

    async def process(candidate: Candidate) -> dict:
        votes = []
        for vote_index in range(3):
            item_count = len(candidate.item_ids)
            item_order, prompt = presentation(candidate, vote_index, product_texts, subcategories)
            judgment = await caller.call(
                key=f"{candidate.candidate_id}/vote/{vote_index}",
                messages=[{"role": "user", "content": prompt}],
                validator=lambda payload, n=item_count: validate_judgment(payload, n),
                max_tokens=900,
                retries=5,
            )
            votes.append({"vote_index": vote_index, "item_order": item_order, "judgment": judgment})
        pass_votes = sum(bool(row["judgment"]["vote_pass"]) for row in votes)
        return {
            **candidate.as_json(),
            "pass_votes": pass_votes,
            "total_votes": 3,
            "accepted": pass_votes >= 2,
            "votes": votes,
        }

    decisions: list[dict] = []
    try:
        for batch in _chunks(candidates):
            decisions.extend(await asyncio.gather(*(process(candidate) for candidate in batch)))
    finally:
        await caller.close()
    decisions.sort(key=lambda row: row["candidate_id"])
    write_jsonl(run_dir / "coherence_decisions.jsonl", decisions)
    accepted = [row for row in decisions if row["accepted"]]
    write_jsonl(run_dir / "accepted_bundles.jsonl", accepted)
    summary = {
        "stage": "judged",
        "candidate_count": len(decisions),
        "accepted_count": len(accepted),
        "rejected_count": len(decisions) - len(accepted),
        "unresolved_count": 0,
    }
    atomic_json(run_dir / "judge_summary.json", summary)
    return summary


async def generate_stage(
    *, profile_name: str, run_dir: str | Path, model: str, concurrency: int = 16
) -> dict:
    profile = load_profile(profile_name)
    run_dir = Path(run_dir)
    _assert_policy(run_dir, profile, model)
    candidates = _candidates(run_dir / "accepted_bundles.jsonl")
    product_texts, subcategories = _load_catalog(run_dir)
    caller = JsonCaller(
        model=model,
        calls_path=run_dir / "calls/query_generation.jsonl",
        errors_path=run_dir / "calls/errors.jsonl",
        concurrency=concurrency,
    )

    async def process(candidate: Candidate) -> dict:
        prior_reason = ""
        for attempt in range(8):
            evidence = evidence_block(candidate, product_texts, subcategories, attempt)
            retry = ""
            if prior_reason:
                retry = (
                    "\nA previous attempt failed automatic validation for this reason: "
                    + clean_text(prior_reason, 240)
                    + ". Correct that issue without adding unsupported details."
                )
            generated = await caller.call(
                key=f"{candidate.candidate_id}/query/{attempt}",
                messages=[
                    {"role": "system", "content": QUERY_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Generation guidance:\n{GENERATOR_GUIDANCE}{retry}\n\nProducts:\n{evidence}",
                    },
                ],
                max_tokens=700,
                retries=4,
            )
            query = " ".join(str(generated.get("query") or "").split())
            valid, prior_reason = validate_query(query, candidate.item_ids)
            if not valid:
                continue
            checked = await caller.call(
                key=f"{candidate.candidate_id}/consistency/{attempt}",
                messages=[
                    {"role": "system", "content": CONSISTENCY_SYSTEM},
                    {
                        "role": "user",
                        "content": f"SHOPPER REQUEST:\n{query}\n\nCANDIDATE PRODUCTS:\n{evidence}",
                    },
                ],
                validator=validate_consistency,
                max_tokens=700,
                retries=4,
            )
            if checked["consistent"]:
                return {
                    **candidate.as_json(),
                    "status": "accepted",
                    "query": query,
                    "attempt": attempt + 1,
                    "consistency": checked,
                }
            prior_reason = f"{checked['reason_code']}: {checked['reason']}"
        return {
            **candidate.as_json(),
            "status": "query_rejected",
            "last_reason": prior_reason,
        }

    decisions: list[dict] = []
    try:
        for batch in _chunks(candidates):
            decisions.extend(await asyncio.gather(*(process(candidate) for candidate in batch)))
    finally:
        await caller.close()
    decisions.sort(key=lambda row: row["candidate_id"])
    write_jsonl(run_dir / "query_decisions.jsonl", decisions)
    accepted = [row for row in decisions if row["status"] == "accepted"]
    summary = {
        "stage": "generated",
        "coherent_bundle_count": len(decisions),
        "accepted_query_count": len(accepted),
        "query_rejected_count": len(decisions) - len(accepted),
        "unresolved_count": 0,
    }
    atomic_json(run_dir / "generation_summary.json", summary)
    return summary


def export_stage(*, profile_name: str, run_dir: str | Path) -> dict:
    profile = load_profile(profile_name)
    run_dir = Path(run_dir)
    decisions = read_jsonl(run_dir / "query_decisions.jsonl")
    by_partition = {name: [] for name in profile["partitions"]}
    for row in decisions:
        if row["partition"] in by_partition:
            by_partition[row["partition"]].append(row)
    order = profile.get("selection_order", profile["deduplicate_sets_in_order"])
    targets = profile.get("targets", {})
    selected = {name: [] for name in profile["partitions"]}
    seen_queries: set[str] = set()
    seen_sets: set[tuple[str, ...]] = set()
    exclusions: list[dict] = []
    for partition in order:
        counts: Counter[int] = Counter()
        partition_rows = sorted(by_partition[partition], key=lambda value: value["candidate_id"])
        expected = targets.get(partition)

        def consider(
            row: dict,
            partition: str = partition,
            counts: Counter[int] = counts,
        ) -> None:
            if row.get("status") != "accepted":
                return
            size = len(row["item_ids"])
            target = int(targets.get(partition, {}).get(str(size), 2**63 - 1))
            if counts[size] >= target:
                return
            query_key = normalize_query(row["query"])
            set_key = tuple(sorted(map(str, row["item_ids"])))
            reasons = []
            if query_key in seen_queries:
                reasons.append("exact_query_overlap")
            if set_key in seen_sets:
                reasons.append("identical_reference_set")
            if reasons:
                exclusions.append({"candidate_id": row["candidate_id"], "reasons": reasons})
                return
            seen_queries.add(query_key)
            seen_sets.add(set_key)
            counts[size] += 1
            selected[partition].append(
                {
                    "candidate_id": row["candidate_id"],
                    "partition": partition,
                    "query": row["query"],
                    "positive_items": list(map(str, row["item_ids"])),
                }
            )

        if expected:
            # Reproduce H&M's fixed scheduler: each iteration takes at most
            # the current shortfall (capped at 16) from each size queue, in
            # size order. Failed candidates are consumed but do not fill a
            # target. Network completion order cannot affect selection.
            queues = {
                int(size): [row for row in partition_rows if len(row["item_ids"]) == int(size)]
                for size in expected
            }
            positions = Counter()
            while any(counts[int(size)] < int(target) for size, target in expected.items()):
                batch = []
                for size_text, target_value in expected.items():
                    size_value = int(size_text)
                    remaining = int(target_value) - counts[size_value]
                    start = positions[size_value]
                    take = min(max(remaining, 0), 16)
                    rows = queues[size_value][start : start + take]
                    batch.extend(rows)
                    positions[size_value] += len(rows)
                if not batch:
                    break
                for row in batch:
                    consider(row)
        else:
            for row in partition_rows:
                consider(row)
        if expected:
            actual = Counter(len(row["positive_items"]) for row in selected[partition])
            missing = {
                size: int(count) - actual[int(size)]
                for size, count in expected.items()
                if actual[int(size)] != int(count)
            }
            if missing:
                raise RuntimeError(f"Target shortfall for {partition}: {missing}")

    output = run_dir / "release"
    output.mkdir(exist_ok=True)
    for partition in profile["partitions"]:
        selected[partition].sort(key=lambda row: row["candidate_id"])
        write_jsonl(output / f"{partition}.jsonl", selected[partition])
    write_jsonl(run_dir / "export_exclusions.jsonl", exclusions)
    summary = {
        "stage": "complete",
        "profile": profile["profile"],
        "counts": {partition: len(rows) for partition, rows in selected.items()},
        "sizes": {
            partition: dict(sorted(Counter(len(row["positive_items"]) for row in rows).items()))
            for partition, rows in selected.items()
        },
        "exact_query_overlap": 0,
        "identical_reference_set_overlap": 0,
        "excluded_overlap_count": len(exclusions),
        "files": {
            partition: {
                "path": str(output / f"{partition}.jsonl"),
                "sha256": sha256(output / f"{partition}.jsonl"),
            }
            for partition in profile["partitions"]
        },
    }
    atomic_json(run_dir / "summary.json", summary)
    return summary
