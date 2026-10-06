# Agent and evaluation protocol

## Interaction

The fixed bundle agent is assembled from `agents/bundle.py`, the protocol
helpers in `agents/protocol.py` and `agents/specific_agent.py`, and the advisory
near-duplicate extension in `agents/diversity.py`. Only the bundle agent is
exposed by the runner; inherited helper classes are not additional experimental
conditions. The complete prompt is assembled from the scaffold, tool
descriptions, fixed `prompts/strategy.txt`, action rules, and diversity guidance.
The strategy file alone is not the entire system prompt.

For each query, the runtime creates a fresh agent. It receives the shopping
request and K, with six tool-decision rounds, minimum two distinct searches and
one complementarity call, and a persistent ledger of at most 60 candidates.
The near-duplicate check at threshold 0.95 is requested but not forced. Its
results are advisory: the agent resolves conflicts with explicit requirements.
Protocol repairs and guard-generated actions are recorded in traces.

The finalizer uses accumulated evidence. The parser removes duplicate and
out-of-catalog identifiers, restricts IDs to observed candidates, and can fill
a short report from remaining observed candidates in stored order, marking
backfilled entries. These are the paper benchmark's rules, not the stricter
validity/reward rules of its separate RL implementation.

`configs/paper_agent.json` stores default decoding and budget settings. The
grammar-enabled backend validates the JSON contract locally even when a server
claims to enforce it. The default script does not train prompts or select an
agent strategy from hidden-query feedback.

## Tools and non-LLM baseline

Utility retrieval uses the frozen query LoRA; semantic retrieval disables it.
Utility complementarity uses the released learned anchor/complement vectors;
semantic expansion uses base item vectors with the same cross-subcategory
exclusions. Both conditions use identical same-subcategory near-duplicate
pruning at cosine similarity above 0.95. Tool descriptions and agent prompts
are unchanged between these conditions.

The non-LLM baseline retrieves `max(20, 5*K)` candidates, selects up to three
subcategory-distinct anchors, and expands complements. Its greedy score is
`0.60*relevance + 0.40*complementarity + 0.08*new_subcategory`.
It forms a stable union of hybrid top-K followed by retrieval top-K, prunes
near duplicates, takes the first K, and restores original-order candidates
when needed to reach K. No reference IDs influence selection.

## Post-report feedback

SetHit@K is fractional reference recall, averaged equally over queries. The
first K positions are selected before removing duplicates/invalid IDs, so later
positions cannot earn credit by shifting forward. Missing or failed reports
receive zero. Generation summaries cover the selected query prefix; standalone
evaluation covers the full selected partition.

The direct-relevance judge receives only the request, K, and 300-character
catalog evidence per selected item. IDs are replaced with P1, P2, etc., after
deterministic shuffling using seed 20260925 and the candidate ID. Explanations,
purchase history, gold references, tool provenance and exact-recovery metrics
are omitted. It scores the usefulness of selections, not bundle completeness
or explanation quality. Its integer 0–4 score is divided by four and averaged;
this is an ordinal summary, not a calibrated probability.

Each report is judged separately. A failed judgment remains unresolved, never
a semantic zero. Summaries expose the evaluated coverage. The optional
`evaluation.metrics.paired_bootstrap` function compares matched per-query score
dictionaries using 10,000 paired resamples; it refuses mismatched query IDs.
The fixed rubric must not be retuned on evaluation outputs to produce a desired
tool ranking. Neither semantic suitability nor exact recovery measures live
user satisfaction by itself.

## Outputs

`generate_traces.py` writes one JSON object per selected query:

- `candidate_id`, `status`, and `predicted_item_ids`;
- for LLM agents, the report, tool trace, protocol statistics, and model-call
  messages/outputs/usage;
- `feedback.sethit_at_k`, computed only after the agent has finished;
- on failure, `error_type` without dumping endpoint URLs or credentials.

The adjacent `.summary.json` records counts, settings, dataset/config hashes,
and aggregate recovery. `judge_traces.py` writes one row per judgment attempt,
including the accepted payload and normalized quality when successful, plus
its own coverage/model/prompt-hash summary. Logs can contain catalog text and
model-generated reasoning; review them before redistributing them.
