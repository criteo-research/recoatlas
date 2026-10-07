"""Deterministic retrieval and three-tool baseline (copied from recoatlas-v1/baselines.py)."""
import json
from typing import Mapping

def deterministic_baselines(query: str, top_k: int, retrieval, comp, subcategories: Mapping[str, str]) -> dict[str, list[str]]:
    pool_k = max(20, top_k * 5)
    retrieval_rows = json.loads(retrieval.search(query, pool_k))
    retrieval_only = [row["item_id"] for row in retrieval_rows[:top_k]]
    anchors = []
    seen_subcategories = set()
    for row in retrieval_rows:
        item_id = row["item_id"]
        subcategory = subcategories.get(item_id, "")
        if subcategory and subcategory in seen_subcategories:
            continue
        anchors.append(item_id)
        seen_subcategories.add(subcategory)
        if len(anchors) >= min(3, top_k):
            break
    if not anchors:
        anchors = retrieval_only[:1]
    comp_rows = json.loads(comp.get_complementary(anchors, pool_k))

    retrieval_score = {row["item_id"]: float(row.get("score", 0)) for row in retrieval_rows}
    comp_score = {row["item_id"]: float(row.get("score", 0)) for row in comp_rows}
    candidates = list(dict.fromkeys([row["item_id"] for row in retrieval_rows + comp_rows]))
    selected = []
    selected_subcategories = set()
    while candidates and len(selected) < top_k:
        best = max(
            candidates,
            key=lambda item_id: (
                0.60 * retrieval_score.get(item_id, 0.0)
                + 0.40 * comp_score.get(item_id, 0.0)
                + (0.08 if subcategories.get(item_id, "") not in selected_subcategories else 0.0),
                -candidates.index(item_id),
            ),
        )
        selected.append(best)
        selected_subcategories.add(subcategories.get(best, ""))
        candidates.remove(best)
    return {"retrieval": retrieval_only, "retrieval_comp": selected}


def three_tool_baseline(query, top_k, retrieval, comp, diversity, subcategories):
    systems = deterministic_baselines(query, top_k, retrieval, comp, subcategories)
    pool = list(dict.fromkeys(systems['retrieval_comp'] + systems['retrieval']))
    result = json.loads(diversity.get_substitutes(pool, similarity_threshold=0.95))
    kept = [row['item_id'] for row in result['kept']][:top_k]
    kept.extend(i for i in pool if i not in kept)
    return kept[:top_k]


class _EpisodeTools:
    """The tool interface the v1 baseline expects, routed through an Episode so its calls are recorded."""

    def __init__(self, episode):
        self.episode = episode

    def search(self, query, top_k=5):
        return self.episode.search_products(query, top_k=top_k)

    def get_complementary(self, item_ids, top_k=5):
        return self.episode.get_complementary_products(item_ids, top_k=top_k)

    def get_substitutes(self, item_ids, similarity_threshold=0.95):
        return self.episode.get_substitute_products(item_ids, similarity_threshold=similarity_threshold)


def run_baseline(episode) -> dict:
    """Run the v1 no-LLM three-tool baseline on one episode, submit its products and return the score."""
    tools = _EpisodeTools(episode)
    ids = three_tool_baseline(episode.request, episode.k, tools, tools, tools, episode.env.item_to_subcategory)
    episode.submit([{"product_id": i, "reasoning": "no-LLM baseline"} for i in ids])
    return episode.score()
