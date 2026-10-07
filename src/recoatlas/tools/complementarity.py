"""Complementarity tool for finding products frequently bought together."""

import json
import torch

from ._noise import apply_slot_noise, validate_noise_args


class ComplementarityTool:
    """Find complementary products using embedding similarity.

    When ``trained=True`` (default), uses separate anchor/complement
    projections from a trained complementarity model.  When
    ``trained=False``, uses base pretrained embeddings for both sides
    (cosine similarity + cross-subcategory filtering).

    When ``subcategories_path`` is provided, items sharing the anchor's
    subcategory are excluded so complements come from different product types.

    When ``noise > 0``, a fraction ``round(top_k * noise)`` of the final
    ranked top-k is replaced with plausible-but-wrong items. Anchor item_ids
    are excluded from the replacement pool (no anchor-as-own-complement).
    Supported ``noise_mode`` values:
      - ``"subcategory"`` (default): random item from the same subcategory.
      - ``"random"``: uniform from the full catalog.
      - ``"worst_subcategory"``: adversarial — sample from the bottom 25% of
        the replaced item's subcategory ranked by mean-anchor complementarity
        (``mean(anchor_embs[anchors]) @ complement_embs.T``), i.e. items in
        the same subcategory that are the worst complements for the call's
        anchor set.
    """

    def __init__(
        self,
        product_texts_path: str,
        complement_embeddings_path: str,
        subcategories_path: str = None,
        trained: bool = True,
        device: str = None,
        progress_cb=None,
        noise: float = 0.0,
        noise_mode: str = "subcategory",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if progress_cb:
            progress_cb("product_texts_comp", "start", {"path": product_texts_path})
        with open(product_texts_path) as f:
            self.product_reps = json.load(f)
        if progress_cb:
            progress_cb("product_texts_comp", "end", {"items": len(self.product_reps)})

        if progress_cb:
            progress_cb("comp_embeddings", "start", {"path": complement_embeddings_path})
        data = torch.load(complement_embeddings_path, map_location=self.device, weights_only=True)
        self.all_items = list(data["item_ids"])
        self.item_to_idx = {item: i for i, item in enumerate(self.all_items)}
        if trained:
            self.anchor_embs = data["anchor_embs"].to(self.device, dtype=torch.float32)
            self.complement_embs = data["complement_embs"].to(self.device, dtype=torch.float32)
        else:
            base_embs = data["item_embs"].to(self.device, dtype=torch.float32)
            self.anchor_embs = base_embs
            self.complement_embs = base_embs
        if progress_cb:
            progress_cb("comp_embeddings", "end", {"items": len(self.all_items)})

        self.item_to_subcategory = {}
        self.subcategory_to_items = {}
        self._subcat_masks = {}
        if subcategories_path:
            with open(subcategories_path) as f:
                sc_data = json.load(f)
            self.item_to_subcategory = sc_data["item_to_subcategory"]
            self.subcategory_to_items = sc_data.get("subcategory_to_items", {})
            n_items = len(self.all_items)
            for sc, items in self.subcategory_to_items.items():
                mask = torch.zeros(n_items, dtype=torch.bool, device=self.device)
                for item in items:
                    if item in self.item_to_idx:
                        mask[self.item_to_idx[item]] = True
                self._subcat_masks[sc] = mask

        self.noise, self.noise_mode = validate_noise_args(noise, noise_mode)

    @staticmethod
    def _extract_title(text: str) -> str:
        if text.startswith("Title: "):
            sep = text.find(" | Description:")
            if sep != -1:
                return text[7:sep]
        return ""

    def get_complementary(self, item_ids: list[str], top_k: int = 5) -> str:
        seen = set(item_ids)
        all_candidates = []

        for item_id in item_ids:
            if item_id not in self.item_to_idx:
                continue

            idx = self.item_to_idx[item_id]
            anchor_emb = self.anchor_embs[idx : idx + 1]

            scores = (anchor_emb @ self.complement_embs.T).squeeze(0)

            mean_s = scores.mean().item()
            max_s = scores.max().item()
            denom = max_s - mean_s + 1e-8

            scores[idx] = -float("inf")
            subcat = self.item_to_subcategory.get(item_id)
            if subcat and subcat in self._subcat_masks:
                scores[self._subcat_masks[subcat]] = -float("inf")

            per_item_k = min(top_k, len(self.all_items) - 1)
            top_scores, top_indices = torch.topk(scores, per_item_k)

            for raw_score, idx_i in zip(top_scores.tolist(), top_indices.tolist()):
                if raw_score == -float("inf"):
                    break
                cand_id = self.all_items[idx_i]
                if cand_id in seen:
                    continue
                seen.add(cand_id)
                normed = max(0.0, (raw_score - mean_s) / denom)
                text = self.product_reps.get(cand_id, "")
                all_candidates.append({
                    "item_id": cand_id,
                    "score": round(normed, 3),
                    "title": self._extract_title(text),
                    "description": text[:300],
                })

        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        final = all_candidates[:top_k]

        if self.noise > 0.0:
            candidate_scores = None
            if self.noise_mode == "worst_subcategory":
                # Rank catalog items by their mean-anchor complementarity.
                # One matmul over the anchor set, averaged; items with low
                # scores are the worst complements for this call's anchors.
                anchor_idxs = [
                    self.item_to_idx[a] for a in item_ids if a in self.item_to_idx
                ]
                if anchor_idxs:
                    with torch.no_grad():
                        mean_anchor = self.anchor_embs[anchor_idxs].mean(
                            dim=0, keepdim=True
                        )
                        scores = (mean_anchor @ self.complement_embs.T).squeeze(0)
                    candidate_scores = dict(zip(self.all_items, scores.tolist()))
            apply_slot_noise(
                final,
                tool_name="get_complementary_products",
                noise=self.noise,
                mode=self.noise_mode,
                seed_inputs=(tuple(sorted(item_ids)), top_k),
                extra_exclude=item_ids,
                item_to_subcategory=self.item_to_subcategory,
                subcategory_to_items=self.subcategory_to_items,
                all_items=self.all_items,
                product_reps=self.product_reps,
                candidate_scores=candidate_scores,
            )

        return json.dumps(final, indent=2)


COMPLEMENTARITY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_complementary_products",
        "description": (
            "Find complementary products for the given items using a trained "
            "complementarity model (dual-encoder fine-tuned on PMI co-purchase data)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Product IDs to find complements for",
                },
                "top_k": {"type": "integer", "description": "Number of results", "default": 5}
            },
            "required": ["item_ids"]
        },
        "strict": False,
    },
}
