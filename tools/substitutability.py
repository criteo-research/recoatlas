"""Substitutability tool for finding similar/interchangeable products."""

import json
import torch
import torch.nn.functional as F

from ._noise import seeded_rng


class SubstitutabilityTool:
    """Find similar/interchangeable products using embedding similarity.

    Hard exact-subcategory filtering: only items sharing the anchor's
    subcategory are considered.  Scores are normalized against the
    same-subcategory catalog slice with max(0, (s - mean) / (max - mean)).

    When ``noise > 0``, the tool simulates a weak duplicate-detector via
    false-negatives: a fraction ``round(len(removed) * noise)`` of items that
    the clean pass flagged as duplicates is moved back into ``kept``, with
    ``duplicate_of`` and ``similarity`` stripped so the output is
    indistinguishable from a legitimately-kept item.
    """

    def __init__(
        self, product_texts_path: str, subcategories_path: str,
        embeddings_path: str, item_order_path: str = None,
        device: str = None,
        noise: float = 0.0,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        with open(product_texts_path) as f:
            self.product_reps = json.load(f)
        with open(subcategories_path) as f:
            data = json.load(f)

        self.item_to_subcategory = data['item_to_subcategory']
        self.subcategory_to_items = data['subcategory_to_items']

        if item_order_path:
            with open(item_order_path) as f:
                self.all_items = json.load(f)
        else:
            self.all_items = list(self.product_reps.keys())
        self.item_to_idx = {item: i for i, item in enumerate(self.all_items)}

        data = torch.load(embeddings_path, map_location=self.device, weights_only=True)
        if list(data['item_ids']) != self.all_items:
            raise ValueError('Diversity embeddings are not aligned with the catalog')
        embeddings = data['item_embs']
        self.item_embs = F.normalize(embeddings.float(), p=2, dim=-1)

        self._build_subcat_masks()

        self.noise = max(0.0, min(1.0, float(noise)))

    def _build_subcat_masks(self):
        """Precompute a boolean mask per subcategory for fast filtering."""
        n_items = len(self.all_items)
        self._subcat_masks = {}
        for sc, items in self.subcategory_to_items.items():
            mask = torch.zeros(n_items, dtype=torch.bool, device=self.device)
            for item in items:
                if item in self.item_to_idx:
                    mask[self.item_to_idx[item]] = True
            self._subcat_masks[sc] = mask

    def _pairwise_similarity(self, id_a: str, id_b: str) -> float:
        idx_a = self.item_to_idx.get(id_a)
        idx_b = self.item_to_idx.get(id_b)
        if idx_a is None or idx_b is None:
            return 0.0
        sc_a = self.item_to_subcategory.get(id_a)
        sc_b = self.item_to_subcategory.get(id_b)
        if not sc_a or not sc_b or sc_a != sc_b:
            return 0.0
        return (self.item_embs[idx_a] @ self.item_embs[idx_b]).item()

    @staticmethod
    def _extract_title(text: str) -> str:
        if text.startswith("Title: "):
            sep = text.find(" | Description:")
            if sep != -1:
                return text[7:sep]
        return ""

    def get_substitutes(
        self, item_ids: list[str], similarity_threshold: float = 0.95,
    ) -> str:
        """Remove near-duplicate items from a candidate set.

        Walks ``item_ids`` in order (first = highest priority).  For each
        item, if any already-kept item is in the same subcategory and their
        cosine similarity exceeds ``similarity_threshold``, the item is
        removed.  Returns kept/removed partitions with similarity details.
        """
        valid_ids = [i for i in item_ids if i in self.item_to_idx]
        if len(valid_ids) <= 1:
            items = []
            for item_id in valid_ids:
                text = self.product_reps.get(item_id, "")
                items.append({
                    "item_id": item_id,
                    "title": self._extract_title(text),
                    "subcategory": self.item_to_subcategory.get(item_id, ""),
                    "description": text[:300],
                })
            return json.dumps({
                "kept": items,
                "removed": [],
                "similarity_threshold": similarity_threshold,
            }, indent=2)

        kept = []
        removed = []
        for item_id in valid_ids:
            sc = self.item_to_subcategory.get(item_id)
            duplicate_of = None
            dup_sim = 0.0
            for kept_item in kept:
                sim = self._pairwise_similarity(item_id, kept_item["item_id"])
                if sim > similarity_threshold:
                    duplicate_of = kept_item["item_id"]
                    dup_sim = sim
                    break

            text = self.product_reps.get(item_id, "")
            entry = {
                "item_id": item_id,
                "title": self._extract_title(text),
                "subcategory": sc or "",
                "description": text[:300],
            }
            if duplicate_of is None:
                kept.append(entry)
            else:
                entry["duplicate_of"] = duplicate_of
                entry["similarity"] = round(dup_sim, 4)
                removed.append(entry)

        if self.noise > 0.0 and removed:
            n_unremove = round(len(removed) * self.noise)
            if n_unremove > 0:
                rng = seeded_rng(
                    "get_substitute_products",
                    (tuple(valid_ids), similarity_threshold),
                )
                unremove_entries = rng.sample(removed, n_unremove)
                unremove_ids = {e["item_id"] for e in unremove_entries}
                removed = [e for e in removed if e["item_id"] not in unremove_ids]
                for entry in unremove_entries:
                    entry.pop("duplicate_of", None)
                    entry.pop("similarity", None)
                kept_ids = {e["item_id"] for e in kept} | unremove_ids
                by_id = {e["item_id"]: e for e in kept}
                by_id.update({e["item_id"]: e for e in unremove_entries})
                kept = [by_id[i] for i in valid_ids if i in kept_ids]

        return json.dumps({
            "kept": kept,
            "removed": removed,
            "similarity_threshold": similarity_threshold,
        }, indent=2)


SUBSTITUTABILITY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_substitute_products",
        "description": (
            "Remove near-duplicate products from a candidate set. Items in the same "
            "subcategory with cosine similarity above the threshold are pruned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Candidate product IDs to deduplicate (order = priority)",
                },
                "similarity_threshold": {
                    "type": "number",
                    "description": "Similarity above which items are considered duplicates",
                    "default": 0.95,
                },
            },
            "required": ["item_ids"]
        },
        "strict": False,
    },
}
