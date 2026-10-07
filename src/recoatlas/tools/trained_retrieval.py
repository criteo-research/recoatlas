from __future__ import annotations
import json
import threading
from pathlib import Path

class TrainedRetrievalTool:
    """Inference adapter for the split-trained multi-positive retrieval checkpoint."""

    def __init__(self, texts_path: Path, embeddings_path: Path, checkpoint_path: Path, device: str,
                 encoder_repo: str, encoder_revision: str):
        import torch
        import torch.nn.functional as functional
        from peft import set_peft_model_state_dict
        from .query_encoder import QueryRetrievalModel

        self.product_reps = json.loads(Path(texts_path).read_text())
        data = torch.load(embeddings_path, map_location="cpu", weights_only=True)
        self.all_items = list(data["item_ids"])
        if self.all_items != list(self.product_reps):
            raise ValueError("retrieval embeddings are not aligned with product_texts")
        self.item_to_idx = {item_id: index for index, item_id in enumerate(self.all_items)}
        self.item_embs = functional.normalize(data["item_embs"].float(), dim=-1).to(device)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = checkpoint.get("model_config", {})
        encoder = {'repo_id': encoder_repo, 'revision': encoder_revision}
        if config.get('model_name', encoder['repo_id']) != encoder['repo_id'] or config.get('backbone', 'qwen') != 'qwen':
            raise ValueError('Checkpoint does not match the pinned query encoder')
        self.checkpoint_metadata = {
            "epoch": checkpoint.get("epoch"),
            "baseline_winner": checkpoint.get("baseline_winner", False),
            "objective": config.get("objective"),
        }
        model = QueryRetrievalModel(
            backbone=config.get("backbone", "qwen"),
            model_name=config.get("model_name"),
            lora_r=int(config.get("lora_r", 32)),
            lora_alpha=int(config.get("lora_alpha", 32)),
            lora_dropout=float(config.get("lora_dropout", 0.1)),
            query_instruction=config.get("query_instruction", "Given a shopping query, retrieve relevant products."),
            query_max_length=int(config.get("query_max_length", 256)),
            item_max_length=int(config.get("item_max_length", 512)),
            lora_target_modules=config.get("lora_target_modules"),
            inference_only=True,
            revision=encoder['revision'],
        ).to(device)
        set_peft_model_state_dict(model.query_encoder, checkpoint["adapter_state_dict"])
        self.model = model.eval()
        self.lock = threading.Lock()

    def search(self, query: str, top_k: int = 5) -> str:
        return self.search_many([query], [top_k])[0]

    def search_many(self, queries: list[str], top_ks: list[int]) -> list[str]:
        """Several searches with one encoder pass. Each result is the same JSON as `search`."""
        import torch

        with self.lock, torch.no_grad():
            scores = self.model.encode_queries(queries).float() @ self.item_embs.T
            means, maxes = scores.mean(dim=1).tolist(), scores.max(dim=1).values.tolist()
            tops = [row.topk(min(k, len(self.all_items))) for row, k in zip(scores, top_ks)]
            tops = [(top.indices.tolist(), top.values.tolist()) for top in tops]
        outputs = []
        for (indices, raws), mean_score, max_score in zip(tops, means, maxes):
            denominator = max_score - mean_score + 1e-8
            rows = [
                {
                    "item_id": self.all_items[index],
                    "score": round(max(0.0, (raw - mean_score) / denominator), 4),
                    "description": self.product_reps[self.all_items[index]][:300],
                }
                for index, raw in zip(indices, raws)
            ]
            outputs.append(json.dumps(rows))
        return outputs
