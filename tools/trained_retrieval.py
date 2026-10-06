from __future__ import annotations
import json
import threading
from pathlib import Path

class TrainedRetrievalTool:
    """Inference adapter for the split-trained multi-positive retrieval checkpoint."""

    def __init__(self, run_dir: Path, device: str, checkpoint_path: Path):
        import torch
        import torch.nn.functional as functional
        from peft import set_peft_model_state_dict
        from tools.query_encoder import QueryRetrievalModel
        from artifacts import source_config

        texts_path = run_dir / "data/product_texts.json"
        embeddings_path = run_dir / "embeddings/item_embeddings_qwen.pt"
        self.product_reps = json.loads(texts_path.read_text())
        data = torch.load(embeddings_path, map_location="cpu", weights_only=True)
        self.all_items = list(data["item_ids"])
        if self.all_items != list(self.product_reps):
            raise ValueError("retrieval embeddings are not aligned with product_texts")
        self.item_to_idx = {item_id: index for index, item_id in enumerate(self.all_items)}
        self.item_embs = functional.normalize(data["item_embs"].float(), dim=-1).to(device)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = checkpoint.get("model_config", {})
        encoder = source_config()['query_encoder']
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
        import torch

        with self.lock, torch.no_grad():
            query_embedding = self.model.encode_queries([query]).float()
            scores = (query_embedding @ self.item_embs.T).squeeze(0)
            indices = scores.topk(min(top_k, len(self.all_items))).indices.tolist()
            mean_score = scores.mean().item()
            max_score = scores.max().item()
            denominator = max_score - mean_score + 1e-8
        rows = []
        for index in indices:
            item_id = self.all_items[index]
            raw = scores[index].item()
            rows.append({
                "item_id": item_id,
                "score": round(max(0.0, (raw - mean_score) / denominator), 4),
                "description": self.product_reps[item_id][:300],
            })
        return json.dumps(rows)


class LockedComplementarityTool:
    def __init__(self, tool):
        self.tool = tool
        self.lock = threading.Lock()

    def get_complementary(self, item_ids: list[str], top_k: int = 5) -> str:
        with self.lock:
            return self.tool.get_complementary(item_ids, top_k)

    def __getattr__(self, name: str):
        return getattr(self.tool, name)
