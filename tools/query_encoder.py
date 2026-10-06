"""Query encoder used by the frozen retrieval tool (no training pipeline)."""
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QUERY_INSTRUCTION = "Given a shopping query, retrieve relevant products."
QWEN_EOD_TOKEN = "<|endoftext|>"

BACKBONE_CONFIGS = {
    "qwen": {
        "model_name":   "Qwen/Qwen3-Embedding-0.6B",
        "pooling":      "last_token",
        "query_prefix": "Instruct: {instruction}\nQuery: ",
        "item_prefix":  "",
        "eod_token":    "<|endoftext|>",
        "pad_side":     "left",
        "lora_targets": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    },
}


def format_query_for_instruction(query: str, instruction: str) -> str:
    return f"Instruct: {instruction}\nQuery: {query}"


def append_qwen_eod(text: str) -> str:
    text = (text or "").rstrip()
    return text if text.endswith(QWEN_EOD_TOKEN) else f"{text}{QWEN_EOD_TOKEN}"


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool final non-padding token for both left- and right-padded batches."""
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return last_hidden_state[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    return last_hidden_state[
        torch.arange(last_hidden_state.size(0), device=last_hidden_state.device), seq_lens
    ]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class QueryRetrievalModel(nn.Module):
    """Dual-encoder model with LoRA fine-tuning on the query tower.

    The item tower is always frozen.  All embeddings are L2-normalized.
    """

    def __init__(
        self,
        backbone: str = "qwen",
        model_name: str = None,
        lora_r: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        query_max_length: int = 256,
        item_max_length: int = 512,
        lora_target_modules: Optional[List[str]] = None,
        inference_only: bool = True,
        revision: str | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone_cfg = BACKBONE_CONFIGS[backbone]
        model_name = model_name or self.backbone_cfg["model_name"]
        self.model_name = model_name
        self.query_instruction = query_instruction
        self.query_max_length = query_max_length
        self.item_max_length = item_max_length
        self.pooling = self.backbone_cfg["pooling"]
        pad_side = self.backbone_cfg["pad_side"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=pad_side, revision=revision)

        query_base = AutoModel.from_pretrained(model_name, revision=revision)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules or self.backbone_cfg["lora_targets"],
            bias="none",
        )
        self.query_encoder = get_peft_model(query_base, lora_config)

        if inference_only:
            self.base_encoder = None
        else:
            self.base_encoder = AutoModel.from_pretrained(model_name, revision=revision).eval()
            for param in self.base_encoder.parameters():
                param.requires_grad = False

        models = [self.query_encoder] + ([self.base_encoder] if self.base_encoder else [])
        self._ensure_padding_token(self.tokenizer, *models)

    @staticmethod
    def _ensure_padding_token(tokenizer, *models):
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is not None:
            for model in models:
                model.config.pad_token_id = tokenizer.pad_token_id

    def train(self, mode=True):
        super().train(mode)
        if mode and self.base_encoder is not None:
            self.base_encoder.eval()
        return self

    def _pool(self, output, attention_mask):
        if self.pooling == "last_token":
            pooled = last_token_pool(output.last_hidden_state, attention_mask)
        else:
            pooled = mean_pool(output.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=-1)

    def _format_query(self, text: str) -> str:
        prefix = self.backbone_cfg["query_prefix"]
        if "{instruction}" in prefix:
            prefix = prefix.format(instruction=self.query_instruction)
        eod = self.backbone_cfg["eod_token"] or ""
        return f"{prefix}{text}{eod}"

    def _format_item(self, text: str) -> str:
        prefix = self.backbone_cfg["item_prefix"]
        eod = self.backbone_cfg["eod_token"] or ""
        return f"{prefix}{(text or '').rstrip()}{eod}"

    def _encode(self, texts: List[str], encoder, format_fn, max_length: int) -> torch.Tensor:
        device = next(encoder.parameters()).device
        inputs = self.tokenizer(
            [format_fn(t) for t in texts],
            padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        return self._pool(encoder(**inputs), inputs['attention_mask'])

    def encode_queries(self, texts: List[str]) -> torch.Tensor:
        return self._encode(texts, self.query_encoder, self._format_query, self.query_max_length)

    def encode_items(self, texts: List[str]) -> torch.Tensor:
        return self._encode(texts, self.base_encoder, self._format_item, self.item_max_length)

    def encode_base_queries(self, texts: List[str]) -> torch.Tensor:
        """Encode queries with frozen base model for baseline comparison."""
        return self._encode(texts, self.base_encoder, self._format_query, self.query_max_length)
