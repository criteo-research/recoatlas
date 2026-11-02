import inspect

from tools_builder.recsys_model import (
    BACKBONE_CONFIGS,
    QueryRetrievalModel,
    train_query_retrieval,
)


def test_qwen_defaults_match_tuned_retrieval_winner():
    model_defaults = inspect.signature(QueryRetrievalModel).parameters
    train_defaults = inspect.signature(train_query_retrieval).parameters

    assert BACKBONE_CONFIGS["qwen"]["lora_targets"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert model_defaults["lora_r"].default == 32
    assert model_defaults["lora_alpha"].default == 32
    assert train_defaults["batch_size"].default == 32
    assert train_defaults["grad_accum_steps"].default == 2
    assert train_defaults["lr"].default == 7.406576281631978e-5
    assert train_defaults["temperature"].default == 0.020090056606207288
    assert train_defaults["lr_schedule"].default == "constant"
    assert train_defaults["warmup_ratio"].default == 0.0
