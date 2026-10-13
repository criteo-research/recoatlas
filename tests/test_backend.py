import json

import pytest
from conftest import fake_client

from recoatlas.backend import paper_backend
from recoatlas.output_contract import validate_contract


@pytest.mark.parametrize("thinking,limit", [("off", 1000), ("on", 16384)])
def test_paper_decoding_and_thinking_switch(thinking, limit):
    client, calls = fake_client(["{}"])
    backend = paper_backend(client, thinking=thinking, grammar=False)
    backend.complete(model="Qwen/test", messages=[], temperature=0, response_format={"type": "json_object"})
    call = calls[0]
    assert call["temperature"] == 1 and call["max_completion_tokens"] == limit
    assert call["extra_body"]["chat_template_kwargs"]["enable_thinking"] == (thinking == "on")
    assert ("thinking_token_budget" in call["extra_body"]) == (thinking == "on")


def test_grammar_contract_includes_near_duplicate_call():
    data = {
        "thought": "Check",
        "step_goal": "Remove redundancy",
        "action": "get_substitute_products",
        "action_input": {"item_ids": ["a", "b"], "similarity_threshold": 0.95},
    }
    assert validate_contract(json.dumps(data), 3) == data
    data["action_input"]["similarity_threshold"] = 0.9
    with pytest.raises(ValueError):
        validate_contract(json.dumps(data), 3)


def test_final_contract_requires_exact_k_and_does_not_accept_tools():
    data = {"report_explanation": "Set", "results": [{"product_id": "a", "reasoning": "Fits"}]}
    with pytest.raises(ValueError):
        validate_contract(json.dumps(data), 3, final_only=True)
