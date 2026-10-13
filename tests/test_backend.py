import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from conftest import fake_client

from recoatlas.backend import paper_backend
from recoatlas.model_profiles import resolve_profile
from recoatlas.output_contract import validate_contract


@pytest.mark.parametrize("thinking,limit", [("off", 1000), ("on", 16384)])
def test_paper_decoding_and_thinking_switch(thinking, limit):
    client, calls = fake_client(["{}"])
    backend = paper_backend(client, thinking=thinking, grammar=False)
    backend.complete(model="Qwen/Qwen3.8-27B-FP8", messages=[], temperature=0, response_format={"type": "json_object"})
    call = calls[0]
    # Pin the complete original Qwen request, not just the newly generalized switch.
    body = {"chat_template_kwargs": {"enable_thinking": thinking == "on", "preserve_thinking": False}}
    if thinking == "on":
        body["thinking_token_budget"] = 4096
    assert call == {
        "model": "Qwen/Qwen3.8-27B-FP8",
        "messages": [],
        "temperature": 1,
        "max_completion_tokens": limit,
        "timeout": 120,
        "response_format": {"type": "json_object"},
        "extra_body": body,
    }


MODELS = [
    ("Qwen/Qwen3.8-27B-FP8", "qwen", "qwen3", {"preserve_thinking": False}),
    ("google/gemma-4-31B-it", "gemma4", "gemma4", {}),
    ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "nemotron", "nemotron_v3", {}),
    ("zai-org/GLM-4.7-Flash", "glm4_7", "glm45", {"clear_thinking": True}),
]


@pytest.mark.parametrize("model,profile,parser,kwargs", MODELS)
@pytest.mark.parametrize("thinking", ["on", "off"])
@pytest.mark.parametrize("grammar", [True, False])
def test_controls_and_output_contract_travel_together(model, profile, parser, kwargs, thinking, grammar):
    response = {
        "thought": "Search",
        "step_goal": "Find a product",
        "action": "search_products",
        "action_input": {"query": "guitar", "top_k": 20},
    }
    client, calls = fake_client([response])
    backend = paper_backend(client, thinking=thinking, grammar=grammar)
    backend.set_bundle_size(3)
    messages = [{"role": "system", "content": "Frozen agent prompt"}, {"role": "user", "content": "Query"}]
    backend.complete(model=model, messages=messages, response_format={"type": "json_object"})
    call = calls[0]
    assert call["messages"] == messages  # model controls never rewrite the paper prompt
    assert call["extra_body"]["chat_template_kwargs"] == {**kwargs, "enable_thinking": thinking == "on"}
    assert call["extra_body"].get("thinking_token_budget") == (4096 if thinking == "on" else None)
    assert ("structured_outputs" in call["extra_body"]) == grammar
    assert ("response_format" in call) != grammar
    assert backend.configuration(model)["required_server"]["reasoning_parser"] == parser
    assert backend.stats()["model_traces"][0]["model_profile"] == profile


@pytest.mark.parametrize("model,profile,parser,kwargs", MODELS)
def test_explicit_profile_for_served_alias(model, profile, parser, kwargs):
    client, calls = fake_client(["{}"])
    backend = paper_backend(client, thinking="on", grammar=False, model_profile=profile)
    backend.complete(model="my-local-model", messages=[])
    expected = paper_backend(None, thinking="on", grammar=False).configuration(model)
    assert calls[0]["extra_body"] == expected["extra_body"]
    assert calls[0]["model"] == "my-local-model"


@pytest.mark.parametrize("thinking", ["off", "on"])
def test_unknown_local_model_is_rejected_before_request(thinking):
    client, calls = fake_client([])
    backend = paper_backend(client, thinking=thinking, grammar=False)
    with pytest.raises(ValueError, match="--model-profile"):
        backend.complete(model="unrecognized-alias", messages=[])
    assert calls == []


def test_reject_mismatched_profile_and_hosted_overrides():
    with pytest.raises(ValueError, match="not the requested profile"):
        resolve_profile("google/gemma-4-31B-it", "qwen")
    with pytest.raises(ValueError, match="Unknown model profile"):
        resolve_profile("my-model", "typo")
    with pytest.raises(ValueError, match="--hosted"):
        resolve_profile("my-model", "gemma4", local=False)
    with pytest.raises(ValueError, match="hosted reasoning"):
        paper_backend(None, local=False, thinking="on")


def test_hosted_mode_remains_provider_managed():
    client, calls = fake_client(["{}"])
    backend = paper_backend(client, local=False, grammar=False)
    backend.complete(model="some-deployment", messages=[], response_format={"type": "json_object"})
    assert "extra_body" not in calls[0]
    assert backend.configuration("some-deployment")["thinking"] is None


def response_client(content="{}", reasoning=None, finish="stop", field="reasoning"):
    response = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content, **{field: reasoning}))],
        usage=None,
    )
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=response))))


@pytest.mark.parametrize("model,profile,parser,kwargs", MODELS)
@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_separated_reasoning_is_logged_not_passed_to_agent(model, profile, parser, kwargs, field):
    backend = paper_backend(response_client(reasoning="Consider the roles.", field=field), thinking="on", grammar=False)
    assert backend.complete(model=model, messages=[]) == "{}"
    assert backend.stats()["usage"]["responses_with_reasoning"] == 1
    assert backend.stats()["model_traces"][0]["reasoning"] == "Consider the roles."


@pytest.mark.parametrize("model,profile,parser,kwargs", MODELS)
def test_server_ignoring_thinking_off_fails_loudly(model, profile, parser, kwargs):
    backend = paper_backend(response_client(reasoning="Still reasoning"), thinking="off", grammar=False)
    with pytest.raises(ValueError, match="THINKING_MODE_MISMATCH"):
        backend.complete(model=model, messages=[])


@pytest.mark.parametrize("thinking", ["on", "off"])
@pytest.mark.parametrize(
    "content", ["<think>unfinished", '</think>{"answer":1}', "<|channel>thought\ntext<channel|>{}", ""]
)
def test_unparsed_reasoning_and_empty_answers_are_not_silently_accepted(thinking, content):
    backend = paper_backend(response_client(content=content), thinking=thinking, grammar=False)
    with pytest.raises(ValueError, match="THINKING_PARSER_ERROR"):
        backend.complete(model="google/gemma-4-31B-it", messages=[])


def test_truncated_thinking_response_is_rejected_even_with_valid_json():
    backend = paper_backend(response_client(finish="length"), thinking="on", grammar=False)
    with pytest.raises(ValueError, match="THINKING_OUTPUT_LIMIT"):
        backend.complete(model="zai-org/GLM-4.7-Flash", messages=[])


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
