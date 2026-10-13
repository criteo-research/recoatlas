import copy
import json
from types import SimpleNamespace

from recoatlas.env import Env, Settings


class FakeEnv:
    """Synthetic catalog and deterministic tools; no downloads or model calls."""

    def __init__(self):
        self.settings = Settings()
        self.catalog = {f"i{i}": f"Catalog product {i}: role {i % 3}." for i in range(100)}
        self.product_reps = self.catalog  # original tool interface, for recorded parity probes
        self.item_to_subcategory = {i: str(n % 3) for n, i in enumerate(self.catalog)}
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append(["search_products", {"query": query, "top_k": top_k}])
        return json.dumps(
            [
                {"item_id": f"i{i}", "score": 1 - i / 100, "description": self.catalog[f"i{i}"]}
                for i in range(min(top_k, 100))
            ]
        )

    search_products = search

    def get_complementary(self, item_ids, top_k=5):
        self.calls.append(["get_complementary_products", {"item_ids": item_ids, "top_k": top_k}])
        return json.dumps(
            [
                {"item_id": f"i{i}", "score": 1 - i / 100, "description": self.catalog[f"i{i}"]}
                for i in range(10, min(10 + top_k, 100))
            ]
        )

    get_complementary_products = get_complementary

    def get_substitutes(self, item_ids, similarity_threshold=0.95):
        self.calls.append(
            ["get_substitute_products", {"item_ids": item_ids, "similarity_threshold": similarity_threshold}]
        )
        return json.dumps(
            {
                "kept": [{"item_id": i} for i in item_ids[:-1]],
                "removed": [{"item_id": item_ids[-1], "duplicate_of": item_ids[0], "similarity": 0.99}],
            }
        )

    get_substitute_products = get_substitutes

    call = Env.call
    returned_ids = staticmethod(Env.returned_ids)
    score = Env.score


class ScriptedBackend:
    def __init__(self, responses):
        self.responses = copy.deepcopy(responses)
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(copy.deepcopy({k: v for k, v in kwargs.items() if k != "timeout"}))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)


def fake_client(responses):
    calls = []
    remaining = iter(responses)

    def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        response = next(remaining)
        if isinstance(response, Exception):
            raise response
        content = json.dumps(response) if isinstance(response, dict) else response
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))], usage=None
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls
