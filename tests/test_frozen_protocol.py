import hashlib
import json
from pathlib import Path
import pytest
from agents.bundle import build_optimized_agent_class
from agents.diversity import extend_agent
from test_release import Retrieval, Comp, Diversity, Backend

ROOT = Path(__file__).resolve().parents[1]
FROZEN = json.loads((ROOT / 'prompts/frozen_checksums.json').read_text())


def test_frozen_agent_source():
    for relative, expected in FROZEN['files'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative


@pytest.mark.parametrize('k', range(3, 8))
def test_complete_system_prompt(k):
    agent = extend_agent(build_optimized_agent_class(), Diversity())(
        Retrieval(), Comp(), None, backend=Backend(), top_k=k,
        max_tool_rounds=6, use_harness_actions=False, planning=False,
        optimized_strategy=(ROOT / 'prompts/strategy.txt').read_text().strip(),
        min_distinct_searches=2, min_comp_calls=1)
    assert hashlib.sha256(agent._system_prompt().encode()).hexdigest() == FROZEN['system_prompt_sha256'][str(k)]


@pytest.mark.parametrize('k', range(3, 8))
def test_grammar_compiles(k):
    pytest.importorskip('xgrammar')
    from bundle_output_contract import compiled_contract
    for final_only in (False, True):
        grammar, digest = compiled_contract(k, final_only)
        assert grammar
        assert hashlib.sha256(grammar.encode()).hexdigest() == digest
