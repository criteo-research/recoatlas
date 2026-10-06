import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import artifacts
import generate_traces
import judge_traces
from test_release import Retrieval, Comp, Diversity, Backend


def test_pins_and_available_data():
    config = artifacts.source_config()
    for source in config['sources'].values():
        assert len(source['revision']) == 40
        assert source['repo_type'] in ('model', 'dataset')
    assert not config['categories']['amazon-video-games']['query_counts']
    assert config['categories']['amazon-musical-instruments']['query_counts']['agent_hidden'] == 1960


@pytest.mark.parametrize('hash_type', ['sha256', 'git_blob_sha1'])
def test_artifact_hashes_and_tampering(tmp_path, hash_type):
    path = tmp_path / 'asset.json'
    path.write_bytes(b'abc')
    digest = hashlib.sha256(b'abc').hexdigest() if hash_type == 'sha256' else hashlib.sha1(b'blob 3\0abc').hexdigest()
    spec = {'size': 3, hash_type: digest}
    artifacts.verify_file(path, spec)
    path.write_bytes(b'abd')
    with pytest.raises(ValueError, match='checksum'):
        artifacts.verify_file(path, spec)
    path.write_bytes(b'abcd')
    with pytest.raises(ValueError, match='size'):
        artifacts.verify_file(path, spec)


def test_unavailable_partition_fails_before_download(tmp_path):
    with pytest.raises(ValueError, match='No pinned'):
        artifacts.resolve('amazon-video-games', tmp_path, required=['queries/agent_hidden.jsonl'])


def test_query_schema_preserves_zero_padding(tmp_path):
    (tmp_path / 'queries').mkdir()
    row = {'candidate_id': 'sample', 'partition': 'agent_hidden', 'query': 'A shopping request',
           'positive_items': ['0000000001', '0000000002', '0000000003']}
    file = tmp_path / 'queries/agent_hidden.jsonl'
    file.write_text(json.dumps(row) + '\n')
    assert artifacts.load_dataset(tmp_path, 'agent_hidden')[0]['positive_items'] == row['positive_items']
    file.write_text((json.dumps(row) + '\n') * 2)
    with pytest.raises(ValueError):
        artifacts.load_dataset(tmp_path, 'agent_hidden')


def fake_artifacts(tmp_path, monkeypatch):
    root = tmp_path / 'assets'
    (root / 'data').mkdir(parents=True)
    (root / 'queries').mkdir()
    catalog = {**Retrieval.product_reps, 'hidden_reference': 'An unobserved valid product'}
    (root / 'data/product_texts.json').write_text(json.dumps(catalog))
    (root / 'data/subcategories.json').write_text(json.dumps({'item_to_subcategory': {x:x for x in catalog}}))
    task = {'candidate_id':'sample','query':'I want to record a guitar performance.',
            'partition':'agent_hidden','positive_items':['a','b','hidden_reference']}
    (root / 'queries/agent_hidden.jsonl').write_text(json.dumps(task) + '\n')
    resolve = lambda *args, **kwargs: (root, {})
    monkeypatch.setattr(generate_traces, 'resolve', resolve)
    monkeypatch.setattr(judge_traces, 'resolve', resolve)
    import tools.load
    monkeypatch.setattr(tools.load, 'build_tools', lambda *args: (Retrieval(), Comp(), Diversity()))
    return task


def test_cli_agent_to_feedback_to_judge(tmp_path, monkeypatch):
    fake_artifacts(tmp_path, monkeypatch)
    policy = Backend()
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        payload = policy.complete(**kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=payload), finish_reason='stop')], usage=None)

    import llm_client
    monkeypatch.setattr(llm_client, 'get_sync_client', lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete))))
    traces = tmp_path / 'traces.jsonl'
    assert generate_traces.main(['--category','amazon-musical-instruments','--no-grammar','--output',str(traces)]) == 0
    row = json.loads(traces.read_text())
    assert row['feedback']['sethit_at_k'] == pytest.approx(2/3)
    assert 'hidden_reference' not in json.dumps(calls)
    assert 'sethit' not in json.dumps(calls).lower()
    assert len(row['report']['trace']) <= 6
    assert row['protocol']['diversity_check_performed']
    before = len(calls)
    assert judge_traces.main(['--category','amazon-musical-instruments','--traces',str(traces),'--dry-run']) == 0
    assert len(calls) == before
    judge_calls = []

    def judge(**kwargs):
        judge_calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(
            {'score':3,'reason':'The products fit the expressed need.','evidence_items':['P1']})), finish_reason='stop')])

    monkeypatch.setattr(llm_client, 'get_sync_client', lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=judge))))
    output = tmp_path / 'judgments.jsonl'
    assert judge_traces.main(['--category','amazon-musical-instruments','--traces',str(traces),
                             '--model','test-only','--output',str(output)]) == 0
    assert json.loads(output.read_text())['quality'] == .75
    request = json.dumps(judge_calls)
    assert all(x not in request for x in ['hidden_reference','sethit','report_explanation','utility'])
    with pytest.raises(SystemExit):
        generate_traces.main(['--no-grammar','--output',str(traces)])


def test_baseline_cli(tmp_path, monkeypatch):
    fake_artifacts(tmp_path, monkeypatch)
    output = tmp_path / 'baseline.jsonl'
    assert generate_traces.main(['--agent','non-llm','--output',str(output)]) == 0
    assert len(json.loads(output.read_text())['predicted_item_ids']) == 3


def test_invalid_judge_trace_makes_no_requests(tmp_path, monkeypatch):
    fake_artifacts(tmp_path, monkeypatch)
    import llm_client
    def forbidden(**kwargs):
        pytest.fail('Must reject invalid input before constructing API client')
    monkeypatch.setattr(llm_client, 'get_sync_client', forbidden)
    traces = tmp_path / 'bad.jsonl'
    traces.write_text(json.dumps({'candidate_id':'unknown','status':'complete','predicted_item_ids':['a','b','c']})+'\n')
    with pytest.raises(SystemExit):
        judge_traces.main(['--category','amazon-musical-instruments','--traces',str(traces),'--dry-run'])
