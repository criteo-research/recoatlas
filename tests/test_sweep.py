import json
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from recoatlas import sweep


def arguments(tmp_path, **overrides):
    return SimpleNamespace(
        **{
            "output": tmp_path / "run",
            "scratch": tmp_path / "scratch",
            "server_python": "/python",
            "gpu_groups": ["0", "1", "2", "3,4"],
            "concurrency": 8,
            "num_shards": 1,
            "shard_index": 0,
            "port": 8100,
            "startup_minutes": 30,
            "resume": False,
            "smoke_only": False,
            **overrides,
        }
    )


def test_full_dry_run_has_all_16_arms_without_processes_downloads_or_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sweep, "server_preflight", Mock(side_effect=AssertionError("No processes")))
    monkeypatch.setattr(sweep.assets, "load_split", Mock(side_effect=AssertionError("No downloads")))
    output = tmp_path / "unused"
    assert sweep.main(["--gpu-groups", "0", "1", "2", "3,4", "--output", str(output), "--dry-run"]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert len(planned["arms"]) == 16
    assert sum(arm["queries_in_shard"] for arm in planned["arms"]) == 31360
    assert len({(r["model"], r["dataset"], r["thinking"]) for r in planned["arms"]}) == 16
    assert {r["thinking"] for r in planned["arms"]} == {"on", "off"}
    assert not output.exists()


def test_gpu_assignment_never_overlaps_or_puts_tp2_on_one_gpu():
    groups = ["0", "1", "2", "3,4"]
    assigned = sweep.assign_models(groups, sweep.CONFIG["models"])
    assert sorted(key for keys in assigned for key in keys) == sorted(sweep.CONFIG["models"])
    assert assigned[-1] == ["glm4_7"]
    for bad in (["0", "0,1"], ["0", "1"], [""], ["0,0"], ["0", " 1"]):
        with pytest.raises(ValueError):
            sweep.assign_models(bad, sweep.CONFIG["models"])


def test_selected_models_use_disjoint_complete_query_shards(tmp_path, capsys):
    totals = {}
    for model, shards, gpus in [("qwen", 4, "0"), ("glm4_7", 3, "0,1")]:
        for index in range(shards):
            assert (
                sweep.main(
                    [
                        "--models",
                        model,
                        "--gpu-groups",
                        gpus,
                        "--dry-run",
                        "--num-shards",
                        str(shards),
                        "--shard-index",
                        str(index),
                    ]
                )
                == 0
            )
            planned = json.loads(capsys.readouterr().out)
            assert set(planned["config"]["models"]) == {model}
            for arm in planned["arms"]:
                key = (arm["model"], arm["dataset"], arm["thinking"])
                totals[key] = totals.get(key, 0) + arm["queries_in_shard"]
    assert len(totals) == 8
    assert set(totals.values()) == {1960}


@pytest.mark.parametrize("key", sweep.CONFIG["models"])
def test_server_and_client_use_matching_pins_reasoning_parser_and_modes(key):
    spec = sweep.CONFIG["models"][key]
    command = sweep.server_command("python", spec, sweep.CONFIG["server"], 8100)
    assert command[command.index("--revision") + 1] == spec["revision"]
    assert command[command.index("--tokenizer-revision") + 1] == spec["revision"]
    for mode in ("off", "on"):
        controls = sweep.paper_backend(None, thinking=mode, model_profile=spec["profile"]).configuration(spec["model"])
        assert controls["extra_body"]["chat_template_kwargs"]["enable_thinking"] == (mode == "on")
        assert command[command.index("--reasoning-parser") + 1] == controls["required_server"]["reasoning_parser"]
        assert (
            json.loads(command[command.index("--reasoning-config") + 1])
            == controls["required_server"]["reasoning_config"]
        )


def mock_serving(monkeypatch):
    server = SimpleNamespace(poll=lambda: None)
    ended = []

    @contextmanager
    def managed(*args, **kwargs):
        try:
            yield server
        finally:
            ended.append(True)

    socket = Mock()
    socket.__enter__ = Mock(return_value=socket)
    socket.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(sweep.socket, "socket", lambda: socket)
    monkeypatch.setattr(sweep, "managed_process", managed)
    monkeypatch.setattr(sweep, "wait_ready", lambda *a, **kw: None)
    return ended


@pytest.mark.parametrize("fail_smoke,smoke_only", [(False, False), (True, False), (False, True)])
def test_all_dev_arms_precede_hidden_and_smoke_failure_blocks_hidden(tmp_path, monkeypatch, fail_smoke, smoke_only):
    ended = mock_serving(monkeypatch)
    phases = []

    def arm(a, spec, dataset, thinking, phase, *args):
        phases.append((phase, dataset, thinking))
        if fail_smoke:
            raise RuntimeError("Development smoke failed")
        return {"failed": 0}

    monkeypatch.setattr(sweep, "run_arm", arm)
    a = arguments(tmp_path, smoke_only=smoke_only)
    if fail_smoke:
        with pytest.raises(RuntimeError, match="smoke"):
            sweep.run_model(a, "gemma4", "0", 8100, {}, threading.Event(), time.monotonic() + 60)
        assert all(phase == "smoke" for phase, *_ in phases)
    else:
        result = sweep.run_model(a, "gemma4", "0", 8100, {}, threading.Event(), time.monotonic() + 60)
        assert result["status"] == "finished"
        assert [p[0] for p in phases] == ["smoke"] * 4 + ([] if smoke_only else ["hidden"] * 4)
    assert ended == [True]


def test_stopping_or_deadline_prevents_further_work():
    stop = threading.Event()
    with pytest.raises(InterruptedError):
        sweep.check_stop(stop, time.monotonic() - 1)
    stop.set()
    with pytest.raises(InterruptedError):
        sweep.check_stop(stop, time.monotonic() + 60)


def test_private_keys_are_not_forwarded_to_hub_and_not_written(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-write-token")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-hosted-token")
    environment = sweep.child_environment("0", tmp_path, "fake-local-key")
    assert "HF_TOKEN" not in environment
    assert environment["OPENAI_API_KEY"] == "fake-local-key"
    assert environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert not any(p.is_file() for p in tmp_path.rglob("*"))


def test_cuda_allocation_cannot_be_escaped(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    sweep.validate_allocation(["4,5"])
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        sweep.validate_allocation(["0,1"])


def smoke_records(mode, *, reasoning=True):
    return [
        {
            "requested_count": k,
            "status": "complete",
            "predicted_item_ids": [str(i) for i in range(k)],
            "report": {
                "trace": [{"tool_name": "search_products", "tool_result": [{"item_id": str(i)} for i in range(k)]}]
            },
            "protocol": {"minimum_met": True},
            "api": {
                "model_traces": [{"thinking": mode, "reasoning": "Reasoning" if reasoning and mode == "on" else None}]
            },
        }
        for k in range(3, 8)
    ]


@pytest.mark.parametrize("mode", ["on", "off"])
def test_smoke_does_not_gate_on_recovery_but_rejects_missing_items_and_bad_modes(tmp_path, mode):
    path = tmp_path / "smoke.jsonl"
    rows = smoke_records(mode)
    for row in rows:
        row["feedback"] = {"sethit_at_k": 0}
    path.write_text("\n".join(json.dumps(row) for row in rows))
    sweep.validate_smoke(path, mode)  # All-zero SetHit is not a failed smoke.
    rows[0]["protocol"] = {"minimum_met": False, "missing": ["complementarity calls: 0/1"]}
    path.write_text("\n".join(json.dumps(row) for row in rows))
    sweep.validate_smoke(path, mode)  # Paper budget exhaustion is an observed policy outcome.
    rows[0]["predicted_item_ids"].pop()
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(RuntimeError, match="invalid/incomplete"):
        sweep.validate_smoke(path, mode)
    rows = smoke_records(mode)
    for row in rows:
        row["api"]["model_traces"][0]["reasoning"] = "Unexpected" if mode == "off" else None
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(RuntimeError, match="reasoning"):
        sweep.validate_smoke(path, mode)


def test_server_start_failure_releases_owned_process(tmp_path, monkeypatch):
    ended = mock_serving(monkeypatch)
    monkeypatch.setattr(sweep, "wait_ready", Mock(side_effect=RuntimeError("vLLM load failed")))
    monkeypatch.setattr(sweep, "run_arm", Mock(side_effect=AssertionError("No evaluations before server readiness")))
    with pytest.raises(RuntimeError, match="load failed"):
        sweep.run_model(arguments(tmp_path), "gemma4", "0", 8100, {}, threading.Event(), time.monotonic() + 60)
    assert ended == [True]


def test_main_prepares_all_modes_and_keeps_other_models_when_one_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(sweep, "server_preflight", lambda *a: {"vllm": "test"})
    import recoatlas.output_contract

    monkeypatch.setattr(recoatlas.output_contract, "compiled_contract", lambda *a: None)
    calls = []

    def model(a, key, *args):
        calls.append(key)
        if key == "qwen":
            raise RuntimeError("Synthetic startup failure")
        return {"model": key, "status": "finished", "hidden_arms": [{"failed": 0}]}

    monkeypatch.setattr(sweep, "run_model", model)
    output = tmp_path / "run"
    argv = ["--output", str(output), "--gpu-groups", "0", "1", "2", "3,4"]
    assert sweep.main(argv) == 1
    assert set(calls) == set(sweep.CONFIG["models"])
    summary = json.loads((output / "shard-0-of-1/summary.json").read_text())
    assert {m["model"]: m["status"] for m in summary["models"]} == {
        "qwen": "error",
        "gemma4": "finished",
        "nemotron": "finished",
        "glm4_7": "finished",
    }
    with pytest.raises(SystemExit):
        sweep.main(argv)
    assert "--resume" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        sweep.main([*argv, "--resume", "--concurrency", "32"])
    assert "manifest differs" in capsys.readouterr().err
