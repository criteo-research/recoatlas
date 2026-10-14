"""One supervised overnight sweep: four open models, two datasets, thinking off/on.

Run on an allocated GPU host. This starts local vLLM servers, not Ray clusters;
it never uploads results, calls hosted models, judges reports or trains policies.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from . import assets
from .backend import paper_backend
from .run_store import code_fingerprint, write_json

CONFIG = json.loads(files("recoatlas").joinpath("configs/open_models.json").read_text())


def assign_models(groups, models):
    """Assign each model once to a disjoint, sufficiently large group of allocated GPUs."""
    parsed = [g.split(",") for g in groups]
    flat = [gpu for group in parsed for gpu in group]
    if not parsed or any(not gpu or gpu.strip() != gpu for gpu in flat) or len(flat) != len(set(flat)):
        raise ValueError("GPU groups must be nonempty and disjoint (e.g. --gpu-groups 0 1 2 3,4)")
    jobs = [[] for _ in groups]
    # Put the largest model allocation first; then distribute the remaining models evenly.
    for key in sorted(models, key=lambda k: -models[k]["tensor_parallel_size"]):
        eligible = [i for i, group in enumerate(parsed) if len(group) >= models[key]["tensor_parallel_size"]]
        if not eligible:
            raise ValueError(f"{key} needs a group of at least {models[key]['tensor_parallel_size']} GPUs")
        index = min(eligible, key=lambda i: (len(jobs[i]), len(parsed[i]), i))
        jobs[index].append(key)
    return jobs


def validate_allocation(groups):
    """Do not escape an allocation's explicit CUDA visibility mask."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        requested = {gpu for group in groups for gpu in group.split(",")}
        if not requested.issubset(visible.split(",")):
            raise ValueError("GPU groups must use the IDs/UUIDs listed in CUDA_VISIBLE_DEVICES, not remapped indices")


def validate_smoke(path, thinking):
    """Check transport/protocol/validity, never threshold on hidden recovery or dev scores."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if {row["requested_count"] for row in rows} != set(range(3, 8)) or len(rows) != 5:
        raise RuntimeError("Development smoke must cover exactly one query per size K=3,...,7")
    reasoning_observed = False
    for row in rows:
        report = row.get("report", {})
        picks = row.get("predicted_item_ids", [])
        seen = {
            item["item_id"]
            for step in report.get("trace", [])
            if step.get("tool_name") in {"search_products", "get_complementary_products"}
            and isinstance(step.get("tool_result"), list)
            for item in step["tool_result"]
            if isinstance(item, dict) and "item_id" in item
        }
        if (
            row.get("status") != "complete"
            or len(picks) != row["requested_count"]
            or len(set(picks)) != len(picks)
            or not set(picks).issubset(seen)
        ):
            raise RuntimeError("Development smoke produced an invalid/incomplete report")
        # The frozen agent can exhaust its six rounds without meeting a prompted tool minimum.
        # Record this model behavior; it is not a transport failure or a reason to resample.
        calls = row.get("api", {}).get("model_traces", [])
        if not calls or any(call.get("thinking") != thinking for call in calls):
            raise RuntimeError("Development smoke lacks matching model-call provenance")
        has_reasoning = any(str(call.get("reasoning") or "").strip() for call in calls)
        if thinking == "off" and has_reasoning:
            raise RuntimeError("Thinking-off returned reasoning")
        reasoning_observed |= has_reasoning
    if thinking == "on" and not reasoning_observed:
        raise RuntimeError("No separated reasoning observed in thinking-on smoke; check server/template configuration")


def server_command(python, spec, server, port):
    profile = paper_backend(None, thinking="on", model_profile=spec["profile"]).configuration(spec["model"])
    required = profile["required_server"]
    return [
        python,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        spec["model"],
        "--revision",
        spec["revision"],
        "--tokenizer-revision",
        spec["revision"],
        "--served-model-name",
        spec["model"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        spec["dtype"],
        "--tensor-parallel-size",
        str(spec["tensor_parallel_size"]),
        "--max-model-len",
        str(server["max_model_len"]),
        "--gpu-memory-utilization",
        str(server["gpu_memory_utilization"]),
        "--max-num-seqs",
        str(server["max_num_seqs"]),
        "--generation-config",
        server["generation_config"],
        "--seed",
        str(server["seed"]),
        "--reasoning-parser",
        required["reasoning_parser"],
        "--reasoning-config",
        json.dumps(required["reasoning_config"], separators=(",", ":")),
    ]


def plan(a, config=CONFIG):
    jobs = assign_models(a.gpu_groups, config["models"])
    return {
        "config": config,
        "code_sha256": code_fingerprint(),
        "artifact_sources": assets.SOURCES,
        "concurrency_per_server": a.concurrency,
        "num_shards": a.num_shards,
        "shard_index": a.shard_index,
        "groups": [
            {
                "visible_gpus": group,
                "models_in_order": keys,
                "commands": [
                    server_command(a.server_python, config["models"][k], config["server"], a.port + i) for k in keys
                ],
            }
            for i, (group, keys) in enumerate(zip(a.gpu_groups, jobs, strict=True))
            if keys
        ],
        "arms": [
            {
                "model": key,
                "dataset": dataset,
                "thinking": thinking,
                "queries_in_shard": len(
                    range(a.shard_index, assets.DATASETS[dataset]["splits"]["agent_hidden"], a.num_shards)
                ),
            }
            for key in config["models"]
            for dataset in config["datasets"]
            for thinking in config["thinking"]
        ],
        "smoke_queries_per_model": 5 * len(config["datasets"]) * len(config["thinking"]),
        "hf_upload": False,
    }


def _stop(process):
    """Stop only a process group created by this launcher, never an existing server."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # The leader can exit before one of its GPU workers; reap the entire owned group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


@contextmanager
def managed_process(command, *, environment, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as stream:
        process = subprocess.Popen(
            command, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            yield process
        finally:
            _stop(process)


def check_stop(stop, deadline):
    if stop.is_set() or time.monotonic() >= deadline:
        raise InterruptedError("Sweep stopped or --max-hours reached; checkpoints are retained")


def wait_ready(process, base_url, model, api_key, *, stop, deadline):
    opener = build_opener(ProxyHandler({}))
    while True:
        check_stop(stop, deadline)
        if process.poll() is not None:
            raise RuntimeError("vLLM exited before becoming ready; inspect server.log")
        try:
            request = Request(base_url + "/models", headers={"Authorization": "Bearer " + api_key})
            with opener.open(request, timeout=2) as response:
                data = json.load(response)
            if model not in {row["id"] for row in data["data"]}:
                raise RuntimeError("Endpoint served an unexpected model")
            return
        except (URLError, TimeoutError, ConnectionError):
            stop.wait(1)


def child_environment(gpus, scratch, api_key):
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpus,
            "OPENAI_API_KEY": api_key,
            "VLLM_API_KEY": api_key,
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "2",
        }
    )
    # The pinned artifacts/models are public. Do not pass a Hub write credential to serving workers.
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "OPENAI_BASE_URL"):
        environment.pop(key, None)
    for key, name in {
        "TMPDIR": "tmp",
        "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
        "TRITON_CACHE_DIR": "triton",
        "VLLM_CACHE_ROOT": "vllm",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    }.items():
        path = scratch / name
        path.mkdir(parents=True, exist_ok=True)
        environment[key] = str(path)
    return environment


def run_arm(a, spec, dataset, thinking, phase, base_url, runtime, environment, stop, deadline, config=CONFIG):
    output = runtime.parent / phase / dataset / f"thinking-{thinking}.jsonl"
    command = [
        sys.executable,
        "-m",
        "recoatlas.run_agent",
        "--dataset",
        dataset,
        "--split",
        "agent_dev" if phase == "smoke" else "agent_hidden",
        "--model",
        spec["model"],
        "--model-revision",
        spec["revision"],
        "--model-profile",
        spec["profile"],
        "--base-url",
        base_url,
        "--thinking",
        thinking,
        "--device",
        "cuda",
        "--tools",
        "utility",
        "--concurrency",
        str(a.concurrency),
        "--runtime",
        str(runtime),
        "--output",
        str(output),
    ]
    if phase == "smoke":
        command.append("--smoke")
    else:
        command.extend(["--shard-index", str(a.shard_index), "--num-shards", str(a.num_shards)])
    if a.resume:
        command.append("--resume")
    print(f"{phase}: {spec['model']} | {dataset} | thinking {thinking}", flush=True)
    with managed_process(command, environment=environment, log=output.with_suffix(".log")) as process:
        while process.poll() is None:
            check_stop(stop, deadline)
            stop.wait(1)
        if process.returncode not in {0, 1}:
            raise RuntimeError(f"Agent runner failed (exit {process.returncode}); see {output.with_suffix('.log')}")
        summary_path = output.with_suffix(".summary.json")
        if not summary_path.exists():
            raise RuntimeError(f"Agent runner produced no summary: {summary_path}")
        summary = json.loads(summary_path.read_text())
        # A CLI/runtime failure must not be mistaken for an evaluation that legitimately scored zero.
        if summary.get("pending") != 0:
            raise RuntimeError(f"Incomplete arm: {output}")
        if phase == "smoke" and (process.returncode != 0 or summary["failed"]):
            raise RuntimeError(f"Development smoke failed: {output}; no hidden run for this model")
        if phase == "smoke":
            validate_smoke(output, thinking)
        return summary


def run_model(a, key, group, port, software, stop, deadline, config=CONFIG):
    spec = config["models"][key]
    gpus = ",".join(group.split(",")[: spec["tensor_parallel_size"]])
    directory = a.output / f"shard-{a.shard_index}-of-{a.num_shards}" / key
    directory.mkdir(parents=True, exist_ok=True)
    runtime = directory / "runtime.json"
    command = server_command(a.server_python, spec, config["server"], port)
    metadata = {
        "model": spec,
        "server": config["server"],
        "software": software,
        "launch_command": command,
        "cuda_visible_devices": gpus,
    }
    if runtime.exists() and json.loads(runtime.read_text()) != metadata:
        raise ValueError(f"Server runtime differs from saved run: {runtime}; use a new output")
    write_json(runtime, metadata)
    api_key = secrets.token_hex(32)  # loopback endpoint; key lives in process environments only
    environment = child_environment(gpus, a.scratch / key, api_key)
    environment["HF_HOME"] = str(a.scratch / "huggingface")
    # Keep the venv's bin directory, not the target of its python symlink (ninja lives in the venv).
    environment["PATH"] = str(Path(a.server_python).absolute().parent) + os.pathsep + environment.get("PATH", "")
    # Check for an occupied port. Never connect to or terminate another user's endpoint.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    with managed_process(command, environment=environment, log=directory / "server.log") as server:
        base_url = f"http://127.0.0.1:{port}/v1"
        wait_ready(
            server,
            base_url,
            spec["model"],
            api_key,
            stop=stop,
            deadline=min(deadline, time.monotonic() + a.startup_minutes * 60),
        )
        summaries = []
        # Every local replica passes both modes and both domains before evaluating hidden queries.
        for phase in ["smoke"] if a.smoke_only else ["smoke", "hidden"]:
            for dataset in config["datasets"]:
                for thinking in config["thinking"]:
                    check_stop(stop, deadline)
                    if server.poll() is not None:
                        raise RuntimeError("vLLM exited; inspect server.log and resume missing queries")
                    summary = run_arm(
                        a, spec, dataset, thinking, phase, base_url, runtime, environment, stop, deadline, config
                    )
                    if phase == "hidden":
                        summaries.append(summary)
        return {"model": key, "status": "finished", "hidden_arms": summaries}


def server_preflight(python, expected):
    program = (
        "import json,importlib.metadata as m; "
        "print(json.dumps({n:m.version(n) for n in ('vllm','torch','transformers','xgrammar')}))"
    )
    process = subprocess.run([python, "-c", program], capture_output=True, text=True, timeout=30, check=False)
    if process.returncode:
        raise ValueError("Server Python needs vLLM/torch/transformers/xgrammar installed; see ELI5.md")
    software = json.loads(process.stdout)
    if software["vllm"] != expected:
        raise ValueError(
            f"Expected recorded vLLM {expected}, found {software['vllm']}; use the pinned serving environment"
        )
    program = (
        "import json,torch; "
        "print(json.dumps({'cuda':torch.version.cuda, 'gpus':["
        "{'name':torch.cuda.get_device_properties(i).name,"
        "'memory_bytes':torch.cuda.get_device_properties(i).total_memory,"
        "'capability':list(torch.cuda.get_device_capability(i))}"
        " for i in range(torch.cuda.device_count())]}))"
    )
    hardware = subprocess.run([python, "-c", program], capture_output=True, text=True, timeout=30, check=False)
    if hardware.returncode:
        raise ValueError("Cannot initialize CUDA in the serving environment; check the driver and GPU allocation")
    software["hardware"] = json.loads(hardware.stdout)
    if not software["hardware"]["gpus"]:
        raise ValueError("No CUDA GPUs visible; run on an allocated GPU host, not the login host/CPU driver")
    return software


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("runs/open-models-mi-hm"))
    p.add_argument(
        "--scratch", type=Path, help="Large local disk for compilation/temp files; defaults to OUTPUT/scratch"
    )
    p.add_argument("--server-python", default=sys.executable, help="Python in the separate pinned vLLM environment")
    p.add_argument("--gpu-groups", nargs="+", required=True, help="Disjoint GPU IDs/UUIDs on THIS host; e.g. 0 1 2 3,4")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--models", nargs="+", choices=CONFIG["models"], help="Run only these pinned models on this host")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--num-shards", type=int, default=1, help="Use the same count on every allocated host")
    p.add_argument("--shard-index", type=int, default=0, help="This host's query shard (0 based)")
    p.add_argument("--startup-minutes", type=float, default=60, help="Maximum model download/load/compile wait")
    p.add_argument(
        "--max-hours", type=float, default=14, help="Stop this invocation and release its local model processes"
    )
    p.add_argument("--smoke-only", action="store_true", help="Only development checks; never evaluate hidden queries")
    p.add_argument("--resume", action="store_true", help="Resume missing queries; never rerun saved failures")
    p.add_argument(
        "--dry-run", action="store_true", help="Print all arms and server commands without files/downloads/GPUs"
    )
    a = p.parse_args(argv)
    if min(a.concurrency, a.num_shards, a.max_hours, a.startup_minutes) <= 0:
        p.error("Concurrency, shard count and time limits must be positive")
    if not 0 <= a.shard_index < a.num_shards:
        p.error("Require 0 <= --shard-index < --num-shards")
    if not 1024 <= a.port <= 65535 - len(a.gpu_groups):
        p.error("Port range must be between 1024 and 65535")
    a.output = a.output.resolve()
    a.scratch = (a.scratch or a.output / "scratch").resolve()
    config = {**CONFIG, "models": {k: CONFIG["models"][k] for k in (a.models or CONFIG["models"])}}
    try:
        planned = plan(a, config)
        if a.dry_run:
            print(json.dumps(planned, indent=2))
            return 0
        validate_allocation(a.gpu_groups)
        software = server_preflight(a.server_python, config["server"]["vllm_version"])
        from .output_contract import compiled_contract

        for k in range(3, 8):
            compiled_contract(k)
            compiled_contract(k, True)
        root = a.output / f"shard-{a.shard_index}-of-{a.num_shards}"
        if root.exists() and not a.resume:
            raise ValueError(f"Output already exists: {root}; use --resume or a new path")
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = root / "manifest.json"
            if manifest.exists() and json.loads(manifest.read_text()) != planned:
                raise ValueError("Sweep manifest differs; use a new output directory")
            write_json(manifest, planned)
            stop = threading.Event()
            deadline = time.monotonic() + a.max_hours * 3600
            previous = {s: signal.signal(s, lambda *_: stop.set()) for s in (signal.SIGINT, signal.SIGTERM)}

            def run_group(index):
                results = []
                for key in assign_models(a.gpu_groups, config["models"])[index]:
                    try:
                        check_stop(stop, deadline)
                        result = run_model(
                            a, key, a.gpu_groups[index], a.port + index, software, stop, deadline, config
                        )
                    except Exception as error:  # noqa: BLE001 -- one failed model must not discard other arms
                        # No provider response bodies or credentials in this status file.
                        result = {
                            "model": key,
                            "status": "stopped" if isinstance(error, InterruptedError) else "error",
                            "error_type": type(error).__name__,
                        }
                        print(f"{key}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                    write_json(root / key / "status.json", result)
                    results.append(result)
                return results

            try:
                with ThreadPoolExecutor(max_workers=len(a.gpu_groups)) as pool:
                    results = [r for group in pool.map(run_group, range(len(a.gpu_groups))) for r in group]
            finally:
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
            summary = {
                "smoke_only": a.smoke_only,
                "num_shards": a.num_shards,
                "shard_index": a.shard_index,
                "models": results,
                "hf_upload": False,
            }
            write_json(root / "summary.json", summary)
            print(f"Summary: {root / 'summary.json'}", flush=True)
            return (
                0
                if all(
                    r["status"] == "finished" and all(not arm["failed"] for arm in r["hidden_arms"]) for r in results
                )
                else 1
            )
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        p.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
