"""Serving controls only: these profiles never modify the frozen agent messages."""

from __future__ import annotations

import copy
import json
from fnmatch import fnmatchcase
from importlib.resources import files

PROFILES = json.loads(files("recoatlas").joinpath("configs/model_profiles.json").read_text())


def resolve_profile(model: str, name: str = "auto", *, local: bool = True) -> dict | None:
    if name != "auto" and name not in PROFILES:
        raise ValueError(f"Unknown model profile {name!r}; choose from {', '.join(PROFILES)}")
    if not local:
        if name != "auto":
            raise ValueError("--model-profile controls local vLLM serving; do not combine it with --hosted")
        return None
    basename = model.rsplit("/", 1)[-1].lower()
    matches = [key for key, spec in PROFILES.items() if any(fnmatchcase(basename, p) for p in spec["model_patterns"])]
    if name == "auto":
        if len(matches) != 1:
            raise ValueError(
                f"No unambiguous thinking profile for {model!r}. "
                "For a served alias, set --model-profile to its actual supported model family. "
                "Other families need a verified profile; --thinking off is not a generic no-reasoning guarantee."
            )
        name = matches[0]
    elif matches and name not in matches:
        raise ValueError(f"Model {model!r} matches {matches[0]!r}, not the requested profile {name!r}")
    return {"name": name, **copy.deepcopy(PROFILES[name])}


def request_controls(profile: dict | None, thinking: str, budget: int | None) -> dict:
    if profile is None:
        return {}
    body = {"chat_template_kwargs": {**profile["chat_template_kwargs"], "enable_thinking": thinking == "on"}}
    if thinking == "on" and budget is not None:
        body["thinking_token_budget"] = budget
    return body
