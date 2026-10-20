"""RecoAtlas: a recommendation environment for LLM agents."""

from .agent import Agent
from .env import Env, Episode, Settings


def load(dataset: str = "hm-clothing", **settings) -> Env:
    """Load tools and requests; optional `splits` restricts downloads, other keywords are `Settings` fields."""
    return Env(dataset, **settings)


__all__ = ["Agent", "Env", "Episode", "Settings", "load"]
