"""RecoAtlas: a recommendation environment for LLM agents."""

from .env import Env, Episode, Settings


def load(dataset: str = "hm-clothing", **settings) -> Env:
    """Download (once) and load a dataset's tools and requests. Keyword arguments are `Settings` fields."""
    return Env(dataset, **settings)


__all__ = ["Env", "Episode", "Settings", "load"]
