__all__ = [
    "QueryRetrievalModel",
    "train_query_retrieval",
]


def __getattr__(name):
    """Load supported public APIs without eagerly importing ML dependencies."""
    if name in __all__:
        from . import recsys_model

        return getattr(recsys_model, name)
    raise AttributeError(name)
