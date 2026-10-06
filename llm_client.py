"""Standard OpenAI-compatible transport; no institution-specific client."""
import os


def resolve_model(model):
    return model


def get_openai_user():
    return 'anonymous'


def get_sync_client(api_key=None, base_url=None, model=None):
    from openai import OpenAI
    url = base_url or os.getenv('OPENAI_BASE_URL')
    key = api_key or os.getenv('OPENAI_API_KEY')
    if not key and not url:
        raise ValueError('Set OPENAI_API_KEY or an OpenAI-compatible OPENAI_BASE_URL.')
    return OpenAI(api_key=key or 'EMPTY', base_url=url, max_retries=2, timeout=120)
