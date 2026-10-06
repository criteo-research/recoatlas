"""Read pinned benchmark assets; reference sets stay outside the agent state."""
import hashlib
import json
import os
from pathlib import Path

SOURCES = Path(__file__).parent / 'artifact_sources.json'
PARTITIONS = ('shared_train', 'agent_dev', 'agent_hidden')


def source_config():
    return json.loads(SOURCES.read_text())


CATEGORIES = tuple(source_config()['categories'])


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_file(path, spec):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size != spec['size']:
        raise ValueError(f'Missing, symlinked, or wrong-size artifact: {path.name}')
    if 'sha256' in spec:
        actual, expected = sha256(path), spec['sha256']
    else:
        digest = hashlib.sha1(f"blob {spec['size']}\0".encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        actual, expected = digest.hexdigest(), spec['git_blob_sha1']
    if actual != expected:
        raise ValueError(f'Artifact checksum mismatch: {path.name}')


def resolve(category, root=None, *, required=None):
    config = source_config()
    if category not in config['categories']:
        raise ValueError('Unsupported category')
    metadata = config['categories'][category]
    category_root = Path(root or os.getenv('RECO_ATLAS_ARTIFACTS', 'artifacts')).resolve() / category
    for rel in (required if required is not None else metadata['files']):
        if rel not in metadata['files']:
            raise ValueError(f'No pinned {rel} for {category}. ' + metadata.get('unavailable_reason', 'This partition is not published.'))
        try:
            verify_file(category_root / rel, metadata['files'][rel])
        except ValueError as exc:
            raise ValueError(f'{exc}. Run prepare_artifacts.py --category {category} first.') from exc
    return category_root, metadata


def load_dataset(category_root, partition):
    if partition not in PARTITIONS:
        raise ValueError('Unsupported partition')
    path = Path(category_root) / 'queries' / f'{partition}.jsonl'
    if not path.is_file():
        raise ValueError(f'No released {partition} query/reference set for {Path(category_root).name}')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen = set()
    for row in rows:
        ids = row['positive_items']
        if (row['partition'] != partition or not isinstance(row['candidate_id'], str)
                or row['candidate_id'] in seen or not isinstance(row.get('query'), str)
                or not row['query'].strip() or not isinstance(ids, list)
                or not 3 <= len(ids) <= 7 or any(not isinstance(i, str) for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('Invalid dataset row')
        seen.add(row['candidate_id'])
    return rows
