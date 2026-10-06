"""Download only frozen inference/evaluation assets, never training runs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from artifacts import CATEGORIES, source_config, verify_file, sha256, SOURCES


def prepare(category, output, *, tools_only=False, metadata_only=False):
    from huggingface_hub import hf_hub_download
    config = source_config()
    category_spec = config['categories'][category]
    if not category_spec['query_counts'] and not tools_only:
        raise ValueError(category_spec['unavailable_reason'] + ' Use --tools-only to download catalog tools without claiming benchmark evaluation.')
    root = Path(output).resolve() / category
    selected = {rel: spec for rel, spec in category_spec['files'].items()
                if not (tools_only and rel.startswith('queries/'))
                and not (metadata_only and rel.endswith('.pt'))}
    for rel, spec in selected.items():
        target = root / rel
        if target.exists() or target.is_symlink():
            verify_file(target, spec)  # Never silently overwrite changed local data.
            print(f'Verified {category}/{rel}', flush=True)
            continue
        src = config['sources'][spec['source']]
        cached = hf_hub_download(**src, filename=spec['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.download-', dir=target.parent)
        os.close(fd)
        temporary = Path(temporary)
        try:
            shutil.copyfile(cached, temporary)
            verify_file(temporary, spec)
            # Exclusive creation prevents a concurrent run from overwriting a file.
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'Downloaded {category}/{rel}', flush=True)
    return {'category': category, 'files': len(selected), 'artifact_sources_sha256': sha256(SOURCES)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category', choices=CATEGORIES, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--tools-only', action='store_true', help='Download tools even when query/reference sets are unavailable')
    mode.add_argument('--metadata-only', action='store_true', help='Catalog and queries only, sufficient to score/judge existing traces')
    args = parser.parse_args()
    try:
        result = prepare(args.category, args.output, tools_only=args.tools_only, metadata_only=args.metadata_only)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
