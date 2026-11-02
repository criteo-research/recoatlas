#!/usr/bin/env python3
"""Publish one completed tool run to the clean RecoAtlas HF dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools_builder.profiles import available_profiles, load_profile
from tools_builder.release import publish_release, validate_release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=available_profiles())
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo", help="Override the profile's release repository")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--keep-extra-files",
        action="store_true",
        help=(
            "Do not remove unexpected files below the profile's tools folder. "
            "The default is an exact clean synchronization."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the five-file plan without network access.",
    )
    args = parser.parse_args()
    profile = load_profile(args.profile)
    if args.dry_run:
        artifacts = validate_release(profile, args.run_dir)
        print(
            json.dumps(
                {
                    "repo_id": args.repo or profile["release_repo"],
                    "repo_type": "dataset",
                    "revision": args.revision,
                    "clean_tools_prefix": not args.keep_extra_files,
                    "artifacts": [item.public_record() for item in artifacts],
                },
                indent=2,
            )
        )
        return
    result = publish_release(
        profile,
        args.run_dir,
        repo_id=args.repo,
        revision=args.revision,
        clean=not args.keep_extra_files,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
