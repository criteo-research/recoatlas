#!/usr/bin/env python3
"""Run one paper tool profile through its frozen domain launcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools_builder.profiles import available_profiles, load_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=available_profiles())
    parser.add_argument(
        "--show-profile",
        action="store_true",
        help="Print the resolved profile without launching training.",
    )
    args, forwarded = parser.parse_known_args()
    profile = load_profile(args.profile)
    if args.show_profile:
        print(json.dumps(profile, indent=2))
        return
    launcher = ROOT / profile["launcher"]
    command = [sys.executable, "-u", str(launcher)]
    if profile["domain"] == "amazon":
        command.extend(["--profile", args.profile])
    command.extend(forwarded)
    print("Profile:", args.profile, flush=True)
    print("Command:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
