"""One-shot refresh: rebuild the universe, recompute returns, regenerate data.

Usage:
    uv run python refresh.py            # full refresh (universe + returns)
    uv run python refresh.py --quick    # reuse existing universe.json, returns only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run(script: str) -> None:
    print(f"\n=== {script} ===", file=sys.stderr)
    r = subprocess.run([sys.executable, str(HERE / script)])
    if r.returncode != 0:
        sys.exit(f"{script} failed (exit {r.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip universe refresh")
    args = ap.parse_args()

    if not args.quick:
        run("fetch_universe.py")
    elif not (HERE / "universe.json").exists():
        run("fetch_universe.py")
    run("compute_returns.py")
    print("\nDone. Open index.html in a browser.", file=sys.stderr)


if __name__ == "__main__":
    main()
