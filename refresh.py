"""One-shot refresh: rebuild universe(s), recompute returns & scores, regenerate data.

By default refreshes BOTH markets (US then KR). Flags:
    uv run python refresh.py            # both
    uv run python refresh.py --us       # US only
    uv run python refresh.py --kr       # KR only
    uv run python refresh.py --quick    # both, skip universe refresh
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run(*args: str) -> None:
    print(f"\n=== {' '.join(args)} ===", file=sys.stderr)
    r = subprocess.run([sys.executable, *args])
    if r.returncode != 0:
        sys.exit(f"failed (exit {r.returncode})")


def refresh_market(market: str, quick: bool) -> None:
    fetcher = HERE / f"fetch_universe_{market}.py"
    universe = HERE / f"universe-{market}.json"
    if not quick or not universe.exists():
        run(str(fetcher))
    run(str(HERE / "compute_returns.py"), "--market", market)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--us", action="store_true", help="US market only")
    ap.add_argument("--kr", action="store_true", help="KR market only")
    ap.add_argument("--quick", action="store_true", help="skip universe refresh")
    args = ap.parse_args()

    if args.us:
        markets = ["us"]
    elif args.kr:
        markets = ["kr"]
    else:
        markets = ["us", "kr"]

    for m in markets:
        refresh_market(m, args.quick)

    print("\nDone. Open index.html in a browser.", file=sys.stderr)


if __name__ == "__main__":
    main()
