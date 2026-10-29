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
    # 그룹판·셋업 스캔 — compute_returns가 남긴 OHLCV 캐시(.cache/)를 재사용한다.
    run(str(HERE / "scanner.py"), "--market", market)
    # 산업 ETF 보드는 미국(vs SPY)·국내(vs KODEX 200) 두 묶음을 각자 최신 마감으로 받는다 →
    # 어느 시장을 갱신하든 함께 돌린다 (ETF별 lastDate·묶음별 asOf를 payload에 기록).
    run(str(HERE / "compute_etf.py"))


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
