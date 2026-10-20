"""Industry/sector ETF relative-strength board -> data-etf.json / data-etf.js.

Answers "how is the market pricing each industry right now?" — not via raw daily
returns but via each ETF's return RELATIVE TO SPY and how its 3M RS rank has
moved over the last 4 weeks (rotation). Same price basis as the stock screener:
yfinance Close with auto_adjust=False = split-adjusted, dividends excluded.

    uv run python compute_etf.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import yfinance as yf

from compute_returns import download_chunk, extract_ticker_frame, returns_for, write_series

HERE = Path(__file__).parent
PERIODS = ["1W", "1M", "3M", "6M"]
RANK_PERIOD = "3M"
RANK_LOOKBACK_DAYS = 28   # "4주 전" rank snapshot
LOOKBACK_DAYS = 600       # 6M return dated 4 weeks back + MA50, plus SERIES_DAYS of chart history
SERIES_DAYS = 380         # trailing closes kept for the modal chart (same as the stock screener)
BENCH = "SPY"

# (ticker, group, Korean label). Groups drive the filter chips in the UI.
UNIVERSE = [
    # 미국 섹터 (Select Sector SPDR, GICS 11)
    ("XLK", "us-sector", "기술"), ("XLF", "us-sector", "금융"), ("XLV", "us-sector", "헬스케어"),
    ("XLE", "us-sector", "에너지"), ("XLB", "us-sector", "소재"), ("XLI", "us-sector", "산업재"),
    ("XLY", "us-sector", "경기소비재"), ("XLP", "us-sector", "필수소비재"), ("XLU", "us-sector", "유틸리티"),
    ("XLRE", "us-sector", "부동산"), ("XLC", "us-sector", "커뮤니케이션"),
    # 미국 산업 세분화
    ("SMH", "us-industry", "반도체"), ("IGV", "us-industry", "소프트웨어"),
    ("XHB", "us-industry", "주택건설·자재"), ("ITB", "us-industry", "주택건설"),
    ("KRE", "us-industry", "지역은행"), ("XME", "us-industry", "금속·광업"),
    ("XOP", "us-industry", "석유·가스 E&P"), ("COPX", "us-industry", "구리 광산"),
    ("URA", "us-industry", "우라늄·원전"), ("ITA", "us-industry", "항공우주·방산"),
    ("IBB", "us-industry", "바이오텍"), ("XRT", "us-industry", "소매유통"),
    ("JETS", "us-industry", "항공사"), ("TAN", "us-industry", "태양광"),
    ("PAVE", "us-industry", "인프라 건설"),
    # 글로벌 섹터 (iShares Global)
    ("IXN", "global-sector", "글로벌 기술"), ("IXC", "global-sector", "글로벌 에너지"),
    ("MXI", "global-sector", "글로벌 소재"), ("IXG", "global-sector", "글로벌 금융"),
    ("IXJ", "global-sector", "글로벌 헬스케어"), ("EXI", "global-sector", "글로벌 산업재"),
    ("RXI", "global-sector", "글로벌 경기소비재"), ("KXI", "global-sector", "글로벌 필수소비재"),
    ("JXI", "global-sector", "글로벌 유틸리티"),
    # 벤치마크 (순위 계산에서 제외)
    ("SPY", "bench", "S&P 500"), ("ACWI", "bench", "MSCI 전세계"),
]
GROUPS = {
    "us-sector": "미국 섹터",
    "us-industry": "미국 산업",
    "global-sector": "글로벌 섹터",
    "bench": "벤치마크",
}


def rel(r: float | None, b: float | None) -> float | None:
    """Return relative to the benchmark: (1+r)/(1+b) − 1, in %."""
    if r is None or b is None:
        return None
    return round(((1 + r / 100) / (1 + b / 100) - 1) * 100, 2)


def ranks(values: dict[str, float | None]) -> dict[str, int | None]:
    """1 = strongest. Tickers with no value get None."""
    have = sorted((v, t) for t, v in values.items() if v is not None)
    have.reverse()
    out: dict[str, int | None] = {t: None for t in values}
    for i, (_, t) in enumerate(have):
        out[t] = i + 1
    return out


def fetch_holdings(tickers: list[str]) -> dict[str, dict]:
    """Current top-10 holdings + expense ratio via yfinance funds_data. Best-effort: a miss leaves []."""
    out: dict[str, dict] = {}
    for t in tickers:
        rec: dict = {"top": [], "expense": None}
        try:
            fd = yf.Ticker(t).funds_data
            th = fd.top_holdings
            if th is not None and not th.empty:
                rec["top"] = [{"t": str(sym), "name": str(r["Name"]), "pct": round(float(r["Holding Percent"]) * 100, 2)}
                              for sym, r in th.head(10).iterrows()]
            ov = getattr(fd, "fund_overview", None)
            if isinstance(ov, dict) and ov.get("expenseRatio") is not None:
                rec["expense"] = round(float(ov["expenseRatio"]) * 100, 2)
        except Exception as e:  # noqa: BLE001
            print(f"  holdings miss {t}: {e}", file=sys.stderr)
        out[t] = rec
    return out


def main() -> None:
    tickers = [t for t, _, _ in UNIVERSE]
    start = (pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    df = download_chunk(tickers, start)
    if df is None or df.empty:
        sys.exit("ETF download failed")

    closes: dict[str, pd.Series] = {}
    for t in tickers:
        sub = extract_ticker_frame(df, t, False)
        if sub is None or "Close" not in sub or sub["Close"].dropna().empty:
            print(f"  no data: {t}", file=sys.stderr)
            continue
        closes[t] = sub["Close"].dropna()
    if BENCH not in closes:
        sys.exit("benchmark SPY missing")

    as_of = closes[BENCH].index[-1]
    prev_date = as_of - pd.Timedelta(days=RANK_LOOKBACK_DAYS)

    now_r: dict[str, dict] = {}
    prev_r: dict[str, dict] = {}
    for t, c in closes.items():
        now_r[t] = returns_for(c)[0]
        prev_r[t] = returns_for(c[c.index <= prev_date])[0]

    ranked = [t for t, g, _ in UNIVERSE if g != "bench" and t in closes]
    rank_now = ranks({t: rel(now_r[t][RANK_PERIOD], now_r[BENCH][RANK_PERIOD]) for t in ranked})
    rank_prev = ranks({t: rel(prev_r[t][RANK_PERIOD], prev_r[BENCH][RANK_PERIOD]) for t in ranked})

    holdings = fetch_holdings(tickers)
    print(f"  holdings: {sum(1 for h in holdings.values() if h['top'])}/{len(tickers)} ETFs", file=sys.stderr)

    etfs = []
    for t, g, label in UNIVERSE:
        if t not in closes:
            continue
        c = closes[t]
        ma50 = c.rolling(50).mean().iloc[-1]
        dev50 = round((c.iloc[-1] / ma50 - 1) * 100, 1) if pd.notna(ma50) and ma50 > 0 else None
        r = {p: now_r[t][p] for p in PERIODS}
        rn, rp = rank_now.get(t), rank_prev.get(t)
        etfs.append({
            "ticker": t,
            "label": label,
            "group": g,
            "price": round(float(c.iloc[-1]), 2),
            "lastDate": c.index[-1].strftime("%Y-%m-%d"),
            "r": r,
            "rel": {p: (None if t == BENCH else rel(r[p], now_r[BENCH][p])) for p in PERIODS},
            "rank": rn,
            "rankPrev": rp,
            "rankChg": (rp - rn) if rn is not None and rp is not None else None,  # + = 순위 상승
            "dev50": dev50,
            "holdings": holdings.get(t, {}).get("top", []),
            "expense": holdings.get(t, {}).get("expense"),
        })

    payload = {
        "asOf": as_of.strftime("%Y-%m-%d"),
        "prevDate": closes[BENCH][closes[BENCH].index <= prev_date].index[-1].strftime("%Y-%m-%d"),
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark": BENCH,
        "rankPeriod": RANK_PERIOD,
        "rankCount": len([e for e in etfs if e["rank"] is not None]),
        "returnBasis": "price return, split-adjusted, dividends excluded",
        "groups": GROUPS,
        "etfs": etfs,
    }
    # chart series (same compact format as series-us.js: shared date axis + per-ticker offset/closes)
    axis_idx = closes[BENCH].index[-SERIES_DAYS:]
    axis = [d.strftime("%Y-%m-%d") for d in axis_idx]
    series = {}
    for t, c in closes.items():
        cc = c.reindex(axis_idx)
        first = int(cc.notna().values.argmax()) if cc.notna().any() else None
        if first is None:
            continue
        series[t] = {"o": first, "c": [round(float(v), 4) if pd.notna(v) else None for v in cc.iloc[first:]]}
    write_series(HERE / "series-etf.js", "etf", payload["asOf"] if False else as_of.strftime("%Y-%m-%d"), axis, series)

    (HERE / "data-etf.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (HERE / "data-etf.js").write_text(
        "window.SCREENER_DATA_ETF = " + json.dumps(payload, ensure_ascii=False) + ";", encoding="utf-8")
    print(f"wrote {len(etfs)} ETFs to data-etf.json (asOf {payload['asOf']}, rank vs {payload['prevDate']})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
