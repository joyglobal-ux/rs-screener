"""Compute split-adjusted price returns (dividends excluded) for every ticker
in universe.json over 1W / 1M / 3M / 6M / 1Y / YTD, and write data.json.

Accuracy notes:
- yfinance "Close" with auto_adjust=False is already SPLIT-adjusted but NOT
  dividend-adjusted (verified: NFLX close is continuous across its 10:1 split).
  That is exactly a *price* return basis, matching how Google/Yahoo headline %
  changes read, so we use Close directly. (auto_adjust=True would fold in
  dividends -> total return, which we do not want.)
- Each period anchors on the latest trading day and looks back to the nearest
  trading day on or before the target calendar date (holiday-safe via asof).
- YTD anchors on the last trading day of the previous calendar year.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent

LOOKBACK_DAYS = 420  # > 365 buffer for 1Y + holidays
CHUNK = 100
PERIOD_KEYS = ["1W", "1M", "3M", "6M", "1Y", "YTD"]

MARKET_META = {
    "us": {
        "code": "us",
        "label": "US · NYSE / Nasdaq / AMEX",
        "currency": "USD",
        "currencySymbol": "$",
        "minMarketCap": 1_000_000_000,
        "minMarketCapLabel": "$1B",
        "tickerUrl": "https://finance.yahoo.com/quote/{ticker}",
    },
    "kr": {
        "code": "kr",
        "label": "KR · KOSPI + KOSDAQ",
        "currency": "KRW",
        "currencySymbol": "₩",
        "minMarketCap": 100_000_000_000,
        "minMarketCapLabel": "1,000억 원",
        "tickerUrl": "https://finance.naver.com/item/main.naver?code={krxCode}",
    },
}


def paths_for(market: str) -> dict[str, Path]:
    return {
        "universe": HERE / f"universe-{market}.json",
        "data_json": HERE / f"data-{market}.json",
        "data_js": HERE / f"data-{market}.js",
    }

# RS Score: trend-following tilt on intermediate-term momentum.
# 1W is excluded (short-term mean reversion) and YTD is excluded (variable
# length). 1M is kept small despite short-term reversal, per user choice.
# "1Y" is the 12-month leg. Weights sum to 100.
SCORE_WEIGHTS = {"1M": 10, "3M": 36, "6M": 32, "1Y": 22}
SCORE_REQUIRED = "6M"  # need ~6 months of history to earn a score
SCORE_BASIS = "1M·3M·6M·12M 가중 백분위 (추세추종 틸트 10/36/32/22), 1W·YTD 제외"


def compute_scores(stocks: list[dict]) -> None:
    """Add an IBD-style 1-99 'score' to each stock, in place.

    For each scoring period, rank stocks into a 0-100 cross-sectional percentile
    (robust to outliers like a +8000% mover). Take the weighted average of the
    available period percentiles (weights renormalized when a period is missing),
    requiring at least ~6 months of history. Finally re-rank that composite into
    a 1-99 score so 99 = strongest relative strength.
    """
    if not stocks:
        return
    df = pd.DataFrame([{p: s["r"].get(p) for p in SCORE_WEIGHTS} for s in stocks])
    pct = pd.DataFrame({p: df[p].rank(pct=True) * 100.0 for p in SCORE_WEIGHTS})
    weights = pd.Series({p: float(w) for p, w in SCORE_WEIGHTS.items()})

    composite = []
    for i in range(len(df)):
        row = pct.iloc[i].dropna()
        if SCORE_REQUIRED not in row.index:
            composite.append(float("nan"))
            continue
        w = weights[row.index]
        composite.append(float((row * w).sum() / w.sum()))

    final = (pd.Series(composite).rank(pct=True) * 98 + 1).round()
    for i, s in enumerate(stocks):
        v = final.iloc[i]
        s["score"] = int(v) if pd.notna(v) else None


def period_targets(last_date: pd.Timestamp) -> dict[str, pd.Timestamp]:
    return {
        "1W": last_date - pd.Timedelta(days=7),
        "1M": last_date - pd.DateOffset(months=1),
        "3M": last_date - pd.DateOffset(months=3),
        "6M": last_date - pd.DateOffset(months=6),
        "1Y": last_date - pd.DateOffset(years=1),
        "YTD": pd.Timestamp(year=last_date.year - 1, month=12, day=31),
    }


def is_halted(close: pd.Series) -> bool:
    """True if the stock was effectively halted (suspended) for much of the
    trailing year — its price stuck at one value across the whole window.

    Such names (e.g. a KOSDAQ suspension that resumes after a capital reduction)
    show a flat line that then jumps once on resumption; the period returns then
    just capture that one structural jump (often == the reverse-split ratio),
    not a real price trend. Threshold is deliberately extreme (one identical
    close on >50% of trailing-year days) so normal low-volatility names are
    never caught — a real stock essentially never repeats one exact price for
    half a year.
    """
    tail = close.tail(252)
    if len(tail) < 60:
        return False
    modal_frac = tail.round(4).value_counts(normalize=True).iloc[0]
    return modal_frac > 0.5


def returns_for(close: pd.Series) -> tuple[dict[str, float | None], float | None, pd.Timestamp | None]:
    close = close.dropna()
    if len(close) < 2:
        return {k: None for k in PERIOD_KEYS}, None, None
    last_date = close.index[-1]
    current = float(close.iloc[-1])
    if is_halted(close):
        # Keep the latest price for display, but null the returns — they would be
        # a halt/capital-reduction artifact, not a real return.
        return {k: None for k in PERIOD_KEYS}, current, last_date
    out: dict[str, float | None] = {}
    for key, target in period_targets(last_date).items():
        past = close.asof(target)
        if pd.isna(past) or past <= 0 or target < close.index[0]:
            out[key] = None
        else:
            out[key] = round((current / float(past) - 1.0) * 100.0, 2)
    return out, current, last_date


def trend_quality_and_accel(close: pd.Series) -> tuple[float | None, float | None]:
    """Return (trend_quality_R2, acceleration).

    - Trend quality = R^2 of log(price) vs time over ~6 months (1.0 = a perfectly
      straight up/down line; low = choppy). Measures *smoothness*, independent of
      how big the move was — complements the RS Score (which captures the level).
    - Acceleration = recent 3M return minus the prior 3M return (months -6..-3),
      in percentage points. Positive = trend speeding up.
    """
    close = close.dropna()
    if len(close) < 2 or is_halted(close):
        return None, None
    last = close.index[-1]

    quality: float | None = None
    w6 = close[close.index >= last - pd.DateOffset(months=6)]
    if len(w6) >= 60:
        y = np.log(w6.to_numpy(dtype=float))
        x = np.arange(len(y), dtype=float)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot > 0:
            slope, intercept = np.polyfit(x, y, 1)
            ss_res = float(np.sum((y - (slope * x + intercept)) ** 2))
            quality = round(1.0 - ss_res / ss_tot, 2)

    accel: float | None = None
    w3 = close[close.index >= last - pd.DateOffset(months=3)]
    seg = close[(close.index >= last - pd.DateOffset(months=6))
                & (close.index <= last - pd.DateOffset(months=3))]
    if len(w3) >= 2 and len(seg) >= 5 and w3.iloc[0] > 0 and seg.iloc[0] > 0:
        recent3 = w3.iloc[-1] / w3.iloc[0] - 1
        prior3 = seg.iloc[-1] / seg.iloc[0] - 1
        accel = round((recent3 - prior3) * 100.0, 1)

    return quality, accel


def extract_ticker_frame(data: pd.DataFrame, ticker: str, single: bool) -> pd.DataFrame | None:
    """Pull a single ticker's OHLC+actions frame out of a yf.download result."""
    if single:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        if ticker not in data.columns.get_level_values(0):
            return None
        return data[ticker]
    return None


def download_chunk(tickers: list[str], start: str) -> pd.DataFrame:
    return yf.download(
        tickers,
        start=start,
        auto_adjust=False,  # Close stays split-adjusted, dividends excluded
        actions=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )


def write_outputs(payload: dict, paths: dict[str, Path]) -> None:
    paths["data_json"].write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    # data-<market>.js lets index.html load via <script> (works from file:// too).
    global_name = "SCREENER_DATA_" + payload["market"]["code"].upper()
    paths["data_js"].write_text(
        f"window.{global_name} = " + json.dumps(payload, ensure_ascii=False) + ";",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["us", "kr"], default="us",
                    help="which market universe to process")
    ap.add_argument("--test", action="store_true", help="run a small verification set")
    ap.add_argument("--limit", type=int, default=0, help="cap number of tickers")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute RS scores from existing data file (no download)")
    args = ap.parse_args()

    paths = paths_for(args.market)

    if args.rescore:
        payload = json.loads(paths["data_json"].read_text(encoding="utf-8"))
        compute_scores(payload["stocks"])
        payload["scoreWeights"] = SCORE_WEIGHTS
        payload["scoreBasis"] = SCORE_BASIS
        payload["market"] = MARKET_META[args.market]
        payload["minMarketCap"] = MARKET_META[args.market]["minMarketCap"]
        write_outputs(payload, paths)
        print(f"rescored {len(payload['stocks'])} {args.market.upper()} stocks", file=sys.stderr)
        return

    if args.test:
        meta = [
            {"ticker": t, "name": t, "sector": "", "industry": "", "marketCap": 0}
            for t in ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL",
                       "BRK-B", "JPM", "WMT", "AVGO", "LLY"]
        ]
    else:
        meta = json.loads(paths["universe"].read_text(encoding="utf-8"))
    if args.limit:
        meta = meta[: args.limit]

    tickers = [m["ticker"] for m in meta]
    start = (pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    print(f"computing returns for {len(tickers)} tickers since {start}", file=sys.stderr)

    results: dict[str, dict] = {}
    failed: list[str] = []
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i : i + CHUNK]
        single = len(chunk) == 1
        df = None
        for attempt in range(3):
            try:
                df = download_chunk(chunk, start)
                if df is not None and not df.empty:
                    break
            except Exception as e:  # noqa: BLE001
                print(f"  chunk {i//CHUNK} attempt {attempt}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
        if df is None or df.empty:
            failed.extend(chunk)
            continue

        for t in chunk:
            sub = extract_ticker_frame(df, t, single)
            if sub is None or "Close" not in sub:
                failed.append(t)
                continue
            rets, price, last_date = returns_for(sub["Close"])
            if price is None:
                failed.append(t)
                continue
            quality, accel = trend_quality_and_accel(sub["Close"])
            results[t] = {
                "r": rets,
                "price": round(price, 2),
                "lastDate": last_date.strftime("%Y-%m-%d"),
                "quality": quality,
                "accel": accel,
            }
        print(f"  {min(i+CHUNK, len(tickers))}/{len(tickers)} done", file=sys.stderr)

    if args.test:
        print(f"\n{'ticker':8}{'price':>10}  " + "".join(f"{k:>9}" for k in PERIOD_KEYS))
        for t in tickers:
            if t not in results:
                print(f"{t:8}  (no data)")
                continue
            r = results[t]
            cells = "".join(
                (f"{r['r'][k]:>8.1f}%" if r["r"][k] is not None else f"{'n/a':>9}")
                for k in PERIOD_KEYS
            )
            print(f"{t:8}{r['price']:>10.2f}  {cells}")
        print(f"\nfailed: {failed}", file=sys.stderr)
        return

    # Merge metadata + returns into final payload.
    last_dates = [v["lastDate"] for v in results.values()]
    as_of = max(last_dates) if last_dates else None
    stocks = []
    for m in meta:
        t = m["ticker"]
        if t not in results:
            continue
        r = results[t]
        stock = {
            "ticker": t,
            "name": m["name"],
            "sector": m["sector"],
            "industry": m["industry"],
            "marketCap": m["marketCap"],
            "price": r["price"],
            "lastDate": r["lastDate"],
            "quality": r["quality"],
            "accel": r["accel"],
            "r": r["r"],
        }
        # Carry market-specific identifiers through to the data file so the UI
        # can build proper links (e.g., KR uses Naver, which keys off krxCode).
        for k in ("krxCode", "subMarket"):
            if k in m:
                stock[k] = m[k]
        if "market" in m:  # KOSPI / KOSDAQ tag on KR rows
            stock["subMarket"] = m["market"]
        stocks.append(stock)

    compute_scores(stocks)

    market_meta = MARKET_META[args.market]
    payload = {
        "market": market_meta,
        "asOf": as_of,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "minMarketCap": market_meta["minMarketCap"],
        "returnBasis": "price return, split-adjusted, dividends excluded",
        "periods": PERIOD_KEYS,
        "scoreWeights": SCORE_WEIGHTS,
        "scoreBasis": SCORE_BASIS,
        "signalsBasis": "품질 = 6M 추세 직선성(log가격 R², 1=완벽직선) · 가속 = 최근3M − 직전3M (%p)",
        "count": len(stocks),
        "stocks": stocks,
    }
    write_outputs(payload, paths)
    print(
        f"wrote {len(stocks)} {args.market.upper()} stocks to {paths['data_json'].name} (asOf {as_of}); "
        f"failed {len(failed)}",
        file=sys.stderr,
    )
    if failed:
        print(f"failed tickers (first 30): {failed[:30]}", file=sys.stderr)


if __name__ == "__main__":
    main()
