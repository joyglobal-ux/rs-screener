"""Fetch the US-listed equity universe (NYSE/Nasdaq/AMEX) from the Nasdaq
screener API, filter to market cap >= MIN_MARKET_CAP, and write universe.json.

Output rows carry the metadata the screener UI needs (name, sector, industry,
market cap, exchange) plus a Yahoo-compatible ticker for the price step.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

MIN_MARKET_CAP = 1_000_000_000  # $1B
OUT = Path(__file__).parent / "universe-us.json"

URL = "https://api.nasdaq.com/api/screener/stocks"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}


def to_yahoo_symbol(sym: str) -> str:
    """Nasdaq uses 'BRK/B'; Yahoo uses 'BRK-B'. Drop preferred/when-issued."""
    return sym.strip().upper().replace("/", "-").replace(".", "-")


def parse_market_cap(raw: str) -> float | None:
    raw = (raw or "").replace(",", "").replace("$", "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v > 0 else None


def fetch_rows() -> list[dict]:
    params = {"tableonly": "false", "limit": "0", "download": "true"}
    last_err = None
    for attempt in range(4):
        try:
            r = requests.get(URL, headers=HEADERS, params=params, timeout=30)
            r.raise_for_status()
            payload = r.json()
            rows = payload["data"]["rows"]
            if rows:
                return rows
            last_err = "empty rows"
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Nasdaq screener fetch failed: {last_err}")


def main() -> None:
    rows = fetch_rows()
    print(f"screener returned {len(rows)} rows", file=sys.stderr)

    out: list[dict] = []
    skipped_no_cap = 0
    for row in rows:
        sym = (row.get("symbol") or "").strip().upper()
        if not sym:
            continue
        # Skip preferred shares / warrants / units / when-issued artifacts.
        if any(ch in sym for ch in ("^", "~")) or sym.endswith((".W", ".U", ".R")):
            continue
        cap = parse_market_cap(row.get("marketCap", ""))
        if cap is None:
            skipped_no_cap += 1
            continue
        if cap < MIN_MARKET_CAP:
            continue
        out.append(
            {
                "ticker": to_yahoo_symbol(sym),
                "nasdaqSymbol": sym,
                "name": (row.get("name") or "").replace(" Common Stock", "").strip(),
                "sector": (row.get("sector") or "").strip() or "Unknown",
                "industry": (row.get("industry") or "").strip() or "Unknown",
                "marketCap": cap,
                "country": (row.get("country") or "").strip(),
                "ipoYear": (row.get("ipoyear") or "").strip(),
            }
        )

    # De-dup by ticker (screener can repeat dual-listed names), keep larger cap.
    by_ticker: dict[str, dict] = {}
    for r in out:
        prev = by_ticker.get(r["ticker"])
        if prev is None or r["marketCap"] > prev["marketCap"]:
            by_ticker[r["ticker"]] = r
    final = sorted(by_ticker.values(), key=lambda r: r["marketCap"], reverse=True)

    OUT.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"kept {len(final)} tickers >= ${MIN_MARKET_CAP/1e9:.0f}B "
        f"(skipped {skipped_no_cap} without market cap) -> {OUT.name}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
