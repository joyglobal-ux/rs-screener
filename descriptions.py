"""One-line Korean business descriptions for setup-card tickers -> desc-<market>.json.

Raw source (fetched once per ticker, cached forever in desc-<m>.json):
    KR  WISEfn 기업개요 via navercomp.wisereport.co.kr (the text behind 네이버 금융 '기업개요'), Korean
    US  yfinance Ticker.info['longBusinessSummary'], English

Summary: Claude (claude-opus-5-5, low effort, structured JSON output) compresses each raw text
into 1~2 Korean sentences for the card. Needs ANTHROPIC_API_KEY (repo secret in Actions).
Without a key the card falls back to the first sentence(s) of the raw text and the entry is
marked "fallback": true so a later run with a key upgrades it.

    uv run python descriptions.py --market kr 005930.KS 000660.KS   # ad-hoc check
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
MODEL = "claude-opus-5-5"
BATCH = 20                 # tickers per Claude request
MAX_FETCH_PER_RUN = 150    # raw-text fetches per run (each ~1s)
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36"

SYSTEM = (
    "너는 한국 개인투자자의 종목 카드에 들어갈 '회사 한 줄 소개'를 쓴다. "
    "입력은 회사명과 공시·데이터 제공사의 사업 설명(한국어 또는 영어)이다. "
    "각 회사에 대해 무엇을 만들고 누구에게 파는지(핵심 제품·서비스·고객)를 한국어 1~2문장, 60~110자로 쓴다. "
    "회사명으로 문장을 시작하지 말고 '~업체', '~를 만든다'처럼 바로 사업으로 들어간다. "
    "투자 의견·전망·수식어(선도적, 글로벌 리더 등)·설립연도·자회사 수는 넣지 않는다. "
    "제품명·기술 용어는 원어(영문)를 그대로 써도 된다. 설명이 비어 있거나 알 수 없으면 ko를 빈 문자열로 둔다."
)
SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}, "ko": {"type": "string"}},
                "required": ["ticker", "ko"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _get(url: str, referer: str | None = None, timeout: int = 15) -> str:
    headers = {"User-Agent": UA}
    if referer:
        headers["Referer"] = referer
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="ignore")


def fetch_raw_kr(ticker: str) -> str | None:
    """WISEfn 기업개요 — the <ul class="dot_cmp"> bullets on the WiseReport company page."""
    code = ticker.split(".")[0]
    page = _get(f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={code}",
                referer="https://finance.naver.com/")
    m = re.search(r'<ul[^>]*class="dot_cmp"[^>]*>(.*?)</ul>', page, re.S)
    if not m:
        return None
    items = [html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", li))).strip()
             for li in re.findall(r"<li[^>]*>(.*?)</li>", m.group(1), re.S)]
    text = " ".join(i for i in items if i)
    return text or None


def fetch_raw_us(ticker: str) -> str | None:
    import yfinance as yf
    info = yf.Ticker(ticker).info or {}
    text = info.get("longBusinessSummary")
    return text.strip() if text else None


def fallback_ko(raw: str, market: str) -> str:
    """No API key: first sentence(s) of the raw text, trimmed."""
    if market == "kr":
        parts = re.split(r"(?<=[음임됨다])\.\s*", raw)
        out = ". ".join(p.strip() for p in parts[:2] if p.strip())
        return (out[:118] + "…") if len(out) > 120 else (out + ("" if out.endswith(".") else "."))
    first = re.split(r"(?<=[.!?])\s+", raw)[0]
    return (first[:158] + "…") if len(first) > 160 else first


def summarize(items: list[tuple[str, str, str]]) -> dict[str, str]:
    """[(ticker, name, raw)] -> {ticker: ko} via Claude. Raises on API failure (caller falls back)."""
    import anthropic
    client = anthropic.Anthropic()
    out: dict[str, str] = {}
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        user = "\n\n".join(f"[{t}] {n}\n{raw[:1500]}" for t, n, raw in chunk)
        kwargs = dict(
            model=MODEL, max_tokens=4000, system=SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": user}],
        )
        try:   # server-side refusal fallback (routes a declined request to another model inside the same call)
            resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        except anthropic.BadRequestError:
            resp = client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            log(f"  desc: refusal on batch {i // BATCH} — falling back for {len(chunk)} tickers")
            continue
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            log(f"  desc: bad JSON on batch {i // BATCH}")
            continue
        for it in data.get("items", []):
            ko = (it.get("ko") or "").strip()
            if ko:
                out[it["ticker"]] = ko
    return out


def load_cache(market: str) -> dict:
    p = HERE / f"desc-{market}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {}


def ensure_descriptions(market: str, wanted: list[tuple[str, str]]) -> dict[str, str]:
    """Return {ticker: ko} for the wanted (ticker, name) pairs, fetching/summarizing what the cache lacks."""
    cache = load_cache(market)
    have_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # 1) raw text for tickers never fetched (cap per run; a miss is remembered so it is not retried daily)
    need_raw = [(t, n) for t, n in wanted if t not in cache][:MAX_FETCH_PER_RUN]
    t0 = time.time()
    for t, n in need_raw:
        raw = None
        try:
            raw = fetch_raw_kr(t) if market == "kr" else fetch_raw_us(t)
        except Exception as e:  # noqa: BLE001
            log(f"  desc: raw miss {t}: {e}")
        cache[t] = {"name": n, "raw": raw, "ko": None, "fallback": None, "at": now}
    if need_raw:
        log(f"  desc: fetched raw text for {len(need_raw)} tickers ({sum(1 for t,_ in need_raw if cache[t]['raw'])} hits) in {time.time()-t0:.0f}s")

    # 2) summaries: anything with raw text and no Claude summary yet (fallbacks get upgraded once a key is present)
    pending = [(t, cache[t]["name"], cache[t]["raw"]) for t, _ in wanted
               if cache.get(t, {}).get("raw") and (cache[t].get("ko") is None or cache[t].get("fallback"))]
    if pending and have_key:
        t0 = time.time()
        try:
            got = summarize(pending)
            for t, ko in got.items():
                if t in cache:
                    cache[t].update({"ko": ko, "fallback": False, "at": now, "model": MODEL})
            log(f"  desc: Claude summarized {len(got)}/{len(pending)} in {time.time()-t0:.0f}s")
        except Exception as e:  # noqa: BLE001
            log(f"  desc: Claude failed ({type(e).__name__}: {str(e)[:120]}) → fallback text")
    elif pending and not have_key:
        log(f"  desc: ANTHROPIC_API_KEY not set → {len(pending)} fallback summaries (first sentences)")
    for t, _, raw in pending:
        if cache[t].get("ko") is None:
            cache[t].update({"ko": fallback_ko(raw, market), "fallback": True, "at": now})

    (HERE / f"desc-{market}.json").write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    return {t: cache[t]["ko"] for t, _ in wanted if cache.get(t, {}).get("ko")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["us", "kr"], required=True)
    ap.add_argument("tickers", nargs="+")
    args = ap.parse_args()
    out = ensure_descriptions(args.market, [(t, t) for t in args.tickers])
    for t in args.tickers:
        print(f"{t}: {out.get(t) or '(없음)'}")


if __name__ == "__main__":
    main()
