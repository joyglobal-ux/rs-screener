"""Industry/sector ETF relative-strength board -> data-etf.json / data-etf.js.

Answers "how is the market pricing each industry right now?" — not via raw daily
returns but via each ETF's return RELATIVE TO ITS BENCHMARK and how its 3M RS
rank has moved over the last 4 weeks (rotation). Two pools, ranked separately:

    us  — US sector / US industry / global sector ETFs, vs SPY
    kr  — 국내 섹터 ETF (KODEX/TIGER …, yfinance "<code>.KS"), vs KODEX 200

Same price basis as the stock screener: yfinance Close with auto_adjust=False =
split-adjusted, dividends excluded. Each pool has its own asOf (US close vs KR
close) and its own 4-weeks-ago snapshot date.

    uv run python compute_etf.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import yfinance as yf

from compute_returns import download_chunk, extract_ticker_frame, returns_for

HERE = Path(__file__).parent
PERIODS = ["1W", "1M", "3M", "6M"]
RANK_PERIOD = "3M"
RANK_LOOKBACK_DAYS = 28   # "4주 전" rank snapshot
LOOKBACK_DAYS = 600       # 6M return dated 4 weeks back + MA50, plus SERIES_DAYS of chart history
SERIES_DAYS = 380         # trailing closes kept for the modal chart (same as the stock screener)

# Pools: each group belongs to one pool; each pool has one benchmark that drives
# rel / rank for every non-bench ETF in it. Bench rows themselves get rel vs the
# pool bench (so ACWI/코스닥150 still read against SPY/KODEX200) but no rank.
POOLS = {
    "us": {"bench": "SPY", "benchLabel": "SPY", "label": "미국·글로벌"},
    "kr": {"bench": "069500.KS", "benchLabel": "KODEX 200", "label": "국내"},
}
GROUPS = {
    "us-sector": "미국 섹터",
    "us-industry": "미국 산업",
    "global-sector": "글로벌 섹터",
    "kr-sector": "국내 섹터",
    "bench": "벤치마크",
}
GROUP_POOL = {"us-sector": "us", "us-industry": "us", "global-sector": "us", "kr-sector": "kr"}

# (ticker, group, Korean label[, fund name]). Groups drive the filter chips in the UI.
# KR codes verified against yfinance + Naver (2026-10-20); a code that stops
# returning prices is skipped at build time with a stderr note.
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
    # 국내 섹터 (KRX 상장, vs KODEX 200)
    ("091160.KS", "kr-sector", "반도체", "KODEX 반도체"),
    ("139260.KS", "kr-sector", "IT(코스피200)", "TIGER 200 IT"),
    ("157490.KS", "kr-sector", "소프트웨어", "TIGER 소프트웨어"),
    ("305720.KS", "kr-sector", "2차전지", "KODEX 2차전지산업"),
    ("091180.KS", "kr-sector", "자동차", "KODEX 자동차"),
    ("494670.KS", "kr-sector", "조선", "TIGER 조선TOP10"),
    ("449450.KS", "kr-sector", "방산", "PLUS K방산"),
    ("445290.KS", "kr-sector", "로봇", "KODEX 로봇액티브"),
    ("102960.KS", "kr-sector", "기계장비", "KODEX 기계장비"),
    ("434730.KS", "kr-sector", "원자력", "HANARO 원자력iSelect"),
    ("117700.KS", "kr-sector", "건설", "KODEX 건설"),
    ("117680.KS", "kr-sector", "철강", "KODEX 철강"),
    ("117460.KS", "kr-sector", "에너지화학", "KODEX 에너지화학"),
    ("139270.KS", "kr-sector", "금융(코스피200)", "TIGER 200 금융"),
    ("091170.KS", "kr-sector", "은행", "KODEX 은행"),
    ("102970.KS", "kr-sector", "증권", "KODEX 증권"),
    ("140700.KS", "kr-sector", "보험", "KODEX 보험"),
    ("143860.KS", "kr-sector", "헬스케어", "TIGER 헬스케어"),
    ("244580.KS", "kr-sector", "바이오", "KODEX 바이오"),
    ("228790.KS", "kr-sector", "화장품", "TIGER 화장품"),
    ("228810.KS", "kr-sector", "미디어컨텐츠", "TIGER 미디어컨텐츠"),
    ("300950.KS", "kr-sector", "게임", "KODEX 게임산업"),
    ("228800.KS", "kr-sector", "여행레저", "TIGER 여행레저"),
    ("140710.KS", "kr-sector", "운송", "KODEX 운송"),
    # 벤치마크 (순위 계산에서 제외)
    ("SPY", "bench", "S&P 500"), ("ACWI", "bench", "MSCI 전세계"),
    ("069500.KS", "bench", "코스피 200", "KODEX 200"), ("229200.KS", "bench", "코스닥 150", "KODEX 코스닥150"),
]
BENCH_POOL = {"SPY": "us", "ACWI": "us", "069500.KS": "kr", "229200.KS": "kr"}

# 미국 ETF → 국내 대응. (국내 ETF 티커 or None, WICS 섹터 목록 — 종목 뷰 KR 섹터 드롭다운 값).
# None/[] = 국내 대응 없음 (지역은행·주택건설 등). WICS 이름은 sector_map_kr.json에 대조한다.
KR_MAP: dict[str, tuple[str | None, list[str]]] = {
    "XLK":  ("139260.KS", ["반도체와반도체장비", "소프트웨어", "IT서비스"]),
    "XLF":  ("139270.KS", ["은행", "증권", "손해보험"]),
    "XLV":  ("143860.KS", ["제약", "생물공학", "건강관리장비와용품"]),
    "XLE":  ("117460.KS", ["석유와가스", "에너지장비및서비스"]),
    "XLB":  ("117680.KS", ["화학", "철강", "비철금속"]),
    "XLI":  ("102960.KS", ["기계", "조선", "우주항공과국방", "건설"]),
    "XLY":  ("091180.KS", ["자동차", "자동차부품", "호텔,레스토랑,레저"]),
    "XLP":  (None, ["식품", "음료", "화장품"]),
    "XLU":  (None, ["전기유틸리티", "가스유틸리티"]),
    "XLRE": (None, ["부동산"]),
    "XLC":  ("228810.KS", ["방송과엔터테인먼트", "게임엔터테인먼트", "양방향미디어와서비스", "무선통신서비스"]),
    "SMH":  ("091160.KS", ["반도체와반도체장비"]),
    "IGV":  ("157490.KS", ["소프트웨어", "IT서비스"]),
    "XHB":  (None, []), "ITB": (None, []), "KRE": (None, []), "TAN": (None, []),
    "XME":  ("117680.KS", ["철강", "비철금속"]),
    "XOP":  (None, ["석유와가스"]),
    "COPX": (None, ["비철금속"]),
    "URA":  ("434730.KS", []),
    "ITA":  ("449450.KS", ["우주항공과국방"]),
    "IBB":  ("244580.KS", ["생물공학", "제약"]),
    "XRT":  (None, ["백화점과일반상점", "판매업체", "인터넷과카탈로그소매"]),
    "JETS": ("140710.KS", ["항공사"]),
    "PAVE": ("117700.KS", ["건설"]),
    "IXN":  ("139260.KS", ["반도체와반도체장비", "소프트웨어", "IT서비스"]),
    "IXC":  ("117460.KS", ["석유와가스", "에너지장비및서비스"]),
    "MXI":  ("117680.KS", ["화학", "철강", "비철금속"]),
    "IXG":  ("139270.KS", ["은행", "증권", "손해보험"]),
    "IXJ":  ("143860.KS", ["제약", "생물공학", "건강관리장비와용품"]),
    "EXI":  ("102960.KS", ["기계", "조선", "우주항공과국방", "건설"]),
    "RXI":  ("091180.KS", ["자동차", "자동차부품", "호텔,레스토랑,레저"]),
    "KXI":  (None, ["식품", "음료", "화장품"]),
    "JXI":  (None, ["전기유틸리티", "가스유틸리티"]),
}
# 국내 ETF → WICS 섹터 (종목 뷰 점프용). 미국 대응은 KR_MAP을 역으로 찾는다.
KR_WICS: dict[str, list[str]] = {
    "091160.KS": ["반도체와반도체장비"], "139260.KS": ["반도체와반도체장비", "소프트웨어", "IT서비스"],
    "157490.KS": ["소프트웨어", "IT서비스"], "305720.KS": ["전기제품", "화학"],
    "091180.KS": ["자동차", "자동차부품"], "494670.KS": ["조선"], "449450.KS": ["우주항공과국방"],
    "445290.KS": ["기계"], "102960.KS": ["기계"], "434730.KS": ["전기장비", "기계"],
    "117700.KS": ["건설"], "117680.KS": ["철강"], "117460.KS": ["화학", "석유와가스"],
    "139270.KS": ["은행", "증권", "손해보험"], "091170.KS": ["은행"], "102970.KS": ["증권"],
    "140700.KS": ["손해보험", "생명보험"], "143860.KS": ["제약", "생물공학", "건강관리장비와용품"],
    "244580.KS": ["생물공학", "제약"], "228790.KS": ["화장품"], "228810.KS": ["방송과엔터테인먼트"],
    "300950.KS": ["게임엔터테인먼트"], "228800.KS": ["호텔,레스토랑,레저"], "140710.KS": ["항공사", "해운사"],
}


def is_kr(t: str) -> bool:
    return t.endswith(".KS") or t.endswith(".KQ")


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


def fetch_holdings_us(t: str) -> dict:
    """Top-10 holdings + expense ratio via yfinance funds_data (US-listed only)."""
    rec: dict = {"top": [], "expense": None}
    fd = yf.Ticker(t).funds_data
    th = fd.top_holdings
    if th is not None and not th.empty:
        rec["top"] = [{"t": str(sym), "name": str(r["Name"]), "pct": round(float(r["Holding Percent"]) * 100, 2)}
                      for sym, r in th.head(10).iterrows()]
    ov = getattr(fd, "fund_overview", None)
    if isinstance(ov, dict) and ov.get("expenseRatio") is not None:
        rec["expense"] = round(float(ov["expenseRatio"]) * 100, 2)
    return rec


def fetch_holdings_kr(t: str) -> dict:
    """Top-10 constituents + total fee from Naver's mobile ETF analysis API (yfinance has no KR fund data).
    Holding tickers are bare KRX codes (e.g. 005930); the UI resolves them against the KR stock list."""
    code = t.split(".")[0]
    req = urllib.request.Request(f"https://m.stock.naver.com/api/stock/{code}/etfAnalysis",
                                 headers={"User-Agent": "Mozilla/5.0"})
    d = json.load(urllib.request.urlopen(req, timeout=15))
    rec: dict = {"top": [], "expense": None}
    for h in (d.get("etfTop10MajorConstituentAssets") or [])[:10]:
        try:
            pct = float(str(h.get("etfWeight", "")).rstrip("%").replace(",", ""))
        except ValueError:
            continue
        rec["top"].append({"t": str(h.get("itemCode", "")), "name": str(h.get("itemName", "")), "pct": round(pct, 2)})
    fee = d.get("totalFee")
    if fee is not None:
        try:
            rec["expense"] = round(float(fee), 2)
        except (TypeError, ValueError):
            pass
    return rec


def fetch_holdings(tickers: list[str]) -> dict[str, dict]:
    """Best-effort: a miss leaves [] / None."""
    out: dict[str, dict] = {}
    for t in tickers:
        try:
            out[t] = fetch_holdings_kr(t) if is_kr(t) else fetch_holdings_us(t)
        except Exception as e:  # noqa: BLE001
            print(f"  holdings miss {t}: {e}", file=sys.stderr)
            out[t] = {"top": [], "expense": None}
    return out


def check_wics() -> None:
    """Warn (don't fail) when a mapped WICS name is not in the current KR sector map."""
    p = HERE / "sector_map_kr.json"
    if not p.exists():
        return
    have = set(json.loads(p.read_text(encoding="utf-8")).get("map", {}).values())
    names = {w for _, ws in KR_MAP.values() for w in ws} | {w for ws in KR_WICS.values() for w in ws}
    for w in sorted(names - have):
        print(f"  WICS not in sector_map_kr.json: {w}", file=sys.stderr)


def pool_of(t: str, g: str) -> str:
    return BENCH_POOL[t] if g == "bench" else GROUP_POOL[g]


def main() -> None:
    check_wics()
    tickers = [u[0] for u in UNIVERSE]
    names = {u[0]: (u[3] if len(u) > 3 else None) for u in UNIVERSE}
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
    for pid, p in POOLS.items():
        if p["bench"] not in closes:
            sys.exit(f"benchmark {p['bench']} ({pid}) missing")

    # per-pool dates: asOf = bench's last close, prevDate = last bench close ≤ 4 weeks before
    pool_info: dict[str, dict] = {}
    for pid, p in POOLS.items():
        b = closes[p["bench"]]
        as_of = b.index[-1]
        prev_cut = as_of - pd.Timedelta(days=RANK_LOOKBACK_DAYS)
        pool_info[pid] = {"asOf": as_of, "prevCut": prev_cut,
                          "prevDate": b[b.index <= prev_cut].index[-1]}

    now_r: dict[str, dict] = {}
    prev_r: dict[str, dict] = {}
    for u in UNIVERSE:
        t, g = u[0], u[1]
        if t not in closes:
            continue
        c = closes[t]
        now_r[t] = returns_for(c)[0]
        prev_r[t] = returns_for(c[c.index <= pool_info[pool_of(t, g)]["prevCut"]])[0]

    rank_now: dict[str, int | None] = {}
    rank_prev: dict[str, int | None] = {}
    rank_count: dict[str, int] = {}
    for pid, p in POOLS.items():
        b = p["bench"]
        members = [u[0] for u in UNIVERSE if u[1] != "bench" and GROUP_POOL[u[1]] == pid and u[0] in closes]
        rn = ranks({t: rel(now_r[t][RANK_PERIOD], now_r[b][RANK_PERIOD]) for t in members})
        rp = ranks({t: rel(prev_r[t][RANK_PERIOD], prev_r[b][RANK_PERIOD]) for t in members})
        rank_now.update(rn)
        rank_prev.update(rp)
        rank_count[pid] = sum(1 for v in rn.values() if v is not None)

    holdings = fetch_holdings([t for t in tickers if t in closes])
    print(f"  holdings: {sum(1 for h in holdings.values() if h['top'])}/{len(holdings)} ETFs", file=sys.stderr)

    # reverse of KR_MAP: 국내 ETF → 대응 미국 ETF 목록 (US sector first, then global)
    us_peers: dict[str, list[str]] = {}
    for us_t, (kr_t, _) in KR_MAP.items():
        if kr_t:
            us_peers.setdefault(kr_t, []).append(us_t)

    etfs = []
    for u in UNIVERSE:
        t, g, label = u[0], u[1], u[2]
        if t not in closes:
            continue
        pid = pool_of(t, g)
        b = POOLS[pid]["bench"]
        c = closes[t]
        ma50 = c.rolling(50).mean().iloc[-1]
        dev50 = round((c.iloc[-1] / ma50 - 1) * 100, 1) if pd.notna(ma50) and ma50 > 0 else None
        r = {p: now_r[t][p] for p in PERIODS}
        rn, rp = rank_now.get(t), rank_prev.get(t)
        if g == "bench":
            peer = None
        elif pid == "kr":
            peer = {"etf": us_peers.get(t, []), "wics": KR_WICS.get(t, [])}
        else:
            kr_t, wics = KR_MAP.get(t, (None, []))
            peer = {"etf": [kr_t] if kr_t and kr_t in closes else [], "wics": wics}
        etfs.append({
            "ticker": t,
            "label": label,
            "name": names[t],
            "group": g,
            "pool": pid,
            "bench": None if t == b else b,
            "market": "kr" if is_kr(t) else "us",
            "price": round(float(c.iloc[-1]), 2),
            "lastDate": c.index[-1].strftime("%Y-%m-%d"),
            "r": r,
            "rel": {p: (None if t == b else rel(r[p], now_r[b][p])) for p in PERIODS},
            "rank": rn,
            "rankPrev": rp,
            "rankChg": (rp - rn) if rn is not None and rp is not None else None,  # + = 순위 상승
            "dev50": dev50,
            "holdings": holdings.get(t, {}).get("top", []),
            "expense": holdings.get(t, {}).get("expense"),
            "peer": peer,
        })

    pools_out = {pid: {"bench": p["bench"], "benchLabel": p["benchLabel"], "label": p["label"],
                       "asOf": pool_info[pid]["asOf"].strftime("%Y-%m-%d"),
                       "prevDate": pool_info[pid]["prevDate"].strftime("%Y-%m-%d"),
                       "rankCount": rank_count[pid]}
                 for pid, p in POOLS.items()}
    payload = {
        # top-level asOf/prevDate/benchmark/rankCount = US pool (kept for the hub / older readers)
        "asOf": pools_out["us"]["asOf"],
        "prevDate": pools_out["us"]["prevDate"],
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark": POOLS["us"]["bench"],
        "rankPeriod": RANK_PERIOD,
        "rankCount": rank_count["us"],
        "returnBasis": "price return, split-adjusted, dividends excluded",
        "groups": GROUPS,
        "groupPool": GROUP_POOL,
        "pools": pools_out,
        "etfs": etfs,
    }

    # chart series: one shared date axis per pool (US and KR trading calendars differ),
    # same compact {o, c[]} format as series-us.js. Pool "us" is the top-level
    # dates/series; pool "kr" sits under "kr" so series-etf.js stays one file.
    def pool_series(pid: str) -> tuple[list[str], dict]:
        members = [u[0] for u in UNIVERSE if u[0] in closes and pool_of(u[0], u[1]) == pid]
        axis_idx = closes[POOLS[pid]["bench"]].index[-SERIES_DAYS:]
        out = {}
        for t in members:
            cc = closes[t].reindex(axis_idx)
            if not cc.notna().any():
                continue
            first = int(cc.notna().values.argmax())
            out[t] = {"o": first, "c": [round(float(v), 4) if pd.notna(v) else None for v in cc.iloc[first:]]}
        return [d.strftime("%Y-%m-%d") for d in axis_idx], out

    us_axis, us_ser = pool_series("us")
    kr_axis, kr_ser = pool_series("kr")
    series_payload = {"asOf": pools_out["us"]["asOf"], "dates": us_axis, "series": us_ser,
                      "kr": {"asOf": pools_out["kr"]["asOf"], "dates": kr_axis, "series": kr_ser}}
    (HERE / "series-etf.js").write_text(
        "window.SCREENER_SERIES_ETF = " + json.dumps(series_payload, ensure_ascii=False, separators=(",", ":")) + ";",
        encoding="utf-8")

    (HERE / "data-etf.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (HERE / "data-etf.js").write_text(
        "window.SCREENER_DATA_ETF = " + json.dumps(payload, ensure_ascii=False) + ";", encoding="utf-8")
    print(f"wrote {len(etfs)} ETFs to data-etf.json "
          f"(US asOf {pools_out['us']['asOf']} rank vs {pools_out['us']['prevDate']} · "
          f"KR asOf {pools_out['kr']['asOf']} rank vs {pools_out['kr']['prevDate']})", file=sys.stderr)


if __name__ == "__main__":
    main()
