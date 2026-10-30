"""Daily "same charts, what changed?" scanner -> groups-<m>.json + setups-<m>.json.

Two products per market (--market us|kr):

1. 그룹판 (groups-<m>.json): one card per index / industry group / sector ETF —
   2-year weekly closes + 30-week line, trend badges on three clocks
   (일 = 50-day line, 주 = 30-week line, 월 = 10-month line: up / down / mixed),
   whether each badge changed versus the previous trading day, 1M return,
   3M return relative to the market index (%p), distance from the 52-week high.
     KR groups = WICS sectors (sector_map_kr.json) folded into ~17 themes, each a
              cap-weighted chain-linked index of the screener universe's constituents
              (shares held fixed at today's marketCap/price, so the weight of each
              name floats with its price like a real cap-weighted index).
     US groups = the US sector / industry ETFs of compute_etf.py (SPY-pool), with n =
              number of universe stocks mapped to that ETF (sector/industry table below).
2. 셋업 (setups-<m>.json): price-action setups scanned over the same OHLCV that
   compute_returns.py downloaded (.cache/ohlcv-<m>.pkl, so no second bulk download):
     일봉  gap_up        거래량 터진 갭 상승
          ma200_cross   200일선 돌파
          base_breakout 베이스 돌파
          high_26w      26주 신고가 (52주 신고가는 주봉 쪽으로)
     주봉  bottom_escape 바닥 탈출 — 최근 26주 중 ≥20주 30주선 아래 → 30주선과 12주 고점 동시 돌파
          ma30w_reclaim 30주선 리클레임
          high_52w      52주 신고가
     월봉  ma10m_cross   10개월선 돌파
   Weekly / monthly conditions are evaluated EVERY DAY on daily-updated bars (the
   running week/month is the last bar), so a weekly setup can be "new today".
   An event found inside its lookback window stays listed while its hold rule
   is still true (e.g. still above the 30-week line). setups-history-<m>.json
   remembers when each (ticker, scan, event) was first seen so the UI can split
   "오늘 새로" from "그 밖에". setups-events-<m>.json is the permanent append-only log of
   every hit (entry = close on the day the card was first seen) with 5/10/20-day
   outcomes, and setups-stats-<m>.json the per-scan summary for the '스캔별 성과' panel. Hit tickers get 2y daily / 5y weekly / 10y monthly
   bars (weekly+monthly are extra yfinance calls for hits only), an RS line vs
   the market index, RS scores from data-<m>.json and (US) the next earnings date.

Thresholds live in THRESH / windows in WINDOW — meant to be tuned; the run log
prints how many cards each scan produced.

    uv run python scanner.py --market kr
    uv run python scanner.py --market us --no-extra   # skip weekly/monthly/earnings fetches (fast local check)
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

import compute_etf
from compute_returns import (CACHE_DIR, LOOKBACK_DAYS, MARKET_META, download_chunk, is_halted)

HERE = Path(__file__).parent

# ----------------------------------------------------------------------------- tunables
THRESH = {
    "gap_pct": 5.0,          # 시가 갭 ≥ +5% (전일 종가 대비)
    "gap_volx": 3.0,         # 거래량 ≥ 20일 평균 3배
    "gap_close_pos": 2 / 3,  # 종가가 당일 범위 상위 1/3
    "cross_volx": 1.5,       # 200일선 돌파 · 베이스 돌파 거래량 배수
    "base_days": 20,         # 베이스 = 직전 20일 고가
    "base_range_max": 25.0,  # 베이스 폭 (고가/저가 − 1) ≤ 25% — 더 넓으면 베이스가 아님
    "high26_volx": 1.0,      # 26주 신고가는 평균 이상 거래량이면 충분
    "bottom_below_weeks": 20,  # 최근 26주 중 30주선 아래였던 주 ≥ 20
    "reclaim_below_weeks": 4,  # 30주선 리클레임: 직전 8주 중 ≥ 4주 아래
    "month_below_months": 3,   # 10개월선 돌파: 직전 6개월 중 ≥ 3개월 아래
    "min_bars": 260,         # 1년 미만 상장은 스캔하지 않음 (52주 고점·26주 비교 불가)
}
WINDOW = {"d": 10, "w": 15, "m": 22}   # 이벤트를 찾는 되돌아보기 거래일 수 (일봉 / 주봉 / 월봉 스캔)
# 카드 정책(2026-10-30): 스캔은 넓게 잡아 전부 기록하되(성과 추적용), 카드는 좁게 보여준다.
#   · 핵심 4스캔(CORE)에 걸린 종목만 카드, 종목당 1장(대표 스캔 = CORE 순서), 나머지 스캔은 태그.
#   · 이평 돌파 3종은 카드 없이 '그룹 × 스캔 군집 표'와 한 줄 리스트로만.
#   · 차트(2y 일봉·5y 주봉·10y 월봉)는 CHART_CAP 종목까지: 오늘 새로 → 기본 필터 통과 → RS점수 순. 나머지는 한 줄 리스트.
#   · 기본 필터(UI 토글, 여기 상수를 json으로 전달): 그룹판 3개월 상대 상위 5 그룹 ∩ RS점수 ≥ RS_FLOOR
#     (바닥 탈출은 RS 면제 대신 이번 주 거래량 ≥ BOTTOM_VOLX 배).
CORE = ["bottom_escape", "gap_up", "base_breakout", "high_52w"]
CHART_CAP = 90
RS_FLOOR = 70
BOTTOM_VOLX = 1.5
TOP_GROUPS = 5
HISTORY_DAYS = 20                      # setups-history 보관 거래일 수
DAILY_BARS = 504                       # 2년 일봉
WEEKLY_YEARS, MONTHLY_YEARS = 5, 10
GROUP_WEEKS = 105                      # 그룹판 2년 주봉 + 진행 중인 주
BENCH = {"us": "SPY", "kr": "^KS11"}   # RS 라인 · 3개월 상대수익 기준
INDEX_CARDS = {
    "us": [("SPY", "S&P 500"), ("QQQ", "나스닥100"), ("IWM", "러셀2000"), ("RSP", "S&P 500 동일가중")],
    "kr": [("^KS11", "코스피"), ("^KQ11", "코스닥")],
}

SCANS = [  # id, timeframe, label, one-line rule, how Jay uses it (shown under the scan title)
    ("gap_up", "d", "거래량 터진 갭 상승", f"시가 갭 ≥ +{THRESH['gap_pct']:.0f}% · 거래량 ≥ 20일 평균 {THRESH['gap_volx']:.0f}배 · 고가권(상위 1/3) 마감",
     "갭은 정보지 매수 신호가 아니다. 갭 당일 추격 금지 → 갭일 저가 지키며 3~10일 횡보 → 그 고점 돌파에 진입, 손절 = 갭일 저가. 52주 고점 −5% 이내에서 터진 갭이 바닥 갭보다 낫다."),
    ("ma200_cross", "d", "200일선 돌파", f"어제 200일선 아래 → 오늘 위 · 거래량 ≥ {THRESH['cross_volx']}배",
     "단독 매수 근거 아님. 같은 그룹에서 같은 날 3장 이상 뜨면 섹터 전환 신호 → 그룹판에서 칩이 따라오는지 확인. 혼자 뜬 한 장은 무시."),
    ("base_breakout", "d", "베이스 돌파", f"{THRESH['base_days']}일 고가 돌파 · 베이스 폭 ≤ {THRESH['base_range_max']:.0f}% · 거래량 ≥ {THRESH['cross_volx']}배 · 50일선 위",
     "오닐 피봇. '피봇 위 +N%'가 5%를 넘으면 끝난 카드(추격 금지). 거래량은 본인 기준 2배 이상만. 손절 = 피봇 −8%, 매수와 동시에 시세포착 주문."),
    ("high_26w", "d", "26주 신고가", "26주 종가 신고가(어제는 아님) · 평균 이상 거래량 · 52주 신고가면 주봉 쪽",
     "신규 진입 리스트가 아니라 보유 점검용. 내 보유 종목이 아직 리더인가, 리더 명단이 바뀌었나."),
    ("bottom_escape", "w", "바닥 탈출 · 30주선과 12주 고점 돌파", f"최근 26주 중 ≥{THRESH['bottom_below_weeks']}주 30주선 아래 → 이번 주 종가가 30주선과 12주 고점을 동시에 상회",
     "가장 값진 스캔(2단계 시작, 가장 길게 탈 수 있는 자리). RS는 낮은 게 정상이니 RS≥90 대신 '그룹판 상위 그룹 소속 + 이번 주 거래량 1.5배 이상'을 요구. 거래량 없는 바닥 탈출은 반등일 뿐. 손절 = 30주선 종가 이탈."),
    ("ma30w_reclaim", "w", "30주선 리클레임", f"직전 8주 중 ≥{THRESH['reclaim_below_weeks']}주 30주선 아래 → 30주선 위로 복귀",
     "추세 전환 관찰용. 군집으로만 쓴다 — 같은 그룹 여러 장이면 섹터가 도는 중, 한 장이면 노이즈."),
    ("high_52w", "w", "52주 신고가", "52주 종가 신고가(어제는 아님)",
     "보유 점검 + 리더 명단. 여기서 새로 사지 않는다(추격). 보유 종목이 이 목록에서 빠지면 50일선·RS 룰 점검."),
    ("ma10m_cross", "m", "10개월선 돌파", f"직전 6개월 중 ≥{THRESH['month_below_months']}개월 10개월선 아래 → 10개월선 위로",
     "가장 느린 시계. 단독 진입 X. 바닥 탈출·리클레임과 겹치는 종목은 긴 추세의 시작일 수 있으니 후보 그룹에 담아 관찰."),
]
SCAN_TF = {s[0]: s[1] for s in SCANS}

# ----------------------------------------------------------------------------- group tables
# KR: WICS 소분류 → 그룹판 카드. 여기 없는 WICS는 '기타'.
KR_GROUPS: dict[str, list[str]] = {
    "반도체": ["반도체와반도체장비"],
    "IT하드웨어·전자부품": ["전자장비와기기", "디스플레이장비및부품", "디스플레이패널", "핸드셋", "통신장비",
                     "컴퓨터와주변기기", "전자제품", "사무용전자제품"],
    "2차전지·전기장비": ["전기제품", "전기장비"],
    "바이오·헬스케어": ["제약", "생물공학", "건강관리장비와용품", "생명과학도구및서비스", "건강관리업체및서비스", "건강관리기술"],
    "인터넷·소프트웨어": ["소프트웨어", "IT서비스", "양방향미디어와서비스", "인터넷과카탈로그소매"],
    "기계·산업재": ["기계", "우주항공과국방", "상업서비스와공급품", "복합기업", "무역회사와판매업체"],
    "건설·건자재": ["건설", "건축자재", "건축제품"],
    "철강·비철금속": ["철강", "비철금속"],
    "화학·소재": ["화학", "포장재", "종이와목재"],
    "에너지·유틸리티": ["석유와가스", "에너지장비및서비스", "가스유틸리티", "전기유틸리티", "복합유틸리티"],
    "자동차": ["자동차", "자동차부품"],
    "조선·운송": ["조선", "해운사", "항공사", "항공화물운송과물류", "도로와철도운송", "운송인프라"],
    "화장품·의류·생활용품": ["화장품", "섬유,의류,신발,호화품", "가정용기기와용품", "가정용품", "가구", "레저용장비와제품", "문구류"],
    "음식료·유통": ["식품", "음료", "담배", "식품과기본식료품소매", "백화점과일반상점", "판매업체", "전문소매"],
    "미디어·엔터·게임": ["방송과엔터테인먼트", "게임엔터테인먼트", "광고", "출판", "무선통신서비스", "다각화된통신서비스"],
    "금융": ["은행", "증권", "손해보험", "생명보험", "기타금융", "창업투자", "카드", "부동산"],
    "레저·교육·서비스": ["호텔,레스토랑,레저", "교육서비스", "다각화된소비자서비스"],
}
KR_WICS_TO_GROUP = {w: g for g, ws in KR_GROUPS.items() for w in ws}

# US: Nasdaq-screener (sector, industry) → ETF card. Industry overrides first, then sector.
US_INDUSTRY_ETF: list[tuple[str, str]] = [  # (substring of industry, ETF)
    ("Semiconductor", "SMH"), ("Computer Software", "IGV"), ("EDP Services", "IGV"),
    ("Biotechnology", "IBB"), ("Homebuilding", "ITB"), ("Precious Metals", "XME"), ("Metal Mining", "XME"),
    ("Steel", "XME"), ("Aluminum", "XME"), ("Coal", "XME"), ("Oil & Gas Production", "XOP"),
    ("Aerospace", "ITA"), ("Military", "ITA"), ("Commercial Banks", "KRE"), ("Savings Institutions", "KRE"),
    ("Retail", "XRT"), ("Stores", "XRT"), ("Engineering & Construction", "PAVE"), ("Building Materials", "PAVE"),
    ("Air Freight", "IYT"), ("Trucking", "IYT"), ("Railroads", "IYT"),
]
US_SECTOR_ETF = {
    "Technology": "XLK", "Health Care": "XLV", "Finance": "XLF", "Consumer Discretionary": "XLY",
    "Industrials": "XLI", "Energy": "XLE", "Basic Materials": "XLB", "Consumer Staples": "XLP",
    "Utilities": "XLU", "Real Estate": "XLRE", "Telecommunications": "XLC",
}


def us_group_of(sector: str, industry: str) -> str | None:
    ind = industry or ""
    for sub, etf in US_INDUSTRY_ETF:
        if sub.lower() in ind.lower():
            return etf
    return US_SECTOR_ETF.get(sector)


def kr_group_of(sector: str) -> str:
    return KR_WICS_TO_GROUP.get(sector, "기타")


# ----------------------------------------------------------------------------- helpers
def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def rnd(v, nd: int = 4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float):
        v = round(v, nd)
        return int(v) if v == int(v) and abs(v) < 1e15 else v
    return v


def arr(a, nd: int = 4) -> list:
    return [rnd(x, nd) for x in np.asarray(a, dtype="float64")]


def pct(a: float, b: float) -> float | None:
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or b <= 0:
        return None
    return round((a / b - 1) * 100, 2)


def asof_return(close: pd.Series, offset) -> float | None:
    """Return from the last close ≤ (last_date − offset) to the latest close."""
    if len(close) < 2:
        return None
    target = close.index[-1] - offset
    if target < close.index[0]:
        return None
    past = close.asof(target)
    return pct(float(close.iloc[-1]), float(past)) if pd.notna(past) else None


def weekly(close: pd.Series) -> pd.Series:
    return close.resample("W-FRI").last().dropna()


def monthly(close: pd.Series) -> pd.Series:
    return close.resample("ME").last().dropna()


def badge(close: pd.Series, n: int, slope_k: int) -> str:
    """up = 이평 위 & 이평 상승, down = 이평 아래 & 이평 하락, else mixed(엇갈림)."""
    ma = close.rolling(n).mean().dropna()
    if len(ma) < slope_k + 1:
        return "mixed"
    c, m, mp = float(close.iloc[-1]), float(ma.iloc[-1]), float(ma.iloc[-1 - slope_k])
    above, rising = c > m, m > mp
    if above and rising:
        return "up"
    if not above and not rising:
        return "down"
    return "mixed"


def badges_for(close: pd.Series) -> dict[str, str]:
    return {"d": badge(close, 50, 5), "w": badge(weekly(close), 30, 1), "m": badge(monthly(close), 10, 1)}


def download_frames(tickers: list[str], start: str, interval: str = "1d") -> dict[str, pd.DataFrame]:
    """OHLCV frames for a list of tickers (chunked, best effort). Close is split-adjusted, not dividend-adjusted."""
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        df = None
        for attempt in range(3):
            try:
                df = yf.download(chunk, start=start, interval=interval, auto_adjust=False, actions=False,
                                 group_by="ticker", threads=True, progress=False)
                if df is not None and not df.empty:
                    break
            except Exception as e:  # noqa: BLE001
                log(f"  download attempt {attempt} failed: {e}")
            time.sleep(2 * (attempt + 1))
        if df is None or df.empty:
            continue
        for t in chunk:
            # yfinance(group_by="ticker")는 종목이 하나여도 (ticker, field) 2단 컬럼을 준다 → 단일 종목(벤치마크)도 level-0으로 꺼낸다.
            # (예전 single=True 경로는 2단 컬럼을 그대로 돌려줘 "Close"를 못 찾았고, 그래서 RS 라인·벤치마크가 조용히 빠졌다.)
            if isinstance(df.columns, pd.MultiIndex):
                sub = df[t] if t in df.columns.get_level_values(0) else None
            else:
                sub = df if len(chunk) == 1 else None
            if sub is None or "Close" not in sub:
                continue
            sub = sub[[c for c in ("Open", "High", "Low", "Close", "Volume") if c in sub]].dropna(subset=["Close"])
            if not sub.empty:
                out[t] = sub.astype("float64")
    return out


# ----------------------------------------------------------------------------- 그룹판
def chain_index(frames: dict[str, pd.DataFrame], caps: dict[str, float]) -> pd.Series | None:
    """Cap-weighted, chain-linked daily index (base 100) of the given constituents.
    Weight of each name on day t = its market cap at t−1 (shares fixed at latest cap/price)."""
    closes = pd.DataFrame({t: frames[t]["Close"] for t in frames if t in caps and caps[t] > 0}).sort_index()
    if closes.empty:
        return None
    last = closes.ffill().iloc[-1]
    shares = pd.Series({t: caps[t] / last[t] for t in closes.columns if pd.notna(last[t]) and last[t] > 0})
    closes = closes[shares.index]
    cap = closes * shares
    ret = closes / closes.shift(1) - 1
    w = cap.shift(1)
    valid = ret.notna() & w.notna()
    num = (ret.where(valid) * w.where(valid)).sum(axis=1)
    den = w.where(valid).sum(axis=1)
    r = (num / den).where(den > 0)
    r = r[den > 0]
    if r.empty:
        return None
    return 100.0 * (1.0 + r.fillna(0.0)).cumprod()


def group_card(gid: str, name: str, kind: str, close: pd.Series, bench: pd.Series | None,
               n: int | None = None, ticker: str | None = None) -> dict | None:
    close = close.dropna()
    if len(close) < 60:
        return None
    wk = weekly(close)
    wk_ma = wk.rolling(30).mean()
    wk_tail, ma_tail = wk.tail(GROUP_WEEKS), wk_ma.tail(GROUP_WEEKS)
    now = badges_for(close)
    prev = badges_for(close.iloc[:-1])
    r3 = asof_return(close, pd.DateOffset(months=3))
    b3 = asof_return(bench, pd.DateOffset(months=3)) if bench is not None else None
    rel3 = round(r3 - b3, 1) if r3 is not None and b3 is not None and bench is not close else None
    hi52 = float(close.tail(252).max())
    return {
        "id": gid, "name": name, "kind": kind, "ticker": ticker, "n": n,
        "last": rnd(float(close.iloc[-1])), "lastDate": close.index[-1].strftime("%Y-%m-%d"),
        "chg1D": pct(float(close.iloc[-1]), float(close.iloc[-2])),
        "r1M": asof_return(close, pd.DateOffset(months=1)),
        "r3M": r3, "rel3M": rel3,
        "hi52": pct(float(close.iloc[-1]), hi52),
        "badges": now, "prev": prev, "changed": {k: now[k] != prev[k] for k in now},
        "wk": {"start": wk_tail.index[0].strftime("%Y-%m-%d"), "end": wk_tail.index[-1].strftime("%Y-%m-%d"),
               "dates": [d.strftime("%Y-%m-%d") for d in wk_tail.index],
               "c": arr(wk_tail.values), "ma": arr(ma_tail.values)},
    }


def build_groups(market: str, ohlcv: dict[str, pd.DataFrame], universe: list[dict], as_of: str) -> dict:
    start = (pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1100)).strftime("%Y-%m-%d")
    idx_tickers = [t for t, _ in INDEX_CARDS[market]]
    if market == "us":
        etf_rows = [u for u in compute_etf.UNIVERSE if u[1] != "bench" and compute_etf.GROUP_POOL[u[1]] == "us"]
        extra = ["ACWI"]
    else:
        etf_rows = [u for u in compute_etf.UNIVERSE if u[1] == "kr-sector"]
        extra = []
    tickers = list(dict.fromkeys(idx_tickers + [u[0] for u in etf_rows] + extra))
    log(f"  groups: downloading {len(tickers)} index/ETF series since {start}")
    frames = download_frames(tickers, start)
    # 데이터 기준일(as_of) 이후 봉은 버린다 — 장중에 돌려도 카드가 종목 리스트·셋업과 같은 날짜로 맞춰진다.
    cut = pd.Timestamp(as_of)
    frames = {t: f[f.index <= cut] for t, f in frames.items() if (f.index <= cut).any()}
    bench_t = BENCH[market]
    bench = frames[bench_t]["Close"] if bench_t in frames else None
    if bench is None:
        sys.exit(f"benchmark {bench_t} missing")

    cards: list[dict] = []
    for t, name in INDEX_CARDS[market]:
        if t in frames:
            c = group_card(t, name, "index", frames[t]["Close"], bench, ticker=t)
            if c:
                cards.append(c)

    group_of_stock: dict[str, str] = {}
    if market == "kr":
        members: dict[str, list[str]] = {}
        caps = {u["ticker"]: float(u.get("marketCap") or 0) for u in universe}
        for u in universe:
            g = kr_group_of(u.get("sector") or "")
            group_of_stock[u["ticker"]] = g
            members.setdefault(g, []).append(u["ticker"])
        order = list(KR_GROUPS) + ["기타"]
        for g in order:
            ts = [t for t in members.get(g, []) if t in ohlcv]
            if len(ts) < 3:
                continue
            idx = chain_index({t: ohlcv[t] for t in ts}, caps)
            if idx is None:
                continue
            c = group_card("kr:" + g, g, "group", idx, bench, n=len(ts))
            if c:
                c["wics"] = KR_GROUPS.get(g, [])
                cards.append(c)
        for t, g, label, *rest in etf_rows:
            if t in frames:
                c = group_card(t, label, "etf", frames[t]["Close"], bench, ticker=t)
                if c:
                    c["fund"] = rest[0] if rest else None
                    cards.append(c)
    else:
        counts: dict[str, int] = {}
        for u in universe:
            g = us_group_of(u.get("sector") or "", u.get("industry") or "")
            if g:
                group_of_stock[u["ticker"]] = g
                counts[g] = counts.get(g, 0) + 1
        for t, g, label, *rest in etf_rows:
            if t not in frames:
                continue
            kind = "group" if g in ("us-sector", "us-industry") else "etf"
            c = group_card(t, label, kind, frames[t]["Close"], bench, n=counts.get(t) if kind == "group" else None, ticker=t)
            if c:
                c["etfGroup"] = g
                cards.append(c)
        if "ACWI" in frames:
            c = group_card("ACWI", "MSCI 전세계", "etf", frames["ACWI"]["Close"], bench, ticker="ACWI")
            if c:
                c["etfGroup"] = "global-sector"
                cards.append(c)

    changed = sum(1 for c in cards for k in ("d", "w", "m") if c["changed"][k])
    log(f"  groups: {len(cards)} cards ({sum(1 for c in cards if c['kind']=='index')} index · "
        f"{sum(1 for c in cards if c['kind']=='group')} group · {sum(1 for c in cards if c['kind']=='etf')} etf) · "
        f"{changed} badge changes vs yesterday")
    return {
        "market": market, "asOf": as_of, "bench": bench_t,
        "benchLabel": "코스피" if market == "kr" else "S&P 500 (SPY)",
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weeks": GROUP_WEEKS,
        "legend": "일=50일선 · 주=30주선 · 월=10개월선 · ▲ 이평 위·이평 상승 / ▼ 이평 아래·이평 하락 / 엇갈림 · 3개월 상대 = 그룹 3M − 지수 3M (%p)",
        "groupOf": group_of_stock if market == "kr" else {},   # KR: ticker → group id (US uses the sector table client-side via stock.group)
        "cards": cards,
    }, group_of_stock


# ----------------------------------------------------------------------------- 셋업 스캔
def _week_map(series: pd.Series, idx: pd.DatetimeIndex) -> np.ndarray:
    """Align a week-indexed (PeriodIndex) series onto trading days."""
    return series.reindex(idx.to_period("W-FRI")).to_numpy(dtype="float64")


def _month_map(series: pd.Series, idx: pd.DatetimeIndex) -> np.ndarray:
    return series.reindex(idx.to_period("M")).to_numpy(dtype="float64")


def scan_frame(df: pd.DataFrame) -> dict[str, dict]:
    """Return {scan_id: {eventDate, stats}} for one ticker (empty if nothing)."""
    n = len(df)
    if n < THRESH["min_bars"]:
        return {}
    idx = df.index
    o, h, l, c, v = (df[k].to_numpy(dtype="float64") for k in ("Open", "High", "Low", "Close", "Volume"))
    cs, hs, ls, vs = df["Close"], df["High"], df["Low"], df["Volume"]
    ma50 = cs.rolling(50).mean().to_numpy()
    ma200 = cs.rolling(200).mean().to_numpy()
    avg20_prev = vs.rolling(20).mean().shift(1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        volx = np.where(avg20_prev > 0, v / avg20_prev, np.nan)
        gap = np.full(n, np.nan)
        gap[1:] = (o[1:] / c[:-1] - 1) * 100
        rng = h - l
        pos = np.where(rng > 0, (c - l) / np.where(rng > 0, rng, 1), 1.0)
    hi20_prev = hs.rolling(THRESH["base_days"]).max().shift(1).to_numpy()
    lo20_prev = ls.rolling(THRESH["base_days"]).min().shift(1).to_numpy()
    hi130_prev = cs.rolling(130).max().shift(1).to_numpy()
    hi252_prev = cs.rolling(252).max().shift(1).to_numpy()

    # weekly aggregates, evaluated daily: prior COMPLETED weeks + today's close as the running week
    wk = idx.to_period("W-FRI")
    wc, wh, wv = cs.groupby(wk).last(), hs.groupby(wk).max(), vs.groupby(wk).mean()   # wv = 주별 일평균 거래량 (진행 중인 주도 공정하게)
    sma30d = (_week_map(wc.rolling(29).sum().shift(1), idx) + c) / 30
    h12_prev = _week_map(wh.rolling(12).max().shift(1), idx)
    wma30 = wc.rolling(30).mean()
    below = (wc < wma30).astype("float64").where(wma30.notna())
    below26_prev = _week_map(below.rolling(26).sum().shift(1), idx)
    below8_prev = _week_map(below.rolling(8).sum().shift(1), idx)
    wvx = _week_map(wv / wv.rolling(10).mean().shift(1), idx)   # 이번 주 일평균 ÷ 직전 10주 일평균
    # monthly
    mo = idx.to_period("M")
    mc = cs.groupby(mo).last()
    sma10d = (_month_map(mc.rolling(9).sum().shift(1), idx) + c) / 10
    mma10 = mc.rolling(10).mean()
    mbelow = (mc < mma10).astype("float64").where(mma10.notna())
    below6m_prev = _month_map(mbelow.rolling(6).sum().shift(1), idx)

    T = n - 1
    date = lambda i: idx[i].strftime("%Y-%m-%d")  # noqa: E731
    hi52_now = pct(c[T], float(np.nanmax(c[-252:])))
    hi2y_now = pct(c[T], float(np.nanmax(c[-DAILY_BARS:])))
    out: dict[str, dict] = {}

    def put(sid: str, t: int, stats: dict) -> None:
        if sid in out:        # keep the earliest still-valid event in the window
            return
        out[sid] = {"eventDate": date(t), "stats": {k: rnd(x, 2) for k, x in stats.items()}}

    def cond_be(t: int) -> bool:
        return (np.isfinite(below26_prev[t]) and below26_prev[t] >= THRESH["bottom_below_weeks"]
                and np.isfinite(sma30d[t]) and c[t] > sma30d[t] and np.isfinite(h12_prev[t]) and c[t] > h12_prev[t])

    for t in range(max(1, n - WINDOW["d"]), n):
        if (np.isfinite(gap[t]) and gap[t] >= THRESH["gap_pct"] and np.isfinite(volx[t]) and volx[t] >= THRESH["gap_volx"]
                and pos[t] >= THRESH["gap_close_pos"] and c[t] > c[t - 1] and c[T] >= l[t]):
            put("gap_up", t, {"gap": gap[t], "dayChg": (c[t] / c[t - 1] - 1) * 100, "volx": volx[t], "closePos": pos[t] * 100,
                              "prevClose": c[t - 1], "hi52": hi52_now})
        if (np.isfinite(ma200[t]) and np.isfinite(ma200[t - 1]) and c[t - 1] < ma200[t - 1] and c[t] > ma200[t]
                and np.isfinite(volx[t]) and volx[t] >= THRESH["cross_volx"] and c[T] > ma200[T]):
            put("ma200_cross", t, {"ma200": ma200[t], "dist": (c[t] / ma200[t] - 1) * 100, "volx": volx[t],
                                   "dayChg": (c[t] / c[t - 1] - 1) * 100, "hi52": hi52_now})
        if (np.isfinite(hi20_prev[t]) and c[t] > hi20_prev[t] and not (np.isfinite(hi20_prev[t - 1]) and c[t - 1] > hi20_prev[t - 1])
                and np.isfinite(volx[t]) and volx[t] >= THRESH["cross_volx"] and np.isfinite(ma50[t]) and c[t] > ma50[t]
                and np.isfinite(lo20_prev[t]) and lo20_prev[t] > 0 and (hi20_prev[t] / lo20_prev[t] - 1) * 100 <= THRESH["base_range_max"]
                and c[T] >= hi20_prev[t]):
            put("base_breakout", t, {"pivot": hi20_prev[t], "baseRange": (hi20_prev[t] / lo20_prev[t] - 1) * 100, "volx": volx[t],
                                     "dayChg": (c[t] / c[t - 1] - 1) * 100, "abovePivot": (c[T] / hi20_prev[t] - 1) * 100, "hi52": hi52_now})
        if (np.isfinite(hi130_prev[t]) and c[t] > hi130_prev[t] and not (np.isfinite(hi130_prev[t - 1]) and c[t - 1] > hi130_prev[t - 1])
                and np.isfinite(volx[t]) and volx[t] >= THRESH["high26_volx"] and c[T] >= hi130_prev[t]
                and not (np.isfinite(hi252_prev[t]) and c[t] > hi252_prev[t])):
            put("high_26w", t, {"prevHigh": hi130_prev[t], "volx": volx[t], "dayChg": (c[t] / c[t - 1] - 1) * 100, "hi52": hi52_now})

    for t in range(max(1, n - WINDOW["w"]), n):
        be = cond_be(t)
        if be and not cond_be(t - 1) and c[T] > sma30d[T]:
            put("bottom_escape", t, {"belowWeeks": below26_prev[t], "sma30": sma30d[t], "h12": h12_prev[t], "weekVolx": wvx[T],
                                     "hi2y": hi2y_now, "hi52": hi52_now})
        if (not be and np.isfinite(sma30d[t]) and np.isfinite(sma30d[t - 1]) and c[t] > sma30d[t] and c[t - 1] <= sma30d[t - 1]
                and np.isfinite(below8_prev[t]) and below8_prev[t] >= THRESH["reclaim_below_weeks"] and c[T] > sma30d[T]):
            put("ma30w_reclaim", t, {"belowWeeks8": below8_prev[t], "sma30": sma30d[t], "dist": (c[T] / sma30d[T] - 1) * 100,
                                     "weekVolx": wvx[T], "hi52": hi52_now})
        if (np.isfinite(hi252_prev[t]) and c[t] > hi252_prev[t] and not (np.isfinite(hi252_prev[t - 1]) and c[t - 1] > hi252_prev[t - 1])
                and c[T] >= hi252_prev[t]):
            put("high_52w", t, {"prevHigh": hi252_prev[t], "volx": volx[t], "weekVolx": wvx[T], "dayChg": (c[t] / c[t - 1] - 1) * 100})

    for t in range(max(1, n - WINDOW["m"]), n):
        if (np.isfinite(sma10d[t]) and np.isfinite(sma10d[t - 1]) and c[t] > sma10d[t] and c[t - 1] <= sma10d[t - 1]
                and np.isfinite(below6m_prev[t]) and below6m_prev[t] >= THRESH["month_below_months"] and c[T] > sma10d[T]):
            put("ma10m_cross", t, {"belowMonths": below6m_prev[t], "sma10m": sma10d[t], "dist": (c[T] / sma10d[T] - 1) * 100, "hi52": hi52_now})
    return out


def pack_bars(df: pd.DataFrame, bench: pd.Series | None, mas: tuple[int, ...], n_bars: int,
              shared_axis: list[str] | None = None, bar_end=None) -> dict:
    """Compact OHLCV + MA lines + RS ratio (stock/bench, start=1) for the last n_bars.
    bench is the DAILY benchmark close; bar_end shifts a weekly/monthly bar label (yfinance
    labels them by period START) to the period end before looking the benchmark up."""
    cs = df["Close"]
    out: dict = {}
    for m in mas:
        out[f"ma{m}"] = arr(cs.rolling(m).mean().tail(n_bars).to_numpy(), 2)
    tail = df.tail(n_bars)
    if shared_axis is not None:
        first = tail.index[0].strftime("%Y-%m-%d")
        out["o"] = shared_axis.index(first) if first in shared_axis else None
        if out["o"] is None:
            out["t"] = [d.strftime("%Y-%m-%d") for d in tail.index]
    else:
        out["t"] = [d.strftime("%Y-%m-%d") for d in tail.index]
    for k, col in (("O", "Open"), ("H", "High"), ("L", "Low"), ("C", "Close"), ("V", "Volume")):
        out[k] = arr(tail[col].to_numpy(), 0 if col == "Volume" else 2)
    if bench is not None:
        at = tail.index + bar_end if bar_end is not None else tail.index
        b = pd.Series(bench.reindex(at, method="ffill").to_numpy(), index=tail.index)
        rs = tail["Close"] / b
        rs = rs / rs.dropna().iloc[0] if rs.notna().any() else rs
        out["rs"] = arr(rs.to_numpy(), 3)
    return out


def fetch_earnings(tickers: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for t in tickers:
        try:
            cal = yf.Ticker(t).calendar or {}
            ds = cal.get("Earnings Date") or []
            out[t] = ds[0].strftime("%Y-%m-%d") if ds else None
        except Exception:  # noqa: BLE001
            out[t] = None
    return out


def load_history(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {"runs": {}}


def top_groups(groups: dict | None) -> set[str]:
    """Group ids (KR: group name, US: ETF ticker) of the TOP_GROUPS best 3M-relative groups — same rule as the UI toggle."""
    if not groups:
        return set()
    cards = [c for c in groups["cards"] if c["kind"] == "group" and c.get("rel3M") is not None]
    cards.sort(key=lambda c: -c["rel3M"])
    return {c["id"].replace("kr:", "", 1) for c in cards[:TOP_GROUPS]}


def build_setups(market: str, ohlcv: dict[str, pd.DataFrame], stocks: list[dict], group_of: dict[str, str],
                 as_of: str, extra: bool, top5: set[str] | None = None) -> dict:
    t0 = time.time()
    meta = {s["ticker"]: s for s in stocks}
    hits: dict[str, dict[str, dict]] = {}
    scanned = 0
    for t, df in ohlcv.items():
        if t not in meta:
            continue
        s = meta[t]
        if not s.get("marketCap") or df.index[-1].strftime("%Y-%m-%d") != as_of:
            continue                      # no bar on the as-of date (halt / delisting) or no cap → skip
        if is_halted(df["Close"]):
            continue
        scanned += 1
        res = scan_frame(df)
        if res:
            hits[t] = res
    log(f"  setups: scanned {scanned} tickers in {time.time()-t0:.1f}s → {len(hits)} tickers with ≥1 hit")

    # first-seen bookkeeping (history keeps the last HISTORY_DAYS runs)
    hist_path = HERE / f"setups-history-{market}.json"
    hist = load_history(hist_path)
    runs: dict[str, list] = hist.get("runs", {})
    first_run = not runs
    first_seen: dict[tuple[str, str, str], str] = {}
    for day in sorted(runs):
        for rec in runs[day]:
            key = (rec[0], rec[1], rec[2])
            first_seen.setdefault(key, rec[3] if len(rec) > 3 else day)
    today_recs = []
    for t, res in hits.items():
        for sid, r in res.items():
            key = (t, sid, r["eventDate"])
            fs = first_seen.get(key) or (r["eventDate"] if first_run else as_of)
            r["firstSeen"] = fs
            r["isNew"] = fs == as_of
            today_recs.append([t, sid, r["eventDate"], fs])
    runs[as_of] = today_recs
    for day in sorted(runs)[:-HISTORY_DAYS]:
        del runs[day]
    hist_path.write_text(json.dumps({"market": market, "asOf": as_of, "runs": runs}, ensure_ascii=False, separators=(",", ":")),
                         encoding="utf-8")

    # representative scan per ticker (CORE order) → card; non-core-only tickers → cluster table / list only
    rep_of: dict[str, str | None] = {t: next((c for c in CORE if c in res), None) for t, res in hits.items()}
    scan_entries: list[dict] = []
    for sid, tf, label, rule, use in SCANS:
        members = sorted((t for t in hits if sid in hits[t]), key=lambda t: -(meta[t].get("score") or 0))
        new_n = sum(1 for t in members if hits[t][sid]["isNew"])
        scan_entries.append({"id": sid, "tf": tf, "label": label, "rule": rule, "use": use, "core": sid in CORE,
                             "total": len(members), "new": new_n, "hits": members})
        log(f"    {tf} {sid:14s} total {len(members):4d} · new {new_n:3d}{'  (card)' if sid in CORE else ''}")

    def qualifies(t: str) -> bool:
        """Default UI filter: top-5 group ∩ (RS ≥ floor, or bottom-escape with weekly volume ≥ BOTTOM_VOLX)."""
        if top5 and group_of.get(t) not in top5:
            return False
        r = rep_of[t]
        if r == "bottom_escape":
            wv = hits[t][r]["stats"].get("weekVolx")
            return wv is not None and wv >= BOTTOM_VOLX
        return (meta[t].get("score") or 0) >= RS_FLOOR

    card_tickers = [t for t in hits if rep_of[t]]
    card_tickers.sort(key=lambda t: (not hits[t][rep_of[t]]["isNew"], not qualifies(t), -(meta[t].get("score") or 0)))
    shown_tickers: set[str] = set(card_tickers[:CHART_CAP])
    n_new = sum(1 for t in card_tickers if hits[t][rep_of[t]]["isNew"])
    n_q = sum(1 for t in card_tickers if hits[t][rep_of[t]]["isNew"] and qualifies(t))
    log(f"  setups: {len(hits)} hit tickers → {len(card_tickers)} core (new {n_new}, new∩filter {n_q}) · charts for {len(shown_tickers)} (cap {CHART_CAP})")

    # bars for the shown tickers
    bench_t = BENCH[market]
    bstart = (pd.Timestamp.now(tz="UTC").normalize() - pd.DateOffset(years=MONTHLY_YEARS + 1)).strftime("%Y-%m-%d")
    bframes = download_frames([bench_t], bstart)
    bench = bframes[bench_t]["Close"] if bench_t in bframes else None
    # 성과 추적: 영구 이벤트 로그 + 사후 수익률 + 스캔별 집계 (setups-events / setups-stats)
    track_events(market, hits, meta, group_of, ohlcv, bench, as_of, runs)
    axis_set: set = set()
    for t in shown_tickers:
        axis_set.update(ohlcv[t].tail(DAILY_BARS).index)
    axis = [d.strftime("%Y-%m-%d") for d in sorted(axis_set)]
    weekly_frames: dict[str, pd.DataFrame] = {}
    monthly_frames: dict[str, pd.DataFrame] = {}
    earnings: dict[str, str | None] = {}
    if extra and shown_tickers:
        tl = sorted(shown_tickers)
        t1 = time.time()
        weekly_frames = download_frames(tl, (pd.Timestamp.now(tz="UTC").normalize() - pd.DateOffset(years=WEEKLY_YEARS + 1)).strftime("%Y-%m-%d"), "1wk")
        monthly_frames = download_frames(tl, (pd.Timestamp.now(tz="UTC").normalize() - pd.DateOffset(years=MONTHLY_YEARS + 1)).strftime("%Y-%m-%d"), "1mo")
        log(f"  setups: weekly {len(weekly_frames)} · monthly {len(monthly_frames)} frames for {len(tl)} hits in {time.time()-t1:.1f}s")
        if market == "us":
            t1 = time.time()
            earnings = fetch_earnings(tl)
            log(f"  setups: earnings dates for {sum(1 for v in earnings.values() if v)}/{len(tl)} in {time.time()-t1:.1f}s")

    stocks_out: dict[str, dict] = {}
    for t in sorted(hits):
        s, df = meta[t], ohlcv[t]
        res = hits[t]
        rec = {
            "ticker": t, "name": s["name"], "sector": s.get("sector"), "industry": s.get("industry"),
            "marketCap": s.get("marketCap"), "krxCode": s.get("krxCode"), "subMarket": s.get("subMarket"),
            "price": s.get("price"), "chg1D": pct(float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])),
            "score": s.get("score"), "scoreShort": s.get("scoreShort"), "dev": s.get("dev"),
            "group": group_of.get(t), "earnings": earnings.get(t),
            "scans": res,
            "tags": [sid for sid, *_ in SCANS if sid in res],
            "rep": rep_of[t], "q": qualifies(t),
        }
        if t in shown_tickers:
            rec["d"] = pack_bars(df, bench, (50, 200), DAILY_BARS, axis)
        if t in weekly_frames:
            wf = weekly_frames[t]
            rec["w"] = pack_bars(wf, bench, (10, 30), WEEKLY_YEARS * 52 + 1, bar_end=pd.Timedelta(days=4))
        if t in monthly_frames:
            mf = monthly_frames[t]
            rec["m"] = pack_bars(mf, bench, (10,), MONTHLY_YEARS * 12 + 1, bar_end=pd.offsets.MonthEnd(0))
        stocks_out[t] = rec

    return {
        "market": market, "asOf": as_of, "bench": bench_t,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "thresholds": THRESH, "windows": WINDOW,
        "cards": {"core": CORE, "chartCap": CHART_CAP, "rsFloor": RS_FLOOR, "bottomVolx": BOTTOM_VOLX,
                  "topGroups": TOP_GROUPS, "top5": sorted(top5) if top5 else []},
        "scannedTickers": scanned, "firstRun": first_run,
        "axis": axis,
        "scans": scan_entries,
        "stocks": stocks_out,
    }


# ----------------------------------------------------------------------------- 셋업 성과 추적
# setups-events-<m>.json : 히트 이벤트 영구 로그(append-only; (ticker, scan, eventDate) 당 한 번). 매 실행 때 OHLCV 캐시로
#                          5·10·20거래일 후 수익률 · MFE/MAE · 손절 가정 이탈 여부를 채운다(20거래일이 차면 status=done, 이후 고정).
# setups-stats-<m>.json  : 스캔별 집계(건수 · 승률 · 평균/중앙 · 지수 대비 초과 · 손절 이탈률) — 셋업 탭 '스캔별 성과' 패널.
# 기준(entry) = 카드가 처음 '보인' 날(seen)의 종가. 첫 실행이 되돌아보기 창에서 과거 이벤트를 찾을 때는 유지 조건이 살아 있는 것만
# 보이므로(생존 편향) 발생일 종가가 아니라 보인 날 종가에서 재기 시작한다. 평소에는 seen == eventDate.
HORIZONS = (5, 10, 20)        # 거래일 후 지평 — 마지막 값(20)이 '완료' 기준
MIN_SAMPLE = 30               # 완료 표본이 이보다 적으면 '표본 부족'
STOP_RULES = {                # scan → (level, mult, label). fixed = 이벤트 시점 기준가×mult 아래 종가 / 그 외 = 매일 다시 계산한 선 아래 종가
    "gap_up": ("fixed", 1.0, "갭일 저가 종가 이탈"),
    "ma200_cross": ("ma200", 1.0, "200일선 종가 이탈"),
    "base_breakout": ("fixed", 0.92, "피봇 −8% 종가 이탈"),
    "high_26w": ("fixed", 0.92, "돌파 고점 −8% 종가 이탈"),
    "bottom_escape": ("sma30w", 1.0, "30주선 종가 이탈"),
    "ma30w_reclaim": ("sma30w", 1.0, "30주선 종가 이탈"),
    "high_52w": ("fixed", 0.92, "돌파 고점 −8% 종가 이탈"),
    "ma10m_cross": ("sma10m", 1.0, "10개월선 종가 이탈"),
}
REF_STAT = {"base_breakout": "pivot", "high_26w": "prevHigh", "high_52w": "prevHigh", "ma200_cross": "ma200",
            "bottom_escape": "sma30", "ma30w_reclaim": "sma30", "ma10m_cross": "sma10m"}   # gap_up → 갭일 저가(OHLCV)


def line_series(df: pd.DataFrame, kind: str) -> np.ndarray:
    """매일 평가한 이평선(scan_frame과 같은 정의): ma200 / 30주선(완결 29주 + 오늘) / 10개월선(완결 9개월 + 오늘)."""
    cs, idx = df["Close"], df.index
    c = cs.to_numpy(dtype="float64")
    if kind == "ma200":
        return cs.rolling(200).mean().to_numpy(dtype="float64")
    if kind == "sma30w":
        wc = cs.groupby(idx.to_period("W-FRI")).last()
        return (_week_map(wc.rolling(29).sum().shift(1), idx) + c) / 30
    if kind == "sma10m":
        mc = cs.groupby(idx.to_period("M")).last()
        return (_month_map(mc.rolling(9).sum().shift(1), idx) + c) / 10
    raise ValueError(kind)


def _bench_at(bench: pd.Series | None, ts) -> float | None:
    if bench is None:
        return None
    v = bench.asof(ts)
    return float(v) if pd.notna(v) else None


def event_outcome(e: dict, df: pd.DataFrame, line: np.ndarray | None, bench: pd.Series | None) -> dict | None:
    """seen일 종가(entry) 기준 5·10·20거래일 후 수익률, MFE/MAE, 손절 가정 이탈. 분할 조정에 안전하도록 entry·기준가는 매번 현재 데이터로 다시 잡는다."""
    idx = df.index
    i0 = int(idx.searchsorted(pd.Timestamp(e["seen"])))
    if i0 >= len(idx):
        return None
    c, h, l = (df[k].to_numpy(dtype="float64") for k in ("Close", "High", "Low"))
    entry = c[i0]
    if not np.isfinite(entry) or entry <= 0:
        return None
    H = HORIZONS[-1]
    j1 = min(i0 + H, len(idx) - 1)
    days = j1 - i0
    out: dict = {"days": days, "status": "done" if days >= H else "open", "last": rnd(c[j1], 2), "lastDate": idx[j1].strftime("%Y-%m-%d")}
    b0 = _bench_at(bench, idx[i0])
    for N in HORIZONS:
        if days >= N:
            r = pct(c[i0 + N], entry)
            out[f"r{N}"] = r
            bN = _bench_at(bench, idx[i0 + N])
            br = pct(bN, b0) if bN is not None and b0 is not None else None
            out[f"x{N}"] = round(r - br, 2) if r is not None and br is not None else None
        else:
            out[f"r{N}"] = out[f"x{N}"] = None
    if days > 0:
        out["mfe"] = pct(float(np.nanmax(h[i0 + 1:j1 + 1])), entry)
        out["mae"] = pct(float(np.nanmin(l[i0 + 1:j1 + 1])), entry)
    else:
        out["mfe"] = out["mae"] = None
    kind, mult, _ = STOP_RULES[e["scan"]]
    out["stop"], out["stopDay"], out["rStop"] = False, None, None
    for j in range(i0 + 1, j1 + 1):
        level = e["refRel"] * entry * mult if kind == "fixed" else (line[j] if line is not None else np.nan)
        if np.isfinite(level) and c[j] < level:
            out["stop"], out["stopDay"], out["rStop"] = True, j - i0, pct(c[j], entry)
            break
    return out


def _agg(evs: list[dict]) -> dict:
    done = [e for e in evs if e.get("status") == "done"]
    row: dict = {"n": len(evs), "done": len(done), "open": len(evs) - len(done), "lowSample": len(done) < MIN_SAMPLE}
    for N in HORIZONS:
        rs = [e[f"r{N}"] for e in evs if e.get(f"r{N}") is not None]
        xs = [e[f"x{N}"] for e in evs if e.get(f"x{N}") is not None]
        row[f"n{N}"] = len(rs)
        row[f"win{N}"] = round(100 * sum(1 for r in rs if r > 0) / len(rs), 1) if rs else None
        row[f"avg{N}"] = round(float(np.mean(rs)), 2) if rs else None
        row[f"med{N}"] = round(float(np.median(rs)), 2) if rs else None
        row[f"x{N}"] = round(float(np.mean(xs)), 2) if xs else None
        row[f"xwin{N}"] = round(100 * sum(1 for x in xs if x > 0) / len(xs), 1) if xs else None
        row[f"bench{N}"] = round(float(np.mean([e[f"r{N}"] - e[f"x{N}"] for e in evs if e.get(f"x{N}") is not None])), 2) if xs else None
    mfe = [e["mfe"] for e in done if e.get("mfe") is not None]
    mae = [e["mae"] for e in done if e.get("mae") is not None]
    row["mfe"] = round(float(np.mean(mfe)), 2) if mfe else None
    row["mae"] = round(float(np.mean(mae)), 2) if mae else None
    stopped = [e for e in done if e.get("stop")]
    row["stopRate"] = round(100 * len(stopped) / len(done), 1) if done else None
    row["stopAvg"] = round(float(np.mean([e["rStop"] for e in stopped if e.get("rStop") is not None])), 2) if stopped else None
    return row


def track_events(market: str, hits: dict[str, dict[str, dict]], meta: dict[str, dict], group_of: dict[str, str],
                 ohlcv: dict[str, pd.DataFrame], bench: pd.Series | None, as_of: str, runs: dict[str, list]) -> None:
    ev_path, st_path = HERE / f"setups-events-{market}.json", HERE / f"setups-stats-{market}.json"
    data = load_history(ev_path) if ev_path.exists() else {}
    events: list[dict] = data.get("events") or []
    known = {(e["ticker"], e["scan"], e["eventDate"]) for e in events}
    floor = min(runs) if runs else as_of          # 이력이 시작된 날 이전엔 카드를 본 적이 없다
    added = 0
    for t, res in hits.items():
        df = ohlcv[t]
        idx = df.index
        c, l = df["Close"].to_numpy(dtype="float64"), df["Low"].to_numpy(dtype="float64")
        for sid, r in res.items():
            key = (t, sid, r["eventDate"])
            if key in known:
                continue
            seen = max(r.get("firstSeen") or as_of, floor)
            ie = int(idx.searchsorted(pd.Timestamp(r["eventDate"])))
            i0 = min(int(idx.searchsorted(pd.Timestamp(seen))), len(idx) - 1)
            entry = c[i0]
            ref = l[ie] if sid == "gap_up" else r["stats"].get(REF_STAT[sid])
            if not np.isfinite(entry) or entry <= 0 or ref is None or not np.isfinite(ref) or ref <= 0:
                continue
            events.append({
                "ticker": t, "name": meta[t].get("name"), "scan": sid, "eventDate": r["eventDate"],
                "seen": idx[i0].strftime("%Y-%m-%d"), "lag": i0 - ie,
                "close": rnd(c[ie], 2), "entry": rnd(entry, 2), "ref": rnd(float(ref), 2), "refRel": round(float(ref) / entry, 6),
                "group": group_of.get(t), "score": meta[t].get("score"),
            })
            known.add(key)
            added += 1
    line_cache: dict[tuple[str, str], np.ndarray] = {}
    updated = 0
    for e in events:
        if e.get("status") == "done":
            continue                              # 20거래일이 찬 이벤트는 고정
        df = ohlcv.get(e["ticker"])
        if df is None:
            continue                              # 상폐·유니버스 이탈: 마지막 값 유지
        kind = STOP_RULES[e["scan"]][0]
        line = None if kind == "fixed" else line_cache.setdefault((e["ticker"], kind), line_series(df, kind))
        o = event_outcome(e, df, line, bench)
        if o:
            e.update(o)
            updated += 1
    events.sort(key=lambda e: (e["seen"], e["scan"], e["ticker"]))
    ev_path.write_text(json.dumps({"market": market, "asOf": as_of, "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                   "events": events}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    done = sum(1 for e in events if e.get("status") == "done")
    stats = {
        "market": market, "asOf": as_of, "bench": BENCH[market], "benchLabel": "코스피" if market == "kr" else "S&P 500 (SPY)",
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horizons": list(HORIZONS), "minSample": MIN_SAMPLE, "events": len(events), "done": done, "open": len(events) - done,
        "firstSeen": min((e["seen"] for e in events), default=None),
        "basis": f"기준 = 카드가 처음 보인 날 종가 → N거래일 후 종가(배당 미반영) · 초과 = 같은 기간 {BENCH[market]} 수익률 차감(%p) · "
                 f"MFE/MAE = 20거래일 내 고가/저가 최대 도달률 · 손절 이탈 = 20거래일 내 가정 손절선 아래 종가 비율 · 완료 표본 {MIN_SAMPLE}건 미만은 표본 부족",
        "scans": [{"id": sid, "tf": tf, "label": label, "stopRule": STOP_RULES[sid][2], **_agg([e for e in events if e["scan"] == sid])}
                  for sid, tf, label, *_ in SCANS],
        "total": _agg(events),
    }
    st_path.write_text(json.dumps(stats, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    log(f"  events: +{added} new → {len(events)} logged · {updated} outcomes updated · {done} done · wrote {ev_path.name} ({ev_path.stat().st_size/1e3:.0f} KB), {st_path.name}")


# ----------------------------------------------------------------------------- main
def load_ohlcv(market: str) -> tuple[dict[str, pd.DataFrame], str]:
    """OHLCV from compute_returns' cache (same run); re-download only if the cache is missing."""
    path = CACHE_DIR / f"ohlcv-{market}.pkl"
    if path.exists():
        blob = pd.read_pickle(path)
        as_of = modal_last_date(blob["ohlcv"]) or blob["asOf"]
        log(f"  ohlcv cache: {len(blob['ohlcv'])} tickers · asOf {as_of} (cache max {blob['asOf']}) · generated {blob['generatedAt']}")
        return blob["ohlcv"], as_of
    log("  ohlcv cache missing → downloading universe (run compute_returns.py first to avoid this)")
    universe = json.loads((HERE / f"universe-{market}.json").read_text(encoding="utf-8"))
    start = (pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    frames = download_frames([u["ticker"] for u in universe], start)
    return frames, modal_last_date(frames)


def modal_last_date(frames: dict[str, pd.DataFrame]) -> str | None:
    """The trading date most tickers end on. Yahoo occasionally returns the latest
    bar for only a handful of names (2026-10-29 evening: 2 of 1,275 KR tickers had
    the day's bar) — anchoring on the max date would then skip the whole market."""
    dates = pd.Series([f.index[-1] for f in frames.values() if len(f)])
    if dates.empty:
        return None
    mode = dates.mode().iloc[0]
    if mode != dates.max():
        log(f"  note: {int((dates == dates.max()).sum())} tickers end on {dates.max().date()} but most ({int((dates == mode).sum())}) end on {mode.date()} → asOf {mode.date()}")
    return mode.strftime("%Y-%m-%d")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["us", "kr"], default="us")
    ap.add_argument("--no-extra", action="store_true", help="skip weekly/monthly bars + earnings for hits")
    ap.add_argument("--only", choices=["groups", "setups"], default=None)
    args = ap.parse_args()
    m = args.market
    t0 = time.time()
    data = json.loads((HERE / f"data-{m}.json").read_text(encoding="utf-8"))
    universe = json.loads((HERE / f"universe-{m}.json").read_text(encoding="utf-8"))
    ohlcv, as_of = load_ohlcv(m)
    if data.get("asOf") != as_of:
        log(f"  warning: data-{m}.json asOf {data.get('asOf')} ≠ ohlcv asOf {as_of}")
    log(f"scanner {m.upper()} asOf {as_of}")

    group_of: dict[str, str] = {}
    groups = None
    if args.only != "setups":
        groups, group_of = build_groups(m, ohlcv, universe, as_of)
        (HERE / f"groups-{m}.json").write_text(json.dumps(groups, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        log(f"  wrote groups-{m}.json ({len(groups['cards'])} cards)")
    else:
        gp = HERE / f"groups-{m}.json"
        groups = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else None
        for u in universe:
            group_of[u["ticker"]] = kr_group_of(u.get("sector") or "") if m == "kr" else (us_group_of(u.get("sector") or "", u.get("industry") or "") or "")
    if args.only != "groups":
        setups = build_setups(m, ohlcv, data["stocks"], group_of, as_of, extra=not args.no_extra, top5=top_groups(groups))
        out = HERE / f"setups-{m}.json"
        out.write_text(json.dumps(setups, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        log(f"  wrote {out.name} ({out.stat().st_size/1e6:.1f} MB · {len(setups['stocks'])} tickers with charts)")
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
