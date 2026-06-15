"""Fetch the KOSPI + KOSDAQ universe with market cap >= MIN_MARKET_CAP and
write universe-kr.json.

Two FinanceDataReader endpoints give everything we need:
  - "KRX"      → latest snapshot incl. Marcap (KRW)
  - "KRX-DESC" → company description incl. Sector / Industry
Joined on Code. Yahoo-compatible tickers: KOSPI → NNNNNN.KS, KOSDAQ → NNNNNN.KQ.
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
import requests

MIN_MARKET_CAP = 100_000_000_000  # 1,000억 원 (KRW)
HERE = Path(__file__).parent
OUT = HERE / "universe-kr.json"
WICS_MAP = HERE / "sector_map_kr.json"  # cached {6-digit code: WICS sector name}
WICS_MAX_AGE_DAYS = 7  # re-scrape Naver only when the cache is older than this

# Preferred shares end with 우 / 우B / 1우B / 2우B etc. — exclude them so each
# company appears once. Also drop SPACs (스팩) which trade like cash boxes.
PREFERRED_RE = re.compile(r"우$|우B$|[0-9]+우B?$")


# When a stock isn't in Naver's WICS map and we fall back to the KSIC classifier,
# fold its coarse label into the WICS vocabulary so we don't get near-duplicate
# sectors (e.g. "기계/장비" vs WICS "기계").
KSIC_TO_WICS = {
    "기계/장비": "기계",
    "제약/바이오": "제약",
    "바이오/연구개발": "생물공학",
    "소프트웨어/IT서비스": "IT서비스",
    "증권/투자": "증권",
    "의료기기/헬스케어": "건강관리장비와용품",
    "유통/도소매": "유통",
    "철강/금속": "철강",
    "반도체": "반도체와반도체장비",
    "디스플레이": "디스플레이장비및부품",
    "전기전자": "전자장비와기기",
}


def classify_sector(industry: str) -> str:
    """Map a fine-grained KSIC industry string to a coarse sector for the
    UI dropdown. Order matters — checks the more specific patterns first."""
    n = industry or ""
    rules = [
        ("반도체", "반도체"),
        ("디스플레이", "디스플레이"),
        ("전자부품", "전기전자"),
        ("광학", "정밀/광학"), ("정밀기기", "정밀/광학"),
        ("게임", "게임"),
        ("소프트웨어", "소프트웨어/IT서비스"), ("프로그래밍", "소프트웨어/IT서비스"),
        ("시스템 통합", "소프트웨어/IT서비스"), ("정보서비스", "소프트웨어/IT서비스"),
        ("영화", "미디어/엔터"), ("방송프로그램", "미디어/엔터"),
        ("음악", "미디어/엔터"), ("출판", "미디어/엔터"),
        ("통신 및 방송 장비", "통신/방송장비"),
        ("통신업", "통신서비스"),
        ("자동차", "자동차/부품"), ("차량", "자동차/부품"),
        ("선박", "조선"), ("조선", "조선"),
        ("항공", "운송/물류"), ("해상 운송", "운송/물류"), ("해운", "운송/물류"),
        ("운수", "운송/물류"), ("운송", "운송/물류"), ("물류", "운송/물류"),
        ("발전기", "기계/장비"), ("전동기", "기계/장비"),
        ("기계", "기계/장비"), ("장비", "기계/장비"),
        ("철강", "철강/금속"), ("비철금속", "철강/금속"),
        ("금속", "철강/금속"),
        ("건설", "건설"), ("토목", "건설"),
        ("시멘트", "건자재"), ("비금속광물", "건자재"),
        ("의약품", "제약/바이오"), ("의약물질", "제약/바이오"),
        ("의약 관련", "제약/바이오"),
        ("의료용", "의료기기/헬스케어"), ("의료", "의료기기/헬스케어"),
        ("자연과학", "바이오/연구개발"),
        ("화학", "화학/소재"), ("고무", "화학/소재"),
        ("플라스틱", "화학/소재"), ("도료", "화학/소재"),
        ("은행", "은행"),
        ("보험", "보험"),
        ("증권", "증권/투자"), ("투자", "증권/투자"),
        ("금융", "금융"),
        ("도매", "유통/도소매"), ("소매", "유통/도소매"), ("유통", "유통/도소매"),
        ("음식점", "호텔/레저"), ("숙박", "호텔/레저"),
        ("여행", "호텔/레저"), ("레저", "호텔/레저"),
        ("식품", "음식료/담배"), ("음료", "음식료/담배"),
        ("주류", "음식료/담배"), ("담배", "음식료/담배"),
        ("농수산", "농수산"), ("어업", "농수산"), ("농업", "농수산"),
        ("섬유", "섬유/의류"), ("봉제", "섬유/의류"),
        ("의복", "섬유/의류"), ("가죽", "섬유/의류"),
        ("종이", "종이/인쇄"), ("펄프", "종이/인쇄"), ("인쇄", "종이/인쇄"),
        ("가구", "가구"),
        ("광업", "광업/에너지"), ("원유", "광업/에너지"), ("석탄", "광업/에너지"),
        ("가스", "유틸리티"), ("수도", "유틸리티"), ("폐기물", "유틸리티"),
        ("전기, 가스", "유틸리티"),
        ("교육", "교육"),
        ("부동산", "부동산"),
        ("광고", "전문서비스"), ("전문서비스", "전문서비스"),
        ("법무", "전문서비스"), ("회계", "전문서비스"),
    ]
    for kw, sec in rules:
        if kw in n:
            return sec
    return "기타"


def to_yahoo(code: str, market: str) -> str:
    return f"{code}.{'KS' if market == 'KOSPI' else 'KQ'}"


_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
_GROUP_RE = re.compile(r"sise_group_detail\.naver\?type=upjong&no=(\d+)\">([^<]+)<")
_CODE_RE = re.compile(r"/item/main\.naver\?code=(\d{6})")


def scrape_wics_map() -> dict[str, str]:
    """Scrape Naver's 업종(WICS) groups → {6-digit code: sector name}.

    Naver's 업종 = WISE Industry Classification (WICS), which groups by investment
    theme — e.g. "반도체와반도체장비" holds chipmakers AND equipment makers (주성·
    피에스케이·테스), unlike KSIC which buries equipment under 특수목적용기계. ~79
    group pages cover every listed name, names included.
    """
    base = "https://finance.naver.com/sise/sise_group.naver?type=upjong"
    listing = requests.get(base, headers=_UA, timeout=20).content.decode("euc-kr", "replace")
    groups = _GROUP_RE.findall(listing)
    if not groups:
        raise RuntimeError("Naver 업종 목록 파싱 실패")
    out: dict[str, str] = {}
    for no, name in groups:
        name = name.strip()
        url = f"https://finance.naver.com/sise/sise_group_detail.naver?type=upjong&no={no}"
        try:
            html = requests.get(url, headers=_UA, timeout=20).content.decode("euc-kr", "replace")
        except Exception as e:  # noqa: BLE001
            print(f"  WICS group {no} {name} 실패: {e}", file=sys.stderr)
            continue
        for code in _CODE_RE.findall(html):
            out.setdefault(code, name)   # first (= its own group) wins
        time.sleep(0.3)
    if len(out) < 500:
        raise RuntimeError(f"WICS 맵이 너무 작음({len(out)}) — 스크랩 불완전")
    return out


def load_or_refresh_wics_map() -> dict[str, str]:
    """Return the cached WICS map, re-scraping only when missing/stale. Any
    failure falls back to the existing cache (or {}), so the daily run never
    breaks on a Naver hiccup — codes missing from the map use the KSIC classifier.
    """
    cached, fresh = {}, False
    if WICS_MAP.exists():
        try:
            blob = json.loads(WICS_MAP.read_text(encoding="utf-8"))
            cached = blob.get("map", {})
            ts = datetime.fromisoformat(blob.get("scrapedAt", "").replace("Z", "+00:00"))
            fresh = (datetime.now(timezone.utc) - ts).days < WICS_MAX_AGE_DAYS
        except Exception:  # noqa: BLE001
            cached, fresh = cached, False
    if fresh and cached:
        print(f"WICS 맵 캐시 사용 ({len(cached)}종목)", file=sys.stderr)
        return cached
    try:
        m = scrape_wics_map()
        WICS_MAP.write_text(
            json.dumps({"scrapedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "map": m}, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"WICS 맵 재스크랩 완료 ({len(m)}종목) → {WICS_MAP.name}", file=sys.stderr)
        return m
    except Exception as e:  # noqa: BLE001
        print(f"WICS 스크랩 실패({e}) — 캐시 {len(cached)}종목으로 폴백", file=sys.stderr)
        return cached


def is_common_stock(name: str) -> bool:
    n = (name or "").strip()
    if not n:
        return False
    if PREFERRED_RE.search(n):
        return False
    if "스팩" in n:
        return False
    return True


def main() -> None:
    marcap = fdr.StockListing("KRX")
    desc = fdr.StockListing("KRX-DESC")
    print(f"FDR: KRX={len(marcap)}, KRX-DESC={len(desc)}", file=sys.stderr)

    df = marcap.merge(desc[["Code", "Sector", "Industry"]], on="Code", how="left")
    # "KOSDAQ GLOBAL" is the premium KOSDAQ segment (알테오젠·에코프로비엠·주성엔지니어링
    # 등 우량주) — fold it into KOSDAQ so it isn't dropped. KONEX stays excluded.
    df["Market"] = df["Market"].replace({"KOSDAQ GLOBAL": "KOSDAQ"})
    df = df[df["Market"].isin(["KOSPI", "KOSDAQ"])]
    df = df[df["Name"].apply(is_common_stock)]
    df = df[df["Marcap"] >= MIN_MARKET_CAP]
    df = df.sort_values("Marcap", ascending=False)

    wics = load_or_refresh_wics_map()
    out = []
    wics_hits = 0
    for _, row in df.iterrows():
        industry = (row["Industry"] if pd.notna(row["Industry"]) else "") or ""
        industry = industry.strip()
        # WICS (Naver) is the investment-theme taxonomy; KSIC classifier is the
        # fallback for codes Naver didn't list (new/edge names).
        sector = wics.get(row["Code"])
        if sector:
            wics_hits += 1
        else:
            sector = classify_sector(industry)
            sector = KSIC_TO_WICS.get(sector, sector)
        out.append({
            "ticker": to_yahoo(row["Code"], row["Market"]),
            "krxCode": row["Code"],
            "name": row["Name"],
            "sector": sector,
            "industry": industry or "기타",
            "marketCap": int(row["Marcap"]),
            "market": row["Market"],  # KOSPI | KOSDAQ
        })
    print(f"섹터: WICS {wics_hits} / KSIC폴백 {len(out)-wics_hits}", file=sys.stderr)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"kept {len(out)} KR tickers >= {MIN_MARKET_CAP/1e8:,.0f}억 원 → {OUT.name}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
