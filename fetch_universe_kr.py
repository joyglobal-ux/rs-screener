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
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd

MIN_MARKET_CAP = 100_000_000_000  # 1,000억 원 (KRW)
OUT = Path(__file__).parent / "universe-kr.json"

# Preferred shares end with 우 / 우B / 1우B / 2우B etc. — exclude them so each
# company appears once. Also drop SPACs (스팩) which trade like cash boxes.
PREFERRED_RE = re.compile(r"우$|우B$|[0-9]+우B?$")


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
    df = df[df["Market"].isin(["KOSPI", "KOSDAQ"])]
    df = df[df["Name"].apply(is_common_stock)]
    df = df[df["Marcap"] >= MIN_MARKET_CAP]
    df = df.sort_values("Marcap", ascending=False)

    out = []
    for _, row in df.iterrows():
        industry = (row["Industry"] if pd.notna(row["Industry"]) else "") or ""
        industry = industry.strip()
        out.append({
            "ticker": to_yahoo(row["Code"], row["Market"]),
            "krxCode": row["Code"],
            "name": row["Name"],
            "sector": classify_sector(industry),
            "industry": industry or "기타",
            "marketCap": int(row["Marcap"]),
            "market": row["Market"],  # KOSPI | KOSDAQ
        })

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"kept {len(out)} KR tickers >= {MIN_MARKET_CAP/1e8:,.0f}억 원 → {OUT.name}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
