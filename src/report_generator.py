"""
리포트 생성 모듈
최종 분석 결과를 마크다운/HTML 리포트로 포맷팅
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"


def get_report_date_dir(base_dir: Path, date: str) -> Path:
    """
    날짜 문자열을 reports/YYYY/MM/DD 경로로 변환하고 디렉터리 생성 후 반환.

    Args:
        base_dir: reports 베이스 디렉터리 (예: outputs/reports)
        date: YYYYMMDD 또는 YYYY-MM-DD 형식

    Returns:
        base_dir / YYYY / MM / DD (생성됨)
    """
    clean = date.replace("-", "") if "-" in date else date
    if len(clean) < 8:
        clean = clean.ljust(8, "0")
    yyyy, mm, dd = clean[:4], clean[4:6], clean[6:8]
    path = base_dir / yyyy / mm / dd
    path.mkdir(parents=True, exist_ok=True)
    return path


class ReportGenerator:
    """분석 결과 리포트 생성기"""

    def __init__(self):
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def save_markdown_report(
        self,
        report_content: str,
        date: str,
        filename: Optional[str] = None,
    ) -> str:
        """
        마크다운 리포트 파일 저장

        Args:
            report_content: 마크다운 리포트 텍스트
            date: 분석 날짜 (YYYYMMDD)
            filename: 파일명 (기본값: report.md, 날짜 폴더 내)

        Returns:
            str: 저장된 파일 경로
        """
        if not filename:
            date_dir = get_report_date_dir(REPORTS_DIR, date)
            filepath = date_dir / "report.md"
        else:
            filepath = REPORTS_DIR / filename
        filepath.write_text(report_content, encoding="utf-8")

        logger.info(f"[리포트] 마크다운 리포트 저장: {filepath}")
        return str(filepath)

    def save_json_data(
        self,
        data: dict,
        date: str,
        stage: str = "full",
    ) -> str:
        """
        분석 데이터를 JSON 파일로 저장

        Args:
            data: 저장할 데이터
            date: 분석 날짜
            stage: 분석 단계명 (collected -> collected.json, full -> full.json)

        Returns:
            str: 저장된 파일 경로
        """
        date_dir = get_report_date_dir(REPORTS_DIR, date)
        if stage == "collected":
            filepath = date_dir / "collected.json"
        elif stage == "full":
            filepath = date_dir / "full.json"
        else:
            filepath = date_dir / f"{stage}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"[리포트] JSON 데이터 저장: {filepath}")
        return str(filepath)

    def generate_fallback_report(self, analysis_result: dict, date: str) -> str:
        """
        AI 리포트 생성 실패 시 데이터 기반 폴백 리포트 생성

        Args:
            analysis_result: 전체 분석 결과
            date: 분석 날짜

        Returns:
            str: 마크다운 리포트
        """
        logger.info("[리포트] 폴백 리포트 생성 시작")

        formatted_date = self._format_date(date, display=True)

        lines = [
            f"# 📊 일일 주식 시장 분석 리포트",
            f"### {formatted_date} | AI 시장 분석",
            "",
            "---",
            "",
        ]

        # 필터링 결과 요약
        filtered = analysis_result.get("filtered_analysis", {})
        if filtered:
            lines.append("## 🔍 필터링 결과 요약")
            lines.append(
                f"> 거래량/거래대금 기준 필터링된 종목 수: "
                f"**{filtered.get('필터링_종목수', 0)}개**"
            )
            lines.append("")

        # 종목 테이블
        stocks = filtered.get("종목_리스트", [])
        if stocks:
            lines.append("## 📋 필터링 종목 리스트")
            lines.append("")
            lines.append(
                "| 순위 | 종목명 | 종목코드 | 종가 | 등락률(%) | 거래대금 |"
            )
            lines.append("|------|--------|----------|------|-----------|----------|")
            for i, s in enumerate(stocks, 1):
                trading_value = s.get("거래대금", 0)
                tv_display = f"{trading_value / 100000000:,.0f}억"
                lines.append(
                    f"| {i} | {s.get('종목명', '')} | {s.get('종목코드', '')} "
                    f"| {s.get('종가', 0):,} | {s.get('등락률', 0)} | {tv_display} |"
                )
            lines.append("")

        # 재무지표 분석
        fundamental = analysis_result.get("fundamental_analysis", {})
        fund_stocks = fundamental.get("종목_분석", [])
        if fund_stocks:
            lines.append("## 📈 재무지표 분석")
            lines.append("")
            for fs in fund_stocks:
                grade = fs.get("종합등급", "N/A")
                lines.append(f"### {fs.get('종목명', '')} ({fs.get('종목코드', '')})")
                lines.append(f"- **종합등급**: {grade}")
                lines.append(
                    f"- **PER**: {fs.get('PER', 'N/A')} "
                    f"(업종평균: {fs.get('업종평균_PER', 'N/A')})"
                )
                lines.append(
                    f"- **PBR**: {fs.get('PBR', 'N/A')} "
                    f"({fs.get('PBR_판단', '')})"
                )
                lines.append(f"- **ROE**: {fs.get('ROE', 'N/A')}%")
                if fs.get("종합의견"):
                    lines.append(f"- **의견**: {fs['종합의견']}")
                lines.append("")

        # 뉴스 분석
        news_results = analysis_result.get("news_analysis", [])
        if news_results:
            lines.append("## 📰 뉴스 감성 분석")
            lines.append("")
            for nr in news_results:
                score = nr.get("종합_뉴스스코어", 0)
                emoji = "🟢" if score > 25 else "🟡" if score > -25 else "🔴"
                lines.append(
                    f"- {emoji} **{nr.get('종목명', '')}**: "
                    f"뉴스 스코어 {score} — {nr.get('뉴스_요약', '')}"
                )
            lines.append("")

        # 종합 분석
        comprehensive = analysis_result.get("comprehensive_analysis", {})
        if comprehensive:
            # TOP3 주목 종목
            top3 = comprehensive.get("TOP3_주목종목", [])
            if top3:
                lines.append("## 🏆 TOP 3 주목 종목")
                lines.append("")
                for t in top3:
                    lines.append(
                        f"### {t.get('순위', '')}위: {t.get('종목명', '')}"
                    )
                    lines.append(f"- **선정 이유**: {t.get('선정_이유', '')}")
                    lines.append(
                        f"- **핵심 모니터링**: {t.get('핵심_모니터링', '')}"
                    )
                    lines.append("")

            # 리스크 경고
            risk = comprehensive.get("리스크_경고", {})
            market_risks = risk.get("시장_리스크", [])
            if market_risks:
                lines.append("## ⚠️ 리스크 경고")
                lines.append("")
                for r in market_risks:
                    lines.append(f"- {r}")
                lines.append("")

        # 면책조항
        lines.extend(
            [
                "---",
                "",
                "⚠️ **면책조항**: 본 리포트는 AI가 공개 데이터를 분석하여 생성한 "
                "참고 자료입니다. 투자 권유가 아니며, 투자 결정은 본인의 판단과 "
                "책임 하에 이루어져야 합니다.",
                "",
                "---",
            ]
        )

        return "\n".join(lines)

    def save_full_output(self, analysis_result: dict, date: str) -> dict:
        """
        전체 분석 결과 저장 (JSON + 마크다운 리포트)

        Args:
            analysis_result: AI 분석 전체 결과
            date: 분석 날짜

        Returns:
            dict: 저장된 파일 경로들
        """
        saved_files = {}

        # JSON 데이터 저장
        json_path = self.save_json_data(analysis_result, date, "full")
        saved_files["json"] = json_path

        # 마크다운 리포트 저장
        report_md = analysis_result.get("report_markdown", "")
        if not report_md or "raw_response" in str(report_md):
            # AI 리포트가 없으면 폴백 리포트 생성
            report_md = self.generate_fallback_report(analysis_result, date)

        md_path = self.save_markdown_report(report_md, date)
        saved_files["markdown"] = md_path

        logger.info(f"[리포트] 전체 출력 저장 완료: {saved_files}")
        return saved_files

    @staticmethod
    def _format_date(date: str, display: bool = False) -> str:
        """날짜 형식 변환"""
        # 이미 YYYY-MM-DD 형식인 경우
        if "-" in date:
            clean = date.replace("-", "")
        else:
            clean = date

        if display:
            return f"{clean[:4]}년 {clean[4:6]}월 {clean[6:8]}일"
        return f"{clean[:4]}-{clean[4:6]}-{clean[6:8]}"

    def get_recent_reports(self, count: int = 10) -> list[dict]:
        """
        최근 리포트 목록 조회 (날짜 폴더 내 report.md 및 구 구조 report_*.md 지원)

        Args:
            count: 조회할 리포트 수

        Returns:
            list[dict]: 리포트 정보 리스트
        """
        reports = []
        seen_dates = set()

        # 새 구조: outputs/reports/YYYY/MM/DD/report.md
        for filepath in REPORTS_DIR.glob("**/report.md"):
            try:
                rel = filepath.relative_to(REPORTS_DIR)
                parts = rel.parts
                if (
                    len(parts) == 4
                    and parts[-1] == "report.md"
                    and parts[0].isdigit()
                    and parts[1].isdigit()
                    and parts[2].isdigit()
                ):
                    yyyy, mm, dd = parts[0], parts[1], parts[2]
                    date_str = f"{yyyy}-{mm}-{dd}"
                    if date_str not in seen_dates:
                        seen_dates.add(date_str)
                        reports.append(
                            {
                                "filename": filepath.name,
                                "path": str(filepath),
                                "date": date_str,
                                "size": filepath.stat().st_size,
                                "modified": datetime.fromtimestamp(
                                    filepath.stat().st_mtime
                                ).strftime("%Y-%m-%d %H:%M"),
                            }
                        )
            except ValueError:
                continue

        # 구 구조: outputs/reports/report_YYYY-MM-DD.md
        for filepath in REPORTS_DIR.glob("report_*.md"):
            date_str = filepath.stem.replace("report_", "")
            if date_str not in seen_dates:
                seen_dates.add(date_str)
                reports.append(
                    {
                        "filename": filepath.name,
                        "path": str(filepath),
                        "date": date_str,
                        "size": filepath.stat().st_size,
                        "modified": datetime.fromtimestamp(
                            filepath.stat().st_mtime
                        ).strftime("%Y-%m-%d %H:%M"),
                    }
                )

        reports.sort(key=lambda x: x["date"], reverse=True)
        return reports[:count]
