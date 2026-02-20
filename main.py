"""
한국 주식 시장 분석 시스템 - 메인 엔트리포인트

3-Phase 파이프라인:
  Phase 1 (collect): 데이터 수집 (Python only, API 불필요)
  Phase 2 (analyze): AI 종합분석 (Step 4 - 뉴스 감성분석 통합)
  Phase 3 (analyze): AI 리포트 생성 (Step 5)

기본 동작: python main.py → collect → analyze까지 실행하되 API 호출 없이 프롬프트만 저장.
API로 실제 분석을 수행하려면 --api 옵션을 사용하세요.

사용법:
    python main.py                                          # 전체 파이프라인 (기본: 프롬프트만 저장)
    python main.py --api                                    # API 호출로 실제 분석 수행
    python main.py --step collect --date 20260213           # Phase 1만
    python main.py --step analyze --from-data data.json     # 저장된 데이터에서 이어가기
    python main.py --step report --from-data full.json      # Step 5(리포트)만 재실행
    python main.py --step4 gemini --step5 claude            # Step별 프로바이더
    python main.py --schedule --api                         # 매일 자동 실행 (API 호출)
    streamlit run app.py                                    # 대시보드
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collector import StockDataCollector
from src.news_collector import NewsCollector
from src.ai_analyzer import (
    AIAnalyzer,
    StepProviderConfig,
    SUPPORTED_PROVIDERS,
    STEP_DEFINITIONS,
    get_available_providers,
    get_step_provider_summary,
)
from src.report_generator import ReportGenerator, get_report_date_dir
from src.scheduler import AnalysisScheduler


def setup_logging(level: str = "INFO") -> None:
    """로깅 설정"""
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    log_date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]

    # outputs 디렉토리가 있으면 파일 핸들러 추가
    log_dir = PROJECT_ROOT / "outputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers.append(
        logging.FileHandler(log_dir / "analysis.log", encoding="utf-8")
    )

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt=log_date_format,
        handlers=handlers,
    )


logger = logging.getLogger(__name__)


# ============================================================
# Phase 1: 데이터 수집 (Python only)
# ============================================================

def run_collect(date: str = None) -> dict:
    """
    Phase 1: 데이터 수집 (AI API 불필요)

    - 주식 필터링 (거래량/거래대금)
    - 재무지표 수집 + Python 규칙기반 등급 부여
    - 시장지수/외국인매매 수집
    - 뉴스 수집

    Returns:
        dict: 수집된 전체 데이터
    """
    data_collector = StockDataCollector()
    news_collector = NewsCollector()
    report_generator = ReportGenerator()

    # 데이터 수집
    logger.info("[Phase 1] 시장 데이터 수집 시작...")
    collected_data = data_collector.collect_all_data(date)
    analysis_date = collected_data["기준일"]

    filtered_count = collected_data["필터링_결과"]["필터링_종목수"]
    logger.info(f"  -> 필터링 종목 수: {filtered_count}개")

    if filtered_count == 0:
        logger.warning("필터링 조건에 해당하는 종목이 없습니다.")
        return collected_data

    # 뉴스 수집
    logger.info("[Phase 1] 종목별 뉴스 수집 시작...")
    stocks = collected_data["필터링_결과"]["종목_리스트"]
    all_news = news_collector.collect_all_stock_news(stocks, num_articles_per_stock=10)
    collected_data["뉴스_데이터"] = all_news
    logger.info(
        f"  -> 총 {sum(n['수집_뉴스수'] for n in all_news)}건 뉴스 수집 완료"
    )

    # 수집 데이터 저장
    formatted_date = _format_date(analysis_date)
    save_path = report_generator.save_json_data(
        collected_data, analysis_date, "collected"
    )
    logger.info(f"[Phase 1] 수집 데이터 저장: {save_path}")

    return collected_data


# ============================================================
# Phase 2+3: AI 분석 (Step 4 + Step 5)
# ============================================================

def run_analyze(
    collected_data: dict,
    provider: str = None,
    step_config: StepProviderConfig = None,
    show_prompts: bool = False,
    prompt_only: bool = False,
) -> dict:
    """
    Phase 2+3: AI 분석 실행

    Args:
        collected_data: Phase 1에서 수집된 데이터
        provider: 글로벌 AI 프로바이더
        step_config: Step별 프로바이더 설정
        show_prompts: 프롬프트 콘솔 출력 여부
        prompt_only: True이면 API 호출 없이 프롬프트만 저장

    Returns:
        dict: 분석 결과
    """
    report_generator = ReportGenerator()
    analysis_date = collected_data.get("기준일", "")
    default_provider = provider or os.getenv("AI_PROVIDER", "claude")

    try:
        ai_analyzer = AIAnalyzer(
            provider=provider,
            step_config=step_config,
            show_prompts=show_prompts,
            analyze_by_api=not prompt_only,
        )
        analysis_result = ai_analyzer.run_full_analysis(collected_data)
    except ValueError as e:
        logger.error(f"AI 분석기 초기화 실패: {e}")
        logger.info("  -> API 키 없이 데이터 수집 결과만 저장합니다.")
        analysis_result = {
            "ai_providers": {},
            "ai_default_provider": default_provider,
            "filtered_analysis": collected_data.get("필터링_결과", {}),
            "fundamental_analysis": collected_data.get("재무지표", {}),
            "news_analysis": [],
            "comprehensive_analysis": {},
            "report_markdown": "",
        }
    except Exception as e:
        logger.error(f"AI 분석 실패: {e}")
        analysis_result = {
            "ai_providers": {},
            "ai_default_provider": default_provider,
            "filtered_analysis": collected_data.get("필터링_결과", {}),
            "fundamental_analysis": collected_data.get("재무지표", {}),
            "news_analysis": [],
            "comprehensive_analysis": {},
            "report_markdown": "",
        }

    # 결과 저장
    saved_files = report_generator.save_full_output(analysis_result, analysis_date)

    return {
        "analysis_result": analysis_result,
        "saved_files": saved_files,
    }


# ============================================================
# 전체 파이프라인
# ============================================================

def run_full_pipeline(
    date: str = None,
    provider: str = None,
    step_config: StepProviderConfig = None,
    show_prompts: bool = False,
    prompt_only: bool = False,
) -> dict:
    """전체 파이프라인 실행 (Phase 1 + Phase 2+3)"""
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("한국 주식 시장 일일 분석 시작")
    logger.info("=" * 60)

    # Phase 1: 데이터 수집
    collected_data = run_collect(date)
    analysis_date = collected_data["기준일"]

    filtered_count = collected_data["필터링_결과"]["필터링_종목수"]
    if filtered_count == 0:
        return {
            "status": "no_data",
            "date": analysis_date,
            "message": "필터링 조건에 해당하는 종목이 없습니다.",
        }

    # Phase 2+3: AI 분석
    analyze_result = run_analyze(
        collected_data,
        provider=provider,
        step_config=step_config,
        show_prompts=show_prompts,
        prompt_only=prompt_only,
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    saved_files = analyze_result["saved_files"]

    logger.info("\n" + "=" * 60)
    logger.info(f"분석 완료! (소요 시간: {elapsed:.1f}초)")
    logger.info(f"리포트: {saved_files.get('markdown', 'N/A')}")
    logger.info(f"데이터: {saved_files.get('json', 'N/A')}")
    logger.info("=" * 60)

    return {
        "status": "success",
        "date": analysis_date,
        "filtered_count": filtered_count,
        "saved_files": saved_files,
        "elapsed_seconds": elapsed,
    }


# 하위 호환용 alias
def run_daily_analysis(
    date: str = None,
    provider: str = None,
    step_config: StepProviderConfig = None,
    show_prompts: bool = False,
    analyze_by_api: bool = True,
) -> dict:
    """하위 호환용 (app.py 등에서 사용)"""
    return run_full_pipeline(
        date=date,
        provider=provider,
        step_config=step_config,
        show_prompts=show_prompts,
        prompt_only=not analyze_by_api,
    )


# ============================================================
# CLI
# ============================================================

def _format_date(date: str) -> str:
    """YYYYMMDD → YYYY-MM-DD"""
    if "-" in date:
        return date
    if len(date) == 8:
        return f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    return date


def main():
    """CLI 엔트리포인트"""
    provider_choices = list(SUPPORTED_PROVIDERS.keys())

    parser = argparse.ArgumentParser(
        description="한국 주식 시장 AI 분석 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py                                          전체 파이프라인 (기본: 프롬프트만 저장)
  python main.py --api                                    API 호출로 실제 분석 수행
  python main.py --step collect --date 20260213           데이터 수집만
  python main.py --step analyze --from-data data.json     저장된 데이터에서 분석
  python main.py --step report --from-data full.json     Step 5(리포트)만 재실행
  python main.py --step4 gemini --step5 claude            Step별 프로바이더
  python main.py --schedule --api                         매일 자동 실행 (API 호출)
  streamlit run app.py                                    Streamlit 대시보드
        """,
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["collect", "analyze", "report", "all"],
        help="실행할 Phase (collect=데이터수집, analyze=Step4+5, report=Step5만, all=전체)",
    )
    parser.add_argument(
        "--from-data",
        type=str,
        default=None,
        dest="from_data",
        help="저장된 JSON 경로 (analyze=collected.json, report=full.json 등)",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        dest="api",
        help="AI API 호출 수행 (미지정 시 기본은 프롬프트만 저장)",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        dest="prompt_only",
        help="API 호출 없이 프롬프트만 저장 (기본 동작과 동일, 명시적 지정용)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=provider_choices,
        help="글로벌 AI 프로바이더 (미지정 Step에 적용)",
    )
    parser.add_argument(
        "--step4",
        type=str,
        default=None,
        choices=provider_choices,
        help="Step 4 (종합분석) 프로바이더",
    )
    parser.add_argument(
        "--step5",
        type=str,
        default=None,
        choices=provider_choices,
        help="Step 5 (리포트) 프로바이더",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="분석 대상 날짜 (YYYYMMDD 형식, 기본: 최근 거래일)",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="스케줄 모드로 실행 (매일 지정 시각에 자동 분석)",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="15:40",
        help="스케줄 실행 시각 (HH:MM 형식, 기본: 15:40)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="로깅 레벨 (기본: INFO)",
    )
    parser.add_argument(
        "--show-prompts",
        action="store_true",
        help="Step 4/5 프롬프트를 콘솔에 출력",
    )
    # 하위 호환용 (--no-api → --prompt-only)
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="(하위 호환) --prompt-only와 동일",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    # 기본: 프롬프트만 저장. --api 지정 시에만 API 호출.
    prompt_only = not args.api

    # Step별 프로바이더 설정
    step_args = (args.step4, args.step5)
    if any(s is not None for s in step_args):
        step_config = StepProviderConfig(
            step4=args.step4,
            step5=args.step5,
        )
    else:
        step_config = None  # .env에서 로드

    # 사용 가능한 프로바이더 표시
    available = get_available_providers()
    if available:
        logger.info(f"사용 가능한 AI 프로바이더: {', '.join(available)}")
    else:
        logger.warning("설정된 AI API 키가 없습니다. .env 파일을 확인하세요.")

    # ── 스케줄 모드 ──
    if args.schedule:
        scheduler = AnalysisScheduler(
            lambda: run_full_pipeline(
                args.date,
                provider=args.provider,
                step_config=step_config,
                show_prompts=args.show_prompts,
                prompt_only=prompt_only,
            )
        )
        scheduler.start(run_time=args.time)
        return

    # ── collect 모드: Phase 1만 ──
    if args.step == "collect":
        collected_data = run_collect(args.date)
        filtered_count = collected_data["필터링_결과"]["필터링_종목수"]
        print(f"\n데이터 수집 완료: {filtered_count}개 종목 필터링됨")
        if filtered_count > 0:
            reports_base = PROJECT_ROOT / "outputs" / "reports"
            save_path = get_report_date_dir(reports_base, collected_data["기준일"]) / "collected.json"
            print(f"저장 위치: {save_path}")
        return

    # ── analyze 모드: Phase 2+3만 (저장된 데이터에서 이어가기) ──
    if args.step == "analyze":
        if not args.from_data:
            parser.error("--step analyze는 --from-data 옵션이 필요합니다.")

        data_path = Path(args.from_data)
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path
        if not data_path.exists():
            print(f"파일을 찾을 수 없습니다: {data_path}")
            sys.exit(1)

        logger.info(f"[Phase 2+3] 저장된 데이터 로드: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            collected_data = json.load(f)

        analyze_result = run_analyze(
            collected_data,
            provider=args.provider,
            step_config=step_config,
            show_prompts=args.show_prompts,
            prompt_only=prompt_only,
        )
        saved_files = analyze_result["saved_files"]
        print(f"\nAI 분석 완료!")
        print(f"리포트: {saved_files.get('markdown', 'N/A')}")
        print(f"데이터: {saved_files.get('json', 'N/A')}")
        return

    # ── report 모드: Step 5(리포트)만 (full.json 등 분석 결과에서) ──
    if args.step == "report":
        if not args.from_data:
            parser.error("--step report는 --from-data 옵션이 필요합니다.")

        data_path = Path(args.from_data)
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path
        if not data_path.exists():
            print(f"파일을 찾을 수 없습니다: {data_path}")
            sys.exit(1)

        logger.info(f"[Step 5] 분석 결과 로드: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        comprehensive_analysis = data.get("comprehensive_analysis")
        if comprehensive_analysis is None:
            print("오류: JSON에 'comprehensive_analysis' 키가 없습니다. full.json 등 분석 결과 파일을 지정하세요.")
            sys.exit(1)
        if not comprehensive_analysis:
            print("오류: comprehensive_analysis가 비어 있어 리포트를 생성할 수 없습니다.")
            sys.exit(1)

        # 날짜 추출: filtered_analysis.기준일 > 기준일 > 경로(YYYY/MM/DD)
        date_str = None
        fa = data.get("filtered_analysis") or {}
        if isinstance(fa, dict) and fa.get("기준일"):
            date_str = fa["기준일"]
        if not date_str:
            date_str = data.get("기준일")
        if not date_str and "reports" in data_path.parts:
            try:
                idx = data_path.parts.index("reports")
                if idx + 3 <= len(data_path.parts):
                    yyyy, mm, dd = data_path.parts[idx + 1], data_path.parts[idx + 2], data_path.parts[idx + 3]
                    if len(yyyy) == 4 and len(mm) == 2 and len(dd) == 2:
                        date_str = f"{yyyy}{mm}{dd}"
            except (ValueError, IndexError):
                pass
        if not date_str:
            print("오류: 날짜를 추출할 수 없습니다. JSON에 기준일 또는 filtered_analysis.기준일이 있거나, 경로가 .../reports/YYYY/MM/DD/... 형식이어야 합니다.")
            sys.exit(1)

        # YYYY-MM-DD → YYYYMMDD 통일 (report_generator는 둘 다 처리)
        analysis_date = date_str.replace("-", "") if len(date_str) >= 8 else date_str
        if len(analysis_date) == 8 and analysis_date.isdigit():
            pass
        else:
            print(f"오류: 유효하지 않은 날짜 형식입니다: {date_str}")
            sys.exit(1)

        try:
            ai_analyzer = AIAnalyzer(
                provider=args.provider,
                step_config=step_config,
                show_prompts=args.show_prompts,
                analyze_by_api=not prompt_only,
            )
            report_markdown = ai_analyzer.generate_report(comprehensive_analysis, analysis_date)
        except Exception as e:
            logger.error(f"Step 5 리포트 생성 실패: {e}")
            print(f"\n리포트 생성 실패: {e}")
            sys.exit(1)

        report_generator = ReportGenerator()
        md_path = report_generator.save_markdown_report(report_markdown, analysis_date)
        print(f"\nStep 5 완료! 리포트 저장: {md_path}")
        return

    # ── all 모드: 전체 파이프라인 ──
    result = run_full_pipeline(
        args.date,
        provider=args.provider,
        step_config=step_config,
        show_prompts=args.show_prompts,
        prompt_only=prompt_only,
    )
    if result["status"] == "success":
        print(f"\n분석 완료: {result['filtered_count']}개 종목 분석됨")
        print(f"리포트: {result['saved_files'].get('markdown', 'N/A')}")
    else:
        print(f"\n{result.get('message', '분석 실패')}")


if __name__ == "__main__":
    main()
