"""
한국 주식 시장 분석 시스템 - 메인 엔트리포인트

사용법:
    python main.py                                      # .env 설정대로 실행
    python main.py --provider gpt                       # 전체 Step을 GPT로
    python main.py --step1 gemini --step4 claude        # Step별 프로바이더 지정
    python main.py --date 20260206                      # 특정 날짜 분석
    python main.py --schedule                           # 매일 15:40 자동 실행
    streamlit run app.py                                # 대시보드 실행
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 프로젝트 루트 경로 설정
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
from src.report_generator import ReportGenerator
from src.scheduler import AnalysisScheduler


def setup_logging(level: str = "INFO") -> None:
    """로깅 설정"""
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    log_date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt=log_date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                PROJECT_ROOT / "outputs" / "analysis.log",
                encoding="utf-8",
            ),
        ],
    )


logger = logging.getLogger(__name__)


def run_daily_analysis(
    date: str = None,
    provider: str = None,
    step_config: StepProviderConfig = None,
) -> dict:
    """
    일일 분석 파이프라인 실행

    Args:
        date: 분석 날짜 (YYYYMMDD, 기본값: 최근 거래일)
        provider: 글로벌 AI 프로바이더 (모든 Step에 적용, step_config보다 우선도 낮음)
        step_config: Step별 프로바이더 설정 (가장 높은 우선도)

    Returns:
        dict: 분석 결과 및 저장 파일 경로
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("🚀 한국 주식 시장 일일 분석 시작")
    logger.info("=" * 60)

    # 모듈 초기화
    data_collector = StockDataCollector()
    news_collector = NewsCollector()
    report_generator = ReportGenerator()

    # ===== Step 1 & 3: 데이터 수집 (Python) =====
    logger.info("\n📥 [Step 1/3] 시장 데이터 및 재무지표 수집...")
    collected_data = data_collector.collect_all_data(date)
    analysis_date = collected_data["기준일"]

    filtered_count = collected_data["필터링_결과"]["필터링_종목수"]
    logger.info(f"  → 필터링 종목 수: {filtered_count}개")

    if filtered_count == 0:
        logger.warning("⚠️ 필터링 조건에 해당하는 종목이 없습니다.")
        logger.info("  → 조건을 완화하여 재시도하거나 날짜를 확인하세요.")
        return {
            "status": "no_data",
            "date": analysis_date,
            "message": "필터링 조건에 해당하는 종목이 없습니다.",
        }

    # ===== Step 5: 뉴스 수집 (Python) =====
    logger.info("\n📰 [Step 5] 종목별 뉴스 수집...")
    stocks = collected_data["필터링_결과"]["종목_리스트"]
    all_news = news_collector.collect_all_stock_news(stocks, num_articles_per_stock=10)
    collected_data["뉴스_데이터"] = all_news
    logger.info(
        f"  → 총 {sum(n['수집_뉴스수'] for n in all_news)}건 뉴스 수집 완료"
    )

    # ===== AI 분석 (프롬프트 체이닝) =====
    logger.info("\n🤖 AI 프롬프트 체이닝 분석 시작...")
    default_provider = provider or os.getenv("AI_PROVIDER", "claude")
    try:
        ai_analyzer = AIAnalyzer(
            provider=provider,
            step_config=step_config,
        )
        analysis_result = ai_analyzer.run_full_analysis(collected_data)
    except ValueError as e:
        logger.error(f"❌ AI 분석기 초기화 실패: {e}")
        logger.info("  → API 키 없이 데이터 수집 결과만 저장합니다.")
        analysis_result = {
            "ai_providers": {},
            "ai_default_provider": default_provider,
            "filtered_analysis": collected_data["필터링_결과"],
            "fundamental_analysis": collected_data["재무지표"],
            "news_analysis": [],
            "comprehensive_analysis": {},
            "report_markdown": "",
        }
    except Exception as e:
        logger.error(f"❌ AI 분석 실패: {e}")
        analysis_result = {
            "ai_providers": {},
            "ai_default_provider": default_provider,
            "filtered_analysis": collected_data["필터링_결과"],
            "fundamental_analysis": collected_data["재무지표"],
            "news_analysis": [],
            "comprehensive_analysis": {},
            "report_markdown": "",
        }

    # ===== 결과 저장 =====
    logger.info("\n💾 결과 저장 중...")
    saved_files = report_generator.save_full_output(analysis_result, analysis_date)

    raw_data_path = report_generator.save_json_data(
        collected_data, analysis_date, "raw_collected"
    )
    saved_files["raw_data"] = raw_data_path

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "=" * 60)
    logger.info(f"✅ 분석 완료! (소요 시간: {elapsed:.1f}초)")
    logger.info(f"📄 리포트: {saved_files.get('markdown', 'N/A')}")
    logger.info(f"📊 데이터: {saved_files.get('json', 'N/A')}")
    logger.info("=" * 60)

    return {
        "status": "success",
        "date": analysis_date,
        "filtered_count": filtered_count,
        "saved_files": saved_files,
        "elapsed_seconds": elapsed,
    }


def main():
    """CLI 엔트리포인트"""
    provider_choices = list(SUPPORTED_PROVIDERS.keys())

    parser = argparse.ArgumentParser(
        description="한국 주식 시장 AI 분석 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py                                    .env 설정대로 분석
  python main.py --provider gpt                     전체 Step을 GPT로
  python main.py --step1 gemini --step4 claude      Step별 프로바이더 지정
  python main.py --step1 gemini --step2 gemini --step3 gemini \\
                 --step4 claude --step5 claude       데이터=Gemini, 분석=Claude
  python main.py --date 20260206                    특정 날짜 분석
  python main.py --schedule                         매일 자동 실행 (기본 15:40)
  streamlit run app.py                              Streamlit 대시보드
        """,
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=provider_choices,
        help="글로벌 AI 프로바이더 (미지정 Step에 적용)",
    )
    parser.add_argument(
        "--step1",
        type=str,
        default=None,
        choices=provider_choices,
        help="Step 1 (필터링) 프로바이더",
    )
    parser.add_argument(
        "--step2",
        type=str,
        default=None,
        choices=provider_choices,
        help="Step 2 (재무분석) 프로바이더",
    )
    parser.add_argument(
        "--step3",
        type=str,
        default=None,
        choices=provider_choices,
        help="Step 3 (뉴스분석) 프로바이더",
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

    args = parser.parse_args()
    setup_logging(args.log_level)

    # Step별 프로바이더 설정: CLI에서 명시한 것만 사용, 나머지는 .env 로드
    step_args = (args.step1, args.step2, args.step3, args.step4, args.step5)
    if any(s is not None for s in step_args):
        step_config = StepProviderConfig(
            step1=args.step1,
            step2=args.step2,
            step3=args.step3,
            step4=args.step4,
            step5=args.step5,
        )
    else:
        step_config = None  # .env의 STEP1_PROVIDER 등 사용

    # 사용 가능한 프로바이더 표시
    available = get_available_providers()
    if available:
        logger.info(f"사용 가능한 AI 프로바이더: {', '.join(available)}")
    else:
        logger.warning("설정된 AI API 키가 없습니다. .env 파일을 확인하세요.")

    if args.schedule:
        scheduler = AnalysisScheduler(
            lambda: run_daily_analysis(
                args.date,
                provider=args.provider,
                step_config=step_config,
            )
        )
        scheduler.start(run_time=args.time)
    else:
        result = run_daily_analysis(
            args.date,
            provider=args.provider,
            step_config=step_config,
        )
        if result["status"] == "success":
            print(f"\n✅ 분석 완료: {result['filtered_count']}개 종목 분석됨")
            print(f"📄 리포트: {result['saved_files'].get('markdown', 'N/A')}")
        else:
            print(f"\n⚠️ {result.get('message', '분석 실패')}")


if __name__ == "__main__":
    main()
