"""
한국 주식 시장 분석 시스템 - 메인 엔트리포인트

사용법:
    python main.py                  # 즉시 1회 분석 실행
    python main.py --schedule       # 매일 15:40 자동 실행 모드
    python main.py --date 20260206  # 특정 날짜 분석
    streamlit run main.py           # Streamlit 대시보드 실행
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
from src.ai_analyzer import AIAnalyzer
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


def run_daily_analysis(date: str = None) -> dict:
    """
    일일 분석 파이프라인 실행

    프롬프트 체이닝 흐름:
    Step 1: pykrx로 원시 데이터 수집 (Python)
    Step 2: 프롬프트 1 → 필터링 및 정리 (AI)
    Step 3: pykrx + OpenDart로 재무지표 수집 (Python)
    Step 4: 프롬프트 2 → 재무지표 분석 (AI)
    Step 5: 네이버 검색 API로 뉴스 수집 (Python)
    Step 6: 프롬프트 3 → 뉴스 감성 분석 (AI)
    Step 7: 프롬프트 4 → 종합 분석 (AI)
    Step 8: 프롬프트 5 → 최종 리포트 생성 (AI)

    Args:
        date: 분석 날짜 (YYYYMMDD, 기본값: 최근 거래일)

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

    # ===== Step 2/4/6/7/8: AI 분석 (프롬프트 체이닝) =====
    logger.info("\n🤖 [Step 2/4/6/7/8] AI 프롬프트 체이닝 분석 시작...")
    try:
        ai_analyzer = AIAnalyzer()
        analysis_result = ai_analyzer.run_full_analysis(collected_data)
    except ValueError as e:
        logger.error(f"❌ AI 분석기 초기화 실패: {e}")
        logger.info("  → API 키 없이 데이터 수집 결과만 저장합니다.")
        analysis_result = {
            "filtered_analysis": collected_data["필터링_결과"],
            "fundamental_analysis": collected_data["재무지표"],
            "news_analysis": [],
            "comprehensive_analysis": {},
            "report_markdown": "",
        }
    except Exception as e:
        logger.error(f"❌ AI 분석 실패: {e}")
        analysis_result = {
            "filtered_analysis": collected_data["필터링_결과"],
            "fundamental_analysis": collected_data["재무지표"],
            "news_analysis": [],
            "comprehensive_analysis": {},
            "report_markdown": "",
        }

    # ===== 결과 저장 =====
    logger.info("\n💾 결과 저장 중...")
    saved_files = report_generator.save_full_output(analysis_result, analysis_date)

    # 수집 원시 데이터도 별도 저장
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
    parser = argparse.ArgumentParser(
        description="한국 주식 시장 AI 분석 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py                    즉시 1회 분석 실행
  python main.py --date 20260206    특정 날짜 분석
  python main.py --schedule         매일 자동 실행 (기본 15:40)
  python main.py --schedule --time 16:00  매일 16:00에 자동 실행
  streamlit run app.py              Streamlit 대시보드 실행
        """,
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

    if args.schedule:
        # 스케줄 모드
        scheduler = AnalysisScheduler(lambda: run_daily_analysis(args.date))
        scheduler.start(run_time=args.time)
    else:
        # 즉시 실행 모드
        result = run_daily_analysis(args.date)
        if result["status"] == "success":
            print(f"\n✅ 분석 완료: {result['filtered_count']}개 종목 분석됨")
            print(f"📄 리포트: {result['saved_files'].get('markdown', 'N/A')}")
        else:
            print(f"\n⚠️ {result.get('message', '분석 실패')}")


if __name__ == "__main__":
    main()
