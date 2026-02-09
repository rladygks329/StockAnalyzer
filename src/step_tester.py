import argparse
import sys

from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collector import StockDataCollector
from src.news_collector import NewsCollector
from src.ai_analyzer import AIAnalyzer, StepProviderConfig
from src.report_generator import ReportGenerator


def step1_collect_data(date: str | None = None):
    """Step 1: 시장 데이터 및 재무지표 수집만 테스트"""
    collector = StockDataCollector()
    collected_data = collector.collect_all_data(date)

    print("\n[STEP 1] 데이터 수집 결과 요약")
    print(f"- 기준일: {collected_data['기준일']}")
    print(f"- 필터링 종목 수: {collected_data['필터링_결과']['필터링_종목수']}")
    print(f"- 예시 종목 5개: {collected_data['필터링_결과']['종목_리스트'][:5]}")

    return collected_data


def step2_collect_news(collected_data: dict, num_articles_per_stock: int = 10):
    """Step 2: 뉴스 수집만 별도 테스트 (데이터 수집 결과를 입력으로 사용)"""
    news_collector = NewsCollector()
    stocks = collected_data["필터링_결과"]["종목_리스트"]

    print(f"\n[STEP 2] 뉴스 수집 시작 (종목 수: {len(stocks)})")
    all_news = news_collector.collect_all_stock_news(
        stocks, num_articles_per_stock=num_articles_per_stock
    )
    collected_data["뉴스_데이터"] = all_news

    total_news = sum(n["수집_뉴스수"] for n in all_news)
    print(f"- 총 수집 뉴스 수: {total_news}")

    return collected_data


def step3_ai_analysis(collected_data: dict, provider: str | None = None,
                      step_config: StepProviderConfig | None = None):
    """Step 3: AI 분석만 별도 테스트 (저장까지 포함)"""
    analyzer = AIAnalyzer(provider=provider, step_config=step_config)

    print("\n[STEP 3] AI 프롬프트 체이닝 분석 실행...")
    analysis_result = analyzer.run_full_analysis(collected_data)

    print("- 사용된 Provider 정보:", analysis_result.get("ai_providers"))
    print("- 필터링 결과 종목 수:",
          len(analysis_result.get("filtered_analysis", {}).get("종목_리스트", [])))
    print("- 뉴스 분석 결과 길이:", len(analysis_result.get("news_analysis", [])))

    # 원하면 여기서 리포트 저장까지 테스트
    report_generator = ReportGenerator()
    analysis_date = collected_data["기준일"]
    saved_files = report_generator.save_full_output(analysis_result, analysis_date)

    print("\n[STEP 3] 결과 저장 완료")
    print("- Markdown 리포트:", saved_files.get("markdown"))
    print("- JSON 결과:", saved_files.get("json"))

    return analysis_result, saved_files


def run_pipeline(step: str, date: str | None, provider: str | None):
    """
    step:
        - "1": 데이터 수집만
        - "2": 데이터 수집 + 뉴스 수집
        - "3": 데이터 수집 + 뉴스 수집 + AI 분석/리포트 (전체)
    """
    start = datetime.now()

    # Step 1 공통: 데이터 수집
    collected_data = step1_collect_data(date)

    if collected_data["필터링_결과"]["필터링_종목수"] == 0:
        print("\n⚠️ 필터링 조건에 해당하는 종목이 없습니다. 여기서 종료합니다.")
        return

    if step in ("2", "3"):
        # Step 2: 뉴스 수집
        collected_data = step2_collect_news(collected_data)

    if step == "3":
        # Step 3: AI 분석 + 리포트 저장
        step3_ai_analysis(collected_data, provider=provider)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n⏱ 총 소요 시간: {elapsed:.1f}초 (step={step})")


def main():
    parser = argparse.ArgumentParser(
        description="StockAnalyzer 단계별 테스트 스크립트",
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["1", "2", "3"],
        default="3",
        help="테스트할 단계 (1=데이터, 2=데이터+뉴스, 3=전체 파이프라인)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="분석 날짜 (YYYYMMDD, 기본: 최근 거래일 로직 사용)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="AI 글로벌 프로바이더 (예: claude, gpt, gemini 등)",
    )

    args = parser.parse_args()
    run_pipeline(step=args.step, date=args.date, provider=args.provider)


if __name__ == "__main__":
    main()