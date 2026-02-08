"""
뉴스 수집 모듈
네이버 검색 API를 활용한 종목별 뉴스 수집
"""

import os
import re
import logging
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class NewsCollector:
    """네이버 검색 API 기반 뉴스 수집기"""

    NAVER_SEARCH_URL = "https://openapi.naver.com/v1/search/news.json"

    def __init__(self):
        self.client_id = os.getenv("NAVER_CLIENT_ID", "")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET", "")

        if not self.client_id or not self.client_secret:
            logger.warning(
                "[뉴스 수집] 네이버 API 키가 설정되지 않았습니다. "
                "config/.env.example 파일을 참고하여 .env 파일을 생성하세요."
            )

    @property
    def _headers(self) -> dict:
        return {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }

    @staticmethod
    def _clean_html(text: str) -> str:
        """HTML 태그 및 특수 문자 제거"""
        text = re.sub(r"<[^>]+>", "", text)
        text = text.replace("&quot;", '"')
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&apos;", "'")
        return text.strip()

    @staticmethod
    def _parse_naver_date(date_str: str) -> str:
        """네이버 API 날짜 형식 변환"""
        try:
            dt = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return date_str

    def search_news(
        self,
        query: str,
        display: int = 10,
        sort: str = "date",
        start: int = 1,
    ) -> list[dict]:
        """
        네이버 뉴스 검색

        Args:
            query: 검색어 (종목명)
            display: 검색 결과 수 (최대 100)
            sort: 정렬 기준 ("date": 날짜순, "sim": 유사도순)
            start: 검색 시작 위치

        Returns:
            list[dict]: 뉴스 리스트
        """
        if not self.client_id or not self.client_secret:
            logger.error("[뉴스 수집] 네이버 API 키가 설정되지 않았습니다.")
            return []

        params = {
            "query": query,
            "display": min(display, 100),
            "start": start,
            "sort": sort,
        }

        try:
            response = requests.get(
                self.NAVER_SEARCH_URL,
                headers=self._headers,
                params=params,
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            news_items = []
            for item in data.get("items", []):
                news_items.append(
                    {
                        "제목": self._clean_html(item.get("title", "")),
                        "링크": item.get("link", ""),
                        "요약": self._clean_html(item.get("description", "")),
                        "날짜": self._parse_naver_date(item.get("pubDate", "")),
                        "출처": item.get("originallink", ""),
                    }
                )

            logger.info(f'[뉴스 수집] "{query}" 검색 결과 {len(news_items)}건')
            return news_items

        except requests.exceptions.RequestException as e:
            logger.error(f'[뉴스 수집] "{query}" 뉴스 검색 실패: {e}')
            return []

    def collect_stock_news(
        self,
        stock_name: str,
        stock_code: str,
        num_articles: int = 10,
    ) -> dict:
        """
        특정 종목의 뉴스 수집 (프롬프트 3 입력용)

        Args:
            stock_name: 종목명
            stock_code: 종목코드
            num_articles: 수집할 기사 수

        Returns:
            dict: 종목 뉴스 데이터
        """
        logger.info(f"[뉴스 수집] {stock_name}({stock_code}) 뉴스 수집 시작")

        # 종목명으로 검색 (주식 키워드 추가)
        query = f"{stock_name} 주식"
        news_items = self.search_news(query, display=num_articles, sort="date")

        # 종목코드로도 추가 검색하여 보충
        if len(news_items) < num_articles:
            code_news = self.search_news(
                stock_name,
                display=num_articles - len(news_items),
                sort="sim",
            )
            # 중복 제거
            existing_titles = {n["제목"] for n in news_items}
            for n in code_news:
                if n["제목"] not in existing_titles:
                    news_items.append(n)

        return {
            "종목명": stock_name,
            "종목코드": stock_code,
            "수집_뉴스수": len(news_items),
            "뉴스_리스트": news_items[:num_articles],
        }

    def collect_all_stock_news(
        self,
        stocks: list[dict],
        num_articles_per_stock: int = 10,
    ) -> list[dict]:
        """
        필터링된 모든 종목의 뉴스 수집

        Args:
            stocks: 종목 리스트 [{"종목코드": "...", "종목명": "..."}, ...]
            num_articles_per_stock: 종목당 수집 기사 수

        Returns:
            list[dict]: 전체 종목 뉴스 데이터
        """
        logger.info(f"[뉴스 수집] {len(stocks)}개 종목 뉴스 수집 시작")

        all_news = []
        for stock in stocks:
            news_data = self.collect_stock_news(
                stock_name=stock["종목명"],
                stock_code=stock["종목코드"],
                num_articles=num_articles_per_stock,
            )
            all_news.append(news_data)

        logger.info(
            f"[뉴스 수집] 전체 {len(all_news)}개 종목 뉴스 수집 완료 "
            f"(총 {sum(n['수집_뉴스수'] for n in all_news)}건)"
        )
        return all_news

    def collect_market_news(self, num_articles: int = 10) -> list[dict]:
        """
        시장 전체 관련 뉴스 수집

        Args:
            num_articles: 수집할 기사 수

        Returns:
            list[dict]: 시장 뉴스 리스트
        """
        logger.info("[뉴스 수집] 시장 전체 뉴스 수집 시작")

        queries = ["코스피", "코스닥 시장", "주식시장 동향"]
        all_news = []
        existing_titles = set()

        for query in queries:
            news_items = self.search_news(query, display=5, sort="date")
            for n in news_items:
                if n["제목"] not in existing_titles:
                    all_news.append(n)
                    existing_titles.add(n["제목"])

        return all_news[:num_articles]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    collector = NewsCollector()

    # 테스트: 삼성전자 뉴스 수집
    result = collector.collect_stock_news("삼성전자", "005930", num_articles=5)
    print(f"\n{result['종목명']} 뉴스 {result['수집_뉴스수']}건:")
    for news in result["뉴스_리스트"]:
        print(f"  [{news['날짜']}] {news['제목']}")
