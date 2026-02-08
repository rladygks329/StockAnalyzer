"""
AI 분석 모듈
Anthropic Claude API를 활용한 프롬프트 체이닝 분석
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = PROJECT_ROOT / "config" / "prompts"


class AIAnalyzer:
    """Claude API 기반 주식 분석 엔진"""

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY가 설정되지 않았습니다. "
                ".env 파일에 API 키를 설정하세요."
            )
        self.client = Anthropic(api_key=api_key)
        self.model_data = os.getenv("AI_MODEL_DATA", "claude-sonnet-4-20250514")
        self.model_analysis = os.getenv("AI_MODEL_ANALYSIS", "claude-sonnet-4-20250514")
        self.temp_data = float(os.getenv("TEMPERATURE_DATA", "0.1"))
        self.temp_analysis = float(os.getenv("TEMPERATURE_ANALYSIS", "0.4"))
        self.system_prompt = self._load_prompt("system_prompt.txt")

    @staticmethod
    def _load_prompt(filename: str) -> str:
        """프롬프트 파일 로드"""
        filepath = PROMPTS_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {filepath}")
        return filepath.read_text(encoding="utf-8")

    def _call_api(
        self,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
    ) -> str:
        """
        Claude API 호출

        Args:
            user_prompt: 사용자 프롬프트
            model: 사용할 모델 (기본: self.model_data)
            temperature: 응답 온도 (기본: self.temp_data)
            max_tokens: 최대 토큰 수

        Returns:
            str: AI 응답 텍스트
        """
        model = model or self.model_data
        temperature = temperature if temperature is not None else self.temp_data

        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text

        except Exception as e:
            logger.error(f"[AI 분석] API 호출 실패: {e}")
            raise

    @staticmethod
    def _extract_json(text: str) -> dict:
        """AI 응답에서 JSON 추출"""
        # 코드 블록 내 JSON 추출 시도
        import re

        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 전체 텍스트에서 JSON 추출 시도
        try:
            # 첫 번째 { 부터 마지막 } 까지 추출
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            logger.warning("[AI 분석] JSON 파싱 실패, 원본 텍스트 반환")
            return {"raw_response": text}

    def analyze_filtering(self, filtered_data: dict) -> dict:
        """
        프롬프트 1: 거래량/거래대금 필터링 분석

        Args:
            filtered_data: 필터링된 종목 데이터

        Returns:
            dict: AI 분석 결과
        """
        logger.info("[AI 분석] 프롬프트 1: 필터링 및 데이터 정리 시작")

        prompt_template = self._load_prompt("prompt_1_filtering.txt")

        # 데이터를 테이블 형태로 변환
        raw_table = self._format_data_table(filtered_data)

        prompt = prompt_template.replace("{날짜}", filtered_data.get("기준일", ""))
        prompt = prompt.replace("{raw_data_table}", raw_table)

        response = self._call_api(
            prompt, model=self.model_data, temperature=self.temp_data
        )
        result = self._extract_json(response)

        logger.info(
            f"[AI 분석] 프롬프트 1 완료: {result.get('필터링_종목수', 'N/A')}개 종목"
        )
        return result

    def analyze_fundamental(self, fundamental_data: dict) -> dict:
        """
        프롬프트 2: 재무지표 분석

        Args:
            fundamental_data: 재무지표 데이터

        Returns:
            dict: AI 분석 결과
        """
        logger.info("[AI 분석] 프롬프트 2: 재무지표 분석 시작")

        prompt_template = self._load_prompt("prompt_2_fundamental.txt")

        # 재무 데이터를 테이블 형태로 변환
        table = json.dumps(fundamental_data, ensure_ascii=False, indent=2)

        prompt = prompt_template.replace("{fundamental_data_table}", table)

        response = self._call_api(
            prompt, model=self.model_data, temperature=self.temp_data
        )
        result = self._extract_json(response)

        logger.info(
            f"[AI 분석] 프롬프트 2 완료: "
            f"{len(result.get('종목_분석', []))}개 종목 분석"
        )
        return result

    def analyze_news(self, news_data: dict) -> dict:
        """
        프롬프트 3: 뉴스 분석 및 감성 평가 (단일 종목)

        Args:
            news_data: 종목 뉴스 데이터

        Returns:
            dict: AI 분석 결과
        """
        stock_name = news_data.get("종목명", "")
        stock_code = news_data.get("종목코드", "")
        logger.info(f"[AI 분석] 프롬프트 3: {stock_name}({stock_code}) 뉴스 분석 시작")

        prompt_template = self._load_prompt("prompt_3_news.txt")

        # 뉴스 리스트를 텍스트로 변환
        news_list_text = self._format_news_list(news_data.get("뉴스_리스트", []))

        prompt = prompt_template.replace("{종목명}", stock_name)
        prompt = prompt.replace("{종목코드}", stock_code)
        prompt = prompt.replace("{N}", str(news_data.get("수집_뉴스수", 0)))
        prompt = prompt.replace("{news_list}", news_list_text)

        response = self._call_api(
            prompt, model=self.model_data, temperature=self.temp_data
        )
        result = self._extract_json(response)

        logger.info(
            f"[AI 분석] 프롬프트 3 완료: {stock_name} "
            f"뉴스 스코어 {result.get('종합_뉴스스코어', 'N/A')}"
        )
        return result

    def analyze_all_news(self, all_news_data: list[dict]) -> list[dict]:
        """
        프롬프트 3: 모든 종목 뉴스 분석 (병렬 처리 가능)

        Args:
            all_news_data: 전체 종목 뉴스 데이터 리스트

        Returns:
            list[dict]: 전체 뉴스 분석 결과
        """
        logger.info(f"[AI 분석] {len(all_news_data)}개 종목 뉴스 분석 시작")
        results = []
        for news_data in all_news_data:
            try:
                result = self.analyze_news(news_data)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"[AI 분석] {news_data.get('종목명', '')} 뉴스 분석 실패: {e}"
                )
                results.append(
                    {
                        "종목명": news_data.get("종목명", ""),
                        "종목코드": news_data.get("종목코드", ""),
                        "분석_뉴스수": 0,
                        "뉴스_분석": [],
                        "종합_뉴스스코어": 0,
                        "뉴스_요약": "뉴스 분석 실패",
                    }
                )
        return results

    def analyze_comprehensive(
        self,
        filtered_stocks: dict,
        fundamental_analysis: dict,
        news_analysis: list[dict],
        market_index: dict,
        foreign_trading: dict,
    ) -> dict:
        """
        프롬프트 4: 시장 동향 종합 분석

        Args:
            filtered_stocks: 필터링 결과
            fundamental_analysis: 재무지표 분석 결과
            news_analysis: 뉴스 분석 결과
            market_index: 시장 지수 데이터
            foreign_trading: 외국인 매매 동향

        Returns:
            dict: 종합 분석 결과
        """
        logger.info("[AI 분석] 프롬프트 4: 종합 분석 시작")

        prompt_template = self._load_prompt("prompt_4_analysis.txt")

        # 데이터 치환
        prompt = prompt_template.replace(
            "{filtered_stocks_json}",
            json.dumps(filtered_stocks, ensure_ascii=False, indent=2),
        )
        prompt = prompt.replace(
            "{fundamental_analysis_json}",
            json.dumps(fundamental_analysis, ensure_ascii=False, indent=2),
        )
        prompt = prompt.replace(
            "{news_analysis_json}",
            json.dumps(news_analysis, ensure_ascii=False, indent=2),
        )

        # 시장 환경 데이터 치환
        kospi = market_index.get("코스피", {})
        kosdaq = market_index.get("코스닥", {})

        prompt = prompt.replace(
            "{kospi_index}", str(kospi.get("지수", "데이터 없음"))
        )
        prompt = prompt.replace(
            "{kospi_change}", str(kospi.get("등락률", "데이터 없음"))
        )
        prompt = prompt.replace(
            "{kosdaq_index}", str(kosdaq.get("지수", "데이터 없음"))
        )
        prompt = prompt.replace(
            "{kosdaq_change}", str(kosdaq.get("등락률", "데이터 없음"))
        )
        prompt = prompt.replace("{exchange_rate}", "데이터 별도 수집 필요")
        prompt = prompt.replace(
            "{foreign_flow}", foreign_trading.get("외국인_순매수_판단", "데이터 없음")
        )

        response = self._call_api(
            prompt,
            model=self.model_analysis,
            temperature=self.temp_analysis,
            max_tokens=8192,
        )
        result = self._extract_json(response)

        logger.info("[AI 분석] 프롬프트 4 종합 분석 완료")
        return result

    def generate_report(self, final_analysis: dict) -> str:
        """
        프롬프트 5: 최종 리포트 생성

        Args:
            final_analysis: 종합 분석 결과 (프롬프트 4 출력)

        Returns:
            str: 마크다운 리포트 텍스트
        """
        logger.info("[AI 분석] 프롬프트 5: 최종 리포트 생성 시작")

        prompt_template = self._load_prompt("prompt_5_report.txt")

        prompt = prompt_template.replace(
            "{final_analysis_json}",
            json.dumps(final_analysis, ensure_ascii=False, indent=2),
        )

        report = self._call_api(
            prompt,
            model=self.model_analysis,
            temperature=self.temp_analysis,
            max_tokens=8192,
        )

        logger.info("[AI 분석] 프롬프트 5 최종 리포트 생성 완료")
        return report

    def run_full_analysis(self, collected_data: dict) -> dict:
        """
        전체 프롬프트 체이닝 실행

        Args:
            collected_data: data_collector에서 수집한 전체 데이터

        Returns:
            dict: {
                "filtered_analysis": ...,
                "fundamental_analysis": ...,
                "news_analysis": ...,
                "comprehensive_analysis": ...,
                "report_markdown": ...,
            }
        """
        logger.info("=" * 60)
        logger.info("[AI 분석] 전체 프롬프트 체이닝 시작")
        logger.info("=" * 60)

        # Step 1: 필터링 분석 (프롬프트 1)
        filtered_data = collected_data["필터링_결과"]
        filtered_analysis = self.analyze_filtering(filtered_data)

        # Step 2: 재무지표 분석 (프롬프트 2)
        fundamental_data = collected_data["재무지표"]
        fundamental_analysis = self.analyze_fundamental(fundamental_data)

        # Step 3: 뉴스 분석 (프롬프트 3) — 뉴스 데이터가 있는 경우
        news_data = collected_data.get("뉴스_데이터", [])
        news_analysis = self.analyze_all_news(news_data) if news_data else []

        # Step 4: 종합 분석 (프롬프트 4)
        comprehensive_analysis = self.analyze_comprehensive(
            filtered_stocks=filtered_analysis,
            fundamental_analysis=fundamental_analysis,
            news_analysis=news_analysis,
            market_index=collected_data.get("시장_지수", {}),
            foreign_trading=collected_data.get("외국인_매매", {}),
        )

        # Step 5: 최종 리포트 생성 (프롬프트 5)
        report_markdown = self.generate_report(comprehensive_analysis)

        result = {
            "filtered_analysis": filtered_analysis,
            "fundamental_analysis": fundamental_analysis,
            "news_analysis": news_analysis,
            "comprehensive_analysis": comprehensive_analysis,
            "report_markdown": report_markdown,
        }

        logger.info("[AI 분석] 전체 프롬프트 체이닝 완료")
        return result

    # ========== 유틸리티 메서드 ==========

    @staticmethod
    def _format_data_table(data: dict) -> str:
        """필터링 데이터를 읽기 좋은 테이블 형태로 변환"""
        stocks = data.get("종목_리스트", [])
        if not stocks:
            return "필터링된 종목이 없습니다."

        lines = [
            "| 종목코드 | 종목명 | 종가 | 등락률(%) | 거래량 | 거래대금 | 평균거래량대비 |",
            "|----------|--------|------|-----------|--------|----------|---------------|",
        ]
        for s in stocks:
            ratio = s.get("평균거래량_대비_배율")
            ratio_str = f"{ratio}배" if ratio else "N/A"
            lines.append(
                f"| {s['종목코드']} | {s['종목명']} | "
                f"{s['종가']:,} | {s['등락률']} | "
                f"{s['거래량']:,} | {s['거래대금']:,} | {ratio_str} |"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_news_list(news_list: list[dict]) -> str:
        """뉴스 리스트를 텍스트로 변환"""
        if not news_list:
            return "수집된 뉴스가 없습니다."

        lines = []
        for i, news in enumerate(news_list, 1):
            lines.append(f"### 뉴스 {i}")
            lines.append(f"- **제목**: {news.get('제목', 'N/A')}")
            lines.append(f"- **날짜**: {news.get('날짜', 'N/A')}")
            lines.append(f"- **요약**: {news.get('요약', 'N/A')}")
            lines.append(f"- **출처**: {news.get('출처', 'N/A')}")
            lines.append("")
        return "\n".join(lines)
