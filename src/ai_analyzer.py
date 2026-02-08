"""
AI 분석 모듈
멀티 프로바이더(Claude, GPT, Gemini, Grok) 지원 + Step별 프로바이더 분리
"""

import os
import json
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = PROJECT_ROOT / "config" / "prompts"

# ============================================================
# Step 정의
# ============================================================

STEP_DEFINITIONS = {
    1: {"name": "필터링", "desc": "거래량/거래대금 필터링 및 데이터 정리", "env": "STEP1_PROVIDER", "type": "data"},
    2: {"name": "재무분석", "desc": "재무지표 분석 (PER/PBR/EPS/ROE)", "env": "STEP2_PROVIDER", "type": "data"},
    3: {"name": "뉴스분석", "desc": "뉴스 감성 분석", "env": "STEP3_PROVIDER", "type": "data"},
    4: {"name": "종합분석", "desc": "시장 동향 종합 분석", "env": "STEP4_PROVIDER", "type": "analysis"},
    5: {"name": "리포트", "desc": "최종 리포트 생성", "env": "STEP5_PROVIDER", "type": "analysis"},
}

# ============================================================
# 지원 프로바이더 목록 및 설정
# ============================================================
SUPPORTED_PROVIDERS = {
    "claude": {
        "name": "Anthropic Claude",
        "env_key": "ANTHROPIC_API_KEY",
        "model_data_env": "CLAUDE_MODEL_DATA",
        "model_analysis_env": "CLAUDE_MODEL_ANALYSIS",
        "default_model_data": "claude-sonnet-4-20250514",
        "default_model_analysis": "claude-sonnet-4-20250514",
    },
    "gpt": {
        "name": "OpenAI GPT",
        "env_key": "OPENAI_API_KEY",
        "model_data_env": "GPT_MODEL_DATA",
        "model_analysis_env": "GPT_MODEL_ANALYSIS",
        "default_model_data": "gpt-4o",
        "default_model_analysis": "gpt-4o",
    },
    "gemini": {
        "name": "Google Gemini",
        "env_key": "GEMINI_API_KEY",
        "model_data_env": "GEMINI_MODEL_DATA",
        "model_analysis_env": "GEMINI_MODEL_ANALYSIS",
        "default_model_data": "gemini-2.0-flash",
        "default_model_analysis": "gemini-2.0-flash",
    },
    "grok": {
        "name": "xAI Grok",
        "env_key": "GROK_API_KEY",
        "model_data_env": "GROK_MODEL_DATA",
        "model_analysis_env": "GROK_MODEL_ANALYSIS",
        "default_model_data": "grok-3",
        "default_model_analysis": "grok-3",
    },
}


def get_available_providers() -> list[str]:
    """API 키가 설정된 사용 가능한 프로바이더 목록 반환"""
    available = []
    for provider_id, config in SUPPORTED_PROVIDERS.items():
        api_key = os.getenv(config["env_key"], "")
        if api_key and not api_key.startswith("your_"):
            available.append(provider_id)
    return available


def get_provider_display_name(provider_id: str) -> str:
    """프로바이더 표시 이름 반환"""
    return SUPPORTED_PROVIDERS.get(provider_id, {}).get("name", provider_id)


# ============================================================
# Step별 프로바이더 설정
# ============================================================

@dataclass
class StepProviderConfig:
    """Step별 프로바이더 설정"""
    step1: Optional[str] = None  # 필터링
    step2: Optional[str] = None  # 재무분석
    step3: Optional[str] = None  # 뉴스분석
    step4: Optional[str] = None  # 종합분석
    step5: Optional[str] = None  # 리포트

    def get(self, step: int) -> Optional[str]:
        """특정 Step의 프로바이더 반환 (None이면 글로벌 기본값 사용)"""
        return getattr(self, f"step{step}", None)

    def set(self, step: int, provider: Optional[str]):
        """특정 Step의 프로바이더 설정"""
        setattr(self, f"step{step}", provider)

    def to_dict(self) -> dict:
        return {f"step{i}": self.get(i) for i in range(1, 6)}

    @classmethod
    def from_dict(cls, d: dict) -> "StepProviderConfig":
        return cls(
            step1=d.get("step1"),
            step2=d.get("step2"),
            step3=d.get("step3"),
            step4=d.get("step4"),
            step5=d.get("step5"),
        )

    @classmethod
    def from_env(cls) -> "StepProviderConfig":
        """환경 변수에서 Step별 설정 로드"""
        config = cls()
        for step_num, step_def in STEP_DEFINITIONS.items():
            val = os.getenv(step_def["env"], "").strip()
            if val and val in SUPPORTED_PROVIDERS:
                config.set(step_num, val)
        return config

    @classmethod
    def all_same(cls, provider: str) -> "StepProviderConfig":
        """모든 Step을 동일 프로바이더로 설정"""
        return cls(step1=provider, step2=provider, step3=provider,
                   step4=provider, step5=provider)


def get_step_provider_summary(step_config: StepProviderConfig, fallback: str) -> dict:
    """Step별 프로바이더 요약 정보 반환"""
    summary = {}
    for step_num, step_def in STEP_DEFINITIONS.items():
        provider = step_config.get(step_num) or fallback
        summary[step_num] = {
            "step": step_num,
            "name": step_def["name"],
            "desc": step_def["desc"],
            "provider": provider,
            "provider_name": get_provider_display_name(provider),
        }
    return summary


# ============================================================
# 프로바이더별 클라이언트 추상 클래스
# ============================================================


class BaseAIClient(ABC):
    """AI 프로바이더 공통 인터페이스"""

    def __init__(self, provider_id: str):
        self.provider_id = provider_id
        config = SUPPORTED_PROVIDERS[provider_id]
        self.provider_name = config["name"]
        self.model_data = os.getenv(
            config["model_data_env"], config["default_model_data"]
        )
        self.model_analysis = os.getenv(
            config["model_analysis_env"], config["default_model_analysis"]
        )
        self.temp_data = float(os.getenv("TEMPERATURE_DATA", "0.1"))
        self.temp_analysis = float(os.getenv("TEMPERATURE_ANALYSIS", "0.4"))

        logger.info(
            f"[AI] {self.provider_name} 초기화 완료 "
            f"(데이터: {self.model_data}, 분석: {self.model_analysis})"
        )

    @abstractmethod
    def call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """AI API 호출 (프로바이더별 구현)"""
        ...


class ClaudeClient(BaseAIClient):
    """Anthropic Claude 클라이언트"""

    def __init__(self):
        super().__init__("claude")
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        self.client = Anthropic(api_key=api_key)

    def call_api(self, system_prompt, user_prompt, model, temperature, max_tokens):
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text


class GPTClient(BaseAIClient):
    """OpenAI GPT 클라이언트"""

    def __init__(self):
        super().__init__("gpt")
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
        self.client = OpenAI(api_key=api_key)

    def call_api(self, system_prompt, user_prompt, model, temperature, max_tokens):
        response = self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


class GeminiClient(BaseAIClient):
    """Google Gemini 클라이언트"""

    def __init__(self):
        super().__init__("gemini")
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다.")
        genai.configure(api_key=api_key)
        self._genai = genai

    def call_api(self, system_prompt, user_prompt, model, temperature, max_tokens):
        gen_model = self._genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt,
            generation_config=self._genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        response = gen_model.generate_content(user_prompt)
        return response.text


class GrokClient(BaseAIClient):
    """xAI Grok 클라이언트 (OpenAI 호환 API)"""

    GROK_BASE_URL = "https://api.x.ai/v1"

    def __init__(self):
        super().__init__("grok")
        from openai import OpenAI

        api_key = os.getenv("GROK_API_KEY", "")
        if not api_key:
            raise ValueError("GROK_API_KEY가 설정되지 않았습니다.")
        self.client = OpenAI(api_key=api_key, base_url=self.GROK_BASE_URL)

    def call_api(self, system_prompt, user_prompt, model, temperature, max_tokens):
        response = self.client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content


# ============================================================
# 클라이언트 팩토리 + 캐시
# ============================================================

_CLIENT_MAP = {
    "claude": ClaudeClient,
    "gpt": GPTClient,
    "gemini": GeminiClient,
    "grok": GrokClient,
}


class ClientPool:
    """프로바이더 클라이언트 풀 (동일 프로바이더 재사용)"""

    def __init__(self):
        self._clients: dict[str, BaseAIClient] = {}

    def get(self, provider_id: str) -> BaseAIClient:
        """클라이언트를 가져오거나 없으면 생성"""
        if provider_id not in self._clients:
            if provider_id not in _CLIENT_MAP:
                raise ValueError(
                    f"지원하지 않는 AI 프로바이더: '{provider_id}'. "
                    f"사용 가능: {', '.join(_CLIENT_MAP.keys())}"
                )
            self._clients[provider_id] = _CLIENT_MAP[provider_id]()
        return self._clients[provider_id]

    @property
    def active_providers(self) -> list[str]:
        return list(self._clients.keys())


def create_ai_client(provider: Optional[str] = None) -> BaseAIClient:
    """단일 AI 클라이언트 생성 (하위 호환용)"""
    if provider is None:
        provider = os.getenv("AI_PROVIDER", "claude").lower().strip()
    if provider not in _CLIENT_MAP:
        raise ValueError(
            f"지원하지 않는 AI 프로바이더: '{provider}'. "
            f"사용 가능: {', '.join(_CLIENT_MAP.keys())}"
        )
    return _CLIENT_MAP[provider]()


# ============================================================
# AI 분석 엔진 (Step별 프로바이더 분리)
# ============================================================


class AIAnalyzer:
    """Step별 멀티 프로바이더 주식 분석 엔진"""

    def __init__(
        self,
        provider: Optional[str] = None,
        step_config: Optional[StepProviderConfig] = None,
    ):
        """
        Args:
            provider: 글로벌 기본 프로바이더 (None이면 .env의 AI_PROVIDER)
            step_config: Step별 프로바이더 설정 (None이면 .env에서 로드)
        """
        self.default_provider = (
            provider or os.getenv("AI_PROVIDER", "claude").lower().strip()
        )
        self.step_config = step_config or StepProviderConfig.from_env()
        self.client_pool = ClientPool()
        self.system_prompt = self._load_prompt("system_prompt.txt")

        # Step별 설정 로그 출력
        summary = get_step_provider_summary(self.step_config, self.default_provider)
        logger.info("[AI 분석] Step별 프로바이더 설정:")
        for step_num, info in summary.items():
            logger.info(
                f"  Step {step_num} ({info['name']}): {info['provider_name']}"
            )

    def _get_step_client(self, step: int) -> BaseAIClient:
        """해당 Step의 프로바이더 클라이언트 반환"""
        provider = self.step_config.get(step) or self.default_provider
        return self.client_pool.get(provider)

    def _get_step_info(self, step: int) -> dict:
        """해당 Step의 모델/온도 정보 반환"""
        client = self._get_step_client(step)
        step_type = STEP_DEFINITIONS[step]["type"]
        if step_type == "data":
            return {
                "client": client,
                "model": client.model_data,
                "temperature": client.temp_data,
                "max_tokens": 4096,
            }
        else:  # analysis
            return {
                "client": client,
                "model": client.model_analysis,
                "temperature": client.temp_analysis,
                "max_tokens": 8192,
            }

    @staticmethod
    def _load_prompt(filename: str) -> str:
        """프롬프트 파일 로드"""
        filepath = PROMPTS_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {filepath}")
        return filepath.read_text(encoding="utf-8")

    def _call_step_api(self, step: int, user_prompt: str) -> str:
        """Step에 매핑된 프로바이더로 API 호출"""
        info = self._get_step_info(step)
        client: BaseAIClient = info["client"]
        step_def = STEP_DEFINITIONS[step]

        logger.info(
            f"  → Step {step} ({step_def['name']}): "
            f"{client.provider_name} / {info['model']}"
        )

        try:
            return client.call_api(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                model=info["model"],
                temperature=info["temperature"],
                max_tokens=info["max_tokens"],
            )
        except Exception as e:
            logger.error(
                f"[AI 분석] Step {step} ({client.provider_name}) API 호출 실패: {e}"
            )
            raise

    @staticmethod
    def _extract_json(text: str) -> dict:
        """AI 응답에서 JSON 추출"""
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            logger.warning("[AI 분석] JSON 파싱 실패, 원본 텍스트 반환")
            return {"raw_response": text}

    # ========== 프롬프트 체이닝 메서드 ==========

    def analyze_filtering(self, filtered_data: dict) -> dict:
        """프롬프트 1: 거래량/거래대금 필터링 분석"""
        logger.info("[AI 분석] 프롬프트 1: 필터링 및 데이터 정리 시작")

        prompt_template = self._load_prompt("prompt_1_filtering.txt")
        raw_table = self._format_data_table(filtered_data)

        prompt = prompt_template.replace("{날짜}", filtered_data.get("기준일", ""))
        prompt = prompt.replace("{raw_data_table}", raw_table)

        response = self._call_step_api(step=1, user_prompt=prompt)
        result = self._extract_json(response)

        logger.info(
            f"[AI 분석] 프롬프트 1 완료: {result.get('필터링_종목수', 'N/A')}개 종목"
        )
        return result

    def analyze_fundamental(self, fundamental_data: dict) -> dict:
        """프롬프트 2: 재무지표 분석"""
        logger.info("[AI 분석] 프롬프트 2: 재무지표 분석 시작")

        prompt_template = self._load_prompt("prompt_2_fundamental.txt")
        table = json.dumps(fundamental_data, ensure_ascii=False, indent=2)
        prompt = prompt_template.replace("{fundamental_data_table}", table)

        response = self._call_step_api(step=2, user_prompt=prompt)
        result = self._extract_json(response)

        logger.info(
            f"[AI 분석] 프롬프트 2 완료: "
            f"{len(result.get('종목_분석', []))}개 종목 분석"
        )
        return result

    def analyze_news(self, news_data: dict) -> dict:
        """프롬프트 3: 뉴스 분석 및 감성 평가 (단일 종목)"""
        stock_name = news_data.get("종목명", "")
        stock_code = news_data.get("종목코드", "")
        logger.info(f"[AI 분석] 프롬프트 3: {stock_name}({stock_code}) 뉴스 분석 시작")

        prompt_template = self._load_prompt("prompt_3_news.txt")
        news_list_text = self._format_news_list(news_data.get("뉴스_리스트", []))

        prompt = prompt_template.replace("{종목명}", stock_name)
        prompt = prompt.replace("{종목코드}", stock_code)
        prompt = prompt.replace("{N}", str(news_data.get("수집_뉴스수", 0)))
        prompt = prompt.replace("{news_list}", news_list_text)

        response = self._call_step_api(step=3, user_prompt=prompt)
        result = self._extract_json(response)

        logger.info(
            f"[AI 분석] 프롬프트 3 완료: {stock_name} "
            f"뉴스 스코어 {result.get('종합_뉴스스코어', 'N/A')}"
        )
        return result

    def analyze_all_news(self, all_news_data: list[dict]) -> list[dict]:
        """프롬프트 3: 모든 종목 뉴스 분석"""
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
        """프롬프트 4: 시장 동향 종합 분석"""
        logger.info("[AI 분석] 프롬프트 4: 종합 분석 시작")

        prompt_template = self._load_prompt("prompt_4_analysis.txt")

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

        kospi = market_index.get("코스피", {})
        kosdaq = market_index.get("코스닥", {})
        prompt = prompt.replace("{kospi_index}", str(kospi.get("지수", "데이터 없음")))
        prompt = prompt.replace("{kospi_change}", str(kospi.get("등락률", "데이터 없음")))
        prompt = prompt.replace("{kosdaq_index}", str(kosdaq.get("지수", "데이터 없음")))
        prompt = prompt.replace("{kosdaq_change}", str(kosdaq.get("등락률", "데이터 없음")))
        prompt = prompt.replace("{exchange_rate}", "데이터 별도 수집 필요")
        prompt = prompt.replace(
            "{foreign_flow}", foreign_trading.get("외국인_순매수_판단", "데이터 없음")
        )

        response = self._call_step_api(step=4, user_prompt=prompt)
        result = self._extract_json(response)

        logger.info("[AI 분석] 프롬프트 4 종합 분석 완료")
        return result

    def generate_report(self, final_analysis: dict) -> str:
        """프롬프트 5: 최종 리포트 생성"""
        logger.info("[AI 분석] 프롬프트 5: 최종 리포트 생성 시작")

        prompt_template = self._load_prompt("prompt_5_report.txt")
        prompt = prompt_template.replace(
            "{final_analysis_json}",
            json.dumps(final_analysis, ensure_ascii=False, indent=2),
        )

        report = self._call_step_api(step=5, user_prompt=prompt)

        logger.info("[AI 분석] 프롬프트 5 최종 리포트 생성 완료")
        return report

    def run_full_analysis(self, collected_data: dict) -> dict:
        """
        전체 프롬프트 체이닝 실행 (Step별 프로바이더 사용)

        Returns:
            dict: 전체 분석 결과 + Step별 프로바이더 정보
        """
        summary = get_step_provider_summary(self.step_config, self.default_provider)

        logger.info("=" * 60)
        logger.info("[AI 분석] 전체 프롬프트 체이닝 시작")
        for s, info in summary.items():
            logger.info(f"  Step {s} ({info['name']}): {info['provider_name']}")
        logger.info("=" * 60)

        # Step 1: 필터링 분석
        filtered_data = collected_data["필터링_결과"]
        filtered_analysis = self.analyze_filtering(filtered_data)

        # Step 2: 재무지표 분석
        fundamental_data = collected_data["재무지표"]
        fundamental_analysis = self.analyze_fundamental(fundamental_data)

        # Step 3: 뉴스 분석
        news_data = collected_data.get("뉴스_데이터", [])
        news_analysis = self.analyze_all_news(news_data) if news_data else []

        # Step 4: 종합 분석
        comprehensive_analysis = self.analyze_comprehensive(
            filtered_stocks=filtered_analysis,
            fundamental_analysis=fundamental_analysis,
            news_analysis=news_analysis,
            market_index=collected_data.get("시장_지수", {}),
            foreign_trading=collected_data.get("외국인_매매", {}),
        )

        # Step 5: 최종 리포트 생성
        report_markdown = self.generate_report(comprehensive_analysis)

        # Step별 프로바이더 정보 기록
        step_providers = {
            f"step{s}": info["provider"]
            for s, info in summary.items()
        }
        step_provider_names = {
            f"step{s}": info["provider_name"]
            for s, info in summary.items()
        }

        result = {
            "ai_providers": step_providers,
            "ai_provider_names": step_provider_names,
            "ai_default_provider": self.default_provider,
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
