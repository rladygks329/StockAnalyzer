"""
AI 분석 모듈
멀티 프로바이더(Claude, GPT, Gemini, Grok) 지원 + Step별 프로바이더 분리

단순화된 2-Step 파이프라인:
  Step 4: 종합분석 (뉴스 감성분석 통합)
  Step 5: 리포트 생성
"""

import os
import json
import re
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from src.report_generator import get_report_date_dir

load_dotenv()

logger = logging.getLogger(__name__)

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = PROJECT_ROOT / "config" / "prompts"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

# ============================================================
# Step 정의 (AI가 필요한 Step만)
# ============================================================

STEP_DEFINITIONS = {
    4: {"name": "종합분석", "desc": "뉴스 감성분석 + 시장 동향 종합 분석", "env": "STEP4_PROVIDER", "type": "analysis"},
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
    """Step별 프로바이더 설정 (Step 4, 5만)"""
    step4: Optional[str] = None  # 종합분석
    step5: Optional[str] = None  # 리포트

    def get(self, step: int) -> Optional[str]:
        """특정 Step의 프로바이더 반환 (None이면 글로벌 기본값 사용)"""
        return getattr(self, f"step{step}", None)

    def set(self, step: int, provider: Optional[str]):
        """특정 Step의 프로바이더 설정"""
        if hasattr(self, f"step{step}"):
            setattr(self, f"step{step}", provider)

    def to_dict(self) -> dict:
        return {f"step{i}": self.get(i) for i in (4, 5)}

    @classmethod
    def from_dict(cls, d: dict) -> "StepProviderConfig":
        return cls(
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
        return cls(step4=provider, step5=provider)


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
    """Google Gemini 클라이언트 (google.genai SDK)"""

    def __init__(self):
        super().__init__("gemini")
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다.")
        self._client = genai.Client(api_key=api_key)
        self._types = types

    def call_api(self, system_prompt, user_prompt, model, temperature, max_tokens):
        config_dict = {
            "system_instruction": system_prompt,
            "temperature": temperature,
        }
        if max_tokens and max_tokens > 0:
            config_dict["max_output_tokens"] = max(max_tokens, 30000)

        response = self._client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=self._types.GenerateContentConfig(**config_dict),
        )

        text = None
        try:
            if hasattr(response, 'text'):
                text = response.text
        except (ValueError, AttributeError) as e:
            logger.debug(f"Gemini response.text 접근 실패: {e}")

        if not text and hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                parts = getattr(candidate.content, 'parts', None)
                if parts:
                    text_parts = []
                    for part in parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    if text_parts:
                        text = ''.join(text_parts)

        return text or ""


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
# AI 분석 엔진 (2-Step 파이프라인)
# ============================================================


class AIAnalyzer:
    """Step별 멀티 프로바이더 주식 분석 엔진 (Step 4: 종합분석, Step 5: 리포트)"""

    def __init__(
        self,
        provider: Optional[str] = None,
        step_config: Optional[StepProviderConfig] = None,
        show_prompts: Optional[bool] = None,
        analyze_by_api: Optional[bool] = None,
    ):
        self.default_provider = (
            provider or os.getenv("AI_PROVIDER", "claude").lower().strip()
        )
        self.step_config = step_config or StepProviderConfig.from_env()
        self.client_pool = ClientPool()
        self.system_prompt = self._load_prompt("system_prompt.txt")

        if show_prompts is None:
            self.show_prompts = os.getenv("SHOW_STEP_PROMPTS", "false").lower() in ("true", "1", "yes")
        else:
            self.show_prompts = show_prompts

        if analyze_by_api is None:
            self._analyze_by_api = os.getenv("ANALYZE_BY_API", "true").lower() in ("true", "1", "yes")
        else:
            self._analyze_by_api = analyze_by_api

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

        print(
            f"    API 호출 중... Step {step} ({step_def['name']}) "
            f"<- {client.provider_name} / {info['model']}",
            flush=True,
        )
        logger.info(
            f"  -> Step {step} ({step_def['name']}): "
            f"{client.provider_name} / {info['model']}"
        )

        t0 = time.time()
        try:
            result = client.call_api(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                model=info["model"],
                temperature=info["temperature"],
                max_tokens=info["max_tokens"],
            )
            elapsed = time.time() - t0
            print(
                f"    Step {step} ({step_def['name']}) 완료 ({elapsed:.1f}초)",
                flush=True,
            )
            return result
        except Exception as e:
            elapsed = time.time() - t0
            print(
                f"    Step {step} ({step_def['name']}) 실패 ({elapsed:.1f}초): {e}",
                flush=True,
            )
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
            logger.info("원본 텍스트: " + text)
            return {"raw_response": text}

    @staticmethod
    def _save_prompt_file(prompt: str, date: str, step: int) -> str:
        """프롬프트를 날짜 폴더 내 prompt_{step}.txt로 저장"""
        date_dir = get_report_date_dir(REPORTS_DIR, date)
        filepath = date_dir / f"prompt_{step}.txt"
        filepath.write_text(prompt, encoding="utf-8")
        logger.info(f"[AI 분석] 프롬프트 저장: {filepath}")
        return str(filepath)

    # ========== 프롬프트 빌더 ==========

    def build_comprehensive_prompt(self, collected_data: dict) -> str:
        """
        Step 4 종합분석 프롬프트 빌드.
        수집된 전체 데이터(필터링, 재무등급, 원본 뉴스, 시장지수, 외국인매매)를
        prompt_4_analysis.txt 템플릿에 주입하여 완성된 프롬프트 문자열을 리턴.
        """
        prompt_template = self._load_prompt("prompt_4_analysis.txt")

        filtered_stocks = collected_data.get("필터링_결과", {})
        fundamental_data = collected_data.get("재무지표", {})
        news_data = collected_data.get("뉴스_데이터", [])
        market_index = collected_data.get("시장_지수", {})
        foreign_trading = collected_data.get("외국인_매매", {})

        prompt = prompt_template.replace(
            "{filtered_stocks_json}",
            json.dumps(filtered_stocks, ensure_ascii=False, indent=2),
        )
        prompt = prompt.replace(
            "{fundamental_graded_json}",
            json.dumps(fundamental_data, ensure_ascii=False, indent=2),
        )
        prompt = prompt.replace(
            "{raw_news_json}",
            json.dumps(news_data, ensure_ascii=False, indent=2),
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

        return prompt

    def build_report_prompt(self, comprehensive_analysis: dict) -> str:
        """
        Step 5 리포트 프롬프트 빌드.
        종합분석 결과를 prompt_5_report.txt 템플릿에 주입하여 완성된 프롬프트 문자열을 리턴.
        """
        prompt_template = self._load_prompt("prompt_5_report.txt")
        prompt = prompt_template.replace(
            "{final_analysis_json}",
            json.dumps(comprehensive_analysis, ensure_ascii=False, indent=2),
        )
        return prompt

    # ========== 분석 실행 ==========

    def analyze_comprehensive(self, collected_data: dict, date: str) -> dict:
        """Step 4: 종합분석 (뉴스 감성분석 통합) 실행"""
        logger.info("[AI 분석] Step 4: 종합 분석 시작")

        prompt = self.build_comprehensive_prompt(collected_data)

        # 프롬프트 파일 저장 (날짜 폴더 내 prompt_4.txt)
        self._save_prompt_file(prompt, date, 4)

        if self.show_prompts:
            print(f"\n{'─'*55}", flush=True)
            print("Step 4 프롬프트:", flush=True)
            print(f"{'─'*55}", flush=True)
            print(prompt[:2000] + "\n... (이하 생략)" if len(prompt) > 2000 else prompt, flush=True)
            print(f"{'─'*55}\n", flush=True)

        if not self._analyze_by_api:
            logger.info("[AI 분석] Step 4: API 호출 스킵 (프롬프트만 저장)")
            return {
                "시장_전체_판단": "API 호출 스킵됨 (--prompt-only 모드)",
                "주요_이슈": [],
                "종목별_종합_평가": [],
            }

        response = self._call_step_api(step=4, user_prompt=prompt)
        result = self._extract_json(response)

        logger.info("[AI 분석] Step 4 종합 분석 완료")
        return result

    def generate_report(self, comprehensive_analysis: dict, date: str) -> str:
        """Step 5: 최종 리포트 생성"""
        logger.info("[AI 분석] Step 5: 최종 리포트 생성 시작")

        prompt = self.build_report_prompt(comprehensive_analysis)

        # 프롬프트 파일 저장 (날짜 폴더 내 prompt_5.txt)
        self._save_prompt_file(prompt, date, 5)

        if self.show_prompts:
            print(f"\n{'─'*55}", flush=True)
            print("Step 5 프롬프트:", flush=True)
            print(f"{'─'*55}", flush=True)
            print(prompt[:2000] + "\n... (이하 생략)" if len(prompt) > 2000 else prompt, flush=True)
            print(f"{'─'*55}\n", flush=True)

        if not self._analyze_by_api:
            logger.info("[AI 분석] Step 5: API 호출 스킵 (프롬프트만 저장)")
            return "# 리포트 생성 스킵됨\n\n--prompt-only 모드로 실행되었습니다. 저장된 프롬프트 파일을 AI에 직접 입력하여 리포트를 생성하세요."

        report = self._call_step_api(step=5, user_prompt=prompt)

        logger.info("[AI 분석] Step 5 최종 리포트 생성 완료")
        return report

    def run_full_analysis(self, collected_data: dict) -> dict:
        """
        전체 AI 분석 실행 (Step 4 + Step 5)

        Returns:
            dict: 분석 결과 + Step별 프로바이더 정보
        """
        summary = get_step_provider_summary(self.step_config, self.default_provider)
        date = collected_data.get("기준일", "")

        logger.info("=" * 60)
        logger.info("[AI 분석] 전체 AI 분석 시작 (Step 4 + Step 5)")
        for s, info in summary.items():
            logger.info(f"  Step {s} ({info['name']}): {info['provider_name']}")
        logger.info("=" * 60)

        pipeline_start = time.time()

        # Step 4: 종합 분석 (뉴스 감성분석 통합)
        print(f"\n{'='*55}", flush=True)
        print(f"  [Step 4] 종합 분석 (뉴스 감성분석 + 시장 동향)", flush=True)
        print(f"{'='*55}", flush=True)
        comprehensive_analysis = self.analyze_comprehensive(collected_data, date)

        # Step 5: 최종 리포트 생성
        print(f"\n{'='*55}", flush=True)
        print(f"  [Step 5] 최종 리포트 생성", flush=True)
        print(f"{'='*55}", flush=True)
        report_markdown = self.generate_report(comprehensive_analysis, date)

        pipeline_elapsed = time.time() - pipeline_start
        print(f"\n{'='*55}", flush=True)
        print(
            f"  전체 AI 분석 완료! (총 {pipeline_elapsed:.1f}초)",
            flush=True,
        )
        print(f"{'='*55}", flush=True)

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
            "filtered_analysis": collected_data.get("필터링_결과", {}),
            "fundamental_analysis": collected_data.get("재무지표", {}),
            "news_analysis": [],  # 뉴스 감성분석은 Step 4에 통합됨
            "comprehensive_analysis": comprehensive_analysis,
            "report_markdown": report_markdown,
        }

        logger.info("[AI 분석] 전체 AI 분석 완료")
        return result

    @staticmethod
    def _format_date(date: str) -> str:
        """날짜 형식 변환 (YYYYMMDD → YYYY-MM-DD)"""
        if "-" in date:
            return date
        if len(date) == 8:
            return f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        return date
