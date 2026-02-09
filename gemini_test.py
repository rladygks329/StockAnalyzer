"""
Gemini API 연결 및 간단 생성 테스트 (google.genai SDK)
실행: python gemini_test.py
필수: config/.env 또는 .env에 GEMINI_API_KEY 설정
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 기준 env 로드
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# .env 우선, 없으면 config/.env
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / "config" / ".env")


def _is_quota_error(exc: Exception) -> bool:
    """429 또는 quota 초과 메시지 여부"""
    msg = str(exc).lower()
    return "429" in msg or "quota" in msg or "rate" in msg


def _print_quota_help():
    print("  → 무료 한도 초과일 수 있습니다. 잠시 후 재시도하거나")
    print("    https://ai.google.dev/gemini-api/docs/rate-limits 에서 한도/과금 확인.")


def test_gemini_direct():
    """google.genai로 Gemini API 직접 호출 테스트"""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("오류: GEMINI_API_KEY가 설정되지 않았습니다.")
        print("  config/.env 또는 .env에 GEMINI_API_KEY=your_key 형태로 추가하세요.")
        return False

    model = os.getenv("GEMINI_MODEL_ANALYSIS", "gemini-2.0-flash").strip()

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents="한 문장으로 자기소개 해줘.",
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=1024,
            ),
        )
        text = response.text if response.text else "(응답 없음)"
        print("[Direct API] 응답:", text)
        return True
    except Exception as e:
        if _is_quota_error(e):
            print("[Direct API] 오류: 한도 초과 (429 Quota exceeded)")
            _print_quota_help()
        else:
            print("[Direct API] 오류:", e)
        return False


def test_gemini_via_client():
    """프로젝트 GeminiClient를 사용한 테스트 (ai_analyzer 경로 동일 검증)"""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("오류: GEMINI_API_KEY가 설정되지 않았습니다.")
        return False

    try:
        from src.ai_analyzer import create_ai_client

        client = create_ai_client("gemini")
        model = client.model_analysis
        reply = client.call_api(
            system_prompt="당신은 친절한 어시스턴트입니다. 짧게 답하세요.",
            user_prompt="1+1은? 숫자만 답해줘.",
            model=model,
            temperature=0.1,
            max_tokens=64,
        )
        print("[GeminiClient] 응답:", (reply or "").strip())
        return True
    except Exception as e:
        if _is_quota_error(e):
            print("[GeminiClient] 오류: 한도 초과 (429 Quota exceeded)")
            _print_quota_help()
        else:
            print("[GeminiClient] 오류:", e)
        return False


if __name__ == "__main__":
    print("=== Gemini API 테스트 ===\n")
    print("1. Direct API (google.genai)")
    ok1 = test_gemini_direct()
    print()
    print("2. GeminiClient (src.ai_analyzer)")
    ok2 = test_gemini_via_client()
    print()
    if ok1 and ok2:
        print("모든 테스트 통과.")
    else:
        print("일부 테스트 실패.")
        sys.exit(1)
