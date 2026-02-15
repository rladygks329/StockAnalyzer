"""
AI 프로바이더 API 연결 점검 스크립트
각 프로바이더의 API 키 설정 및 연결 상태를 테스트합니다.
"""

import os
import sys
import time
import io
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Windows 콘솔에서 UTF-8 인코딩 설정
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_analyzer import (
    SUPPORTED_PROVIDERS,
    _CLIENT_MAP,
    get_available_providers,
)


def test_provider(provider_id: str) -> dict:
    """
    특정 프로바이더의 API 연결 테스트
    
    Returns:
        dict: 테스트 결과 (status, message, elapsed_time, response_preview)
    """
    result = {
        "provider_id": provider_id,
        "provider_name": SUPPORTED_PROVIDERS[provider_id]["name"],
        "status": "unknown",
        "message": "",
        "elapsed_time": 0.0,
        "response_preview": "",
        "error": None,
    }

    # API 키 확인
    env_key = SUPPORTED_PROVIDERS[provider_id]["env_key"]
    api_key = os.getenv(env_key, "")
    
    if not api_key or api_key.startswith("your_") or api_key == "":
        result["status"] = "no_key"
        result["message"] = f"API 키가 설정되지 않았습니다 ({env_key})"
        return result

    # 클라이언트 초기화 테스트
    try:
        if provider_id not in _CLIENT_MAP:
            result["status"] = "unsupported"
            result["message"] = "지원하지 않는 프로바이더"
            return result

        print(f"  → {result['provider_name']} 클라이언트 초기화 중...", flush=True)
        client = _CLIENT_MAP[provider_id]()
        result["model_data"] = client.model_data
        result["model_analysis"] = client.model_analysis
    except Exception as e:
        result["status"] = "init_failed"
        result["message"] = f"클라이언트 초기화 실패: {str(e)}"
        result["error"] = str(e)
        return result

    # 간단한 API 호출 테스트
    test_prompt = "안녕하세요. 이 메시지를 한 단어로 응답해주세요."
    test_system = "You are a helpful assistant."

    try:
        print(f"  → API 호출 테스트 중...", flush=True)
        start_time = time.time()
        
        # 모든 프로바이더에 대해 client.call_api 사용 (일관성 유지)
        # Gemini의 경우 max_tokens가 너무 작으면 MAX_TOKENS로 종료되어 빈 응답이 올 수 있으므로 최소값 보장
        max_tokens_for_test = 50 if provider_id == "gemini" else 10
        response = client.call_api(
            system_prompt=test_system,
            user_prompt=test_prompt,
            model=client.model_data,
            temperature=0.1,
            max_tokens=max_tokens_for_test,
        )
        
        elapsed = time.time() - start_time
        result["elapsed_time"] = elapsed
        
        # 응답 처리 (프로바이더별로 다를 수 있음)
        if response is None:
            result["response_preview"] = "(응답 없음 - None)"
            result["status"] = "empty_response"
            result["message"] = "None 응답을 받았습니다"
        elif isinstance(response, str):
            result["response_preview"] = response[:100] if response else "(응답 없음 - 빈 문자열)"
            if response and len(response.strip()) > 0:
                result["status"] = "success"
                result["message"] = f"API 호출 성공 ({elapsed:.2f}초)"
            else:
                result["status"] = "empty_response"
                result["message"] = "빈 문자열 응답을 받았습니다"
        else:
            # 응답이 문자열이 아닌 경우 (객체 등)
            result["response_preview"] = str(response)[:100]
            result["status"] = "success"
            result["message"] = f"API 호출 성공 ({elapsed:.2f}초) - 응답 타입: {type(response).__name__}"
            
    except Exception as e:
        result["status"] = "api_failed"
        result["message"] = f"API 호출 실패: {str(e)}"
        result["error"] = str(e)
        if result["elapsed_time"] == 0:
            result["elapsed_time"] = time.time() - start_time

    return result


def print_result(result: dict):
    """테스트 결과를 보기 좋게 출력"""
    provider_name = result["provider_name"]
    status = result["status"]
    
    status_icons = {
        "success": "✅",
        "no_key": "⚪",
        "init_failed": "❌",
        "api_failed": "❌",
        "empty_response": "⚠️",
        "unsupported": "❓",
        "unknown": "❓",
    }
    
    icon = status_icons.get(status, "❓")
    
    print(f"\n{icon} {provider_name} ({result['provider_id']})")
    print(f"   상태: {result['message']}")
    
    if result.get("model_data"):
        print(f"   데이터 모델: {result['model_data']}")
    if result.get("model_analysis"):
        print(f"   분석 모델: {result['model_analysis']}")
    if result["elapsed_time"] > 0:
        print(f"   응답 시간: {result['elapsed_time']:.2f}초")
    if result["response_preview"]:
        preview = result["response_preview"].replace("\n", " ")
        print(f"   응답 미리보기: {preview}")
    if result.get("error"):
        print(f"   오류: {result['error']}")


def main():
    """모든 프로바이더 테스트 실행"""
    print("=" * 60)
    print("AI 프로바이더 API 연결 점검")
    print("=" * 60)
    
    # 사용 가능한 프로바이더 확인
    available = get_available_providers()
    print(f"\n📋 설정된 API 키가 있는 프로바이더: {', '.join(available) if available else '없음'}")
    
    # 모든 프로바이더 테스트
    print(f"\n🔍 프로바이더 연결 테스트 시작...\n")
    
    results = []
    for provider_id in SUPPORTED_PROVIDERS.keys():
        print(f"\n[{provider_id.upper()}] 테스트 중...")
        result = test_provider(provider_id)
        results.append(result)
        print_result(result)
    
    # 요약
    print("\n" + "=" * 60)
    print("📊 테스트 요약")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    no_key_count = sum(1 for r in results if r["status"] == "no_key")
    failed_count = sum(1 for r in results if r["status"] in ("init_failed", "api_failed", "empty_response"))
    
    print(f"✅ 성공: {success_count}개")
    print(f"⚪ API 키 없음: {no_key_count}개")
    print(f"❌ 실패: {failed_count}개")
    
    if success_count > 0:
        print(f"\n✅ 사용 가능한 프로바이더:")
        for r in results:
            if r["status"] == "success":
                print(f"   - {r['provider_name']} ({r['provider_id']})")
    
    if no_key_count > 0:
        print(f"\n⚪ API 키가 필요한 프로바이더:")
        for r in results:
            if r["status"] == "no_key":
                env_key = SUPPORTED_PROVIDERS[r["provider_id"]]["env_key"]
                print(f"   - {r['provider_name']}: {env_key} 설정 필요")
    
    if failed_count > 0:
        print(f"\n❌ 연결 실패한 프로바이더:")
        for r in results:
            if r["status"] in ("init_failed", "api_failed", "empty_response"):
                print(f"   - {r['provider_name']}: {r['message']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
