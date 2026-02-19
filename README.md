# 📊 한국 주식 시장 AI 분석 시스템 (StockAnalyzer)

한국 주식 시장에서 거래량·거래대금 기준 상위 종목을 자동 필터링하고, **재무지표 등급 + 뉴스 감성 분석 + AI 종합 분석**을 수행하는 시스템입니다.

## 주요 기능

| 기능 | 설명 |
|------|------|
| 📥 데이터 수집 | pykrx 기반 KRX 거래 데이터, 재무지표 자동 수집 |
| 🔍 종목 필터링 | 거래량 1,000만주 이상 + 거래대금 100억원 이상 필터링 |
| 💰 재무 등급 | PER, PBR, EPS, ROE 기반 규칙기반 밸류에이션 등급 (A~D) |
| 📰 뉴스 수집 | 네이버 검색 API 기반 종목별 뉴스 수집 |
| 🤖 AI 종합 분석 | 뉴스 감성분석 + 시장 동향 + 종목별 시그널 판정 (1회 AI 호출) |
| 📄 리포트 생성 | 마크다운/JSON 형태의 일일 분석 리포트 자동 생성 |
| 📊 대시보드 | Streamlit 기반 인터랙티브 시각화 대시보드 |
| ⏰ 자동 실행 | 매일 장 마감 후(15:40) 자동 분석 스케줄러 |

## 3-Phase 파이프라인

```
Phase 1: 데이터 수집 (Python only, AI API 불필요)
  ├─ pykrx → 거래량/거래대금 필터링
  ├─ pykrx → 재무지표 수집 + Python 규칙기반 등급(A~D) 부여
  ├─ pykrx → 시장지수/외국인매매 수집
  └─ 네이버 API → 종목별 뉴스 수집
  → outputs/reports/YYYY/MM/DD/collected.json 저장

Phase 2: AI 종합분석 (Step 4)
  ├─ 원본 뉴스 + 재무등급 + 전체 데이터를 프롬프트에 주입
  ├─ 뉴스 감성분석 + 시장 동향 + 종목별 시그널 한 번에 분석
  └─ outputs/reports/YYYY/MM/DD/prompt_4.txt 저장 (API 없이도 사용 가능)

Phase 3: AI 리포트 생성 (Step 5)
  ├─ 종합분석 결과를 프롬프트에 주입
  └─ outputs/reports/YYYY/MM/DD/prompt_5.txt 저장 (API 없이도 사용 가능)

→ 최종 출력: 같은 폴더 내 report.md / full.json
```

**이전 5-Step 대비 개선점:**
- Phase 1은 AI API 없이 Python만으로 처리 (비용 절감)
- 재무등급을 Python 규칙으로 계산 (AI 호출 불필요)
- 뉴스 감성분석을 종합분석에 통합 (종목별 N번 → 1번 호출)
- 중간 데이터 저장으로 Phase별 이어가기 가능

## 설치 방법

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
cp config/.env.example .env
```

`.env` 파일에 아래 API 키를 설정하세요:

| 키 | 필수 | 설명 | 발급처 |
|----|------|------|--------|
| `ANTHROPIC_API_KEY` | AI 분석 시 | Claude AI API 키 | [Anthropic Console](https://console.anthropic.com) |
| `OPENAI_API_KEY` | 선택 | GPT API 키 | [OpenAI Platform](https://platform.openai.com) |
| `GEMINI_API_KEY` | 선택 | Gemini API 키 | [Google AI Studio](https://aistudio.google.com) |
| `GROK_API_KEY` | 선택 | Grok API 키 | [xAI Console](https://console.x.ai) |
| `NAVER_CLIENT_ID` | 뉴스 수집 시 | 네이버 검색 API | [Naver Developers](https://developers.naver.com) |
| `NAVER_CLIENT_SECRET` | 뉴스 수집 시 | 네이버 검색 API | [Naver Developers](https://developers.naver.com) |

**선택적 설정:**

| 키 | 기본값 | 설명 |
|----|--------|------|
| `AI_PROVIDER` | `claude` | 기본 AI 프로바이더 |
| `STEP4_PROVIDER` | (기본값) | Step 4 종합분석 프로바이더 |
| `STEP5_PROVIDER` | (기본값) | Step 5 리포트 프로바이더 |
| `SHOW_STEP_PROMPTS` | `false` | Step 4/5 프롬프트 콘솔 출력 |
| `ANALYZE_BY_API` | `true` | API 호출 여부 (false면 프롬프트만 저장) |

## 사용법

### 전체 파이프라인 실행
```bash
# 기본: 데이터 수집(collect) → AI 분석(analyze)까지 실행하되, API 호출 없이 프롬프트만 파일로 저장
python main.py                    # 최근 거래일 기준
python main.py --date 20260213    # 특정 날짜

# AI API를 호출해 실제 분석·리포트까지 수행하려면 --api
python main.py --api
python main.py --api --date 20260213
```

### Phase별 실행 (스텝별 이어가기)

```bash
# Phase 1만: 데이터 수집 (AI API 불필요)
python main.py --step collect --date 20260213

# 저장된 데이터에서 AI 분석 이어가기 (경로: outputs/reports/YYYY/MM/DD/collected.json)
python main.py --step analyze --from-data outputs/reports/2026/02/13/collected.json

# API 호출 
python main.py --step analyze --from-data outputs/reports/2026/02/13/collected.json --api
```

### Step별 프로바이더 지정

```bash
# 종합분석은 Gemini, 리포트는 Claude (API 사용 시)
python main.py --api --step4 gemini --step5 claude

# 전체 Step을 GPT로
python main.py --api --provider gpt
```

### 프롬프트 확인/디버깅

```bash
# 프롬프트를 콘솔에 출력 (API 사용 시)
python main.py --api --show-prompts
```

### 매일 자동 실행 (스케줄러)
```bash
# API 호출까지 수행하려면 --api 함께 지정 (기본은 프롬프트만 저장)
python main.py --schedule --api             # 기본 15:40
python main.py --schedule --api --time 16:00 # 시각 지정
```

### Streamlit 대시보드
```bash
streamlit run app.py
```

## CLI 옵션 전체

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--step` | 실행할 Phase (`collect`, `analyze`, `all`) | `all` |
| `--from-data` | 저장된 collected JSON 파일 경로 (`outputs/reports/YYYY/MM/DD/collected.json`) | - |
| `--api` | AI API 호출 수행 (미지정 시 프롬프트만 저장) | `false` |
| `--prompt-only` | API 호출 없이 프롬프트만 저장 (기본 동작과 동일) | 기본 동작 |
| `--provider` | 글로벌 AI 프로바이더 | `.env` 설정 |
| `--step4` | Step 4 (종합분석) 프로바이더 | 글로벌 설정 |
| `--step5` | Step 5 (리포트) 프로바이더 | 글로벌 설정 |
| `--date` | 분석 날짜 (YYYYMMDD) | 최근 거래일 |
| `--schedule` | 스케줄 모드 | `false` |
| `--time` | 스케줄 실행 시각 (HH:MM) | `15:40` |
| `--log-level` | 로깅 레벨 | `INFO` |
| `--show-prompts` | Step 4/5 프롬프트 콘솔 출력 | `false` |
| `--no-api` | `--prompt-only`와 동일 (하위 호환) | - |

## 프로젝트 구조

```
StockAnalyzer/
├── config/
│   ├── .env.example              # 환경 변수 예시
│   └── prompts/
│       ├── system_prompt.txt     # 시스템 프롬프트
│       ├── prompt_4_analysis.txt # 종합 분석 (뉴스 감성분석 통합)
│       └── prompt_5_report.txt   # 리포트 생성
├── src/
│   ├── __init__.py
│   ├── data_collector.py         # pykrx 데이터 수집 + 재무등급 계산
│   ├── news_collector.py         # 네이버 API 뉴스 수집
│   ├── ai_analyzer.py            # AI 2-Step 파이프라인 (Step 4 + 5)
│   ├── report_generator.py       # 리포트 포맷팅 및 저장
│   ├── scheduler.py              # 일일 자동 실행 스케줄러
│   └── test_providers.py         # AI 프로바이더 연결 테스트
├── outputs/
│   └── reports/                  # 일별 리포트/데이터/프롬프트 (경로: YYYY/MM/DD)
├── app.py                        # Streamlit 대시보드
├── main.py                       # CLI 엔트리포인트
├── requirements.txt
└── README.md
```

## 출력 파일

모든 파일은 **날짜 폴더** `outputs/reports/YYYY/MM/DD/` 아래에 저장됩니다.

| 파일 | 설명 |
|------|------|
| `collected.json` | Phase 1 수집 데이터 (이어가기 가능) |
| `prompt_4.txt` | 데이터 주입된 Step 4 프롬프트 |
| `prompt_5.txt` | 데이터 주입된 Step 5 프롬프트 |
| `report.md` | 최종 마크다운 리포트 |
| `full.json` | 전체 분석 데이터 (JSON) |

## 분석 출력 형식

### 시그널 판정 (5단계)
| 시그널 | 의미 | 조건 |
|--------|------|------|
| 🟢 강한 긍정 | 적극 주목 | 저평가 + 뉴스 긍정 + 거래량 급증 |
| 🔵 긍정 | 긍정적 관심 | 밸류에이션 적정 + 뉴스 우호적 |
| ⚪ 중립 | 관망 필요 | 혼재된 시그널 |
| 🟡 주의 | 리스크 인지 | 고평가 징후 또는 부정적 뉴스 |
| 🔴 경고 | 위험 경고 | 악재 기반 거래량 급증 + 고평가 |

### 재무등급 (A~D)
| 등급 | 의미 |
|------|------|
| A | 저평가 + 수익성 양호 + 성장성 |
| B | 밸류에이션 적정 + 안정적 수익 |
| C | 고평가이나 성장 모멘텀 존재 |
| D | 고평가 + 수익성 악화 |

## 면책조항

> ⚠️ 본 시스템은 AI가 공개 데이터를 분석하여 생성한 참고 자료입니다.
> 투자 권유가 아니며, 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.


## 성과
| 일자       | 결과 | 실행                         | 비고                     |
|------------|------|------------------------------|--------------------------|
| 2026-02-06 | 한화에너지 솔루션, KD, 코데즈컴바인 | 한화 에너지 솔루션 10주 매입 | 수익률 +12.9%            |
| 2026-02-09 | 한화에너지 솔루션, 미래에셋증권, 삼성전자 |  |             |
| 2026-02-19 | 유진투자증권, 대한해운, 흥아해운 |  |한화 에너지 솔루션 10주 매도 수익률 20.1%, 3종목 60만원치 매입             |