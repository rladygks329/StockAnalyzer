# 📊 한국 주식 시장 AI 분석 시스템 (StockAnalyzer)

한국 주식 시장에서 거래량·거래대금 기준 상위 종목을 자동 필터링하고, **재무지표 분석 + 뉴스 감성 분석 + AI 종합 분석**을 수행하는 시스템입니다.

## 주요 기능

| 기능 | 설명 |
|------|------|
| 📥 데이터 수집 | pykrx 기반 KRX 거래 데이터, 재무지표 자동 수집 |
| 🔍 종목 필터링 | 거래량 1,000만주 이상 + 거래대금 100억원 이상 필터링 |
| 💰 재무 분석 | PER, PBR, EPS, ROE 기반 밸류에이션 평가 (A~D 등급) |
| 📰 뉴스 분석 | 네이버 검색 API 기반 뉴스 수집 + AI 감성 분석 |
| 🤖 AI 종합 분석 | Claude API 프롬프트 체이닝으로 시장 종합 판단 |
| 📄 리포트 생성 | 마크다운/JSON 형태의 일일 분석 리포트 자동 생성 |
| 📊 대시보드 | Streamlit 기반 인터랙티브 시각화 대시보드 |
| ⏰ 자동 실행 | 매일 장 마감 후(15:40) 자동 분석 스케줄러 |

## 프롬프트 체이닝 흐름

```
[매일 15:40 장 마감 후 자동 실행]

Step 1: pykrx로 원시 데이터 수집 (Python)
    ↓
Step 2: 프롬프트 1 → 필터링 및 정리 (AI)
    ↓
Step 3: pykrx로 재무지표 수집 (Python)
    ↓
Step 4: 프롬프트 2 → 재무지표 분석 (AI)
    ↓
Step 5: 네이버 검색 API로 뉴스 수집 (Python)
    ↓
Step 6: 프롬프트 3 → 뉴스 감성 분석 (AI) [종목별 병렬 처리]
    ↓
Step 7: 프롬프트 4 → 종합 분석 (AI) [모든 데이터 통합]
    ↓
Step 8: 프롬프트 5 → 최종 리포트 생성 (AI)
    ↓
[출력: 마크다운 리포트 / JSON / Streamlit 대시보드]
```

## 설치 방법

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
```bash
# .env 파일 생성 (config/.env.example 참고)
cp config/.env.example .env
```

`.env` 파일에 아래 API 키를 설정하세요:

| 키 | 필수 | 설명 | 발급처 |
|----|------|------|--------|
| `ANTHROPIC_API_KEY` | ✅ | Claude AI API 키 | [Anthropic Console](https://console.anthropic.com) |
| `NAVER_CLIENT_ID` | ✅ | 네이버 검색 API 클라이언트 ID | [Naver Developers](https://developers.naver.com) |
| `NAVER_CLIENT_SECRET` | ✅ | 네이버 검색 API 시크릿 | [Naver Developers](https://developers.naver.com) |

## 사용법

### 즉시 1회 분석 실행
```bash
python main.py
```

### 특정 날짜 분석
```bash
python main.py --date 20260206
```

### 매일 자동 실행 (스케줄러)
```bash
python main.py --schedule                # 기본 15:40
python main.py --schedule --time 16:00   # 시각 지정
```

### Streamlit 대시보드
```bash
streamlit run app.py
```

## 프로젝트 구조

```
StockAnalyzer/
├── config/
│   ├── .env.example              # 환경 변수 예시
│   └── prompts/
│       ├── system_prompt.txt     # 시스템 프롬프트
│       ├── prompt_1_filtering.txt   # 거래량/거래대금 필터링
│       ├── prompt_2_fundamental.txt # 재무지표 분석
│       ├── prompt_3_news.txt        # 뉴스 감성 분석
│       ├── prompt_4_analysis.txt    # 종합 분석
│       └── prompt_5_report.txt      # 리포트 생성
├── src/
│   ├── __init__.py
│   ├── data_collector.py         # pykrx 데이터 수집
│   ├── news_collector.py         # 네이버 API 뉴스 수집
│   ├── ai_analyzer.py            # Claude API 프롬프트 체이닝
│   ├── report_generator.py       # 리포트 포맷팅 및 저장
│   └── scheduler.py              # 일일 자동 실행 스케줄러
├── outputs/
│   └── reports/                  # 일별 리포트 저장
├── app.py                        # Streamlit 대시보드
├── main.py                       # CLI 엔트리포인트
├── requirements.txt
└── README.md
```

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