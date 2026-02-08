"""
Streamlit 대시보드
한국 주식 시장 분석 결과를 시각적으로 표시
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

from src.ai_analyzer import (
    SUPPORTED_PROVIDERS,
    STEP_DEFINITIONS,
    StepProviderConfig,
    get_available_providers,
    get_provider_display_name,
)

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# ===== 페이지 설정 =====
st.set_page_config(
    page_title="📊 한국 주식 시장 AI 분석",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== 커스텀 CSS =====
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .signal-green { color: #10b981; font-weight: 700; }
    .signal-blue { color: #3b82f6; font-weight: 700; }
    .signal-gray { color: #6b7280; font-weight: 700; }
    .signal-yellow { color: #f59e0b; font-weight: 700; }
    .signal-red { color: #ef4444; font-weight: 700; }
    .stMetric {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===== 유틸리티 함수 =====
def load_report_data(date_str: str) -> dict:
    """리포트 JSON 데이터 로드"""
    filepath = REPORTS_DIR / f"data_{date_str}_full.json"
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_raw_data(date_str: str) -> dict:
    """원시 수집 데이터 로드"""
    filepath = REPORTS_DIR / f"data_{date_str}_raw_collected.json"
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_markdown_report(date_str: str) -> str:
    """마크다운 리포트 로드"""
    filepath = REPORTS_DIR / f"report_{date_str}.md"
    if filepath.exists():
        return filepath.read_text(encoding="utf-8")
    return ""


def get_available_dates() -> list[str]:
    """사용 가능한 리포트 날짜 목록"""
    dates = set()
    for f in REPORTS_DIR.glob("report_*.md"):
        date = f.stem.replace("report_", "")
        dates.add(date)
    for f in REPORTS_DIR.glob("data_*_full.json"):
        date = f.stem.replace("data_", "").replace("_full", "")
        dates.add(date)
    return sorted(dates, reverse=True)


def format_number(num, unit=""):
    """숫자 포맷팅"""
    if num is None:
        return "N/A"
    if abs(num) >= 1_0000_0000_0000:
        return f"{num / 1_0000_0000_0000:,.1f}조{unit}"
    if abs(num) >= 1_0000_0000:
        return f"{num / 1_0000_0000:,.0f}억{unit}"
    if abs(num) >= 1_0000:
        return f"{num / 1_0000:,.0f}만{unit}"
    return f"{num:,.0f}{unit}"


def get_signal_color(signal: str) -> str:
    """시그널에 따른 색상 반환"""
    if "강한 긍정" in signal or "🟢" in signal:
        return "signal-green"
    elif "긍정" in signal or "🔵" in signal:
        return "signal-blue"
    elif "주의" in signal or "🟡" in signal:
        return "signal-yellow"
    elif "경고" in signal or "🔴" in signal:
        return "signal-red"
    return "signal-gray"


# ===== 사이드바 =====
with st.sidebar:
    st.title("📊 주식 분석 대시보드")
    st.markdown("---")

    # 날짜 선택
    available_dates = get_available_dates()
    if available_dates:
        selected_date = st.selectbox(
            "📅 분석 날짜 선택",
            available_dates,
            format_func=lambda x: f"{x[:4]}-{x[5:7]}-{x[8:10]}" if "-" in x else x,
        )
    else:
        selected_date = None
        st.warning("아직 분석 데이터가 없습니다.")

    st.markdown("---")

    # ===== AI 프로바이더 설정 =====
    st.subheader("🤖 AI 프로바이더")

    provider_options = list(SUPPORTED_PROVIDERS.keys())
    provider_labels = {pid: conf["name"] for pid, conf in SUPPORTED_PROVIDERS.items()}
    provider_icons = {"claude": "🟠", "gpt": "🟢", "gemini": "🔵", "grok": "⚫"}

    def _fmt_provider(x):
        return f"{provider_icons.get(x, '')} {provider_labels.get(x, x)}"

    # 사용 가능한 프로바이더 표시
    available = get_available_providers()
    if available:
        st.caption(f"✅ API 키 설정됨: {', '.join(available)}")
    else:
        st.warning("설정된 API 키가 없습니다")

    # 글로벌 기본 프로바이더
    default_provider = os.getenv("AI_PROVIDER", "claude").lower().strip()
    if default_provider not in provider_options:
        default_provider = "claude"

    selected_default = st.selectbox(
        "기본 프로바이더",
        provider_options,
        index=provider_options.index(default_provider),
        format_func=_fmt_provider,
        help="Step별 미지정 시 사용되는 기본 프로바이더",
    )

    # Step별 프로바이더 설정
    st.markdown("**Step별 프로바이더 설정**")
    step_options = ["(기본값 사용)"] + provider_options
    env_step_config = StepProviderConfig.from_env()

    step_selections = {}
    for step_num, step_def in STEP_DEFINITIONS.items():
        env_val = env_step_config.get(step_num)
        default_idx = step_options.index(env_val) if env_val and env_val in provider_options else 0

        selected = st.selectbox(
            f"Step {step_num}: {step_def['name']}",
            step_options,
            index=default_idx,
            format_func=lambda x: _fmt_provider(x) if x in provider_options else x,
            key=f"step{step_num}_provider",
            help=step_def["desc"],
        )
        step_selections[step_num] = selected if selected != "(기본값 사용)" else None

    # StepProviderConfig 구성
    sidebar_step_config = StepProviderConfig(
        step1=step_selections.get(1),
        step2=step_selections.get(2),
        step3=step_selections.get(3),
        step4=step_selections.get(4),
        step5=step_selections.get(5),
    )

    # 설정 요약
    summary_lines = []
    for step_num in range(1, 6):
        p = step_selections.get(step_num) or selected_default
        icon = provider_icons.get(p, "")
        summary_lines.append(f"S{step_num}: {icon}{p}")
    st.caption(" | ".join(summary_lines))

    # 분석 실행에 필요한 프로바이더 키 확인
    needed_providers = set()
    for step_num in range(1, 6):
        needed_providers.add(step_selections.get(step_num) or selected_default)
    missing_keys = [
        p for p in needed_providers
        if p not in available
    ]
    has_all_keys = len(missing_keys) == 0

    st.markdown("---")

    # ===== 분석 실행 =====
    st.subheader("🚀 분석 실행")
    analysis_date = st.text_input(
        "분석 날짜 (YYYYMMDD)",
        value=datetime.now().strftime("%Y%m%d"),
        help="비워두면 최근 거래일 기준",
    )

    if st.button(
        "▶️ 분석 실행",
        type="primary",
        use_container_width=True,
        disabled=not has_all_keys,
    ):
        with st.spinner("🤖 Step별 AI 분석 진행 중... (수 분 소요될 수 있습니다)"):
            try:
                from main import run_daily_analysis

                result = run_daily_analysis(
                    date=analysis_date if analysis_date else None,
                    provider=selected_default,
                    step_config=sidebar_step_config,
                )
                if result["status"] == "success":
                    st.success(
                        f"✅ 분석 완료! ({result['filtered_count']}개 종목)"
                    )
                    st.rerun()
                else:
                    st.warning(result.get("message", "분석 실패"))
            except Exception as e:
                st.error(f"❌ 분석 실행 중 오류: {e}")

    if not has_all_keys:
        st.caption(f"⚠️ API 키 미설정: {', '.join(missing_keys)}")

    st.markdown("---")
    st.caption("💡 매일 15:40에 자동 분석 실행")
    st.caption("CLI: `python main.py --schedule`")
    st.caption("CLI: `python main.py --step1 gemini --step4 claude`")


# ===== 메인 콘텐츠 =====
st.markdown('<div class="main-header">📊 한국 주식 시장 AI 분석 대시보드</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">거래량·거래대금 기반 상위 종목 필터링 + 재무분석 + 뉴스 감성분석 + AI 종합판단</div>', unsafe_allow_html=True)

if not selected_date:
    st.info(
        "👋 아직 분석 데이터가 없습니다. "
        "사이드바에서 '분석 실행' 버튼을 클릭하거나 "
        "CLI에서 `python main.py`를 실행하세요."
    )
    st.stop()

# 데이터 로드
report_data = load_report_data(selected_date)
raw_data = load_raw_data(selected_date)
markdown_report = load_markdown_report(selected_date)

if not report_data and not raw_data and not markdown_report:
    st.warning(f"📂 {selected_date} 날짜의 데이터를 찾을 수 없습니다.")
    st.stop()

# ===== 탭 구성 =====
tab_overview, tab_stocks, tab_fundamental, tab_news, tab_report = st.tabs(
    ["📈 시장 개요", "📋 필터링 종목", "💰 재무분석", "📰 뉴스 분석", "📄 리포트"]
)

# ===== Tab 1: 시장 개요 =====
with tab_overview:
    # 시장 지수 (원시 데이터에서)
    market_index = raw_data.get("시장_지수", {}) if raw_data else {}
    foreign_data = raw_data.get("외국인_매매", {}) if raw_data else {}

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kospi = market_index.get("코스피", {})
        st.metric(
            "코스피",
            f"{kospi.get('지수', 'N/A'):,.2f}" if kospi.get("지수") else "N/A",
            f"{kospi.get('등락률', 0):.2f}%" if kospi.get("등락률") else None,
        )
    with col2:
        kosdaq = market_index.get("코스닥", {})
        st.metric(
            "코스닥",
            f"{kosdaq.get('지수', 'N/A'):,.2f}" if kosdaq.get("지수") else "N/A",
            f"{kosdaq.get('등락률', 0):.2f}%" if kosdaq.get("등락률") else None,
        )
    with col3:
        filtered_count = 0
        if report_data:
            fa = report_data.get("filtered_analysis", {})
            filtered_count = fa.get("필터링_종목수", 0)
        elif raw_data:
            filtered_count = raw_data.get("필터링_결과", {}).get("필터링_종목수", 0)
        st.metric("필터링 종목", f"{filtered_count}개")
    with col4:
        st.metric(
            "외국인",
            foreign_data.get("외국인_순매수_판단", "N/A"),
        )

    st.markdown("---")

    # 종합 분석 정보
    comprehensive = report_data.get("comprehensive_analysis", {}) if report_data else {}
    if comprehensive:
        market_trend = comprehensive.get("시장_동향", {})

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("🔍 시장 동향")
            if market_trend:
                st.write(f"**섹터 집중도**: {market_trend.get('섹터_집중도', 'N/A')}")
                themes = market_trend.get("주요_테마", [])
                if themes:
                    st.write(f"**주요 테마**: {', '.join(themes)}")
                st.write(f"**수급 주체**: {market_trend.get('수급_주체', 'N/A')}")

                sentiment = market_trend.get("시장_심리_지수", 5)
                st.write(
                    f"**시장 심리**: {sentiment}/10 "
                    f"({market_trend.get('시장_심리_판단', '')})"
                )

                # 시장 심리 게이지
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=sentiment,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "시장 심리 지수"},
                        gauge={
                            "axis": {"range": [1, 10]},
                            "bar": {"color": "#667eea"},
                            "steps": [
                                {"range": [1, 3], "color": "#fee2e2"},
                                {"range": [3, 5], "color": "#fef3c7"},
                                {"range": [5, 7], "color": "#d1fae5"},
                                {"range": [7, 10], "color": "#dbeafe"},
                            ],
                        },
                    )
                )
                fig_gauge.update_layout(height=250, margin=dict(t=50, b=0, l=30, r=30))
                st.plotly_chart(fig_gauge, use_container_width=True)

        with col_b:
            st.subheader("🏆 TOP 3 주목 종목")
            top3 = comprehensive.get("TOP3_주목종목", [])
            if top3:
                for t in top3:
                    medal = ["🥇", "🥈", "🥉"][t.get("순위", 1) - 1] if t.get("순위", 1) <= 3 else ""
                    st.markdown(
                        f"**{medal} {t.get('종목명', '')}** — "
                        f"{t.get('선정_이유', '')}"
                    )
                    st.caption(f"📌 모니터링: {t.get('핵심_모니터링', '')}")
            else:
                st.info("종합 분석 데이터가 없습니다.")

        # 리스크 경고
        risk = comprehensive.get("리스크_경고", {})
        if risk:
            st.markdown("---")
            st.subheader("⚠️ 리스크 경고")
            market_risks = risk.get("시장_리스크", [])
            if market_risks:
                for r in market_risks:
                    st.warning(r)

            warn_stocks = risk.get("주의_종목", [])
            if warn_stocks:
                for ws in warn_stocks:
                    st.error(
                        f"🚨 **{ws.get('종목명', '')}**: "
                        f"{ws.get('경고_사유', '')}"
                    )

# ===== Tab 2: 필터링 종목 =====
with tab_stocks:
    st.subheader("📋 거래량/거래대금 상위 필터링 종목")

    # 데이터 소스 결정
    stocks_data = []
    if report_data and report_data.get("filtered_analysis", {}).get("종목_리스트"):
        stocks_data = report_data["filtered_analysis"]["종목_리스트"]
    elif raw_data and raw_data.get("필터링_결과", {}).get("종목_리스트"):
        stocks_data = raw_data["필터링_결과"]["종목_리스트"]

    if stocks_data:
        df_stocks = pd.DataFrame(stocks_data)

        # 거래대금 차트
        fig_trading = px.bar(
            df_stocks.head(15),
            x="종목명",
            y="거래대금",
            color="등락률",
            color_continuous_scale="RdYlGn",
            title="거래대금 상위 15 종목",
            labels={"거래대금": "거래대금 (원)", "종목명": ""},
        )
        fig_trading.update_layout(height=400)
        st.plotly_chart(fig_trading, use_container_width=True)

        # 등락률 분포
        col1, col2 = st.columns(2)
        with col1:
            fig_change = px.bar(
                df_stocks.head(15),
                x="종목명",
                y="등락률",
                color="등락률",
                color_continuous_scale="RdYlGn",
                title="등락률 분포",
            )
            fig_change.update_layout(height=350)
            st.plotly_chart(fig_change, use_container_width=True)

        with col2:
            if "평균거래량_대비_배율" in df_stocks.columns:
                vol_ratio = df_stocks[
                    df_stocks["평균거래량_대비_배율"].notna()
                ].head(15)
                if not vol_ratio.empty:
                    fig_vol = px.bar(
                        vol_ratio,
                        x="종목명",
                        y="평균거래량_대비_배율",
                        title="평균 거래량 대비 배율",
                        color="평균거래량_대비_배율",
                        color_continuous_scale="Blues",
                    )
                    fig_vol.update_layout(height=350)
                    st.plotly_chart(fig_vol, use_container_width=True)

        # 전체 테이블
        st.dataframe(
            df_stocks,
            use_container_width=True,
            column_config={
                "종가": st.column_config.NumberColumn("종가", format="%d"),
                "등락률": st.column_config.NumberColumn("등락률(%)", format="%.2f"),
                "거래량": st.column_config.NumberColumn("거래량", format="%d"),
                "거래대금": st.column_config.NumberColumn("거래대금", format="%d"),
            },
        )
    else:
        st.info("필터링 종목 데이터가 없습니다.")

# ===== Tab 3: 재무분석 =====
with tab_fundamental:
    st.subheader("💰 재무지표 분석")

    fund_analysis = (
        report_data.get("fundamental_analysis", {}).get("종목_분석", [])
        if report_data
        else []
    )

    if fund_analysis:
        for stock in fund_analysis:
            grade = stock.get("종합등급", "N/A")
            grade_colors = {"A": "🟢", "B": "🔵", "C": "🟡", "D": "🔴"}
            grade_emoji = grade_colors.get(grade, "⚪")

            with st.expander(
                f"{grade_emoji} **{stock.get('종목명', '')}** ({stock.get('종목코드', '')}) — 등급: {grade}",
                expanded=False,
            ):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "PER",
                        stock.get("PER", "N/A"),
                        help=f"업종평균: {stock.get('업종평균_PER', 'N/A')}",
                    )
                    st.caption(f"판단: {stock.get('PER_판단', 'N/A')}")
                with col2:
                    st.metric("PBR", stock.get("PBR", "N/A"))
                    st.caption(f"판단: {stock.get('PBR_판단', 'N/A')}")
                with col3:
                    st.metric("ROE", f"{stock.get('ROE', 'N/A')}%")
                    st.caption(f"판단: {stock.get('ROE_판단', 'N/A')}")
                with col4:
                    st.metric("배당수익률", f"{stock.get('배당수익률', 'N/A')}%")

                if stock.get("종합의견"):
                    st.info(f"📝 **종합 의견**: {stock['종합의견']}")
    else:
        # 원시 재무 데이터 표시
        raw_fund = (
            raw_data.get("재무지표", {}).get("종목_재무지표", []) if raw_data else []
        )
        if raw_fund:
            st.write("원시 재무지표 데이터:")
            st.dataframe(pd.DataFrame(raw_fund), use_container_width=True)
        else:
            st.info("재무분석 데이터가 없습니다.")

# ===== Tab 4: 뉴스 분석 =====
with tab_news:
    st.subheader("📰 뉴스 감성 분석")

    news_analysis = report_data.get("news_analysis", []) if report_data else []

    if news_analysis:
        # 뉴스 스코어 개요
        score_data = [
            {
                "종목명": n.get("종목명", ""),
                "뉴스_스코어": n.get("종합_뉴스스코어", 0),
                "분석_뉴스수": n.get("분석_뉴스수", 0),
            }
            for n in news_analysis
        ]
        df_scores = pd.DataFrame(score_data)

        if not df_scores.empty:
            fig_score = px.bar(
                df_scores,
                x="종목명",
                y="뉴스_스코어",
                color="뉴스_스코어",
                color_continuous_scale="RdYlGn",
                title="종목별 뉴스 감성 스코어 (-100 ~ +100)",
                range_color=[-100, 100],
            )
            fig_score.update_layout(height=400)
            st.plotly_chart(fig_score, use_container_width=True)

        # 종목별 뉴스 상세
        for news in news_analysis:
            with st.expander(
                f"📰 **{news.get('종목명', '')}** — "
                f"스코어: {news.get('종합_뉴스스코어', 0)} | "
                f"{news.get('분석_뉴스수', 0)}건 분석"
            ):
                if news.get("뉴스_요약"):
                    st.info(f"📝 {news['뉴스_요약']}")

                for article in news.get("뉴스_분석", []):
                    sentiment = article.get("감성", "중립")
                    sentiment_emoji = {
                        "매우 긍정": "🟢",
                        "긍정": "🔵",
                        "중립": "⚪",
                        "부정": "🟡",
                        "매우 부정": "🔴",
                    }.get(sentiment, "⚪")

                    st.markdown(
                        f"{sentiment_emoji} **{article.get('제목', '')}** "
                        f"[{article.get('카테고리', '')}] "
                        f"({article.get('날짜', '')})"
                    )
                    st.caption(
                        f"{article.get('요약', '')} | "
                        f"감성: {sentiment} | "
                        f"신뢰도: {article.get('신뢰도', '')}"
                    )
    else:
        st.info("뉴스 분석 데이터가 없습니다.")

# ===== Tab 5: 전체 리포트 =====
with tab_report:
    st.subheader("📄 AI 생성 리포트")

    # 사용된 프로바이더 정보 표시 (Step별)
    if report_data:
        ai_providers = report_data.get("ai_providers", {})
        ai_names = report_data.get("ai_provider_names", {})
        if ai_providers:
            cols = st.columns(5)
            for i, (step_key, provider_id) in enumerate(ai_providers.items()):
                step_num = step_key.replace("step", "")
                name = ai_names.get(step_key, provider_id)
                icon = {"claude": "🟠", "gpt": "🟢", "gemini": "🔵", "grok": "⚫"}.get(provider_id, "")
                with cols[i]:
                    st.caption(f"S{step_num}: {icon}{name}")

    if markdown_report:
        st.markdown(markdown_report)
    else:
        st.info("마크다운 리포트가 없습니다.")

    # JSON 데이터 다운로드
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if report_data:
            st.download_button(
                "📥 분석 데이터 (JSON) 다운로드",
                data=json.dumps(report_data, ensure_ascii=False, indent=2),
                file_name=f"analysis_{selected_date}.json",
                mime="application/json",
            )
    with col2:
        if markdown_report:
            st.download_button(
                "📥 리포트 (Markdown) 다운로드",
                data=markdown_report,
                file_name=f"report_{selected_date}.md",
                mime="text/markdown",
            )

# ===== 푸터 =====
st.markdown("---")
st.caption(
    "⚠️ **면책조항**: 본 대시보드는 AI가 공개 데이터를 분석하여 생성한 참고 자료입니다. "
    "투자 권유가 아니며, 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다."
)
