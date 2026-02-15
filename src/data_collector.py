"""
한국 주식 시장 데이터 수집 모듈
pykrx를 활용한 거래량/거래대금 데이터 및 재무지표 수집
"""

import os
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from pykrx import stock as pykrx_stock
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── pykrx 내부 로거가 쓰레기 에러를 찍는 것을 억제 ──
# pykrx 내부(urllib3, pykrx.website 등)에서 특정 종목/날짜 조합에
# 빈 응답이 돌아올 때 ERROR 레벨 로그를 남기므로, WARNING 이상만 통과시킨다.
for _name in ("pykrx", "pykrx.website", "urllib3", "urllib3.connectionpool"):
    logging.getLogger(_name).setLevel(logging.WARNING)


class StockDataCollector:
    """KRX 주식 데이터 수집기"""

    def __init__(self):
        self.volume_threshold = int(os.getenv("VOLUME_THRESHOLD", "10000000"))
        self.trading_value_threshold = int(
            os.getenv("TRADING_VALUE_THRESHOLD", "10000000000")
        )

    # ── 거래일 유틸리티 ──────────────────────────────────────
    def get_latest_trading_date(self, date: Optional[str] = None) -> str:
        """
        가장 최근 실제 거래일 반환 (주말·공휴일·임시휴장 모두 자동 건너뜀).

        pykrx의 거래일 캘린더를 사용하므로 공휴일까지 정확하게 처리됩니다.
        pykrx 호출이 실패하면 기존의 주말-만 보정 로직으로 폴백합니다.
        """
        if date:
            return date

        today = datetime.now()
        try:
            # 최근 2주 거래일 목록을 조회하여 가장 마지막 거래일을 반환
            start = (today - timedelta(days=14)).strftime("%Y%m%d")
            end = today.strftime("%Y%m%d")
            trading_days = pykrx_stock.get_previous_business_days(fromdate=start, todate=end)
            if trading_days is not None and len(trading_days) > 0:
                latest = trading_days[-1]
                # Timestamp → str
                return pd.Timestamp(latest).strftime("%Y%m%d")
        except Exception as e:
            logger.debug(f"[거래일] pykrx 거래일 캘린더 조회 실패, 주말 보정으로 폴백: {e}")

        # 폴백: 주말만 보정
        if today.weekday() == 5:
            today -= timedelta(days=1)
        elif today.weekday() == 6:
            today -= timedelta(days=2)
        return today.strftime("%Y%m%d")

    def _find_nearest_trading_date(self, date_str: str, search_days: int = 10) -> Optional[str]:
        """
        주어진 날짜 기준으로 가장 가까운 과거 거래일을 반환.

        _get_prev_eps 등에서 비거래일(공휴일·주말)을 자동 보정하는 데 사용합니다.
        """
        try:
            end = datetime.strptime(date_str, "%Y%m%d")
            start = end - timedelta(days=search_days)
            trading_days = pykrx_stock.get_previous_business_days(
                fromdate=start.strftime("%Y%m%d"),
                todate=end.strftime("%Y%m%d"),
            )
            if trading_days is not None and len(trading_days) > 0:
                return pd.Timestamp(trading_days[-1]).strftime("%Y%m%d")
        except Exception:
            pass
        return None

    def get_market_ohlcv(self, date: str, market: str = "ALL") -> pd.DataFrame:
        """
        시장 전체 OHLCV(시가/고가/저가/종가/거래량) 데이터 수집

        Args:
            date: 조회 날짜 (YYYYMMDD)
            market: 시장 구분 ("KOSPI", "KOSDAQ", "ALL")

        Returns:
            pd.DataFrame: OHLCV 데이터
        """
        logger.info(f"[데이터 수집] {date} 기준 {market} OHLCV 데이터 수집 시작")
        try:
            if market == "ALL":
                df_kospi = pykrx_stock.get_market_ohlcv(date, market="KOSPI")
                df_kosdaq = pykrx_stock.get_market_ohlcv(date, market="KOSDAQ")
                df = pd.concat([df_kospi, df_kosdaq])
            else:
                df = pykrx_stock.get_market_ohlcv(date, market=market)

            if df.empty:
                logger.warning(f"[데이터 수집] {date}에 대한 OHLCV 데이터가 없습니다.")
                return pd.DataFrame()

            logger.debug(f"[데이터 수집][원시] OHLCV DataFrame shape: {df.shape}")
            logger.debug(f"[데이터 수집][원시] OHLCV 컬럼: {list(df.columns)}")
            logger.debug(f"[데이터 수집][원시] OHLCV 상위 10행:\n{df.head(10).to_string()}")
            logger.debug(
                f"[데이터 수집][원시] 거래량 통계: "
                f"min={df['거래량'].min():,}, max={df['거래량'].max():,}, "
                f"mean={df['거래량'].mean():,.0f}, median={df['거래량'].median():,.0f}"
            )
            logger.debug(
                f"[데이터 수집][원시] 거래대금 통계: "
                f"min={df['거래대금'].min():,}, max={df['거래대금'].max():,}, "
                f"mean={df['거래대금'].mean():,.0f}"
            )

            # 종목명 추가
            df["종목명"] = df.index.map(
                lambda x: pykrx_stock.get_market_ticker_name(x)
            )
            df.index.name = "종목코드"
            df = df.reset_index()

            logger.info(f"[데이터 수집] 총 {len(df)}개 종목 원시 데이터 수집 완료 (필터링 전)")
            return df

        except Exception as e:
            logger.error(f"[데이터 수집] OHLCV 데이터 수집 실패: {e}")
            raise

    def get_market_trading_value(self, date: str, market: str = "ALL") -> pd.DataFrame:
        """
        거래대금 데이터 수집

        Args:
            date: 조회 날짜 (YYYYMMDD)
            market: 시장 구분

        Returns:
            pd.DataFrame: 거래대금 포함 데이터
        """
        logger.info(f"[데이터 수집] {date} 기준 거래대금 데이터 수집 시작")
        try:
            if market == "ALL":
                df_kospi = pykrx_stock.get_market_ohlcv(date, market="KOSPI")
                df_kosdaq = pykrx_stock.get_market_ohlcv(date, market="KOSDAQ")
                df = pd.concat([df_kospi, df_kosdaq])
            else:
                df = pykrx_stock.get_market_ohlcv(date, market=market)

            if df.empty:
                return pd.DataFrame()

            logger.debug(f"[데이터 수집][원시] 거래대금 DataFrame shape: {df.shape}")
            logger.debug(f"[데이터 수집][원시] 거래대금 상위 10행:\n{df.head(10).to_string()}")

            df["종목명"] = df.index.map(
                lambda x: pykrx_stock.get_market_ticker_name(x)
            )
            df.index.name = "종목코드"
            df = df.reset_index()

            return df

        except Exception as e:
            logger.error(f"[데이터 수집] 거래대금 데이터 수집 실패: {e}")
            raise

    def get_market_cap(self, date: str, market: str = "ALL") -> pd.DataFrame:
        """시가총액 데이터 수집"""
        logger.info(f"[데이터 수집] {date} 기준 시가총액 데이터 수집 시작")
        try:
            if market == "ALL":
                df_kospi = pykrx_stock.get_market_cap(date, market="KOSPI")
                df_kosdaq = pykrx_stock.get_market_cap(date, market="KOSDAQ")
                df = pd.concat([df_kospi, df_kosdaq])
            else:
                df = pykrx_stock.get_market_cap(date, market=market)

            if df.empty:
                return pd.DataFrame()

            logger.debug(f"[데이터 수집][원시] 시가총액 DataFrame shape: {df.shape}")
            logger.debug(f"[데이터 수집][원시] 시가총액 컬럼: {list(df.columns)}")
            logger.debug(f"[데이터 수집][원시] 시가총액 상위 10행:\n{df.head(10).to_string()}")

            df.index.name = "종목코드"
            df = df.reset_index()
            return df

        except Exception as e:
            logger.error(f"[데이터 수집] 시가총액 데이터 수집 실패: {e}")
            raise

    def calculate_avg_volume(
        self, ticker: str, end_date: str, period: int = 20
    ) -> float:
        """
        특정 종목의 최근 N일 평균 거래량 계산

        Args:
            ticker: 종목코드
            end_date: 기준일 (YYYYMMDD)
            period: 평균 산출 기간 (기본 20일)

        Returns:
            float: 평균 거래량
        """
        try:
            start_date = (
                datetime.strptime(end_date, "%Y%m%d") - timedelta(days=period * 2)
            ).strftime("%Y%m%d")
            df = pykrx_stock.get_market_ohlcv(start_date, end_date, ticker)
            if df.empty or len(df) < 2:
                return 0.0
            # 마지막 날(당일) 제외하고 최근 period일
            recent = df.iloc[-(period + 1) : -1]
            return recent["거래량"].mean() if not recent.empty else 0.0
        except Exception as e:
            logger.warning(f"[데이터 수집] {ticker} 평균 거래량 계산 실패: {e}")
            return 0.0

    def filter_stocks(self, date: str) -> dict:
        """
        거래량/거래대금 기준으로 종목 필터링

        Args:
            date: 조회 날짜 (YYYYMMDD)

        Returns:
            dict: 필터링된 종목 데이터 (프롬프트 1 입력용)
        """
        logger.info(
            f"[필터링] 거래량 >= {self.volume_threshold:,}, "
            f"거래대금 >= {self.trading_value_threshold:,} 기준 필터링"
        )

        df = self.get_market_ohlcv(date)
        if df.empty:
            return {"기준일": date, "필터링_종목수": 0, "종목_리스트": []}

        # 등락률 계산 (시가 대비)
        df["등락률"] = df.apply(
            lambda row: round(
                ((row["종가"] - row["시가"]) / row["시가"] * 100), 2
            )
            if row["시가"] > 0
            else 0,
            axis=1,
        )

        # 필터링: 거래량 >= 1,000만 주 AND 거래대금 >= 100억 원
        filtered = df[
            (df["거래량"] >= self.volume_threshold)
            & (df["거래대금"] >= self.trading_value_threshold)
        ].copy()

        logger.info(
            f"[필터링] 전체 {len(df)}개 → 필터링 후 {len(filtered)}개 "
            f"(거래량 >= {self.volume_threshold:,} AND 거래대금 >= {self.trading_value_threshold:,})"
        )
        logger.debug(
            f"[필터링][원시] 필터링된 종목 목록:\n"
            f"{filtered[['종목코드', '종목명', '종가', '거래량', '거래대금']].to_string()}"
            if not filtered.empty else "[필터링][원시] 필터링된 종목 없음"
        )

        # 거래대금 기준 내림차순 정렬
        filtered = filtered.sort_values("거래대금", ascending=False)

        # 평균 거래량 대비 배율 계산
        stock_list = []
        for _, row in filtered.iterrows():
            avg_vol = self.calculate_avg_volume(row["종목코드"], date)
            ratio = round(row["거래량"] / avg_vol, 2) if avg_vol > 0 else None

            stock_list.append(
                {
                    "종목코드": row["종목코드"],
                    "종목명": row["종목명"],
                    "종가": int(row["종가"]),
                    "등락률": row["등락률"],
                    "거래량": int(row["거래량"]),
                    "거래대금": int(row["거래대금"]),
                    "평균거래량_대비_배율": ratio,
                }
            )

        formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        result = {
            "기준일": formatted_date,
            "필터링_종목수": len(stock_list),
            "종목_리스트": stock_list,
        }

        logger.info(f"[필터링] {len(stock_list)}개 종목 필터링 완료")
        for s in stock_list:
            logger.debug(
                f"[필터링][원시] {s['종목명']}({s['종목코드']}): "
                f"종가={s['종가']:,}원, 등락률={s['등락률']}%, "
                f"거래량={s['거래량']:,}, 거래대금={s['거래대금']:,}, "
                f"평균거래량대비={s['평균거래량_대비_배율']}배"
            )
        return result

    # ── 재무지표 안전 래퍼 ─────────────────────────────────
    @staticmethod
    def _safe_get_fundamental(
        date: str,
        market: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        pykrx.get_market_fundamental 호출을 안전하게 감싸는 헬퍼.

        - 빈 DataFrame, 기대 컬럼 누락, 내부 예외를 모두 처리합니다.
        - 반환값은 항상 PER/PBR/EPS/DIV 컬럼이 존재하는 DataFrame이거나 빈 DataFrame입니다.
        """
        REQUIRED_COLS = {"PER", "PBR", "EPS", "DIV"}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if ticker:
                    df = pykrx_stock.get_market_fundamental(date, date, ticker)
                elif market:
                    df = pykrx_stock.get_market_fundamental(date, market=market)
                else:
                    df = pykrx_stock.get_market_fundamental(date)

            if df is None or df.empty:
                return pd.DataFrame()

            # 기대 컬럼이 하나라도 없으면 빈 DataFrame 반환
            if not REQUIRED_COLS.issubset(set(df.columns)):
                missing = REQUIRED_COLS - set(df.columns)
                logger.debug(
                    f"[재무지표] fundamental 응답에 컬럼 누락 {missing} "
                    f"(date={date}, market={market}, ticker={ticker})"
                )
                return pd.DataFrame()

            return df
        except Exception as e:
            logger.debug(
                f"[재무지표] fundamental 조회 예외 무시 "
                f"(date={date}, market={market}, ticker={ticker}): {e}"
            )
            return pd.DataFrame()

    def get_fundamental_data(self, date: str, tickers: list[str]) -> dict:
        """
        필터링된 종목들의 재무지표 수집

        Args:
            date: 조회 날짜 (YYYYMMDD)
            tickers: 종목코드 리스트

        Returns:
            dict: 재무지표 데이터 (프롬프트 2 입력용)
        """
        logger.info(f"[재무지표] {len(tickers)}개 종목 재무지표 수집 시작")

        # 전체 PER/PBR/EPS/DIV 데이터 (안전 래퍼 사용)
        fundamental_kospi = self._safe_get_fundamental(date, market="KOSPI")
        fundamental_kosdaq = self._safe_get_fundamental(date, market="KOSDAQ")

        logger.debug(
            f"[재무지표][원시] KOSPI fundamental shape: {fundamental_kospi.shape}, "
            f"KOSDAQ fundamental shape: {fundamental_kosdaq.shape}"
        )
        if not fundamental_kospi.empty:
            logger.debug(f"[재무지표][원시] KOSPI fundamental 상위 5행:\n{fundamental_kospi.head(5).to_string()}")
        if not fundamental_kosdaq.empty:
            logger.debug(f"[재무지표][원시] KOSDAQ fundamental 상위 5행:\n{fundamental_kosdaq.head(5).to_string()}")

        if not fundamental_kospi.empty and not fundamental_kosdaq.empty:
            fundamental_all = pd.concat([fundamental_kospi, fundamental_kosdaq])
        elif not fundamental_kospi.empty:
            fundamental_all = fundamental_kospi
        elif not fundamental_kosdaq.empty:
            fundamental_all = fundamental_kosdaq
        else:
            fundamental_all = pd.DataFrame()
            logger.warning(f"[재무지표] {date} 전체 시장 재무지표를 가져오지 못했습니다.")

        # 전년 EPS 조회용 날짜 미리 계산 (가장 가까운 과거 거래일로 보정)
        prev_year_raw = (
            datetime.strptime(date, "%Y%m%d") - timedelta(days=365)
        ).strftime("%Y%m%d")
        prev_year_date = self._find_nearest_trading_date(prev_year_raw) or prev_year_raw

        results = []
        skip_prev_eps_count = 0

        for ticker in tickers:
            try:
                ticker_name = pykrx_stock.get_market_ticker_name(ticker)

                # ── 당기 재무지표 추출 ──
                per = pbr = eps = div_yield = None
                if (
                    not fundamental_all.empty
                    and ticker in fundamental_all.index
                ):
                    fund = fundamental_all.loc[ticker]
                    per = fund.get("PER", None)
                    pbr = fund.get("PBR", None)
                    eps = fund.get("EPS", None)
                    div_yield = fund.get("DIV", None)

                # ── 업종 PER 평균 ──
                sector_avg_per = self._get_sector_avg_per(
                    ticker, date, fundamental_all
                )

                # ── 전년 EPS ──
                prev_eps = self._get_prev_eps(ticker, prev_year_date)
                if prev_eps is None:
                    skip_prev_eps_count += 1

                # ── EPS 변화율 ──
                eps_change = None
                if eps and prev_eps and prev_eps != 0:
                    eps_change = round(
                        ((eps - prev_eps) / abs(prev_eps)) * 100, 1
                    )

                # ── ROE 근사 (PBR / PER) ──
                roe = None
                if per and pbr and per > 0:
                    roe = round((pbr / per) * 100, 2)

                result_item = {
                    "종목코드": ticker,
                    "종목명": ticker_name,
                    "PER": round(per, 2) if per else None,
                    "PBR": round(pbr, 2) if pbr else None,
                    "EPS": int(eps) if eps else None,
                    "EPS_전년": int(prev_eps) if prev_eps else None,
                    "EPS_변화율": eps_change,
                    "ROE": roe,
                    "배당수익률": round(div_yield, 2) if div_yield else None,
                    "업종평균_PER": (
                        round(sector_avg_per, 2) if sector_avg_per else None
                    ),
                }
                logger.debug(
                    f"[재무지표][원시] {ticker_name}({ticker}): "
                    f"PER={result_item['PER']}, PBR={result_item['PBR']}, "
                    f"EPS={result_item['EPS']}, EPS전년={result_item['EPS_전년']}, "
                    f"EPS변화율={result_item['EPS_변화율']}%, ROE={result_item['ROE']}%, "
                    f"배당수익률={result_item['배당수익률']}%, "
                    f"업종평균PER={result_item['업종평균_PER']}"
                )
                results.append(result_item)

            except Exception as e:
                logger.warning(f"[재무지표] {ticker} 데이터 수집 실패: {e}")
                # 실패 시에도 ticker_name 조회가 또 실패할 수 있으므로 안전하게
                try:
                    t_name = pykrx_stock.get_market_ticker_name(ticker)
                except Exception:
                    t_name = ticker
                results.append(
                    {
                        "종목코드": ticker,
                        "종목명": t_name,
                        "PER": None,
                        "PBR": None,
                        "EPS": None,
                        "EPS_전년": None,
                        "EPS_변화율": None,
                        "ROE": None,
                        "배당수익률": None,
                        "업종평균_PER": None,
                    }
                )

        if skip_prev_eps_count > 0:
            logger.warning(
                f"[재무지표] {skip_prev_eps_count}/{len(tickers)}개 종목의 "
                f"전년 EPS를 가져오지 못했습니다 (전년 기준일={prev_year_date})."
            )

        logger.info(f"[재무지표] {len(results)}개 종목 재무지표 수집 완료")
        return {"기준일": date, "종목_재무지표": results}

    def _get_sector_avg_per(
        self, ticker: str, date: str, fundamental_all: pd.DataFrame
    ) -> Optional[float]:
        """업종 평균 PER 계산 (동일 시장 평균으로 근사)"""
        try:
            if fundamental_all.empty or "PER" not in fundamental_all.columns:
                return None
            # PER가 0보다 크고 합리적인 범위인 종목만 필터
            valid = fundamental_all[
                (fundamental_all["PER"] > 0) & (fundamental_all["PER"] < 200)
            ]
            if valid.empty:
                return None
            return valid["PER"].median()  # 중앙값 사용 (이상치 영향 최소화)
        except Exception:
            return None

    def _get_prev_eps(self, ticker: str, date: str) -> Optional[float]:
        """
        전년도 EPS 조회.

        - _safe_get_fundamental 래퍼를 사용하여 빈/특이 DataFrame 방어.
        - 비거래일이 넘어온 경우 _find_nearest_trading_date 로 이미
          호출 전에 보정되므로, 여기서는 단순 조회만 수행.
        """
        fund = self._safe_get_fundamental(date, ticker=ticker)
        if fund.empty:
            return None
        try:
            eps_val = fund.iloc[0].get("EPS", None)
            # EPS가 0이면 의미 있는 값이므로 그대로 반환
            return eps_val
        except (IndexError, AttributeError):
            return None

    def get_market_index(self, date: str) -> dict:
        """
        코스피/코스닥 지수 데이터 수집

        Args:
            date: 조회 날짜 (YYYYMMDD)

        Returns:
            dict: 시장 지수 데이터
        """
        logger.info(f"[시장지수] {date} 기준 시장 지수 수집")
        try:
            # 코스피 지수
            kospi = pykrx_stock.get_index_ohlcv(date, date, "1001")  # 코스피 지수 코드
            kosdaq = pykrx_stock.get_index_ohlcv(date, date, "2001")  # 코스닥 지수 코드

            logger.debug(f"[시장지수][원시] KOSPI 지수 DataFrame:\n{kospi.to_string()}" if not kospi.empty else "[시장지수][원시] KOSPI 데이터 없음")
            logger.debug(f"[시장지수][원시] KOSDAQ 지수 DataFrame:\n{kosdaq.to_string()}" if not kosdaq.empty else "[시장지수][원시] KOSDAQ 데이터 없음")

            kospi_close = float(kospi.iloc[-1]["종가"]) if not kospi.empty else None
            kospi_change = float(kospi.iloc[-1]["등락률"]) if not kospi.empty else None
            kosdaq_close = float(kosdaq.iloc[-1]["종가"]) if not kosdaq.empty else None
            kosdaq_change = float(kosdaq.iloc[-1]["등락률"]) if not kosdaq.empty else None

            logger.debug(
                f"[시장지수][원시] 코스피: {kospi_close} (등락률 {kospi_change}%), "
                f"코스닥: {kosdaq_close} (등락률 {kosdaq_change}%)"
            )

            return {
                "코스피": {"지수": kospi_close, "등락률": kospi_change},
                "코스닥": {"지수": kosdaq_close, "등락률": kosdaq_change},
            }
        except Exception as e:
            logger.warning(f"[시장지수] 시장 지수 수집 실패: {e}")
            return {
                "코스피": {"지수": None, "등락률": None},
                "코스닥": {"지수": None, "등락률": None},
            }

    def get_foreign_trading(self, date: str) -> dict:
        """
        외국인 순매수/매도 데이터 수집

        Args:
            date: 조회 날짜 (YYYYMMDD)

        Returns:
            dict: 외국인 매매 동향
        """
        logger.info(f"[외국인매매] {date} 기준 외국인 매매 동향 수집")
        try:
            # pykrx investor: "외국인" (9000). "외국인합계"는 미지원 → KeyError 방지
            df = pykrx_stock.get_market_net_purchases_of_equities(
                date, date, market="KOSPI", investor="외국인"
            )
            if df.empty:
                return {"외국인_순매수_금액": None, "외국인_순매수_판단": "데이터 없음"}

            logger.debug(f"[외국인매매][원시] DataFrame shape: {df.shape}")
            logger.debug(f"[외국인매매][원시] 컬럼: {list(df.columns)}")
            logger.debug(f"[외국인매매][원시] 상위 10행:\n{df.head(10).to_string()}")

            total_buy = df["순매수거래대금"].sum() if "순매수거래대금" in df.columns else 0
            logger.debug(f"[외국인매매][원시] 순매수거래대금 합계: {total_buy:,}원")

            if total_buy > 0:
                judgment = f"외국인 순매수 {total_buy / 100000000:,.0f}억 원"
            elif total_buy < 0:
                judgment = f"외국인 순매도 {abs(total_buy) / 100000000:,.0f}억 원"
            else:
                judgment = "외국인 매매 중립"

            return {"외국인_순매수_금액": int(total_buy), "외국인_순매수_판단": judgment}

        except Exception as e:
            logger.warning(f"[외국인매매] 외국인 매매 동향 수집 실패: {e}")
            return {"외국인_순매수_금액": None, "외국인_순매수_판단": "데이터 없음"}

    # ── 재무등급 규칙 기반 계산 ─────────────────────────────
    @staticmethod
    def grade_fundamental(fund_list: list[dict]) -> list[dict]:
        """
        재무지표에 규칙 기반 등급(A-D)을 부여.

        기존 prompt_2의 AI 분석 로직을 Python으로 이전:
        - PER: 업종평균 대비 저평가/적정/고평가
        - PBR: 자산가치 대비 평가
        - ROE: 수익성 평가
        - EPS 변화율: 성장성 평가
        - 종합등급: A/B/C/D
        """
        graded = []
        for item in fund_list:
            g = dict(item)  # 원본 보존

            per = g.get("PER")
            pbr = g.get("PBR")
            roe = g.get("ROE")
            eps_change = g.get("EPS_변화율")
            sector_per = g.get("업종평균_PER")

            # ── PER 판단 ──
            if per is None:
                g["PER_판단"] = "데이터 없음"
            elif per < 0:
                g["PER_판단"] = "적자 전환 주의"
            elif sector_per and sector_per > 0:
                if per < sector_per * 0.7:
                    g["PER_판단"] = "저평가"
                elif per > sector_per * 1.3:
                    g["PER_판단"] = "고평가"
                else:
                    g["PER_판단"] = "적정"
            else:
                g["PER_판단"] = "업종평균 없음"

            # ── PBR 판단 ──
            if pbr is None:
                g["PBR_판단"] = "데이터 없음"
            elif pbr < 1.0:
                g["PBR_판단"] = "자산가치 대비 저평가 가능성"
            elif pbr > 3.0:
                g["PBR_판단"] = "성장 프리미엄 반영 또는 고평가"
            else:
                g["PBR_판단"] = "적정"

            # ── ROE 판단 ──
            if roe is None:
                g["ROE_판단"] = "데이터 없음"
            elif roe >= 15:
                g["ROE_판단"] = "우수"
            elif roe >= 10:
                g["ROE_판단"] = "양호"
            elif roe >= 0:
                g["ROE_판단"] = "보통"
            else:
                g["ROE_판단"] = "부진"

            # ── EPS 성장성 판단 ──
            if eps_change is None:
                g["EPS_성장"] = "데이터 없음"
            elif eps_change > 20:
                g["EPS_성장"] = "고성장"
            elif eps_change > 0:
                g["EPS_성장"] = "성장"
            elif eps_change > -20:
                g["EPS_성장"] = "둔화"
            else:
                g["EPS_성장"] = "악화"

            # ── 종합등급 (A/B/C/D) ──
            score = 0
            # PER 점수
            per_j = g.get("PER_판단", "")
            if per_j == "저평가":
                score += 2
            elif per_j == "적정":
                score += 1
            elif per_j in ("고평가", "적자 전환 주의"):
                score -= 1

            # PBR 점수
            pbr_j = g.get("PBR_판단", "")
            if "저평가" in pbr_j:
                score += 1
            elif "고평가" in pbr_j:
                score -= 1

            # ROE 점수
            roe_j = g.get("ROE_판단", "")
            if roe_j == "우수":
                score += 2
            elif roe_j == "양호":
                score += 1
            elif roe_j == "부진":
                score -= 1

            # EPS 성장 점수
            eps_g = g.get("EPS_성장", "")
            if eps_g == "고성장":
                score += 2
            elif eps_g == "성장":
                score += 1
            elif eps_g == "악화":
                score -= 1

            if score >= 4:
                g["종합등급"] = "A"
                g["등급_설명"] = "저평가 + 수익성 양호 + 성장성"
            elif score >= 2:
                g["종합등급"] = "B"
                g["등급_설명"] = "밸류에이션 적정 + 안정적 수익"
            elif score >= 0:
                g["종합등급"] = "C"
                g["등급_설명"] = "고평가이나 성장 모멘텀 존재"
            else:
                g["종합등급"] = "D"
                g["등급_설명"] = "고평가 + 수익성 악화"

            graded.append(g)

        return graded

    def collect_all_data(self, date: Optional[str] = None) -> dict:
        """
        모든 데이터를 종합 수집하는 메인 메서드

        Args:
            date: 조회 날짜 (YYYYMMDD, 기본값: 최근 거래일)

        Returns:
            dict: 전체 수집 데이터
        """
        date = self.get_latest_trading_date(date)
        logger.info(f"{'='*50}")
        logger.info(f"[전체 수집] {date} 기준 데이터 수집 시작")
        logger.info(f"{'='*50}")

        # Step 1: 거래량/거래대금 필터링
        filtered_stocks = self.filter_stocks(date)

        # Step 2: 필터링된 종목의 재무지표 수집 + 등급 부여
        tickers = [s["종목코드"] for s in filtered_stocks["종목_리스트"]]
        fundamental_data = (
            self.get_fundamental_data(date, tickers) if tickers else {"종목_재무지표": []}
        )

        # 규칙 기반 재무등급 계산
        raw_fund = fundamental_data.get("종목_재무지표", [])
        graded_fund = self.grade_fundamental(raw_fund)
        fundamental_data["종목_재무지표"] = graded_fund
        logger.info(
            f"[재무등급] {len(graded_fund)}개 종목 등급 부여 완료: "
            + ", ".join(f"{g['종목명']}={g.get('종합등급','?')}" for g in graded_fund[:5])
        )

        # Step 3: 시장 지수 수집
        market_index = self.get_market_index(date)

        # Step 4: 외국인 매매 동향
        foreign_trading = self.get_foreign_trading(date)

        result = {
            "기준일": date,
            "필터링_결과": filtered_stocks,
            "재무지표": fundamental_data,
            "시장_지수": market_index,
            "외국인_매매": foreign_trading,
        }

        logger.info(f"[전체 수집] 데이터 수집 완료")
        logger.debug(f"[전체 수집][원시] 수집 결과 키: {list(result.keys())}")
        logger.debug(
            f"[전체 수집][원시] 필터링 종목 수: {filtered_stocks.get('필터링_종목수', 0)}, "
            f"재무지표 종목 수: {len(fundamental_data.get('종목_재무지표', []))}"
        )
        logger.debug(f"[전체 수집][원시] 시장 지수: {market_index}")
        logger.debug(f"[전체 수집][원시] 외국인 매매: {foreign_trading}")
        return result


if __name__ == "__main__":
    import argparse as _argparse

    _parser = _argparse.ArgumentParser(description="데이터 수집 단독 실행")
    _parser.add_argument("--debug", action="store_true", help="DEBUG 레벨 로그 활성화 (원시 데이터 출력)")
    _parser.add_argument("--date", type=str, default=None, help="분석 날짜 (YYYYMMDD)")
    _args = _parser.parse_args()

    log_level = logging.DEBUG if _args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # DEBUG 모드에서도 pykrx 내부 로거는 억제
    if _args.debug:
        for _name in ("pykrx", "pykrx.website", "urllib3", "urllib3.connectionpool"):
            logging.getLogger(_name).setLevel(logging.WARNING)

    collector = StockDataCollector()
    data = collector.collect_all_data(_args.date)
    print(f"\n필터링 종목 수: {data['필터링_결과']['필터링_종목수']}")
    for stock in data["필터링_결과"]["종목_리스트"][:5]:
        print(f"  {stock['종목명']} ({stock['종목코드']}): 거래대금 {stock['거래대금']:,}")
