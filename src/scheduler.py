"""
스케줄러 모듈
매일 장 마감 후(15:30) 자동으로 분석을 실행
"""

import logging
import time
from datetime import datetime

import schedule

logger = logging.getLogger(__name__)


class AnalysisScheduler:
    """일일 분석 자동 실행 스케줄러"""

    def __init__(self, run_analysis_func):
        """
        Args:
            run_analysis_func: 실행할 분석 함수 (main.py의 run_daily_analysis)
        """
        self.run_analysis = run_analysis_func
        self._is_running = False

    def _job(self):
        """스케줄 작업 실행"""
        now = datetime.now()
        # 주말 제외 (0=월요일, 6=일요일)
        if now.weekday() >= 5:
            logger.info(f"[스케줄러] 주말({now.strftime('%A')})이므로 분석 건너뜀")
            return

        logger.info(f"[스케줄러] 일일 분석 시작: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            self._is_running = True
            self.run_analysis()
            logger.info("[스케줄러] 일일 분석 완료")
        except Exception as e:
            logger.error(f"[스케줄러] 일일 분석 실패: {e}")
        finally:
            self._is_running = False

    def start(self, run_time: str = "15:40"):
        """
        스케줄러 시작

        Args:
            run_time: 실행 시각 (HH:MM 형식, 기본값: "15:40" - 장 마감 10분 후)
        """
        logger.info(f"[스케줄러] 매일 {run_time}에 분석이 자동 실행됩니다.")
        logger.info("[스케줄러] 중지하려면 Ctrl+C를 누르세요.")

        schedule.every().day.at(run_time).do(self._job)

        try:
            while True:
                schedule.run_pending()
                time.sleep(30)  # 30초마다 체크
        except KeyboardInterrupt:
            logger.info("[스케줄러] 스케줄러가 종료되었습니다.")

    def run_once(self):
        """스케줄러 없이 즉시 1회 실행"""
        logger.info("[스케줄러] 즉시 실행 모드")
        self._job()

    @property
    def is_running(self) -> bool:
        """현재 분석 실행 중 여부"""
        return self._is_running
