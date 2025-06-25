"""
통계 관련 API 라우터

사용자별 체형 변화 추이, 자세 개선 통계 등의 기능
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime, timedelta

from app.database.database import get_db
from app.services.statistics_service import StatisticsService
from app.models.schemas import StatisticsResponse, TrendResponse

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

@router.get("/{user_id}", response_model=StatisticsResponse)
async def get_user_statistics(
    user_id: int,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    사용자 통계 조회 API
    
    Args:
        user_id: 사용자 ID
        token: 액세스 토큰
        db: 데이터베이스 세션
        
    Returns:
        StatisticsResponse: 사용자 통계 정보
    """
    try:
        statistics_service = StatisticsService(db)
        stats = await statistics_service.get_user_statistics(user_id, token)
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"통계 조회 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/{user_id}/trend", response_model=List[TrendResponse])
async def get_posture_trend(
    user_id: int,
    days: int = 30,  # 기본 30일
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    자세 변화 추이 조회 API
    
    Args:
        user_id: 사용자 ID
        days: 조회할 일수 (기본 30일)
        token: 액세스 토큰
        db: 데이터베이스 세션
        
    Returns:
        List[TrendResponse]: 자세 변화 추이 데이터
    """
    try:
        statistics_service = StatisticsService(db)
        trend_data = await statistics_service.get_posture_trend(user_id, days, token)
        return trend_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"추이 데이터 조회 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/{user_id}/sessions")
async def get_session_statistics(
    user_id: int,
    start_date: datetime = None,
    end_date: datetime = None,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    세션 통계 조회 API
    
    Args:
        user_id: 사용자 ID
        start_date: 시작 날짜
        end_date: 종료 날짜
        token: 액세스 토큰
        db: 데이터베이스 세션
        
    Returns:
        Dict: 세션 통계 정보
    """
    try:
        statistics_service = StatisticsService(db)
        
        # 날짜 범위 설정
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
            
        session_stats = await statistics_service.get_session_statistics(
            user_id, start_date, end_date, token
        )
        return session_stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"세션 통계 조회 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/{user_id}/improvements")
async def get_improvement_metrics(
    user_id: int,
    period: str = "month",  # week, month, year
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    자세 개선 지표 조회 API
    
    Args:
        user_id: 사용자 ID
        period: 조회 기간 (week, month, year)
        token: 액세스 토큰
        db: 데이터베이스 세션
        
    Returns:
        Dict: 자세 개선 지표
    """
    try:
        statistics_service = StatisticsService(db)
        improvements = await statistics_service.get_improvement_metrics(
            user_id, period, token
        )
        return improvements
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"개선 지표 조회 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/{user_id}/alerts")
async def get_alert_statistics(
    user_id: int,
    alert_type: str = None,  # neck, shoulder, spine
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    알림 통계 조회 API
    
    Args:
        user_id: 사용자 ID
        alert_type: 알림 타입 필터
        token: 액세스 토큰
        db: 데이터베이스 세션
        
    Returns:
        Dict: 알림 통계 정보
    """
    try:
        statistics_service = StatisticsService(db)
        alert_stats = await statistics_service.get_alert_statistics(
            user_id, alert_type, token
        )
        return alert_stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"알림 통계 조회 중 오류가 발생했습니다: {str(e)}"
        ) 