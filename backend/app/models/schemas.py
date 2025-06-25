"""
Pydantic 스키마 정의

API 요청/응답을 위한 Pydantic 모델들
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# 인증 관련 스키마
class UserCreate(BaseModel):
    """사용자 등록 요청 스키마"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    age: Optional[int] = Field(None, ge=1, le=120)
    height: Optional[float] = Field(None, ge=50, le=250)  # cm
    weight: Optional[float] = Field(None, ge=20, le=300)  # kg

class UserLogin(BaseModel):
    """사용자 로그인 요청 스키마"""
    email: EmailStr
    password: str

class Token(BaseModel):
    """토큰 응답 스키마"""
    access_token: str
    token_type: str

class UserResponse(BaseModel):
    """사용자 정보 응답 스키마"""
    id: int
    email: str
    username: str
    full_name: Optional[str]
    age: Optional[int]
    height: Optional[float]
    weight: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True

# 체형 분석 관련 스키마
class BodyAnalysisRequest(BaseModel):
    """체형 분석 요청 스키마"""
    analysis_type: str = Field(..., pattern="^(front|side)$")  # front 또는 side
    image_data: str  # base64 인코딩된 이미지 데이터
    user_id: int

class BodyAnalysisResponse(BaseModel):
    """체형 분석 응답 스키마"""
    id: int
    user_id: int
    analysis_type: str
    image_path: Optional[str]
    
    # 분석 결과
    shoulder_angle: Optional[float]
    spine_angle: Optional[float]
    pelvis_angle: Optional[float]
    neck_angle: Optional[float]
    posture_score: float = Field(..., ge=0, le=100)
    
    # 메타데이터
    analysis_date: datetime
    notes: Optional[str]
    
    # 개선 권장사항
    recommendations: List[str] = []
    
    class Config:
        from_attributes = True

# 자세 분석 관련 스키마
class PostureData(BaseModel):
    """자세 데이터 스키마"""
    ear_x: float
    ear_y: float
    jaw_x: float
    jaw_y: float
    neck_x: float
    neck_y: float
    shoulder_x: float
    shoulder_y: float
    pelvis_x: float
    pelvis_y: float

class PostureAnalysisResponse(BaseModel):
    """자세 분석 응답 스키마"""
    neck_angle: float
    shoulder_angle: float
    spine_angle: float
    posture_score: float = Field(..., ge=0, le=100)
    posture_grade: str = Field(..., pattern="^[ABCD]$")  # A, B, C, D 등급
    
    # 알림 정보
    alert_level: Optional[str] = None  # warning, critical
    alert_message: Optional[str] = None
    
    # 시각적 피드백
    feedback: List[str] = []
    
    # 실시간 데이터
    timestamp: datetime

# 통계 관련 스키마
class StatisticsResponse(BaseModel):
    """통계 응답 스키마"""
    user_id: int
    
    # 전체 통계
    total_sessions: int
    total_analysis_count: int
    avg_posture_score: float
    improvement_rate: float  # 개선률 (%)
    
    # 기간별 통계
    weekly_stats: Dict[str, Any]
    monthly_stats: Dict[str, Any]
    
    # 알림 통계
    total_alerts: int
    alert_by_type: Dict[str, int]
    
    # 최근 활동
    last_session_date: Optional[datetime]
    last_analysis_date: Optional[datetime]

class TrendResponse(BaseModel):
    """추이 데이터 응답 스키마"""
    date: datetime
    avg_posture_score: float
    session_count: int
    alert_count: int
    improvement_indicator: float  # 개선 지표

# WebSocket 관련 스키마
class WebSocketMessage(BaseModel):
    """WebSocket 메시지 스키마"""
    type: str  # posture_data, alert, error
    data: Dict[str, Any]
    timestamp: datetime

class PostureAlert(BaseModel):
    """자세 알림 스키마"""
    alert_type: str  # neck, shoulder, spine
    alert_level: str  # warning, critical
    message: str
    timestamp: datetime

# 세션 관련 스키마
class SessionData(BaseModel):
    """세션 데이터 스키마"""
    session_id: int
    user_id: int
    session_start: datetime
    session_end: Optional[datetime]
    duration_minutes: Optional[int]
    
    # 세션 통계
    avg_posture_score: float
    good_posture_time: int  # 초
    bad_posture_time: int   # 초
    alert_count: int
    
    class Config:
        from_attributes = True

# 설정 관련 스키마
class UserSettings(BaseModel):
    """사용자 설정 스키마"""
    user_id: int
    alert_enabled: bool = True
    alert_threshold: int = Field(30, ge=10, le=60)  # 알림 임계값 (초)
    tts_enabled: bool = True
    auto_session_start: bool = False
    
    class Config:
        from_attributes = True 