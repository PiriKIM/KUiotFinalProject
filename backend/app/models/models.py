"""
데이터베이스 모델 정의

SQLAlchemy ORM을 사용한 데이터베이스 테이블 모델
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.database import Base

class User(Base):
    """사용자 테이블 모델"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(200))
    age = Column(Integer)
    height = Column(Float)  # cm
    weight = Column(Float)  # kg
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 관계 설정
    body_analyses = relationship("BodyAnalysis", back_populates="user")
    posture_sessions = relationship("PostureSession", back_populates="user")
    posture_alerts = relationship("PostureAlert", back_populates="user")

class BodyAnalysis(Base):
    """체형 분석 결과 테이블 모델"""
    __tablename__ = "body_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    analysis_type = Column(String(50), nullable=False)  # "front", "side"
    image_path = Column(String(500))  # 분석된 이미지 경로
    
    # 체형 분석 결과
    shoulder_angle = Column(Float)  # 어깨 각도
    spine_angle = Column(Float)     # 척추 각도
    pelvis_angle = Column(Float)    # 골반 각도
    neck_angle = Column(Float)      # 목 각도
    posture_score = Column(Float)   # 자세 점수 (0-100)
    
    # 분석 메타데이터
    analysis_date = Column(DateTime(timezone=True), server_default=func.now())
    notes = Column(Text)  # 추가 메모
    
    # 관계 설정
    user = relationship("User", back_populates="body_analyses")

class PostureSession(Base):
    """자세 교정 세션 테이블 모델"""
    __tablename__ = "posture_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_start = Column(DateTime(timezone=True), server_default=func.now())
    session_end = Column(DateTime(timezone=True))
    duration_minutes = Column(Integer)  # 세션 지속 시간 (분)
    
    # 세션 통계
    avg_posture_score = Column(Float)  # 평균 자세 점수
    good_posture_time = Column(Integer)  # 좋은 자세 유지 시간 (초)
    bad_posture_time = Column(Integer)   # 나쁜 자세 유지 시간 (초)
    alert_count = Column(Integer, default=0)  # 알림 횟수
    
    # 관계 설정
    user = relationship("User", back_populates="posture_sessions")

class PostureAlert(Base):
    """자세 알림 기록 테이블 모델"""
    __tablename__ = "posture_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("posture_sessions.id"))
    alert_type = Column(String(50), nullable=False)  # "neck", "shoulder", "spine"
    alert_level = Column(String(20), nullable=False)  # "warning", "critical"
    message = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 관계 설정
    user = relationship("User", back_populates="posture_alerts")

class PostureData(Base):
    """실시간 자세 데이터 테이블 모델"""
    __tablename__ = "posture_data"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("posture_sessions.id"))
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # 자세 좌표 데이터
    ear_x = Column(Float)
    ear_y = Column(Float)
    jaw_x = Column(Float)
    jaw_y = Column(Float)
    neck_x = Column(Float)
    neck_y = Column(Float)
    shoulder_x = Column(Float)
    shoulder_y = Column(Float)
    pelvis_x = Column(Float)
    pelvis_y = Column(Float)
    
    # 계산된 각도
    neck_angle = Column(Float)
    shoulder_angle = Column(Float)
    spine_angle = Column(Float)
    posture_score = Column(Float)
    posture_grade = Column(String(5))  # A, B, C, D 