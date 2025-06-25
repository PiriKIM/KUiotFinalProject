"""
데이터베이스 연결 및 설정 모듈

SQLAlchemy를 사용한 데이터베이스 연결 및 세션 관리
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 데이터베이스 URL 설정
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///./jaseyojeong.db"
)

# Alembic용 변수명
SQLALCHEMY_DATABASE_URL = DATABASE_URL

# SQLite를 위한 엔진 설정
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=True  # 개발 환경에서 SQL 쿼리 로그 출력
    )
else:
    # PostgreSQL을 위한 엔진 설정
    engine = create_engine(
        DATABASE_URL,
        echo=True
    )

# 세션 팩토리 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 베이스 클래스 생성
Base = declarative_base()

def get_db():
    """
    데이터베이스 세션 의존성 함수
    
    Yields:
        Session: 데이터베이스 세션
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 