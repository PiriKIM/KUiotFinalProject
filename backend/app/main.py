"""
자세요정 FastAPI 애플리케이션 메인 모듈

AI 기반 체형 분석 및 자세 교정 시스템의 API 서버
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.routers import auth, analyze, statistics
from app.database.database import engine
from app.models import models

# 데이터베이스 테이블 생성
models.Base.metadata.create_all(bind=engine)

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="자세요정 API",
    description="AI 기반 체형 분석 및 자세 교정 시스템 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # 프론트엔드 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 (업로드된 이미지 등)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 라우터 등록
app.include_router(auth.router, prefix="/auth", tags=["인증"])
app.include_router(analyze.router, prefix="/analyze", tags=["분석"])
app.include_router(statistics.router, prefix="/statistics", tags=["통계"])

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "자세요정 API 서버에 오신 것을 환영합니다!",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "service": "자세요정 API"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 