"""
인증 관련 API 라우터

사용자 등록, 로그인, 토큰 갱신 등의 인증 기능
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional

from app.database.database import get_db
from app.models import models
from app.services.auth_service import AuthService
from app.models.schemas import UserCreate, UserLogin, Token, UserResponse

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    사용자 등록 API
    
    Args:
        user_data: 사용자 등록 데이터
        db: 데이터베이스 세션
        
    Returns:
        UserResponse: 등록된 사용자 정보
        
    Raises:
        HTTPException: 이메일 또는 사용자명이 이미 존재하는 경우
    """
    auth_service = AuthService(db)
    
    # 이메일 중복 확인
    if auth_service.get_user_by_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 등록된 이메일입니다."
        )
    
    # 사용자명 중복 확인
    if auth_service.get_user_by_username(user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 사용 중인 사용자명입니다."
        )
    
    # 사용자 생성
    user = auth_service.create_user(user_data)
    return UserResponse.from_orm(user)

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    사용자 로그인 API
    
    Args:
        form_data: 로그인 폼 데이터
        db: 데이터베이스 세션
        
    Returns:
        Token: 액세스 토큰과 토큰 타입
        
    Raises:
        HTTPException: 잘못된 이메일 또는 비밀번호
    """
    auth_service = AuthService(db)
    
    # 사용자 인증
    user = auth_service.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="잘못된 이메일 또는 비밀번호입니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # 액세스 토큰 생성
    access_token = auth_service.create_access_token(data={"sub": user.email})
    
    return Token(
        access_token=access_token,
        token_type="bearer"
    )

@router.post("/refresh", response_model=Token)
async def refresh_token(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """
    토큰 갱신 API
    
    Args:
        token: 현재 액세스 토큰
        db: 데이터베이스 세션
        
    Returns:
        Token: 새로운 액세스 토큰
        
    Raises:
        HTTPException: 유효하지 않은 토큰
    """
    auth_service = AuthService(db)
    
    # 토큰 검증 및 갱신
    try:
        new_token = auth_service.refresh_access_token(token)
        return Token(
            access_token=new_token,
            token_type="bearer"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 토큰입니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """
    현재 로그인한 사용자 정보 조회 API
    
    Args:
        token: 액세스 토큰
        db: 데이터베이스 세션
        
    Returns:
        UserResponse: 현재 사용자 정보
        
    Raises:
        HTTPException: 유효하지 않은 토큰
    """
    auth_service = AuthService(db)
    
    try:
        user = auth_service.get_current_user(token)
        return UserResponse.from_orm(user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 토큰입니다.",
            headers={"WWW-Authenticate": "Bearer"},
        ) 