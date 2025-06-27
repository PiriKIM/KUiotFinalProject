"""
체형 분석 및 자세 교정 API 라우터

체형 분석, 실시간 자세 감지, WebSocket 통신 등의 기능
"""

from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import json

from app.database.database import get_db
from app.services.analyze_service import AnalyzeService
from app.services.posture_service import PostureService
from app.models.schemas import BodyAnalysisRequest, BodyAnalysisResponse, PostureAnalysisResponse

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

@router.post("/body", response_model=BodyAnalysisResponse)
async def analyze_body(
    analysis_request: BodyAnalysisRequest,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    체형 분석 API
    
    Args:
        analysis_request: 체형 분석 요청 데이터
        token: 액세스 토큰
        db: 데이터베이스 세션
        
    Returns:
        BodyAnalysisResponse: 체형 분석 결과
        
    Raises:
        HTTPException: 분석 실패 시
    """
    try:
        print(f"체형 분석 요청 받음 - 사용자 ID: {analysis_request.user_id}")
        print(f"전면 이미지 크기: {len(analysis_request.front_image)}")
        print(f"측면 이미지 크기: {len(analysis_request.side_image)}")
        print(f"분석 타입: {analysis_request.analysis_type}")
        
        analyze_service = AnalyzeService(db)
        result = await analyze_service.analyze_body_posture(analysis_request, token)
        
        print(f"체형 분석 완료 - 전체 점수: {result.overall_score}")
        return result
    except Exception as e:
        print(f"체형 분석 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"체형 분석 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/body/{user_id}", response_model=List[BodyAnalysisResponse])
async def get_body_analysis_history(
    user_id: int,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    사용자 체형 분석 이력 조회 API
    
    Args:
        user_id: 사용자 ID
        token: 액세스 토큰
        db: 데이터베이스 세션
        
    Returns:
        List[BodyAnalysisResponse]: 체형 분석 이력 목록
    """
    try:
        analyze_service = AnalyzeService(db)
        history = analyze_service.get_analysis_history(user_id, token)
        return history
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"분석 이력 조회 중 오류가 발생했습니다: {str(e)}"
        )

@router.post("/posture", response_model=PostureAnalysisResponse)
async def analyze_posture(
    posture_data: Dict[str, Any],
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    자세 분석 API (단일 프레임)
    실제 MediaPipe/OpenCV 분석 결과를 PostureAnalysisResponse로 반환
    """
    from datetime import datetime
    try:
        posture_service = PostureService(db)
        result = await posture_service.analyze_single_frame(posture_data, token)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # 타입 보정
        score = float(result["posture_score"]) if isinstance(result["posture_score"], (int, float, str)) else 0.0
        body_angles = result.get("body_angles", {})
        if isinstance(body_angles, str):
            import json as _json
            body_angles = _json.loads(body_angles)
        recommendations = result.get("recommendations", [])
        if isinstance(recommendations, str):
            recommendations = [recommendations]

        # 등급 산정 로직 (예시)
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        else:
            grade = "D"

        # 알림 레벨/메시지
        if score < 70:
            alert_level = "critical"
            alert_message = "자세가 매우 좋지 않습니다. 즉시 교정이 필요합니다."
        elif score < 80:
            alert_level = "warning"
            alert_message = "자세 개선이 필요합니다."
        else:
            alert_level = None
            alert_message = None

        return PostureAnalysisResponse(
            neck_angle=float(body_angles.get("neck_angle", 0.0)),
            shoulder_angle=float(body_angles.get("shoulder_angle", 0.0)),
            spine_angle=float(body_angles.get("back_angle", 0.0)),
            posture_score=score,
            posture_grade=grade,
            alert_level=alert_level,
            alert_message=alert_message,
            feedback=recommendations,
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"자세 분석 중 오류가 발생했습니다: {str(e)}"
        )

@router.websocket("/ws/body-analysis/{user_id}")
async def websocket_body_analysis(websocket: WebSocket, user_id: int):
    """
    실시간 체형 분석 WebSocket
    웹캠 프레임을 실시간으로 받아 체형 분석 수행
    """
    await websocket.accept()
    try:
        analyze_service = AnalyzeService(db=None)
        session_info = await analyze_service.start_body_analysis_session(user_id)
        session_id = session_info["session_id"] if isinstance(session_info, dict) else session_info
        
        while True:
            # 클라이언트로부터 웹캠 프레임 데이터 수신
            data = await websocket.receive_text()
            frame_data = json.loads(data)
            
            # 실시간 체형 분석 수행
            analysis_result = await analyze_service.analyze_realtime_body(
                frame_data, user_id, session_id
            )
            
            # 분석 결과를 클라이언트로 전송
            await websocket.send_text(json.dumps(analysis_result))
            
    except WebSocketDisconnect:
        # WebSocket 연결 종료 시 세션 종료
        await analyze_service.end_body_analysis_session(session_id)
        print(f"사용자 {user_id}의 체형 분석 WebSocket 연결이 종료되었습니다.")
    except Exception as e:
        # 오류 발생 시 클라이언트에 오류 메시지 전송
        error_message = {"error": str(e)}
        await websocket.send_text(json.dumps(error_message))
        print(f"체형 분석 WebSocket 오류: {str(e)}")

@router.websocket("/ws/posture/{user_id}")
async def websocket_posture_monitoring(websocket: WebSocket, user_id: int):
    """
    실시간 자세 모니터링 WebSocket
    """
    await websocket.accept()
    try:
        # 자세 교정 세션 시작
        posture_service = PostureService()
        session_info = await posture_service.start_posture_session(user_id)
        session_id = session_info["session_id"] if isinstance(session_info, dict) else session_info
        while True:
            # 클라이언트로부터 자세 데이터 수신
            data = await websocket.receive_text()
            posture_data = json.loads(data)
            # 자세 분석 수행
            analysis_result = await posture_service.analyze_realtime_posture(
                posture_data, user_id, session_id
            )
            # 분석 결과를 클라이언트로 전송
            await websocket.send_text(json.dumps(analysis_result))
    except WebSocketDisconnect:
        # WebSocket 연결 종료 시 세션 종료
        await posture_service.end_posture_session(session_id)
        print(f"사용자 {user_id}의 WebSocket 연결이 종료되었습니다.")
    except Exception as e:
        # 오류 발생 시 클라이언트에 오류 메시지 전송
        error_message = {"error": str(e)}
        await websocket.send_text(json.dumps(error_message))
        print(f"WebSocket 오류: {str(e)}")

@router.get("/posture/session/{session_id}")
async def get_posture_session_data(
    session_id: str,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    자세 교정 세션 데이터 조회 API
    
    Args:
        session_id: 세션 ID
        token: 액세스 토큰
        db: 데이터베이스 세션
        
    Returns:
        Dict: 세션 데이터 및 통계
    """
    try:
        posture_service = PostureService(db)
        session_data = posture_service.get_session_data(session_id, token)
        return session_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"세션 데이터 조회 중 오류가 발생했습니다: {str(e)}"
        )

@router.websocket("/ws/test")
async def websocket_test(websocket: WebSocket):
    """
    WebSocket 연결 테스트
    """
    await websocket.accept()
    try:
        await websocket.send_text(json.dumps({"message": "WebSocket 연결 성공!"}))
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(json.dumps({"echo": data}))
    except WebSocketDisconnect:
        print("WebSocket 테스트 연결 종료") 