from sqlalchemy.orm import Session
from app.models.schemas import BodyAnalysisRequest, BodyAnalysisResponse
from datetime import datetime
from typing import Dict, Any, Optional, List
import cv2
import mediapipe as mp
import numpy as np
import base64
import logging
import uuid
from app.services.posture_service import PostureAnalysisService

logger = logging.getLogger(__name__)

class AnalyzeService:
    def __init__(self, db: Optional[Session] = None):
        self.db = db
        self.posture_analyzer = PostureAnalysisService()
        self.active_sessions = {}  # session_id -> session_data
        
        logger.info("AnalyzeService 초기화 완료")
    
    async def analyze_body_posture(self, request: BodyAnalysisRequest, token: str):
        """기존 체형 분석 (단일 이미지)"""
        # TODO: 실제 AI 분석 로직 구현
        return BodyAnalysisResponse(
            id=1,
            user_id=request.user_id,
            analysis_type=request.analysis_type,
            image_path=None,
            shoulder_angle=0.0,
            spine_angle=0.0,
            pelvis_angle=0.0,
            neck_angle=0.0,
            posture_score=85.0,
            analysis_date=datetime.now(),
            notes=None,
            recommendations=[]
        )
    
    def get_analysis_history(self, user_id: int, token: str):
        """분석 이력 조회"""
        # TODO: 실제 데이터베이스 조회 로직 구현
        return []
    
    async def start_body_analysis_session(self, user_id: int) -> Dict[str, Any]:
        """
        체형 분석 세션 시작
        
        Args:
            user_id: 사용자 ID
            
        Returns:
            Dict: 세션 정보
        """
        session_id = str(uuid.uuid4())
        
        session_data = {
            "user_id": user_id,
            "start_time": datetime.now(),
            "frames_analyzed": 0,
            "total_score": 0,
            "measurements_history": [],
            "angles_history": [],
            "is_active": True
        }
        
        self.active_sessions[session_id] = session_data
        
        logger.info(f"체형 분석 세션 시작: {session_id} (사용자: {user_id})")
        
        return {
            "session_id": session_id,
            "message": "체형 분석 세션이 시작되었습니다."
        }
    
    async def analyze_realtime_body(self, frame_data: Dict[str, Any], user_id: int, session_id: str) -> Dict[str, Any]:
        """
        실시간 체형 분석
        
        Args:
            frame_data: 웹캠 프레임 데이터 (base64 이미지)
            user_id: 사용자 ID
            session_id: 세션 ID
            
        Returns:
            Dict: 체형 분석 결과
        """
        if session_id not in self.active_sessions:
            return {"error": "유효하지 않은 세션입니다."}
        
        session_data = self.active_sessions[session_id]
        if not session_data["is_active"]:
            return {"error": "세션이 종료되었습니다."}
        
        try:
            # base64 이미지 디코딩
            image_data = frame_data.get('image')
            if not image_data:
                return {"error": "이미지 데이터가 없습니다."}
            
            # base64 문자열을 바이너리로 변환
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"error": "이미지를 읽을 수 없습니다."}
            
            # MediaPipe 체형 분석 수행
            posture_analysis = self.posture_analyzer.analyze_posture(image)
            
            # 세션 데이터 업데이트
            session_data["frames_analyzed"] += 1
            session_data["total_score"] += posture_analysis.posture_score
            
            # 히스토리 저장
            session_data["measurements_history"].append(
                self._calculate_body_measurements(posture_analysis.landmarks or [], image.shape)
            )
            session_data["angles_history"].append({
                "neck_angle": posture_analysis.neck_angle,
                "shoulder_angle": posture_analysis.shoulder_angle,
                "back_angle": posture_analysis.back_angle
            })
            
            # 평균 점수 계산
            avg_score = session_data["total_score"] / session_data["frames_analyzed"]
            
            # 실시간 피드백 생성
            feedback = self._generate_body_feedback(posture_analysis, session_data["measurements_history"][-1])
            
            return {
                "session_id": session_id,
                "current_score": posture_analysis.posture_score,
                "average_score": round(avg_score, 1),
                "frames_analyzed": session_data["frames_analyzed"],
                "body_measurements": session_data["measurements_history"][-1],
                "angles": {
                    "neck_angle": posture_analysis.neck_angle,
                    "shoulder_angle": posture_analysis.shoulder_angle,
                    "spine_angle": posture_analysis.back_angle,
                    "pelvis_angle": self._calculate_pelvis_angle(posture_analysis.landmarks or [])
                },
                "recommendations": posture_analysis.recommendations,
                "feedback": feedback,
                "is_good_posture": posture_analysis.posture_score >= 80
            }
            
        except Exception as e:
            logger.error(f"실시간 체형 분석 오류: {str(e)}")
            return {"error": f"분석 중 오류가 발생했습니다: {str(e)}"}
    
    async def end_body_analysis_session(self, session_id: str) -> Dict[str, Any]:
        """
        체형 분석 세션 종료
        
        Args:
            session_id: 세션 ID
            
        Returns:
            Dict: 세션 요약 정보
        """
        if session_id not in self.active_sessions:
            return {"error": "유효하지 않은 세션입니다."}
        
        session_data = self.active_sessions[session_id]
        session_data["is_active"] = False
        session_data["end_time"] = datetime.now()
        
        # 세션 요약 계산
        duration = session_data["end_time"] - session_data["start_time"]
        avg_score = session_data["total_score"] / session_data["frames_analyzed"] if session_data["frames_analyzed"] > 0 else 0
        
        # 개선도 계산
        improvement = self._calculate_improvement(session_data)
        
        summary = {
            "session_id": session_id,
            "duration_minutes": round(duration.total_seconds() / 60, 1),
            "frames_analyzed": session_data["frames_analyzed"],
            "average_score": round(avg_score, 1),
            "improvement": improvement,
            "final_measurements": session_data["measurements_history"][-1] if session_data["measurements_history"] else {}
        }
        
        logger.info(f"체형 분석 세션 종료: {session_id}, 평균 점수: {avg_score}")
        
        return summary
    
    def _calculate_body_measurements(self, landmarks: List[tuple], image_shape: tuple) -> Dict[str, float]:
        """
        체형 측정값 계산
        
        Args:
            landmarks: 랜드마크 좌표
            image_shape: 이미지 크기
            
        Returns:
            Dict: 체형 측정값
        """
        if not landmarks or len(landmarks) < 33:  # MediaPipe Pose는 33개 랜드마크
            return {}
        
        height, width = image_shape[:2]
        
        # 랜드마크를 픽셀 좌표로 변환
        points = [(int(x * width), int(y * height)) for x, y in landmarks]
        
        measurements = {}
        
        # 어깨 폭 계산 (11번: left_shoulder, 12번: right_shoulder)
        if len(points) > 12:
            shoulder_width = self._calculate_distance(points[11], points[12])
            measurements["shoulder_width"] = shoulder_width
        
        # 골반 폭 계산 (23번: left_hip, 24번: right_hip)
        if len(points) > 24:
            hip_width = self._calculate_distance(points[23], points[24])
            measurements["hip_width"] = hip_width
        
        # 신체 높이 계산 (0번: nose에서 31번: left_foot_index 또는 32번: right_foot_index)
        if len(points) > 32:
            body_height = self._calculate_distance(points[0], points[31])  # 코에서 발끝까지
            measurements["body_height"] = body_height
        
        return measurements

    def _calculate_distance(self, point1: tuple, point2: tuple) -> float:
        """
        두 점 사이의 거리 계산
        
        Args:
            point1: 첫 번째 점 (x, y)
            point2: 두 번째 점 (x, y)
            
        Returns:
            float: 거리 (픽셀)
        """
        import math
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    def _calculate_pelvis_angle(self, landmarks: List[tuple]) -> float:
        """
        골반 각도 계산
        
        Args:
            landmarks: 랜드마크 좌표
            
        Returns:
            float: 골반 각도 (도)
        """
        if not landmarks or len(landmarks) < 24:
            return 0.0
        
        # 골반 기울기 계산 (23번: left_hip, 24번: right_hip)
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # 수평선과의 각도 계산
        import math
        dx = right_hip[0] - left_hip[0]
        dy = right_hip[1] - left_hip[1]
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        
        return angle

    def _generate_body_feedback(self, posture_analysis, body_measurements: Dict[str, float]) -> str:
        """
        체형 피드백 생성
        
        Args:
            posture_analysis: 자세 분석 결과
            body_measurements: 체형 측정값
            
        Returns:
            str: 피드백 메시지
        """
        score = posture_analysis.posture_score
        
        if score >= 90:
            return "완벽한 체형과 자세입니다! 👍"
        elif score >= 80:
            return "좋은 체형입니다. 자세를 조금만 더 개선해보세요."
        elif score >= 70:
            return "체형 개선이 필요합니다. 꾸준한 운동을 권장합니다."
        else:
            # 가장 문제가 되는 부위 찾기
            angles = {
                "목": posture_analysis.neck_angle,
                "어깨": posture_analysis.shoulder_angle,
                "등": posture_analysis.back_angle
            }
            
            worst_part = max(angles.items(), key=lambda x: x[1])[0]
            return f"{worst_part} 자세를 개선해주세요!"

    def _calculate_improvement(self, session_data: Dict[str, Any]) -> float:
        """
        세션 동안의 개선도 계산
        
        Args:
            session_data: 세션 데이터
            
        Returns:
            float: 개선도 (%)
        """
        if session_data["frames_analyzed"] < 10:
            return 0.0
        
        # 처음 10프레임과 마지막 10프레임의 평균 비교
        angles_history = session_data["angles_history"]
        
        if len(angles_history) < 20:
            return 0.0
        
        # 처음 10프레임의 평균
        early_avg = sum(
            (a["neck_angle"] + a["shoulder_angle"] + a["back_angle"]) / 3
            for a in angles_history[:10]
        ) / 10
        
        # 마지막 10프레임의 평균
        late_avg = sum(
            (a["neck_angle"] + a["shoulder_angle"] + a["back_angle"]) / 3
            for a in angles_history[-10:]
        ) / 10
        
        # 개선도 계산 (각도가 줄어들면 개선)
        improvement = ((early_avg - late_avg) / early_avg) * 100
        
        return max(0, min(100, improvement))  # 0-100% 범위로 제한
    
    def __del__(self):
        """리소스 정리"""
        pass
