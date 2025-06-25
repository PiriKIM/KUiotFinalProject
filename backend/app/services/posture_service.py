from sqlalchemy.orm import Session
from app.models.schemas import PostureAnalysisResponse
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import cv2
import mediapipe as mp
import numpy as np
import logging
from dataclasses import dataclass
import base64
import uuid

logger = logging.getLogger(__name__)

@dataclass
class PostureAnalysis:
    """자세 분석 결과 데이터 클래스"""
    posture_score: float
    neck_angle: float
    shoulder_angle: float
    back_angle: float
    recommendations: List[str]
    landmarks: Optional[List[Tuple[float, float]]] = None

class PostureAnalysisService:
    """MediaPipe를 사용한 자세 분석 서비스"""
    
    def __init__(self):
        """MediaPipe 초기화"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        
        # 자세 기준값 (도)
        self.POSTURE_THRESHOLDS = {
            'neck': {'good': 10, 'warning': 20, 'bad': 30},
            'shoulder': {'good': 5, 'warning': 10, 'bad': 15},
            'back': {'good': 5, 'warning': 10, 'bad': 15}
        }
        
        logger.info("PostureAnalysisService 초기화 완료")
    
    def analyze_posture(self, image: np.ndarray) -> PostureAnalysis:
        """
        이미지에서 자세 분석 수행
        
        Args:
            image: 분석할 이미지 (numpy array)
            
        Returns:
            PostureAnalysis: 자세 분석 결과
        """
        try:
            # BGR to RGB 변환
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # MediaPipe 포즈 감지
            results = self.pose.process(rgb_image)
            
            if not results.pose_landmarks:
                logger.warning("포즈 랜드마크를 감지할 수 없습니다.")
                return self._create_default_analysis()
            
            # 각도 계산
            angles = self._calculate_angles(results.pose_landmarks, image.shape)
            
            # 자세 점수 계산
            posture_score = self._calculate_posture_score(angles)
            
            # 권장사항 생성
            recommendations = self._generate_recommendations(angles)
            
            # 랜드마크 좌표 추출
            landmarks = self._extract_landmarks(results.pose_landmarks)
            
            return PostureAnalysis(
                posture_score=posture_score,
                neck_angle=angles['neck'],
                shoulder_angle=angles['shoulder'],
                back_angle=angles['back'],
                recommendations=recommendations,
                landmarks=landmarks
            )
            
        except Exception as e:
            logger.error(f"자세 분석 중 오류 발생: {str(e)}")
            return self._create_default_analysis()
    
    def _calculate_angles(self, landmarks, image_shape: Tuple[int, ...]) -> Dict[str, float]:
        """
        신체 부위별 각도 계산
        
        Args:
            landmarks: MediaPipe 포즈 랜드마크
            image_shape: 이미지 크기 (height, width, channels)
            
        Returns:
            Dict[str, float]: 부위별 각도
        """
        height, width = image_shape[:2]
        
        # 랜드마크 좌표 추출
        points = {}
        for i, landmark in enumerate(landmarks.landmark):
            points[i] = (landmark.x * width, landmark.y * height)
        
        angles = {}
        
        # 목 각도 계산 (귀-어깨-어깨 중점)
        if all(k in points for k in [2, 11, 12]):  # left_ear, left_shoulder, right_shoulder
            neck_angle = self._calculate_angle(
                points[2],  # left_ear
                points[11],  # left_shoulder
                (points[11][0] + points[12][0]) / 2,  # shoulder midpoint
                (points[11][1] + points[12][1]) / 2
            )
            angles['neck'] = abs(neck_angle)
        
        # 어깨 각도 계산 (어깨-어깨-골반)
        if all(k in points for k in [11, 12, 23, 24]):  # shoulders, hips
            shoulder_angle = self._calculate_angle(
                points[11],  # left_shoulder
                points[12],  # right_shoulder
                points[23],  # left_hip
                points[24]   # right_hip
            )
            angles['shoulder'] = abs(shoulder_angle)
        
        # 등 각도 계산 (어깨-골반-무릎)
        if all(k in points for k in [11, 23, 25]):  # left_shoulder, left_hip, left_knee
            back_angle = self._calculate_angle(
                points[11],  # left_shoulder
                points[23],  # left_hip
                points[25],  # left_knee
                points[25]   # vertical reference
            )
            angles['back'] = abs(back_angle)
        
        return angles
    
    def _calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                        p3: Tuple[float, float], p4: Tuple[float, float]) -> float:
        """
        세 점으로 이루어진 각도 계산
        
        Args:
            p1, p2, p3, p4: 좌표점들
            
        Returns:
            float: 각도 (도)
        """
        # 벡터 계산
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # 각도 계산
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def _calculate_posture_score(self, angles: Dict[str, float]) -> float:
        """
        자세 점수 계산 (0-100점)
        
        Args:
            angles: 부위별 각도
            
        Returns:
            float: 자세 점수
        """
        scores = {}
        
        for part, angle in angles.items():
            thresholds = self.POSTURE_THRESHOLDS[part]
            
            if angle <= thresholds['good']:
                scores[part] = 100
            elif angle <= thresholds['warning']:
                scores[part] = 80 - (angle - thresholds['good']) * 4
            elif angle <= thresholds['bad']:
                scores[part] = 60 - (angle - thresholds['warning']) * 4
            else:
                scores[part] = max(0, 40 - (angle - thresholds['bad']) * 2)
        
        # 가중 평균 계산 (목: 40%, 어깨: 30%, 등: 30%)
        weights = {'neck': 0.4, 'shoulder': 0.3, 'back': 0.3}
        total_score = sum(scores.get(part, 0) * weights[part] for part in weights)
        
        return round(total_score, 1)
    
    def _generate_recommendations(self, angles: Dict[str, float]) -> List[str]:
        """
        자세 개선 권장사항 생성
        
        Args:
            angles: 부위별 각도
            
        Returns:
            List[str]: 권장사항 목록
        """
        recommendations = []
        
        for part, angle in angles.items():
            thresholds = self.POSTURE_THRESHOLDS[part]
            
            if angle > thresholds['bad']:
                if part == 'neck':
                    recommendations.append("목을 더 세우세요. 턱을 가슴에 붙이지 마세요.")
                elif part == 'shoulder':
                    recommendations.append("어깨를 펴고 균형을 맞추세요.")
                elif part == 'back':
                    recommendations.append("등을 곧게 펴고 허리를 세우세요.")
            elif angle > thresholds['warning']:
                if part == 'neck':
                    recommendations.append("목을 조금 더 세우세요.")
                elif part == 'shoulder':
                    recommendations.append("어깨를 조금 더 펴세요.")
                elif part == 'back':
                    recommendations.append("등을 조금 더 곧게 펴세요.")
        
        if not recommendations:
            recommendations.append("좋은 자세입니다! 계속 유지하세요.")
        
        return recommendations
    
    def _extract_landmarks(self, landmarks) -> List[Tuple[float, float]]:
        """
        랜드마크 좌표 추출
        
        Args:
            landmarks: MediaPipe 포즈 랜드마크
            
        Returns:
            List[Tuple[float, float]]: 랜드마크 좌표 목록
        """
        return [(landmark.x, landmark.y) for landmark in landmarks.landmark]
    
    def _create_default_analysis(self) -> PostureAnalysis:
        """기본 분석 결과 생성"""
        return PostureAnalysis(
            posture_score=0.0,
            neck_angle=0.0,
            shoulder_angle=0.0,
            back_angle=0.0,
            recommendations=["포즈를 감지할 수 없습니다. 전체 몸이 보이도록 촬영해주세요."]
        )
    
    def draw_analysis_on_image(self, image: np.ndarray, analysis: PostureAnalysis) -> np.ndarray:
        """
        분석 결과를 이미지에 그리기
        
        Args:
            image: 원본 이미지
            analysis: 분석 결과
            
        Returns:
            np.ndarray: 분석 결과가 그려진 이미지
        """
        # 이미지 복사
        result_image = image.copy()
        
        # 텍스트 정보 추가
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (255, 255, 255)
        thickness = 2
        
        # 자세 점수
        cv2.putText(result_image, f"자세 점수: {analysis.posture_score}", 
                   (10, 30), font, font_scale, color, thickness)
        
        # 각도 정보
        cv2.putText(result_image, f"목 각도: {analysis.neck_angle:.1f}°", 
                   (10, 60), font, font_scale, color, thickness)
        cv2.putText(result_image, f"어깨 각도: {analysis.shoulder_angle:.1f}°", 
                   (10, 90), font, font_scale, color, thickness)
        cv2.putText(result_image, f"등 각도: {analysis.back_angle:.1f}°", 
                   (10, 120), font, font_scale, color, thickness)
        
        return result_image
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'pose'):
            self.pose.close()

class PostureService:
    def __init__(self, db: Optional[Session] = None):
        self.db = db
        self.posture_analyzer = PostureAnalysisService()
        self.active_sessions = {}  # session_id -> session_data

    async def analyze_single_frame(self, posture_data: Dict[str, Any], token: str):
        """
        단일 프레임 자세 분석
        
        Args:
            posture_data: 분석할 자세 데이터
            token: 사용자 토큰
            
        Returns:
            Dict: 분석 결과
        """
        try:
            # 이미지 데이터 추출 (base64 또는 바이너리)
            image_data = posture_data.get('image')
            if not image_data:
                return {"error": "이미지 데이터가 없습니다."}
            
            # base64 디코딩
            if isinstance(image_data, str):
                # base64 문자열을 바이너리로 변환
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                # 바이너리 데이터
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"error": "이미지를 읽을 수 없습니다."}
            
            # 자세 분석 수행
            analysis = self.posture_analyzer.analyze_posture(image)
            
            # 분석 결과를 이미지에 그리기
            result_image = self.posture_analyzer.draw_analysis_on_image(image, analysis)
            
            # 결과 이미지를 base64로 인코딩
            _, buffer = cv2.imencode('.jpg', result_image)
            result_image_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            
            return {
                "posture_score": analysis.posture_score,
                "body_angles": {
                    "neck_angle": analysis.neck_angle,
                    "shoulder_angle": analysis.shoulder_angle,
                    "back_angle": analysis.back_angle
                },
                "recommendations": analysis.recommendations,
                "result_image": result_image_base64,
                "landmarks": analysis.landmarks
            }
            
        except Exception as e:
            logger.error(f"자세 분석 중 오류: {str(e)}")
            return {"error": f"분석 중 오류가 발생했습니다: {str(e)}"}

    async def start_posture_session(self, user_id: int):
        """
        자세 교정 세션 시작
        
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
            "is_active": True
        }
        
        self.active_sessions[session_id] = session_data
        
        logger.info(f"자세 교정 세션 시작: {session_id} (사용자: {user_id})")
        
        return {
            "session_id": session_id,
            "message": "자세 교정 세션이 시작되었습니다."
        }

    async def analyze_realtime_posture(self, posture_data: Dict[str, Any], user_id: int, session_id: str):
        """
        실시간 자세 분석
        
        Args:
            posture_data: 분석할 자세 데이터
            user_id: 사용자 ID
            session_id: 세션 ID (str)
            
        Returns:
            Dict: 실시간 분석 결과
        """
        if session_id not in self.active_sessions:
            return {"error": "유효하지 않은 세션입니다."}
        
        session_data = self.active_sessions[session_id]
        if not session_data["is_active"]:
            return {"error": "세션이 종료되었습니다."}
        
        # 단일 프레임 분석 수행
        analysis_result = await self.analyze_single_frame(posture_data, "")
        
        if "error" in analysis_result:
            return analysis_result
        
        # 세션 데이터 업데이트
        session_data["frames_analyzed"] += 1
        session_data["total_score"] += analysis_result["posture_score"]
        
        # 평균 점수 계산
        avg_score = session_data["total_score"] / session_data["frames_analyzed"]
        
        # 실시간 피드백 생성
        feedback = self._generate_realtime_feedback(analysis_result)
        
        return {
            "session_id": session_id,
            "current_score": analysis_result["posture_score"],
            "average_score": round(avg_score, 1),
            "frames_analyzed": session_data["frames_analyzed"],
            "feedback": feedback,
            "body_angles": analysis_result["body_angles"],
            "recommendations": analysis_result["recommendations"],
            "is_good_posture": analysis_result["posture_score"] >= 80
        }

    async def end_posture_session(self, session_id: str):
        """
        자세 교정 세션 종료
        
        Args:
            session_id: 세션 ID (str)
            
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
        
        summary = {
            "session_id": session_id,
            "duration_minutes": round(duration.total_seconds() / 60, 1),
            "frames_analyzed": session_data["frames_analyzed"],
            "average_score": round(avg_score, 1),
            "best_score": session_data.get("best_score", 0),
            "improvement": self._calculate_improvement(session_data)
        }
        
        logger.info(f"자세 교정 세션 종료: {session_id}, 평균 점수: {avg_score}")
        
        return summary

    def get_session_data(self, session_id: str, token: str):
        """
        세션 데이터 조회
        
        Args:
            session_id: 세션 ID (str)
            token: 사용자 토큰
            
        Returns:
            Dict: 세션 데이터
        """
        if session_id not in self.active_sessions:
            return {"error": "세션을 찾을 수 없습니다."}
        
        return self.active_sessions[session_id]

    def _generate_realtime_feedback(self, analysis_result: Dict) -> str:
        """
        실시간 피드백 생성
        
        Args:
            analysis_result: 분석 결과
            
        Returns:
            str: 피드백 메시지
        """
        score = analysis_result.get("posture_score", 0)
        angles = analysis_result.get("body_angles", {})
        
        if score >= 90:
            return "완벽한 자세입니다! 👍"
        elif score >= 80:
            return "좋은 자세입니다! 조금만 더 노력하세요."
        elif score >= 70:
            return "자세를 조금 더 개선해보세요."
        else:
            # 가장 문제가 되는 부위 찾기
            if angles:
                worst_part = max(angles.items(), key=lambda x: x[1])[0]
                if worst_part == "neck_angle":
                    return "목을 더 세우세요!"
                elif worst_part == "shoulder_angle":
                    return "어깨를 펴세요!"
                else:
                    return "등을 곧게 펴세요!"
            else:
                return "자세를 개선해보세요."

    def _calculate_improvement(self, session_data: Dict) -> float:
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
        # 실제 구현에서는 더 정교한 계산이 필요
        return 5.0  # 임시 값
