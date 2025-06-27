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
        """체형 분석 (전면 + 측면 이미지)"""
        try:
            logger.info(f"체형 분석 시작 - 사용자 ID: {request.user_id}")
            
            # base64 이미지 디코딩
            front_image = self._decode_base64_image(request.front_image)
            side_image = self._decode_base64_image(request.side_image)
            
            if front_image is None or side_image is None:
                raise ValueError("이미지 디코딩에 실패했습니다.")
            
            # 전면 이미지 분석
            front_analysis = self._analyze_front_image(front_image)
            
            # 측면 이미지 분석
            side_analysis = self._analyze_side_image(side_image)
            
            # 전체 점수 계산
            overall_score = (front_analysis["posture_score"] + side_analysis["posture_score"]) / 2
            
            # 전체 피드백 생성
            overall_feedback = self._generate_overall_feedback(front_analysis, side_analysis, overall_score)
            
            # 개선 제안 생성
            improvement_suggestions = self._generate_improvement_suggestions(front_analysis, side_analysis)
            
            logger.info(f"체형 분석 완료 - 전체 점수: {overall_score}")
            
            return BodyAnalysisResponse(
                front_analysis=front_analysis,
                side_analysis=side_analysis,
                overall_score=overall_score,
                overall_feedback=overall_feedback,
                improvement_suggestions=improvement_suggestions
            )
            
        except Exception as e:
            logger.error(f"체형 분석 오류: {str(e)}")
            raise Exception(f"체형 분석 중 오류가 발생했습니다: {str(e)}")
    
    def _decode_base64_image(self, base64_string: str):
        """base64 문자열을 OpenCV 이미지로 변환"""
        try:
            # data:image/jpeg;base64, 제거
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            image_bytes = base64.b64decode(base64_string)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            logger.error(f"이미지 디코딩 오류: {str(e)}")
            return None
    
    def _analyze_front_image(self, image):
        """전면 이미지 분석"""
        try:
            # MediaPipe Pose 초기화
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5
            )
            
            # BGR을 RGB로 변환
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_image)
            
            if not results.pose_landmarks:
                return {
                    "posture_score": 50.0,
                    "angles": {},
                    "recommendations": ["사람이 감지되지 않았습니다. 더 명확한 전면 사진을 촬영해주세요."],
                    "feedback": "전면 사진에서 사람을 감지할 수 없습니다.",
                    "landmarks": []
                }
            
            # 랜드마크 추출
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y))
            
            # 각도 계산
            angles = self._calculate_front_angles(landmarks)
            
            # 자세 점수 계산
            posture_score = self._calculate_front_posture_score(angles)
            
            # 피드백 생성
            feedback = self._generate_front_feedback(angles, posture_score)
            
            # 권장사항 생성
            recommendations = self._generate_front_recommendations(angles, posture_score)
            
            pose.close()
            
            return {
                "posture_score": posture_score,
                "angles": angles,
                "recommendations": recommendations,
                "feedback": feedback,
                "landmarks": landmarks
            }
            
        except Exception as e:
            logger.error(f"전면 이미지 분석 오류: {str(e)}")
            return {
                "posture_score": 50.0,
                "angles": {},
                "recommendations": ["전면 이미지 분석 중 오류가 발생했습니다."],
                "feedback": "전면 이미지 분석에 실패했습니다.",
                "landmarks": []
            }
    
    def _analyze_side_image(self, image):
        """측면 이미지 분석"""
        try:
            # MediaPipe Pose 초기화
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=0.5
            )
            
            # BGR을 RGB로 변환
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_image)
            
            if not results.pose_landmarks:
                return {
                    "posture_score": 50.0,
                    "angles": {},
                    "recommendations": ["사람이 감지되지 않았습니다. 더 명확한 측면 사진을 촬영해주세요."],
                    "feedback": "측면 사진에서 사람을 감지할 수 없습니다.",
                    "landmarks": []
                }
            
            # 랜드마크 추출
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y))
            
            # 각도 계산
            angles = self._calculate_side_angles(landmarks)
            
            # 자세 점수 계산
            posture_score = self._calculate_side_posture_score(angles)
            
            # 피드백 생성
            feedback = self._generate_side_feedback(angles, posture_score)
            
            # 권장사항 생성
            recommendations = self._generate_side_recommendations(angles, posture_score)
            
            pose.close()
            
            return {
                "posture_score": posture_score,
                "angles": angles,
                "recommendations": recommendations,
                "feedback": feedback,
                "landmarks": landmarks
            }
            
        except Exception as e:
            logger.error(f"측면 이미지 분석 오류: {str(e)}")
            return {
                "posture_score": 50.0,
                "angles": {},
                "recommendations": ["측면 이미지 분석 중 오류가 발생했습니다."],
                "feedback": "측면 이미지 분석에 실패했습니다.",
                "landmarks": []
            }
    
    def _calculate_front_angles(self, landmarks):
        """전면 각도 계산"""
        try:
            # MediaPipe Pose 랜드마크 인덱스
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_HIP = 23
            RIGHT_HIP = 24
            
            angles = {}
            
            # 어깨 각도 계산 (수평성)
            if len(landmarks) > max(LEFT_SHOULDER, RIGHT_SHOULDER):
                left_shoulder = landmarks[LEFT_SHOULDER]
                right_shoulder = landmarks[RIGHT_SHOULDER]
                
                # 어깨 기울기 각도 계산
                dx = right_shoulder[0] - left_shoulder[0]
                dy = right_shoulder[1] - left_shoulder[1]
                shoulder_angle = abs(np.degrees(np.arctan2(dy, dx)))
                angles["shoulder_angle"] = shoulder_angle
            
            # 골반 각도 계산 (수평성)
            if len(landmarks) > max(LEFT_HIP, RIGHT_HIP):
                left_hip = landmarks[LEFT_HIP]
                right_hip = landmarks[RIGHT_HIP]
                
                dx = right_hip[0] - left_hip[0]
                dy = right_hip[1] - left_hip[1]
                pelvis_angle = abs(np.degrees(np.arctan2(dy, dx)))
                angles["pelvis_angle"] = pelvis_angle
            
            return angles
            
        except Exception as e:
            logger.error(f"전면 각도 계산 오류: {str(e)}")
            return {}
    
    def _calculate_side_angles(self, landmarks):
        """측면 각도 계산"""
        try:
            # MediaPipe Pose 랜드마크 인덱스
            NOSE = 0
            EAR = 7
            SHOULDER = 11
            HIP = 23
            KNEE = 25
            ANKLE = 27
            
            angles = {}
            
            # 목 각도 계산
            if len(landmarks) > max(NOSE, EAR, SHOULDER):
                nose = landmarks[NOSE]
                ear = landmarks[EAR]
                shoulder = landmarks[SHOULDER]
                
                # 목 기울기 각도 계산
                neck_angle = self._calculate_angle(nose, ear, shoulder)
                angles["neck_angle"] = neck_angle
            
            # 척추 각도 계산
            if len(landmarks) > max(SHOULDER, HIP):
                shoulder = landmarks[SHOULDER]
                hip = landmarks[HIP]
                
                # 척추 기울기 각도 계산
                spine_angle = self._calculate_vertical_angle(shoulder, hip)
                angles["spine_angle"] = spine_angle
            
            return angles
            
        except Exception as e:
            logger.error(f"측면 각도 계산 오류: {str(e)}")
            return {}
    
    def _calculate_angle(self, point1, point2, point3):
        """세 점으로 이루어진 각도 계산"""
        try:
            v1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
            v2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            return angle
        except:
            return 0.0
    
    def _calculate_vertical_angle(self, point1, point2):
        """수직선과의 각도 계산"""
        try:
            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]
            angle = abs(np.degrees(np.arctan2(dx, dy)))
            return angle
        except:
            return 0.0
    
    def _calculate_front_posture_score(self, angles):
        """전면 자세 점수 계산"""
        score = 100.0
        
        # 어깨 각도에 따른 점수 감점
        if "shoulder_angle" in angles:
            shoulder_angle = angles["shoulder_angle"]
            if shoulder_angle > 10:
                score -= min(20, (shoulder_angle - 10) * 2)
        
        # 골반 각도에 따른 점수 감점
        if "pelvis_angle" in angles:
            pelvis_angle = angles["pelvis_angle"]
            if pelvis_angle > 10:
                score -= min(20, (pelvis_angle - 10) * 2)
        
        return max(0, score)
    
    def _calculate_side_posture_score(self, angles):
        """측면 자세 점수 계산"""
        score = 100.0
        
        # 목 각도에 따른 점수 감점
        if "neck_angle" in angles:
            neck_angle = angles["neck_angle"]
            if neck_angle > 15:
                score -= min(30, (neck_angle - 15) * 2)
        
        # 척추 각도에 따른 점수 감점
        if "spine_angle" in angles:
            spine_angle = angles["spine_angle"]
            if spine_angle > 10:
                score -= min(30, (spine_angle - 10) * 3)
        
        return max(0, score)
    
    def _generate_front_feedback(self, angles, score):
        """전면 피드백 생성"""
        if score >= 90:
            return "전면 자세가 매우 좋습니다!"
        elif score >= 80:
            return "전면 자세가 양호합니다."
        elif score >= 70:
            return "전면 자세에 약간의 개선이 필요합니다."
        else:
            return "전면 자세 개선이 필요합니다."
    
    def _generate_side_feedback(self, angles, score):
        """측면 피드백 생성"""
        if score >= 90:
            return "측면 자세가 매우 좋습니다!"
        elif score >= 80:
            return "측면 자세가 양호합니다."
        elif score >= 70:
            return "측면 자세에 약간의 개선이 필요합니다."
        else:
            return "측면 자세 개선이 필요합니다."
    
    def _generate_front_recommendations(self, angles, score):
        """전면 권장사항 생성"""
        recommendations = []
        
        if "shoulder_angle" in angles and angles["shoulder_angle"] > 10:
            recommendations.append("어깨를 수평하게 맞춰주세요.")
        
        if "pelvis_angle" in angles and angles["pelvis_angle"] > 10:
            recommendations.append("골반을 수평하게 유지해주세요.")
        
        if not recommendations:
            recommendations.append("현재 전면 자세를 유지해주세요.")
        
        return recommendations
    
    def _generate_side_recommendations(self, angles, score):
        """측면 권장사항 생성"""
        recommendations = []
        
        if "neck_angle" in angles and angles["neck_angle"] > 15:
            recommendations.append("목을 똑바로 세워주세요.")
        
        if "spine_angle" in angles and angles["spine_angle"] > 10:
            recommendations.append("척추를 곧게 펴주세요.")
        
        if not recommendations:
            recommendations.append("현재 측면 자세를 유지해주세요.")
        
        return recommendations
    
    def _generate_overall_feedback(self, front_analysis, side_analysis, overall_score):
        """전체 피드백 생성"""
        if overall_score >= 90:
            return "전반적으로 자세가 매우 좋습니다! 현재 자세를 유지해주세요."
        elif overall_score >= 80:
            return "전반적으로 자세가 양호합니다. 약간의 개선으로 더 좋은 자세를 만들 수 있습니다."
        elif overall_score >= 70:
            return "자세에 개선이 필요합니다. 권장사항을 참고하여 자세를 교정해주세요."
        else:
            return "자세 개선이 시급합니다. 전문가와 상담을 권장합니다."
    
    def _generate_improvement_suggestions(self, front_analysis, side_analysis):
        """개선 제안 생성"""
        suggestions = []
        
        # 전면 분석 기반 제안
        front_angles = front_analysis.get("angles", {})
        if front_angles.get("shoulder_angle", 0) > 10:
            suggestions.append("어깨 스트레칭 운동을 정기적으로 수행하세요.")
        
        if front_angles.get("pelvis_angle", 0) > 10:
            suggestions.append("골반 교정 운동을 통해 균형을 맞춰주세요.")
        
        # 측면 분석 기반 제안
        side_angles = side_analysis.get("angles", {})
        if side_angles.get("neck_angle", 0) > 15:
            suggestions.append("목 스트레칭과 목 근육 강화 운동을 하세요.")
        
        if side_angles.get("spine_angle", 0) > 10:
            suggestions.append("코어 운동을 통해 척추를 지지하는 근육을 강화하세요.")
        
        if not suggestions:
            suggestions.append("현재 자세를 유지하면서 정기적인 운동을 계속하세요.")
        
        return suggestions
    
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
