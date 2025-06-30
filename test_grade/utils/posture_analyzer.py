import math
import numpy as np
import mediapipe as mp

class PostureAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def calculate_angle(self, a, b, c):
        """세 점으로 각도 계산"""
        a_pt = np.array([a.x, a.y]) if not isinstance(a, np.ndarray) else a
        b_pt = np.array([b.x, b.y]) if not isinstance(b, np.ndarray) else b
        c_pt = np.array([c.x, c.y]) if not isinstance(c, np.ndarray) else c

        ba = a_pt - b_pt
        bc = c_pt - b_pt

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def calculate_distance(self, point1, point2):
        """두 점 간의 거리 계산"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def analyze_turtle_neck_detailed(self, landmarks):
        """목 자세 상세 분석"""
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # 목 중심점 계산
        neck_top_x = (left_ear.x + right_ear.x) / 2
        neck_top_y = (left_ear.y + right_ear.y) / 2
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # 목 각도 계산
        ear_center = np.array([neck_top_x, neck_top_y])
        shoulder_center = np.array([shoulder_center_x, shoulder_center_y])
        vertical = np.array([shoulder_center[0], ear_center[1]])
        neck_angle = self.calculate_angle(ear_center, shoulder_center, vertical)
        
        # 등급 분류
        grade, desc = self.grade_neck_posture(neck_angle)
        
        # 수직 이탈도 계산
        vertical_deviation = abs(neck_top_x - shoulder_center_x)
        
        return {
            'neck_angle': neck_angle,
            'grade': grade,
            'grade_description': desc,
            'vertical_deviation': vertical_deviation,
            'neck_top': (neck_top_x, neck_top_y),
            'shoulder_center': (shoulder_center_x, shoulder_center_y)
        }

    def grade_neck_posture(self, neck_angle):
        """목 자세 등급 분류"""
        if neck_angle <= 5:
            return 'A', "완벽한 자세"
        elif neck_angle <= 10:
            return 'B', "양호한 자세"
        elif neck_angle <= 15:
            return 'C', "보통 자세"
        else:
            return 'D', "나쁜 자세"

    def analyze_spine_curvature(self, landmarks):
        """척추 굴곡 분석"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # 어깨와 엉덩이 중심점
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_x = (left_hip.x + right_hip.x) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        
        # 척추 각도 계산
        spine_angle = math.degrees(math.atan2(hip_center_x - shoulder_center_x, 
                                            hip_center_y - shoulder_center_y))
        
        # 척추 굴곡 판정
        is_hunched = abs(spine_angle) > 12
        
        return {
            'is_hunched': is_hunched,
            'spine_angle': spine_angle,
            'shoulder_center': (shoulder_center_x, shoulder_center_y),
            'hip_center': (hip_center_x, hip_center_y)
        }

    def analyze_shoulder_asymmetry(self, landmarks):
        """어깨 비대칭 분석"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # 어깨 높이 차이 계산
        shoulder_height_diff = abs(left_shoulder.y - right_shoulder.y)
        
        # 어깨 기울기 계산
        shoulder_angle = math.degrees(math.atan2(right_shoulder.y - left_shoulder.y, 
                                               right_shoulder.x - left_shoulder.x))
        
        # 어깨 비대칭 판정
        is_asymmetric = shoulder_height_diff > 0.02
        
        # 어느 쪽이 높은지 판정
        if left_shoulder.y < right_shoulder.y:
            higher_shoulder = "왼쪽"
        else:
            higher_shoulder = "오른쪽"
        
        return {
            'is_asymmetric': is_asymmetric,
            'height_difference': shoulder_height_diff,
            'shoulder_angle': shoulder_angle,
            'higher_shoulder': higher_shoulder,
            'left_shoulder': (left_shoulder.x, left_shoulder.y),
            'right_shoulder': (right_shoulder.x, right_shoulder.y)
        }

    def analyze_pelvic_tilt(self, landmarks):
        """골반 기울기 분석"""
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # 골반 높이 차이 계산
        hip_height_diff = abs(left_hip.y - right_hip.y)
        
        # 골반 기울기 계산
        pelvic_angle = math.degrees(math.atan2(right_hip.y - left_hip.y, 
                                             right_hip.x - left_hip.x))
        
        # 골반 기울어짐 판정
        is_tilted = hip_height_diff > 0.015
        
        # 어느 쪽이 높은지 판정
        if left_hip.y < right_hip.y:
            higher_hip = "왼쪽"
        else:
            higher_hip = "오른쪽"
        
        return {
            'is_tilted': is_tilted,
            'height_difference': hip_height_diff,
            'pelvic_angle': pelvic_angle,
            'higher_hip': higher_hip,
            'left_hip': (left_hip.x, left_hip.y),
            'right_hip': (right_hip.x, right_hip.y)
        }

    def get_comprehensive_grade(self, landmarks):
        """종합 자세 등급 계산"""
        neck_analysis = self.analyze_turtle_neck_detailed(landmarks)
        spine_analysis = self.analyze_spine_curvature(landmarks)
        shoulder_analysis = self.analyze_shoulder_asymmetry(landmarks)
        pelvic_analysis = self.analyze_pelvic_tilt(landmarks)
        
        # 각 부위별 점수 계산
        neck_score = self.get_neck_score(neck_analysis['neck_angle'])
        spine_score = self.get_spine_score(spine_analysis['spine_angle'])
        shoulder_score = self.get_shoulder_score(shoulder_analysis['height_difference'])
        pelvic_score = self.get_pelvic_score(pelvic_analysis['height_difference'])
        
        # 종합 점수 계산 (가중치 적용)
        total_score = (neck_score * 0.4 + spine_score * 0.3 + 
                      shoulder_score * 0.2 + pelvic_score * 0.1)
        
        # 종합 등급 결정
        if total_score >= 90:
            grade = 'A'
        elif total_score >= 80:
            grade = 'B'
        elif total_score >= 70:
            grade = 'C'
        else:
            grade = 'D'
        
        return {
            'total_grade': grade,
            'total_score': total_score,
            'neck_score': neck_score,
            'spine_score': spine_score,
            'shoulder_score': shoulder_score,
            'pelvic_score': pelvic_score,
            'details': {
                'neck': neck_analysis,
                'spine': spine_analysis,
                'shoulder': shoulder_analysis,
                'pelvic': pelvic_analysis
            }
        }

    def get_neck_score(self, neck_angle):
        """목 자세 점수 계산"""
        if neck_angle <= 5:
            return 100
        elif neck_angle <= 10:
            return 90
        elif neck_angle <= 15:
            return 80
        elif neck_angle <= 20:
            return 70
        else:
            return 60

    def get_spine_score(self, spine_angle):
        """척추 자세 점수 계산"""
        abs_angle = abs(spine_angle)
        if abs_angle <= 5:
            return 100
        elif abs_angle <= 10:
            return 90
        elif abs_angle <= 15:
            return 80
        elif abs_angle <= 20:
            return 70
        else:
            return 60

    def get_shoulder_score(self, height_diff):
        """어깨 자세 점수 계산"""
        if height_diff <= 0.01:
            return 100
        elif height_diff <= 0.02:
            return 90
        elif height_diff <= 0.03:
            return 80
        elif height_diff <= 0.04:
            return 70
        else:
            return 60

    def get_pelvic_score(self, height_diff):
        """골반 자세 점수 계산"""
        if height_diff <= 0.005:
            return 100
        elif height_diff <= 0.01:
            return 90
        elif height_diff <= 0.015:
            return 80
        elif height_diff <= 0.02:
            return 70
        else:
            return 60 