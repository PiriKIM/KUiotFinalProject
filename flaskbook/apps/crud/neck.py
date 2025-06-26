import math
import numpy as np

class PostureAnalyzer:
    def __init__(self):
        import mediapipe as mp
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

    def analyze_turtle_neck(self, landmarks):
        """거북목 분석 - 목-어깨의 수직선 이탈 확인 (더 세밀한 분석)"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
        
        # 목 중심점 계산 (귀 중앙에서 어깨 중앙으로)
        neck_top_x = (left_ear.x + right_ear.x) / 2
        neck_top_y = (left_ear.y + right_ear.y) / 2
        
        # 어깨 중심점 계산
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # 목 중간점 계산 (귀와 어깨 중간)
        neck_mid_x = (neck_top_x + shoulder_center_x) / 2
        neck_mid_y = (neck_top_y + shoulder_center_y) / 2
        
        # 수직선에서의 이탈 정도 계산 (목 상단, 중간, 하단)
        vertical_deviation_top = abs(neck_top_x - shoulder_center_x)
        vertical_deviation_mid = abs(neck_mid_x - shoulder_center_x)
        
        # 거북목 판정 (더 세밀한 기준)
        is_turtle_neck = vertical_deviation_top > 0.03 or vertical_deviation_mid > 0.04
        
        return {
            'is_turtle_neck': is_turtle_neck,
            'deviation_top': vertical_deviation_top,
            'deviation_mid': vertical_deviation_mid,
            'neck_top': (neck_top_x, neck_top_y),
            'neck_mid': (neck_mid_x, neck_mid_y),
            'shoulder_center': (shoulder_center_x, shoulder_center_y)
        }

    def calculate_neck_angle(self, landmarks):
        mp = self.mp_pose
        left_ear = landmarks[mp.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[mp.PoseLandmark.RIGHT_EAR]
        left_shoulder = landmarks[mp.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp.PoseLandmark.RIGHT_SHOULDER]
        ear_center = np.array([(left_ear.x + right_ear.x) / 2, (left_ear.y + right_ear.y) / 2])
        shoulder_center = np.array([(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2])
        vertical = np.array([shoulder_center[0], ear_center[1]])
        return self.calculate_angle(ear_center, shoulder_center, vertical)

    def grade_neck_posture(self, neck_angle):
        if neck_angle <= 5:
            return 'A', "완벽한 자세"
        elif neck_angle <= 10:
            return 'B', "양호한 자세"
        elif neck_angle <= 15:
            return 'C', "보통 자세"
        else:
            return 'D', "나쁜 자세"

    def analyze_turtle_neck_detailed(self, landmarks):
        mp = self.mp_pose
        neck_angle = self.calculate_neck_angle(landmarks)
        grade, desc = self.grade_neck_posture(neck_angle)
        neck_top = (
            (landmarks[mp.PoseLandmark.LEFT_EAR].x + landmarks[mp.PoseLandmark.RIGHT_EAR].x) / 2,
            (landmarks[mp.PoseLandmark.LEFT_EAR].y + landmarks[mp.PoseLandmark.RIGHT_EAR].y) / 2
        )
        shoulder_center = (
            (landmarks[mp.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp.PoseLandmark.RIGHT_SHOULDER].x) / 2,
            (landmarks[mp.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp.PoseLandmark.RIGHT_SHOULDER].y) / 2
        )
        vertical_deviation = abs(neck_top[0] - shoulder_center[0])
        return {
            'neck_angle': neck_angle,
            'grade': grade,
            'grade_description': desc,
            'vertical_deviation': vertical_deviation,
            'neck_top': neck_top,
            'shoulder_center': shoulder_center
        }

    def analyze_spine_curvature(self, landmarks):
        """척추 굴곡 분석 - 등이 굽은 상태 확인 (더 세밀한 분석)"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        
        # 어깨와 엉덩이 중심점
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_x = (left_hip.x + right_hip.x) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        
        # 척추 각도 계산 (어깨-엉덩이 수직선과의 각도)
        spine_angle = math.degrees(math.atan2(hip_center_x - shoulder_center_x, 
                                            hip_center_y - shoulder_center_y))
        
        # 척추 굴곡 판정 (더 세밀한 기준)
        is_hunched = abs(spine_angle) > 12  # 12도 이상 기울어지면 굽은 상태
        
        # 척추 중간점 계산 (어깨와 골반 중간)
        spine_mid_x = (shoulder_center_x + hip_center_x) / 2
        spine_mid_y = (shoulder_center_y + hip_center_y) / 2
        
        return {
            'is_hunched': is_hunched,
            'spine_angle': spine_angle,
            'shoulder_center': (shoulder_center_x, shoulder_center_y),
            'hip_center': (hip_center_x, hip_center_y),
            'spine_mid': (spine_mid_x, spine_mid_y)
        }

    def analyze_shoulder_asymmetry(self, landmarks):
        """어깨선 분석 - 어깨의 기울기와 비대칭 확인 (더 세밀한 분석)"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        
        # 어깨 높이 차이 계산
        shoulder_height_diff = abs(left_shoulder.y - right_shoulder.y)
        
        # 어깨 기울기 계산 (수평선과의 각도)
        shoulder_angle = math.degrees(math.atan2(right_shoulder.y - left_shoulder.y, 
                                               right_shoulder.x - left_shoulder.x))
        
        # 어깨 비대칭 판정 (더 세밀한 기준)
        is_asymmetric = shoulder_height_diff > 0.02  # 2% 이상 차이나면 비대칭
        
        # 어느 쪽이 높은지 판정
        if left_shoulder.y < right_shoulder.y:
            higher_shoulder = "왼쪽"
        else:
            higher_shoulder = "오른쪽"
        
        # 어깨선 중간점
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        
        return {
            'is_asymmetric': is_asymmetric,
            'height_difference': shoulder_height_diff,
            'shoulder_angle': shoulder_angle,
            'higher_shoulder': higher_shoulder,
            'shoulder_mid': (shoulder_mid_x, shoulder_mid_y),
            'left_shoulder': (left_shoulder.x, left_shoulder.y),
            'right_shoulder': (right_shoulder.x, right_shoulder.y)
        }

    def analyze_pelvic_tilt(self, landmarks):
        """골반 관절 분석 - 골반의 기울기와 비대칭 확인 (더 세밀한 분석)"""
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        
        # 골반 높이 차이 계산
        hip_height_diff = abs(left_hip.y - right_hip.y)
        
        # 골반 기울기 계산 (수평선과의 각도)
        pelvic_angle = math.degrees(math.atan2(right_hip.y - left_hip.y, 
                                             right_hip.x - left_hip.x))
        
        # 골반 기울어짐 판정 (더 세밀한 기준)
        is_tilted = hip_height_diff > 0.015  # 1.5% 이상 차이나면 기울어짐
        
        # 어느 쪽이 높은지 판정
        if left_hip.y < right_hip.y:
            higher_hip = "왼쪽"
        else:
            higher_hip = "오른쪽"
        
        # 골반 중앙 좌표 계산
        pelvic_center_x = (left_hip.x + right_hip.x) / 2
        pelvic_center_y = (left_hip.y + right_hip.y) / 2
        
        # 골반-무릎 연결선 각도 (측면 자세 확인용)
        left_hip_knee_angle = math.degrees(math.atan2(left_knee.y - left_hip.y, 
                                                    left_knee.x - left_hip.x))
        right_hip_knee_angle = math.degrees(math.atan2(right_knee.y - right_hip.y, 
                                                     right_knee.x - right_hip.x))
        
        return {
            'is_tilted': is_tilted,
            'height_difference': hip_height_diff,
            'pelvic_angle': pelvic_angle,
            'higher_hip': higher_hip,
            'pelvic_center': (pelvic_center_x, pelvic_center_y),
            'left_hip': (left_hip.x, left_hip.y),
            'right_hip': (right_hip.x, right_hip.y),
            'left_hip_knee_angle': left_hip_knee_angle,
            'right_hip_knee_angle': right_hip_knee_angle
        }

    def analyze_spine_twisting(self, landmarks):
        """척추 틀어짐 분석 - 측면에서 본 척추 곡률 확인"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
        
        # 어깨와 골반의 수평 정렬 확인
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        hip_center_x = (left_hip.x + right_hip.x) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        
        # 척추 수직 정렬도 계산
        spine_alignment = abs(shoulder_center_x - hip_center_x)
        
        # 척추 틀어짐 판정
        is_twisted = spine_alignment > 0.03  # 3% 이상 어긋나면 틀어짐
        
        # 측면 자세 분석 (귀-어깨-골반 수직선)
        ear_center_x = (left_ear.x + right_ear.x) / 2
        ear_center_y = (left_ear.y + right_ear.y) / 2
        
        # 측면 척추 각도
        side_spine_angle = math.degrees(math.atan2(hip_center_x - ear_center_x, 
                                                 hip_center_y - ear_center_y))
        
        return {
            'is_twisted': is_twisted,
            'spine_alignment': spine_alignment,
            'side_spine_angle': side_spine_angle,
            'ear_center': (ear_center_x, ear_center_y),
            'shoulder_center': (shoulder_center_x, (left_shoulder.y + right_shoulder.y) / 2),
            'hip_center': (hip_center_x, hip_center_y)
        }
