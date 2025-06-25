import math
import numpy as np
import mediapipe as mp
from .utils import calculate_angle, calculate_distance, get_center_point
from app.models.posture_result import (
    NeckResult, SpineResult, ShoulderResult, PelvicResult, SpineTwistingResult
)

class PostureAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def calculate_neck_angle(self, landmarks):
        """목 각도 계산 - 귀-어깨-수직선 각도"""
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        ear_center = get_center_point(left_ear, right_ear)
        shoulder_center = get_center_point(left_shoulder, right_shoulder)
        vertical = (shoulder_center[0], ear_center[1])
        # 각도 계산 (귀-어깨-수직선)
        class Dummy:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        a = Dummy(*ear_center)
        b = Dummy(*shoulder_center)
        c = Dummy(*vertical)
        return calculate_angle(a, b, c)

    def grade_neck_posture(self, neck_angle):
        """목 각도에 따른 자세 등급 판정"""
        if neck_angle <= 5:
            return 'A', "완벽한 자세"
        elif neck_angle <= 10:
            return 'B', "약간의 거북목"
        elif neck_angle <= 15:
            return 'C', "거북목 주의"
        else:
            return 'D', "심한 거북목"

    def analyze_turtle_neck_detailed(self, landmarks):
        angle = self.calculate_neck_angle(landmarks)
        grade, feedback = self.grade_neck_posture(angle)
        return NeckResult(angle=angle, grade=grade, feedback=feedback)

    def analyze_spine_curvature(self, landmarks):
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        shoulder_center = get_center_point(left_shoulder, right_shoulder)
        hip_center = get_center_point(left_hip, right_hip)
        # 수직선과 어깨-엉덩이 선의 각도
        class Dummy:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        a = Dummy(shoulder_center[0], 0)
        b = Dummy(*shoulder_center)
        c = Dummy(*hip_center)
        curvature = calculate_angle(a, b, c)
        feedback = "정상" if abs(curvature - 90) < 10 else "척추 만곡 주의"
        return SpineResult(curvature=curvature, feedback=feedback)

    def analyze_shoulder_asymmetry(self, landmarks):
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        asymmetry = abs(left_shoulder.y - right_shoulder.y)
        feedback = "정상" if asymmetry < 0.03 else "어깨 높이 비대칭"
        return ShoulderResult(asymmetry=asymmetry, feedback=feedback)

    def analyze_pelvic_tilt(self, landmarks):
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        tilt = abs(left_hip.y - right_hip.y)
        feedback = "정상" if tilt < 0.03 else "골반 기울어짐"
        return PelvicResult(tilt=tilt, feedback=feedback)

    def analyze_spine_twisting(self, landmarks):
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        shoulder_width = calculate_distance(left_shoulder, right_shoulder)
        hip_width = calculate_distance(left_hip, right_hip)
        twisting = abs(shoulder_width - hip_width)
        feedback = "정상" if twisting < 0.05 else "척추 비틀림"
        return SpineTwistingResult(twisting=twisting, feedback=feedback)
