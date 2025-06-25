import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import cv2
import mediapipe as mp
import numpy as np
from yj.Back_End.MediaPipe_test.neck import PostureAnalyzer

# PostureAnalyzer 인스턴스 생성
posture_analyzer = PostureAnalyzer()

def analyze_posture(image):
    """
    Flask 서버에서 호출: 이미지를 받아 거북목 분석 결과만 반환
    """
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # BGR → RGB 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            return {
                'grade': 'None',
                'grade_description': 'No person detected',
                'neck_angle': 0.0
            }
        # 거북목 분석
        turtle_neck_result = posture_analyzer.analyze_turtle_neck_detailed(results.pose_landmarks.landmark)
        return {
            'grade': turtle_neck_result['grade'],
            'grade_description': turtle_neck_result['grade_description'],
            'neck_angle': turtle_neck_result['neck_angle']
        }