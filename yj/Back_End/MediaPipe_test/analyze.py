import cv2
import mediapipe as mp
import numpy as np
import math
import sys

# 이미지 파일명 (사용자가 업로드한 파일명으로 변경)
IMAGE_PATH = '/home/yj/KUiotFinalProject/yj/Back_End/image.png'  # 실제 파일명으로 바꿔주세요

mp_pose = mp.solutions.pose

class PostureAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
    def calculate_neck_angle(self, landmarks):
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        ear_center_x = (left_ear.x + right_ear.x) / 2
        ear_center_y = (left_ear.y + right_ear.y) / 2
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        vertical_x = shoulder_center_x
        vertical_y = ear_center_y
        a = np.array([ear_center_x, ear_center_y])
        b = np.array([shoulder_center_x, shoulder_center_y])
        c = np.array([vertical_x, vertical_y])
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

if __name__ == '__main__':
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f'Image not found: {IMAGE_PATH}')
        sys.exit(1)
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            print('No pose detected in the image.')
            sys.exit(1)
        analyzer = PostureAnalyzer()
        neck_angle = analyzer.calculate_neck_angle(results.pose_landmarks.landmark)
        print(f'Neck angle in the reference image: {neck_angle:.2f} degrees')
        # (Optional) 시각화
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Reference Pose', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 예시: reference_angle = 3.2
        if abs(neck_angle - 3.2) <= 2:  # A등급: 기준±2도
            grade = 'A'
            grade_text = "Perfect Posture"
        elif abs(neck_angle - 3.2) <= 5:
            grade = 'B'
            grade_text = "Good Posture"
        elif abs(neck_angle - 3.2) <= 10:
            grade = 'C'
            grade_text = "Average Posture"
        else:
            grade = 'D'
            grade_text = "Poor Posture"

        print(f'Grade: {grade} - {grade_text}')
        # (Optional) 등급 텍스트 표시
        cv2.putText(image, f'{grade} - {grade_text}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Reference Pose with Grade', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 