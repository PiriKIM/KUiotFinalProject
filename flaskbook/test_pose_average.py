import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import mediapipe as mp
from apps.crud.neck import PostureAnalyzer

# 초기화
pose = mp.solutions.pose.Pose()
analyzer = PostureAnalyzer()
frame_buffer = []
FRAME_BATCH_SIZE = 30

# 웹캠 켜기
cap = cv2.VideoCapture(0)
print("✅ 웹캠을 시작합니다. 'q'를 누르면 종료됩니다.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 프레임을 읽을 수 없습니다.")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        frame_buffer.append(results.pose_landmarks.landmark)

        # 30프레임 평균 분석
        if len(frame_buffer) == FRAME_BATCH_SIZE:
            result = analyzer.analyze_average_posture(frame_buffer)
            print("\n📊 [30프레임 평균 결과]")
            print(f"📏 평균 목각도: {result['avg_angle']}")
            print(f"📈 등급: {result['grade']} - {result['description']}")
            print(f"🎞️ 유효 프레임 수: {result['frame_count']}")
            frame_buffer.clear()

    # 실시간 영상 보여주기
    cv2.imshow('Posture Analyzer - 30 Frame AVG', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
