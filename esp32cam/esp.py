import cv2
import mediapipe as mp
import time
import json
import os

# 스트리밍 주소
ESP32_STREAM_URL = 'http://192.168.0.99:81/stream' # 본인 ip주소 찾아서 변경할것

# 저장 디렉토리 설정
OUTPUT_DIR = "pose_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# 비디오 캡처
cap = cv2.VideoCapture(ESP32_STREAM_URL)

frame_count = 0
prev_time = 0

# OpenPose 포맷으로 저장하는 함수
def save_openpose_json(results, frame_id):
    if not results.pose_landmarks:
        return

    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])  # x, y, z, confidence

    data = {
        "version": 1.0,
        "people": [
            {
                "person_id": [-1],
                "pose_keypoints_3d": keypoints
            }
        ]
    }

    filename = f"frame_{frame_id:05d}_keypoints.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

# 메인 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 가져오지 못했습니다.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    # 결과 저장
    if frame_count % 20 == 0:
        save_openpose_json(results, frame_count)

    # 화면 표시용
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    h, w, _ = frame.shape
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(frame, f"Resolution: {w}x{h}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("ESP32-CAM + MediaPipe Pose", frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 정리
cap.release()
cv2.destroyAllWindows()