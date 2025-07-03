# ✅ esp_module.py
import cv2
import mediapipe as mp
import json
import os
from datetime import datetime

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 저장 디렉토리 설정
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../../pose_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 프레임 저장 조건 변수
SAVE_EVERY_N_FRAMES = 200

def extract_landmarks(landmarks):
    result = []
    for i, lm in enumerate(landmarks.landmark):
        result.append({
            "index": i,
            "x": lm.x,
            "y": lm.y,
            "z": lm.z,
            "visibility": lm.visibility
        })
    return result

def save_pose_data(landmarks, frame_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pose_{timestamp}_f{frame_id}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(landmarks, f, indent=2)
    print(f"✅ 저장 완료: {filename}")

def run_pose_tracking():
    print("▶️ ESP32 스트리밍 시작")
    cap = cv2.VideoCapture("http://192.168.0.99:81/stream")
    frame_id = 0

    if not cap.isOpened():
        print("❌ ESP32 스트리밍 연결 실패")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️ 프레임 읽기 실패")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            if frame_id % SAVE_EVERY_N_FRAMES == 0:
                landmarks = extract_landmarks(results.pose_landmarks)
                save_pose_data(landmarks, frame_id)
            else:
                print(f"[{frame_id}] 분석됨 (저장 생략)")

        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 사용자 종료 요청")
            break

    cap.release()
    cv2.destroyAllWindows()
