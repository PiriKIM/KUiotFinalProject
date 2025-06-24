import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# 몸만 관절 번호 (얼굴 제외)
BODY_LANDMARKS = [
    11, 12,   # 어깨 (left, right)
    13, 14,   # 팔꿈치
    15, 16,   # 손목
    23, 24,   # 골반
    25, 26,   # 무릎
    27, 28    # 발목
]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    
    if results.pose_landmarks:
        # 관절 직접 그리기 (얼굴 제외)
        for id in BODY_LANDMARKS:
            landmark = results.pose_landmarks.landmark[id]
            h, w, c = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            print(f'ID: {id}, X: {cx}, Y: {cy}, Z: {landmark.z:.4f}, Visibility: {landmark.visibility:.4f}')
            
            cv2.circle(frame, (cx, cy), 7, (0, 255, 0), cv2.FILLED)
    
    # 전체 landmark 연결선 그리기 → 주석처리 (얼굴이 자동으로 같이 나와서!)
    # mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.imshow('Body Pose Only', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
Processing triggers for desktop-file-utils (0.26-1ubuntu3) ...
eunkyu@eunkyu-virtual-machine:~/Downloads$ ./Cursor-1.1.3-x86_64.AppImage
libva error: vaGetDriverNameByIndex() failed with unknown libva error, driver_name = (null)
[main 2025-06-24T02:28:00.182Z] updateURL https://api2.cursor.sh/updates/api/update/linux-x64/cursor/1.1.3/24ec02a8bd37d6c7ca303d1bd14ddc557bf5656bb7f634d6fb80e6599ad1f817/stable
[main 2025-06-24T02:28:00.184Z] update#setState idle
[22774:0