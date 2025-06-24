import cv2
import mediapipe as mp

# 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# 몸 관절 번호 (얼굴 제외)
BODY_LANDMARKS = {
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16,
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28
}

# 연결선 정의
BODY_CONNECTIONS = [
    (BODY_LANDMARKS['left_shoulder'], BODY_LANDMARKS['left_elbow']),
    (BODY_LANDMARKS['left_elbow'], BODY_LANDMARKS['left_wrist']),
    (BODY_LANDMARKS['right_shoulder'], BODY_LANDMARKS['right_elbow']),
    (BODY_LANDMARKS['right_elbow'], BODY_LANDMARKS['right_wrist']),
    (BODY_LANDMARKS['left_shoulder'], BODY_LANDMARKS['left_hip']),
    (BODY_LANDMARKS['right_shoulder'], BODY_LANDMARKS['right_hip']),
    (BODY_LANDMARKS['left_hip'], BODY_LANDMARKS['left_knee']),
    (BODY_LANDMARKS['left_knee'], BODY_LANDMARKS['left_ankle']),
    (BODY_LANDMARKS['right_hip'], BODY_LANDMARKS['right_knee']),
    (BODY_LANDMARKS['right_knee'], BODY_LANDMARKS['right_ankle']),
    (BODY_LANDMARKS['left_shoulder'], BODY_LANDMARKS['right_shoulder']),
    (BODY_LANDMARKS['left_hip'], BODY_LANDMARKS['right_hip'])
]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    
    if results.pose_landmarks:
        h, w, c = frame.shape
        landmarks = results.pose_landmarks.landmark
        
        # 좌표 저장
        points = {}
        for name, id in BODY_LANDMARKS.items():
            lm = landmarks[id]
            cx, cy = int(lm.x * w), int(lm.y * h)
            points[name] = (cx, cy)
            
            # 원 그리기 (몸 관절만)
            cv2.circle(frame, (cx, cy), 7, (0, 255, 0), cv2.FILLED)
        
        # 연결선 그리기 (몸 관절만)
        for p1, p2 in BODY_CONNECTIONS:
            x1, y1 = int(landmarks[p1].x * w), int(landmarks[p1].y * h)
            x2, y2 = int(landmarks[p2].x * w), int(landmarks[p2].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        
        # 비대칭 체크
        l_sh_y = points['left_shoulder'][1]
        r_sh_y = points['right_shoulder'][1]
        l_hip_y = points['left_hip'][1]
        r_hip_y = points['right_hip'][1]
        
        shoulder_diff = abs(l_sh_y - r_sh_y)
        hip_diff = abs(l_hip_y - r_hip_y)
        
        cv2.putText(frame, f'Shoulder diff: {shoulder_diff}px', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f'Hip diff: {hip_diff}px', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        print(f'Shoulder diff: {shoulder_diff}px, Hip diff: {hip_diff}px')
    
    # 전체 화면 그대로 보여주기
    cv2.imshow('Body Pose - Fullscreen No Face Landmarks', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
