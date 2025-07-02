import cv2
import math
import mediapipe as mp

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# 픽셀 변환 함수
def to_pixel(x, y, image_width, image_height):
    return int(x * image_width), int(y * image_height)

# ✅ Tragus 위치 보정 (좌측 더 이동)
def estimate_tragus_xy_refined(ear_landmark, image_w, image_h):
    x_offset = -0.035  # 기존보다 더 왼쪽
    y_offset =  0.02
    x = ear_landmark.x + x_offset
    y = ear_landmark.y + y_offset
    return to_pixel(x, y, image_w, image_h)

# C7 보정 (v1.3 기준 그대로 유지)
def estimate_c7_xy_final_refined(left_shoulder, right_shoulder, image_w, image_h):
    mid_x = (left_shoulder.x + right_shoulder.x) / 2
    mid_y = (left_shoulder.y + right_shoulder.y) / 2
    x_offset = -0.04
    y_offset = -0.09
    x = mid_x + x_offset
    y = mid_y + y_offset
    return to_pixel(x, y, image_w, image_h)

# CVA 계산
def calculate_cva(tragus_xy, c7_xy):
    dx = tragus_xy[0] - c7_xy[0]
    dy = tragus_xy[1] - c7_xy[1]
    angle_rad = math.atan2(dy, dx)
    return abs(math.degrees(angle_rad))

# HPD 계산
def calculate_hpd(tragus_xy, shoulder_xy):
    return abs(tragus_xy[0] - shoulder_xy[0])

# 통합 진단
def evaluate_posture(cva, hpd):
    if cva < 43 or hpd > 80:
        return "Severe Forward Head Posture!", (0, 0, 255)
    elif cva < 48 or hpd > 60:
        return "Mild Forward Head Posture.", (0, 165, 255)
    else:
        return "Posture looks normal.", (0, 255, 0)

# 웹캠 실행
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark

        left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_ear = lm[mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR]

        # 좌/우 tragus 판단
        tragus_landmark = left_ear if left_ear.x < 0.5 else right_ear
        shoulder_landmark = left_shoulder if tragus_landmark == left_ear else right_shoulder

        # 좌표 보정
        tragus_xy = estimate_tragus_xy_refined(tragus_landmark, w, h)
        c7_xy = estimate_c7_xy_final_refined(left_shoulder, right_shoulder, w, h)
        shoulder_xy = to_pixel(shoulder_landmark.x, shoulder_landmark.y, w, h)

        # 진단
        cva = calculate_cva(tragus_xy, c7_xy)
        hpd = calculate_hpd(tragus_xy, shoulder_xy)
        msg, color = evaluate_posture(cva, hpd)

        # 시각화
        cv2.circle(frame, tragus_xy, 5, (255, 0, 0), -1)
        cv2.circle(frame, c7_xy, 5, (0, 255, 0), -1)
        cv2.line(frame, tragus_xy, c7_xy, (0, 255, 255), 2)

        cv2.putText(frame, f'CVA: {cva:.1f} deg', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'HPD: {hpd:.0f} px', (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 255), 2)
        cv2.putText(frame, msg, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Posture Diagnosis (v1.4 - refined Tragus + C7)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
