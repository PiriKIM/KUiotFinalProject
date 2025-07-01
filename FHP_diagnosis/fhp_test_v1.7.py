import cv2
import math
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from collections import deque

# MediaPipe pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5
)

# 픽셀 좌표 변환
def to_pixel(x, y, w, h):
    return int(x * w), int(y * h)

# Tragus 보정 (귀 앞쪽)
def estimate_tragus_xy_precise(ear_landmark, w, h):
    x_offset = -0.04
    y_offset =  0.00
    return to_pixel(ear_landmark.x + x_offset, ear_landmark.y + y_offset, w, h)

# C7 보정 (어깨 중간 기준 위·왼쪽 이동)
def estimate_c7_xy_precise(left_shoulder, right_shoulder, w, h):
    mid_x = (left_shoulder.x + right_shoulder.x) / 2
    mid_y = (left_shoulder.y + right_shoulder.y) / 2
    return to_pixel(mid_x - 0.06, mid_y - 0.09, w, h)

# CVA 각도 계산 (벡터 내각 방식)
def calculate_cva(tragus, c7):
    dx, dy = tragus[0] - c7[0], tragus[1] - c7[1]
    angle_rad = math.atan2(dy, dx)
    return abs(math.degrees(angle_rad))

# HPD 수평 거리 계산
def calculate_hpd(tragus, shoulder):
    return abs(tragus[0] - shoulder[0])

# 자세 진단 해석
def evaluate_posture_korean(cva):
    if cva < 40:
        return "중증 거북목", (0, 0, 255)
    elif cva < 45:
        return "중등도 거북목", (0, 0, 200)
    elif cva < 50:
        return "경도 거북목", (0, 165, 255)
    else:
        return "정상 자세", (0, 255, 0)

# 오차 계산
def estimate_cva_error(buffer):
    if len(buffer) < 5:
        return None
    return np.std(buffer)

# 한글 텍스트 출력 (Pillow + 색공간 변환)
def draw_korean_text(frame, text, pos, font_path, size, color):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, size)
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print("❌ draw_korean_text 오류:", e)
        return frame

# ---------- 메인 실행 ----------
cap = cv2.VideoCapture(0)
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
cva_buffer = deque(maxlen=20)

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 프레임 수신 실패")
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

        tragus_landmark = left_ear if left_ear.x < 0.5 else right_ear
        shoulder_landmark = left_shoulder if tragus_landmark == left_ear else right_shoulder

        tragus = estimate_tragus_xy_precise(tragus_landmark, w, h)
        c7 = estimate_c7_xy_precise(left_shoulder, right_shoulder, w, h)
        shoulder = to_pixel(shoulder_landmark.x, shoulder_landmark.y, w, h)

        cva = calculate_cva(tragus, c7)
        hpd = calculate_hpd(tragus, shoulder)
        cva_buffer.append(cva)
        error = estimate_cva_error(cva_buffer)
        diagnosis, color = evaluate_posture_korean(cva)

        # 시각화
        cv2.circle(frame, tragus, 5, (255, 0, 0), -1)
        cv2.circle(frame, c7, 5, (0, 255, 0), -1)
        cv2.line(frame, tragus, c7, (0, 255, 255), 2)

        cv2.putText(frame, f'CVA: {cva:.1f} deg', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'HPD: {hpd:.0f} px', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 255), 2)
        if error:
            cv2.putText(frame, f'CVA 오차: ±{error:.1f} deg', (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)

        frame = draw_korean_text(frame, f'진단 결과: {diagnosis}', (30, 150), font_path, 28, color)

    cv2.imshow('Posture Diagnosis (v1.7.1)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
