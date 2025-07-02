# 추가 기능
# FHP 진단 (tragus-C7 각도 기반)
# 척추 기울기 진단 (C7-pelvis 각도 기반)
# 골반 추정 보정 (골반 미검출 시 어깨 좌표 기반 보정 추정)


import cv2
import math
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from collections import deque

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# 픽셀 변환
def to_pixel(x, y, w, h):
    return int(x * w), int(y * h)

# Tragus 보정
def estimate_tragus_xy(ear, w, h):
    return to_pixel(ear.x - 0.04, ear.y + 0.00, w, h)

# C7 보정
def estimate_c7_xy(left_shoulder, right_shoulder, w, h):
    mid_x = (left_shoulder.x + right_shoulder.x) / 2
    mid_y = (left_shoulder.y + right_shoulder.y) / 2
    return to_pixel(mid_x - 0.06, mid_y - 0.09, w, h)

# 골반 보정: 실제 감지 또는 어깨 기반 추정
def estimate_pelvis_xy(left_hip, right_hip, left_shoulder, right_shoulder, w, h):
    if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
        px = (left_hip.x + right_hip.x) / 2
        py = (left_hip.y + right_hip.y) / 2
    else:
        px = (left_shoulder.x + right_shoulder.x) / 2
        py = (left_shoulder.y + right_shoulder.y) / 2 + 0.25  # 아래로 보정
    return to_pixel(px, py, w, h)

# CVA 계산
def calculate_cva(tragus, c7):
    dx, dy = tragus[0] - c7[0], tragus[1] - c7[1]
    return abs(math.degrees(math.atan2(dy, dx)))

# 척추 기울기 (C7→골반)
def calculate_spine_angle(c7, pelvis):
    dx, dy = pelvis[0] - c7[0], pelvis[1] - c7[1]
    return abs(math.degrees(math.atan2(dy, dx)))

# HPD
def calculate_hpd(tragus, shoulder):
    return abs(tragus[0] - shoulder[0])

# FHP 진단
def evaluate_fhp(cva):
    if cva < 45:
        return "중증 거북목", (0, 0, 255)
    elif cva < 55:
        return "중등도 거북목", (0, 0, 200)
    elif cva < 65:
        return "경도 거북목", (0, 165, 255)
    else:
        return "정상 자세", (0, 255, 0)

# 척추 기울기 진단
def evaluate_spine(angle):
    if angle < 75:
        return "상체 전반 전방 기울어짐", (255, 0, 255)
    else:
        return "", None

# 한글 표시
def draw_korean_text(frame, text, pos, font_path, size, color):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, size)
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except:
        return frame

# ---------- MAIN ----------
cap = cv2.VideoCapture(0)
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
cva_buffer = deque(maxlen=20)

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
        l_sh, r_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip, r_hip = lm[mp_pose.PoseLandmark.LEFT_HIP], lm[mp_pose.PoseLandmark.RIGHT_HIP]
        l_ear, r_ear = lm[mp_pose.PoseLandmark.LEFT_EAR], lm[mp_pose.PoseLandmark.RIGHT_EAR]

        # visibility 기반 귀/어깨 선택
        tragus_lm, shoulder_lm = (
            (l_ear, l_sh) if l_ear.visibility > r_ear.visibility else (r_ear, r_sh)
        )

        tragus = estimate_tragus_xy(tragus_lm, w, h)
        c7 = estimate_c7_xy(l_sh, r_sh, w, h)
        shoulder = to_pixel(shoulder_lm.x, shoulder_lm.y, w, h)
        pelvis = estimate_pelvis_xy(l_hip, r_hip, l_sh, r_sh, w, h)

        cva = calculate_cva(tragus, c7)
        spine_angle = calculate_spine_angle(c7, pelvis)
        hpd = calculate_hpd(tragus, shoulder)

        cva_buffer.append(cva)
        error = np.std(cva_buffer) if len(cva_buffer) >= 5 else None

        fhp_result, fhp_color = evaluate_fhp(cva)
        spine_result, spine_color = evaluate_spine(spine_angle)

        # 시각화
        cv2.circle(frame, tragus, 5, (255, 0, 0), -1)
        cv2.circle(frame, c7, 5, (0, 255, 0), -1)
        cv2.circle(frame, pelvis, 5, (255, 0, 255), -1)
        cv2.line(frame, tragus, c7, (0, 255, 255), 2)
        cv2.line(frame, c7, pelvis, (255, 255, 0), 2)

        cv2.putText(frame, f'CVA: {cva:.1f}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, fhp_color, 2)
        cv2.putText(frame, f'Spine: {spine_angle:.1f}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        if error:
            cv2.putText(frame, f'CVA 오차: ±{error:.1f}', (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 255, 255), 2)

        frame = draw_korean_text(frame, f'거북목 진단: {fhp_result}', (30, 150), font_path, 28, fhp_color)
        if spine_result:
            frame = draw_korean_text(frame, f'추가 진단: {spine_result}', (30, 190), font_path, 28, spine_color)

    else:
        cv2.putText(frame, "사람이 감지되지 않음", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)

    cv2.imshow('Posture Diagnosis (v1.7.4)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
