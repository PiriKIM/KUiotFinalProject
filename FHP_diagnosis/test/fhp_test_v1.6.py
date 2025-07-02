import cv2
import math
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# MediaPipe pose 모듈 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,            # 영상 스트림 기반 입력
    model_complexity=1,                # 기본 모델 복잡도 설정
    min_detection_confidence=0.5       # 최소 검출 신뢰도 설정
)

# 정규화된 좌표(0~1)를 픽셀 단위로 변환하는 함수
def to_pixel(x, y, image_width, image_height):
    return int(x * image_width), int(y * image_height)

# Tragus(귀 앞 연골) 위치 보정 함수
# MediaPipe의 귀 좌표는 귀의 뒷부분 또는 측면이라 좌표를 왼쪽/위로 보정
def estimate_tragus_xy_precise(ear_landmark, image_w, image_h):
    x_offset = -0.04  # X축으로 왼쪽 이동 (귀 앞쪽으로)
    y_offset =  0.00  # Y축은 위/아래 보정 없음 (기준에 따라 조정 가능)
    x = ear_landmark.x + x_offset
    y = ear_landmark.y + y_offset
    return to_pixel(x, y, image_w, image_h)

# C7 (제7경추) 위치 보정 함수
# 어깨 중간점에서 왼쪽으로 이동, 위로 이동하여 해부학적 목뼈 기준으로 근접
def estimate_c7_xy_precise(left_shoulder, right_shoulder, image_w, image_h):
    mid_x = (left_shoulder.x + right_shoulder.x) / 2
    mid_y = (left_shoulder.y + right_shoulder.y) / 2
    x_offset = -0.05  # X축으로 왼쪽 이동
    y_offset = -0.09  # Y축으로 위로 이동
    x = mid_x + x_offset
    y = mid_y + y_offset
    return to_pixel(x, y, image_w, image_h)

# CVA (Craniovertebral Angle) 계산 함수
# Tragus와 C7을 잇는 선과 수평선 사이의 각도 계산
def calculate_cva(tragus_xy, c7_xy):
    dx = tragus_xy[0] - c7_xy[0]
    dy = tragus_xy[1] - c7_xy[1]
    angle_rad = math.atan2(dy, dx)
    return abs(math.degrees(angle_rad))  # 양의 각도로 반환

# HPD (Head Protrusion Distance) 계산 함수
# 귀와 어깨의 X축 거리 차이를 계산하여 머리 전방 돌출 정도 측정
def calculate_hpd(tragus_xy, shoulder_xy):
    return abs(tragus_xy[0] - shoulder_xy[0])

# CVA에 따른 진단 결과 반환 (한글 문자열 + 색상)
def evaluate_posture_korean(cva):
    if cva < 40:
        return "중증 거북목", (0, 0, 255)        # 빨강
    elif cva < 45:
        return "중등도 거북목", (0, 0, 200)      # 어두운 빨강/파랑 계열
    elif cva < 50:
        return "경도 거북목", (0, 165, 255)      # 주황
    else:
        return "정상 자세", (0, 255, 0)          # 초록

# 한글 텍스트를 이미지에 출력하기 위한 함수 (Pillow 사용)
def draw_korean_text(img, text, position, font_path, font_size, color):
    img_pil = Image.fromarray(img)  # OpenCV 이미지를 PIL 이미지로 변환
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)  # 다시 OpenCV 이미지로 변환

# 웹캠 캡처 시작
cap = cv2.VideoCapture(0)
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # 나눔고딕 폰트 경로

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 좌우 반전 (거울 모드)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR → RGB
    result = pose.process(rgb)  # 포즈 추론 실행

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark

        # 어깨 및 귀 좌표 추출
        left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_ear = lm[mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR]

        # 얼굴 방향 기준으로 tragus, shoulder 선택
        tragus_landmark = left_ear if left_ear.x < 0.5 else right_ear
        shoulder_landmark = left_shoulder if tragus_landmark == left_ear else right_shoulder

        # 보정된 좌표 계산
        tragus_xy = estimate_tragus_xy_precise(tragus_landmark, w, h)
        c7_xy = estimate_c7_xy_precise(left_shoulder, right_shoulder, w, h)
        shoulder_xy = to_pixel(shoulder_landmark.x, shoulder_landmark.y, w, h)

        # 분석 결과 계산
        cva = calculate_cva(tragus_xy, c7_xy)
        hpd = calculate_hpd(tragus_xy, shoulder_xy)
        diagnosis_text, color = evaluate_posture_korean(cva)

        # 시각화 요소
        cv2.circle(frame, tragus_xy, 5, (255, 0, 0), -1)       # Tragus: 파란 점
        cv2.circle(frame, c7_xy, 5, (0, 255, 0), -1)           # C7: 초록 점
        cv2.line(frame, tragus_xy, c7_xy, (0, 255, 255), 2)    # 연결선: 노랑

        # 영어 텍스트 (CVA, HPD)
        cv2.putText(frame, f'CVA: {cva:.1f} deg', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'HPD: {hpd:.0f} px', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 255), 2)

        # 진단 결과 (한글 표시)
        frame = draw_korean_text(frame, f'진단 결과: {diagnosis_text}', (30, 120), font_path, 28, color)

    # 결과 출력
    cv2.imshow('Posture Diagnosis (v1.6 - Korean Medical CVA)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
