import cv2
import math
import mediapipe as mp

# 초기 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

# 픽셀 변환 함수
def to_pixel(x, y, image_width, image_height):
    return int(x * image_width), int(y * image_height)

def estimate_tragus_xy_fixed(right_ear, image_w, image_h):
    # 경험적 보정값: X는 왼쪽으로 (음수), Y는 아래로 (양수)
    x_offset = -0.02  # 왼쪽 이동
    y_offset =  0.02  # 아래 이동

    x = right_ear.x + x_offset
    y = right_ear.y + y_offset

    return to_pixel(x, y, image_w, image_h)

# C7 위치 추정 (어깨 중간보다 약간 위)
def estimate_c7_xy(left_shoulder, right_shoulder, image_w, image_h):
    x = (left_shoulder.x + right_shoulder.x) / 2
    y = (left_shoulder.y + right_shoulder.y) / 2 - 0.03
    return to_pixel(x, y, image_w, image_h)

# CVA 계산 (각도)
def calculate_cva(tragus_xy, c7_xy):
    dx = tragus_xy[0] - c7_xy[0]
    dy = tragus_xy[1] - c7_xy[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = abs(math.degrees(angle_rad))
    return angle_deg

# 귀–어깨 수평 거리 계산
def calculate_hpd(tragus_xy, shoulder_xy):
    return abs(tragus_xy[0] - shoulder_xy[0])  # 수평 거리 (픽셀)

# 통합 진단
def evaluate_posture(cva, hpd):
    if cva < 43 or hpd > 80:
        return "Severe Forward Head Posture!", (0, 0, 255)
    elif cva < 48 or hpd > 60:
        return "Mild Forward Head Posture.", (0, 165, 255)
    else:
        return "Posture looks normal.", (0, 255, 0)

# 웹캠 시작
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

        # 랜드마크 추출
        left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR]
        right_eye = lm[mp_pose.PoseLandmark.RIGHT_EYE]

        # 보정된 위치 계산
        tragus_xy = estimate_tragus_xy_fixed(right_ear, w, h)
        c7_xy = estimate_c7_xy(left_shoulder, right_shoulder, w, h)
        right_shoulder_xy = to_pixel(right_shoulder.x, right_shoulder.y, w, h)

        # 분석
        cva = calculate_cva(tragus_xy, c7_xy)
        hpd = calculate_hpd(tragus_xy, right_shoulder_xy)

        # 통합 진단
        msg, color = evaluate_posture(cva, hpd)

        # 시각화
        cv2.circle(frame, tragus_xy, 5, (255, 0, 0), -1)  # 파란색 - tragus
        cv2.circle(frame, c7_xy, 5, (0, 255, 0), -1)      # 초록색 - C7
        cv2.line(frame, tragus_xy, c7_xy, (0, 255, 255), 2)  # 노란색 라인

        cv2.putText(frame, f'CVA: {cva:.1f} deg', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(frame, f'HPD: {hpd:.0f} px', (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 255), 2)
        cv2.putText(frame, msg, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('CVA + HPD Posture Diagnosis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
