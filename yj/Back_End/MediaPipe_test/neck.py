import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
import os

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 자세 분석 클래스
class PostureAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
         
    def calculate_angle(self, a, b, c):
        """세 점으로 각도 계산"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def calculate_neck_angle(self, landmarks):
        """목 각도 계산 - 귀-어깨-수직선 각도"""
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # 귀 중심점
        ear_center_x = (left_ear.x + right_ear.x) / 2
        ear_center_y = (left_ear.y + right_ear.y) / 2
        
        # 어깨 중심점
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # 수직선 기준점 (어깨와 같은 x좌표, 귀와 같은 y좌표)
        vertical_x = shoulder_center_x
        vertical_y = ear_center_y
        
        # 목 각도 계산 (귀-어깨-수직선)
        a = np.array([ear_center_x, ear_center_y])
        b = np.array([shoulder_center_x, shoulder_center_y])
        c = np.array([vertical_x, vertical_y])
        
        # 각도 계산
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def grade_neck_posture(self, neck_angle):
        """목 각도에 따른 자세 등급 판정"""
        # 목 각도가 작을수록 좋은 자세 (수직에 가까울수록)
        if neck_angle <= 5:  # 5도 이하: 완벽한 자세
            return 'A', "완벽한 자세"
        elif neck_angle <= 10:  # 5-10도: 양호한 자세
            return 'B', "양호한 자세"
        elif neck_angle <= 15:  # 10-15도: 보통 자세
            return 'C', "보통 자세"
        else:  # 15도 이상: 나쁜 자세
            return 'D', "나쁜 자세"
    
    def analyze_turtle_neck_detailed(self, landmarks):
        """거북목 상세 분석 - 목 각도 기반"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
        
        # 목 각도 계산
        neck_angle = self.calculate_neck_angle(landmarks)
        
        # 등급 판정
        grade, grade_description = self.grade_neck_posture(neck_angle)
        
        # 목 중심점 계산
        neck_top_x = (left_ear.x + right_ear.x) / 2
        neck_top_y = (left_ear.y + right_ear.y) / 2
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # 수직 이탈도 계산
        vertical_deviation = abs(neck_top_x - shoulder_center_x)
        
        return {
            'neck_angle': neck_angle,
            'grade': grade,
            'grade_description': grade_description,
            'vertical_deviation': vertical_deviation,
            'neck_top': (neck_top_x, neck_top_y),
            'shoulder_center': (shoulder_center_x, shoulder_center_y)
        }
    
    def calculate_distance(self, point1, point2):
        """두 점 간의 거리 계산"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def analyze_spine_curvature(self, landmarks):
        """척추 굴곡 분석 - 등이 굽은 상태 확인 (더 세밀한 분석)"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        
        # 어깨와 엉덩이 중심점
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_center_x = (left_hip.x + right_hip.x) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        
        # 척추 각도 계산 (어깨-엉덩이 수직선과의 각도)
        spine_angle = math.degrees(math.atan2(hip_center_x - shoulder_center_x, 
                                            hip_center_y - shoulder_center_y))
        
        # 척추 굴곡 판정 (더 세밀한 기준)
        is_hunched = abs(spine_angle) > 12  # 12도 이상 기울어지면 굽은 상태
        
        # 척추 중간점 계산 (어깨와 골반 중간)
        spine_mid_x = (shoulder_center_x + hip_center_x) / 2
        spine_mid_y = (shoulder_center_y + hip_center_y) / 2
        
        return {
            'is_hunched': is_hunched,
            'spine_angle': spine_angle,
            'shoulder_center': (shoulder_center_x, shoulder_center_y),
            'hip_center': (hip_center_x, hip_center_y),
            'spine_mid': (spine_mid_x, spine_mid_y)
        }
    
    def analyze_shoulder_asymmetry(self, landmarks):
        """어깨선 분석 - 어깨의 기울기와 비대칭 확인 (더 세밀한 분석)"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        
        # 어깨 높이 차이 계산
        shoulder_height_diff = abs(left_shoulder.y - right_shoulder.y)
        
        # 어깨 기울기 계산 (수평선과의 각도)
        shoulder_angle = math.degrees(math.atan2(right_shoulder.y - left_shoulder.y, 
                                               right_shoulder.x - left_shoulder.x))
        
        # 어깨 비대칭 판정 (더 세밀한 기준)
        is_asymmetric = shoulder_height_diff > 0.02  # 2% 이상 차이나면 비대칭
        
        # 어느 쪽이 높은지 판정
        if left_shoulder.y < right_shoulder.y:
            higher_shoulder = "왼쪽"
        else:
            higher_shoulder = "오른쪽"
        
        # 어깨선 중간점
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        
        return {
            'is_asymmetric': is_asymmetric,
            'height_difference': shoulder_height_diff,
            'shoulder_angle': shoulder_angle,
            'higher_shoulder': higher_shoulder,
            'shoulder_mid': (shoulder_mid_x, shoulder_mid_y),
            'left_shoulder': (left_shoulder.x, left_shoulder.y),
            'right_shoulder': (right_shoulder.x, right_shoulder.y)
        }
    
    def analyze_pelvic_tilt(self, landmarks):
        """골반 관절 분석 - 골반의 기울기와 비대칭 확인 (더 세밀한 분석)"""
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        
        # 골반 높이 차이 계산
        hip_height_diff = abs(left_hip.y - right_hip.y)
        
        # 골반 기울기 계산 (수평선과의 각도)
        pelvic_angle = math.degrees(math.atan2(right_hip.y - left_hip.y, 
                                             right_hip.x - left_hip.x))
        
        # 골반 기울어짐 판정 (더 세밀한 기준)
        is_tilted = hip_height_diff > 0.015  # 1.5% 이상 차이나면 기울어짐
        
        # 어느 쪽이 높은지 판정
        if left_hip.y < right_hip.y:
            higher_hip = "왼쪽"
        else:
            higher_hip = "오른쪽"
        
        # 골반 중앙 좌표 계산
        pelvic_center_x = (left_hip.x + right_hip.x) / 2
        pelvic_center_y = (left_hip.y + right_hip.y) / 2
        
        # 골반-무릎 연결선 각도 (측면 자세 확인용)
        left_hip_knee_angle = math.degrees(math.atan2(left_knee.y - left_hip.y, 
                                                    left_knee.x - left_hip.x))
        right_hip_knee_angle = math.degrees(math.atan2(right_knee.y - right_hip.y, 
                                                     right_knee.x - right_hip.x))
        
        return {
            'is_tilted': is_tilted,
            'height_difference': hip_height_diff,
            'pelvic_angle': pelvic_angle,
            'higher_hip': higher_hip,
            'pelvic_center': (pelvic_center_x, pelvic_center_y),
            'left_hip': (left_hip.x, left_hip.y),
            'right_hip': (right_hip.x, right_hip.y),
            'left_hip_knee_angle': left_hip_knee_angle,
            'right_hip_knee_angle': right_hip_knee_angle
        }
    
    def analyze_spine_twisting(self, landmarks):
        """척추 틀어짐 분석 - 측면에서 본 척추 곡률 확인"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
        
        # 어깨와 골반의 수평 정렬 확인
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        hip_center_x = (left_hip.x + right_hip.x) / 2
        hip_center_y = (left_hip.y + right_hip.y) / 2
        
        # 척추 수직 정렬도 계산
        spine_alignment = abs(shoulder_center_x - hip_center_x)
        
        # 척추 틀어짐 판정
        is_twisted = spine_alignment > 0.03  # 3% 이상 어긋나면 틀어짐
        
        # 측면 자세 분석 (귀-어깨-골반 수직선)
        ear_center_x = (left_ear.x + right_ear.x) / 2
        ear_center_y = (left_ear.y + right_ear.y) / 2
        
        # 측면 척추 각도
        side_spine_angle = math.degrees(math.atan2(hip_center_x - ear_center_x, 
                                                 hip_center_y - ear_center_y))
        
        return {
            'is_twisted': is_twisted,
            'spine_alignment': spine_alignment,
            'side_spine_angle': side_spine_angle,
            'ear_center': (ear_center_x, ear_center_y),
            'shoulder_center': (shoulder_center_x, (left_shoulder.y + right_shoulder.y) / 2),
            'hip_center': (hip_center_x, hip_center_y)
        }

def put_english_text(img, text, position, font_size=0.8, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
    return img

# 자세 분석기 초기화
posture_analyzer = PostureAnalyzer()

# 웹캠 열기 (여러 인덱스 시도)
cap = None
for camera_index in [0, 1, 2]:
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        print(f"웹캠 {camera_index}에 연결되었습니다.")
        break
    else:
        cap.release()

if cap is None or not cap.isOpened():
    print("웹캠에 연결할 수 없습니다. 테스트용 비디오 파일을 사용하거나 웹캠 권한을 확인해주세요.")
    print("테스트를 위해 샘플 비디오 파일을 다운로드하거나 웹캠 설정을 확인해주세요.")
    exit(1)

# 고해상도 설정으로 더 넓은 영역 캡처
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# 현재는 기본 해상도로 설정됨 (일반적으로 640x480 또는 1280x720)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR → RGB 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 포즈 추론 실행
        results = pose.process(image)

        # 결과 표시를 위해 다시 BGR로 변환
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 좌우반전 (거울 모드)
        image = cv2.flip(image, 1)
        
        # 화면 크기 조정 (더 크게 표시)
        scale_factor = 0.9  # 90% 크기로 조정
        h, w, _ = image.shape
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        image = cv2.resize(image, (new_w, new_h))

        # 관절 랜드마크 그리기 (얼굴 제외, 상체만)
        if results.pose_landmarks:
            # 상체 랜드마크 그리기 (얼굴 제외, 귀는 포함)
            upper_body_landmarks = []
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                # 얼굴 랜드마크 제외하되 귀는 포함 (귀: 7, 8번, 어깨부터: 11번)
                if i >= 7:  # 귀부터 시작 (7: LEFT_EAR, 8: RIGHT_EAR, 11: 어깨)
                    upper_body_landmarks.append(landmark)
            
            # 상체 랜드마크만 그리기
            if upper_body_landmarks:
                # 상체 연결선 정의 (얼굴 제외, 더 많은 관절 포함)
                upper_body_connections = [
                    # 어깨선
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                    
                    # 팔 연결선
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
                    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
                    
                    # 몸통 연결선
                    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                    
                    # 골반선
                    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
                    
                    # 다리 연결선
                    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
                    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
                    
                    # 목 연결선 (귀-어깨)
                    (mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.LEFT_SHOULDER),
                    (mp_pose.PoseLandmark.RIGHT_EAR, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                ]
                
                # 상체 랜드마크 그리기
                for connection in upper_body_connections:
                    start_idx = connection[0].value
                    end_idx = connection[1].value
                    
                    if start_idx >= 7 and end_idx >= 7:  # 귀부터 시작 (얼굴 제외)
                        start_point = results.pose_landmarks.landmark[start_idx]
                        end_point = results.pose_landmarks.landmark[end_idx]
                        
                        # 축소된 화면 크기 사용
                        h, w, _ = image.shape
                        # 좌우반전을 고려한 좌표 변환 (x좌표를 반전)
                        start_pixel = (int((1 - start_point.x) * w), int(start_point.y * h))
                        end_pixel = (int((1 - end_point.x) * w), int(end_point.y * h))
                        
                        cv2.line(image, start_pixel, end_pixel, (0, 255, 0), 2)
                
                # 상체 랜드마크 점 그리기
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    if i >= 7:  # 귀부터 시작
                        # 축소된 화면 크기 사용
                        h, w, _ = image.shape
                        # 좌우반전을 고려한 좌표 변환 (x좌표를 반전)
                        pixel = (int((1 - landmark.x) * w), int(landmark.y * h))
                        cv2.circle(image, pixel, 3, (255, 0, 0), -1)

            # 자세 분석 실행 (거북목 중점)
            turtle_neck_result = posture_analyzer.analyze_turtle_neck_detailed(results.pose_landmarks.landmark)
            spine_result = posture_analyzer.analyze_spine_curvature(results.pose_landmarks.landmark)
            shoulder_result = posture_analyzer.analyze_shoulder_asymmetry(results.pose_landmarks.landmark)
            pelvic_result = posture_analyzer.analyze_pelvic_tilt(results.pose_landmarks.landmark)
            spine_twisting_result = posture_analyzer.analyze_spine_twisting(results.pose_landmarks.landmark)

            # 등급에 따른 색상 설정
            grade_colors = {
                'A': (0, 255, 0),    # 초록색
                'B': (0, 255, 255),  # 노란색
                'C': (0, 165, 255),  # 주황색
                'D': (0, 0, 255)     # 빨간색
            }
            
            # 거북목 등급 표시 (큰 글씨로 중앙에)
            grade = turtle_neck_result['grade']
            grade_color = grade_colors[grade]
            
            # Grade display (move to top-right)
            h, w, _ = image.shape
            # Estimate text width for alignment (approximate, since OpenCV doesn't provide exact width for all fonts)
            grade_text = f"Grade: {grade}"
            description_text = turtle_neck_result['grade_description']
            # Use cv2.getTextSize to get width
            (grade_text_width, _), _ = cv2.getTextSize(grade_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            (desc_text_width, _), _ = cv2.getTextSize(description_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            # Set right margin
            right_margin = 20
            image = put_english_text(image, grade_text, (w - grade_text_width - right_margin, 40), 1.2, grade_color, 3)
            image = put_english_text(image, description_text, (w - desc_text_width - right_margin, 80), 0.9, grade_color, 2)

            # 분석 결과를 화면에 표시 (한글)
            y_offset = 30
            
            # 거북목 상태 (목 각도 포함)
            neck_angle_text = f"Turtle Neck: {turtle_neck_result['grade_description']} (Angle: {turtle_neck_result['neck_angle']:.1f} deg)"
            image = put_english_text(image, neck_angle_text, (10, y_offset), 0.7, grade_color, 2)
            
            # 척추 굴곡 상태
            y_offset += 40
            spine_color = (0, 0, 255) if spine_result['is_hunched'] else (0, 255, 0)
            spine_text = f"Spine Curvature: {'Detected' if spine_result['is_hunched'] else 'Normal'}"
            image = put_english_text(image, spine_text, (10, y_offset), 0.7, spine_color, 2)
            
            # 어깨 비대칭 상태
            y_offset += 40
            shoulder_color = (0, 0, 255) if shoulder_result['is_asymmetric'] else (0, 255, 0)
            shoulder_text = f"Shoulder Asymmetry: {'Detected' if shoulder_result['is_asymmetric'] else 'Normal'}"
            image = put_english_text(image, shoulder_text, (10, y_offset), 0.7, shoulder_color, 2)
            
            # 골반 기울어짐 상태
            y_offset += 40
            pelvic_color = (0, 0, 255) if pelvic_result['is_tilted'] else (0, 255, 0)
            pelvic_text = f"Pelvic Tilt: {'Detected' if pelvic_result['is_tilted'] else 'Normal'}"
            image = put_english_text(image, pelvic_text, (10, y_offset), 0.7, pelvic_color, 2)
            
            # 척추 틀어짐 상태
            y_offset += 40
            twisting_color = (0, 0, 255) if spine_twisting_result['is_twisted'] else (0, 255, 0)
            twisting_text = f"Spine Twisting: {'Detected' if spine_twisting_result['is_twisted'] else 'Normal'}"
            image = put_english_text(image, twisting_text, (10, y_offset), 0.7, twisting_color, 2)

            # 분석선 그리기
            h, w, _ = image.shape  # 축소된 화면 크기 사용
            
            # 거북목 분석선 (목-어깨 수직선) - 좌우반전 고려
            head_x = int((1 - turtle_neck_result['neck_top'][0]) * w)
            head_y = int(turtle_neck_result['neck_top'][1] * h)
            shoulder_x = int((1 - turtle_neck_result['shoulder_center'][0]) * w)
            shoulder_y = int(turtle_neck_result['shoulder_center'][1] * h)
            
            # 목 각도 시각화 (수직선과 목선)
            vertical_x = shoulder_x
            vertical_y = head_y
            
            # 수직선 그리기
            cv2.line(image, (vertical_x, vertical_y), (shoulder_x, shoulder_y), (255, 255, 255), 2)
            # 목선 그리기
            cv2.line(image, (head_x, head_y), (shoulder_x, shoulder_y), grade_color, 3)
            
            # 각도 표시
            angle_text = f"{turtle_neck_result['neck_angle']:.1f}°"
            cv2.putText(image, angle_text, (vertical_x + 10, (vertical_y + shoulder_y) // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, grade_color, 2)
            
            # 척추 분석선 (어깨-골반) - 좌우반전 고려
            spine_shoulder_x = int((1 - spine_result['shoulder_center'][0]) * w)
            spine_shoulder_y = int(spine_result['shoulder_center'][1] * h)
            spine_hip_x = int((1 - spine_result['hip_center'][0]) * w)
            spine_hip_y = int(spine_result['hip_center'][1] * h)
            
            cv2.line(image, (spine_shoulder_x, spine_shoulder_y), (spine_hip_x, spine_hip_y), (0, 255, 255), 2)
            
            # 골반 중앙점 표시 - 좌우반전 고려
            pelvic_x = int((1 - pelvic_result['pelvic_center'][0]) * w)
            pelvic_y = int(pelvic_result['pelvic_center'][1] * h)
            cv2.circle(image, (pelvic_x, pelvic_y), 5, (255, 255, 0), -1)

            # 콘솔에 상세 정보 출력
            print(f"=== 거북목 분석 결과 ===")
            print(f"목 각도: {turtle_neck_result['neck_angle']:.1f}도")
            print(f"등급: {grade} - {turtle_neck_result['grade_description']}")
            print(f"수직 이탈도: {turtle_neck_result['vertical_deviation']:.3f}")
            print(f"척추 굴곡: {spine_result['is_hunched']} (각도: {spine_result['spine_angle']:.1f}도)")
            print(f"어깨 비대칭: {shoulder_result['is_asymmetric']} (차이: {shoulder_result['height_difference']:.3f})")
            print(f"골반 기울어짐: {pelvic_result['is_tilted']} (차이: {pelvic_result['height_difference']:.3f})")
            print(f"척추 틀어짐: {spine_twisting_result['is_twisted']} (차이: {spine_twisting_result['spine_alignment']:.3f})")
            print("=" * 30)
        else:
            # 포즈가 감지되지 않을 때 안내 메시지
            h, w, _ = image.shape
            put_english_text(image, "No person detected. Please stand in front of the camera.", (w//2 - 200, h//2), 0.8, (0, 255, 255), 2)

        # 화면에 출력
        cv2.imshow('Posture Analysis', image)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC 키 종료
            break

cap.release()
cv2.destroyAllWindows()
