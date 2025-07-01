# ===============================================
# 📌 이 코드는 pose_classifier_robust.py 기반입니다.
#
# ✅ 이유:
# - 실시간 예측에 적합하도록 경량화된 특징 추출 구조 사용
# - 전체 관절 대신 상체 중심 랜드마크만 사용 (0~12번)
# - 특징 수를 약 28개로 제한하여 예측 속도 향상
# - 정규화된 좌표, 어깨 대칭, 어깨 비율 등 핵심 피처만 활용
# - RandomForest + Scaler 모델을 Pickle 형태로 저장·로드하여 효율적인 재사용 가능
#
# ❌ 참고로 pose_classification_model.py는:
# - 고정밀 분석용으로 특징 수가 73개 이상이며,
# - 전체 랜드마크 + 다양한 각도, 비율, 대칭성 포함
# - 실시간 시스템에는 과적합 또는 느린 처리 속도 발생 가능성 있음
#
# 따라서 본 실시간 분류 시스템에는 pose_classifier_robust.py 구조가 더 적합합니다.
# ===============================================


import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 전역 폰트 변수import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from PIL import Image, ImageDraw, ImageFont
import warnings
warnings.filterwarnings('ignore')

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 전역 폰트 변수
_global_font = None
_font_loaded = False

def get_korean_font(font_size=20):
    """한글 폰트를 한 번만 로드하여 재사용"""
    global _global_font, _font_loaded
    
    if _font_loaded and _global_font:
        return _global_font
    
    # 한글 폰트 로드 (여러 폰트 경로 시도)
    font_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Arial.ttf",  # macOS
        "C:/Windows/Fonts/malgun.ttf",      # Windows
        "/home/yj/kuuniv.bigdata2025/opencv/data/NanumPenScript-Regular.ttf"  # 사용자 지정 경로
    ]
    
    for font_path in font_paths:
        try:
            _global_font = ImageFont.truetype(font_path, font_size)
            _font_loaded = True
            print(f"한글 폰트 로드 성공: {font_path}")
            break
        except Exception as e:
            continue
    
    if not _font_loaded:
        # 폰트 로드 실패 시 기본 폰트 사용
        _global_font = ImageFont.load_default()
        print("기본 폰트 사용")
        _font_loaded = True
    
    return _global_font

def put_korean_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """PIL을 사용한 한글 텍스트 렌더링 함수"""
    try:
        # PIL 이미지로 변환
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 폰트 가져오기
        font = get_korean_font(font_size)
        
        # 색상을 RGB로 변환 (PIL은 RGB 사용)
        color_rgb = (color[2], color[1], color[0])  # BGR to RGB
        
        # 텍스트 그리기
        draw.text(position, text, font=font, fill=color_rgb)
        
        # OpenCV 이미지로 변환
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_cv
        
    except Exception as e:
        print(f"PIL 한글 텍스트 렌더링 오류: {e}")
        # 오류 시 기본 OpenCV 텍스트 사용
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

class RealTimePoseClassifier:
    def __init__(self, model_path='pose_classifier_model.pkl'):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.load_model(model_path)
        
    def normalize_coordinates(self, landmarks):
        """어깨 중심 기준으로 좌표 정규화"""
        normalized = []
        
        # 어깨 중심점 계산 (랜드마크 11, 12)
        if landmarks[11] != -1 and landmarks[12] != -1:
            shoulder_center_x = (landmarks[11] + landmarks[12]) / 2
            shoulder_center_y = (landmarks[11+1] + landmarks[12+1]) / 2
        else:
            shoulder_center_x, shoulder_center_y = 0, 0
            
        # 어깨 너비로 정규화
        if landmarks[11] != -1 and landmarks[12] != -1:
            shoulder_width = abs(landmarks[11] - landmarks[12])
        else:
            shoulder_width = 1.0
            
        # 각 랜드마크를 어깨 중심 기준으로 정규화
        for i in range(0, len(landmarks), 2):
            if landmarks[i] != -1 and landmarks[i+1] != -1:
                norm_x = (landmarks[i] - shoulder_center_x) / max(shoulder_width, 0.001)
                norm_y = (landmarks[i+1] - shoulder_center_y) / max(shoulder_width, 0.001)
                normalized.extend([norm_x, norm_y])
            else:
                normalized.extend([0, 0])
                
        return normalized
    
    def calculate_angles(self, landmarks):
        """주요 각도 계산"""
        angles = []
        
        # 목-어깨-팔꿈치 각도 (랜드마크 0-11-12)
        if landmarks[0] != -1 and landmarks[11] != -1 and landmarks[12] != -1:
            dx1 = landmarks[11] - landmarks[0]
            dy1 = landmarks[11+1] - landmarks[0+1]
            dx2 = landmarks[12] - landmarks[11]
            dy2 = landmarks[12+1] - landmarks[11+1]
            
            dot_product = dx1*dx2 + dy1*dy2
            mag1 = np.sqrt(dx1*dx1 + dy1*dy1)
            mag2 = np.sqrt(dx2*dx2 + dy2*dy2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(np.degrees(angle))
            else:
                angles.append(0)
        else:
            angles.append(0)
        
        return angles
    
    def extract_features(self, landmarks):
        """특징 추출"""
        features = []
        
        # 1. 정규화된 좌표 (중요한 부위만 선택)
        normalized_coords = self.normalize_coordinates(landmarks)
        
        # 머리, 목, 어깨 부위만 선택 (랜드마크 0-12)
        important_landmarks = []
        for i in range(0, 26, 2):  # 0-12번 랜드마크만
            important_landmarks.extend([normalized_coords[i], normalized_coords[i+1]])
        
        features.extend(important_landmarks)
        
        # 2. 각도 특징
        angles = self.calculate_angles(landmarks)
        features.extend(angles)
        
        # 3. 어깨 비율
        if landmarks[11] != -1 and landmarks[12] != -1:
            shoulder_width = abs(landmarks[11] - landmarks[12])
            shoulder_height = abs(landmarks[11+1] - landmarks[12+1])
            shoulder_ratio = shoulder_width / max(shoulder_height, 0.001)
            features.append(shoulder_ratio)
        else:
            features.append(1.0)
        
        # 4. 대칭성 특징
        if landmarks[11] != -1 and landmarks[12] != -1:
            shoulder_symmetry = abs(landmarks[11+1] - landmarks[12+1])
            features.append(shoulder_symmetry)
        else:
            features.append(0)
        
        return features
    
    def load_model(self, model_path):
        """모델 로드"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            
            print(f"모델이 {model_path}에서 로드되었습니다.")
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            print("먼저 모델을 학습시켜주세요.")
            self.is_trained = False
    
    def predict(self, landmarks):
        """실시간 예측"""
        if not self.is_trained or self.model is None or self.scaler is None:
            return None, None
        
        try:
            features = self.extract_features(landmarks)
            features_scaled = self.scaler.transform([features])
            
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            return prediction, probability
        except Exception as e:
            print(f"예측 오류: {e}")
            return None, None

def landmarks_to_array(landmarks):
    """MediaPipe landmarks를 배열로 변환"""
    landmarks_array = []
    for landmark in landmarks:
        landmarks_array.extend([landmark.x, landmark.y])
    return landmarks_array

def main():
    print("실시간 자세 분류 시스템 시작...")
    
    # 자세 분류기 초기화
    classifier = RealTimePoseClassifier()
    
    if not classifier.is_trained:
        print("모델이 로드되지 않았습니다. 먼저 모델을 학습시켜주세요.")
        return
    
    # 웹캠 열기 (여러 카메라 시도)
    cap = None
    for camera_id in [0, 1, 2]:
        try:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                print(f"카메라 {camera_id} 연결 성공")
                break
            else:
                cap.release()
        except Exception as e:
            print(f"카메라 {camera_id} 연결 실패: {e}")
            if cap:
                cap.release()
    
    if cap is None or not cap.isOpened():
        print("사용 가능한 카메라를 찾을 수 없습니다.")
        return
    
    # 카메라 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 창 생성 및 설정
    window_name = '실시간 자세 분류 시스템'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    # MediaPipe Pose 초기화
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        print("웹캠이 시작되었습니다. ESC 키를 눌러 종료하세요.")
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            frame_count += 1
            
            # BGR → RGB 변환
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # 포즈 추론 실행
            results = pose.process(image)
            
            # 결과 표시를 위해 다시 BGR로 변환
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 좌우반전 (거울 모드) - 랜드마크 그리기 전에 수행
            image = cv2.flip(image, 1)
            
            # 화면에 기본 정보 표시
            h, w, _ = image.shape
            
            # 제목 표시
            title_text = "실시간 자세 분류 시스템"
            image = put_korean_text(image, title_text, (10, 30), 24, (255, 255, 255))
            
            # 관절 랜드마크 그리기
            if results.pose_landmarks:
                # 랜드마크를 배열로 변환
                landmarks_array = landmarks_to_array(results.pose_landmarks.landmark)
                
                # 자세 분류 예측
                prediction, probability = classifier.predict(landmarks_array)
                
                if prediction is not None and probability is not None:
                    # 예측 결과 표시
                    pose_type = "정면" if prediction == 1 else "측면"
                    confidence = max(probability)
                    
                    # 결과 텍스트
                    result_text = f"자세: {pose_type}"
                    confidence_text = f"신뢰도: {confidence:.1%}"
                    
                    # 색상 설정 (정면: 초록, 측면: 파랑)
                    color = (0, 255, 0) if prediction == 1 else (255, 0, 0)
                    
                    # 화면에 결과 표시
                    image = put_korean_text(image, result_text, (10, 70), 32, color)
                    image = put_korean_text(image, confidence_text, (10, 110), 20, (255, 255, 255))
                    
                    # 상세 확률 정보
                    front_prob = probability[0] if len(probability) > 0 else 0
                    side_prob = probability[1] if len(probability) > 1 else 0
                    
                    detail_text = f"정면 확률: {front_prob:.1%}"
                    image = put_korean_text(image, detail_text, (10, 140), 16, (0, 255, 0))
                    
                    detail_text2 = f"측면 확률: {side_prob:.1%}"
                    image = put_korean_text(image, detail_text2, (10, 160), 16, (255, 0, 0))
                    
                    # 콘솔에 결과 출력 (1초마다)
                    current_time = time.time()
                    if not hasattr(main, 'last_print_time'):
                        main.last_print_time = 0
                    
                    if current_time - main.last_print_time >= 1.0:  # 1초마다 출력
                        print(f"[{time.strftime('%H:%M:%S')}] 자세: {pose_type} (신뢰도: {confidence:.1%})")
                        print(f"  - 정면 확률: {front_prob:.1%}")
                        print(f"  - 측면 확률: {side_prob:.1%}")
                        
                        # 주요 랜드마크 좌표 출력
                        print("  - 주요 랜드마크 좌표:")
                        key_landmarks = [
                            (0, "NOSE", "코"),
                            (7, "LEFT_EAR", "왼쪽 귀"),
                            (8, "RIGHT_EAR", "오른쪽 귀"),
                            (11, "LEFT_SHOULDER", "왼쪽 어깨"),
                            (12, "RIGHT_SHOULDER", "오른쪽 어깨"),
                            (23, "LEFT_HIP", "왼쪽 엉덩이"),
                            (24, "RIGHT_HIP", "오른쪽 엉덩이")
                        ]
                        
                        for idx, name, korean in key_landmarks:
                            if idx * 2 < len(landmarks_array):
                                x = landmarks_array[idx * 2]
                                y = landmarks_array[idx * 2 + 1]
                                print(f"    {name:15s} ({korean:8s}): x={x:.3f}, y={y:.3f}")
                        
                        print("-" * 40)
                        main.last_print_time = current_time
                
                # 랜드마크 그리기 - 좌우반전된 이미지에 맞춰 좌표 변환
                # 좌우반전을 고려하여 x좌표를 반전
                flipped_landmarks = results.pose_landmarks
                for landmark in flipped_landmarks.landmark:
                    # x좌표를 반전 (1 - x)
                    landmark.x = 1.0 - landmark.x
                
                # 반전된 랜드마크로 그리기
                mp_drawing.draw_landmarks(
                    image, 
                    flipped_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # 어깨 거리 정보 표시 (디버깅용) - 반전된 좌표 사용
                left_shoulder = flipped_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = flipped_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                shoulder_distance = abs(left_shoulder.x - right_shoulder.x)
                distance_text = f"어깨 거리: {shoulder_distance:.3f}"
                image = put_korean_text(image, distance_text, (10, 200), 14, (255, 255, 255))
                
            else:
                # 사람이 감지되지 않을 때
                no_person_text = "사람이 감지되지 않습니다"
                image = put_korean_text(image, no_person_text, (10, 70), 24, (0, 0, 255))
                
                # 콘솔에 상태 출력 (1초마다)
                current_time = time.time()
                if not hasattr(main, 'last_print_time'):
                    main.last_print_time = 0
                
                if current_time - main.last_print_time >= 1.0:
                    print(f"[{time.strftime('%H:%M:%S')}] 사람이 감지되지 않습니다")
                    print("-" * 40)
                    main.last_print_time = current_time
            
            # 프레임 번호 표시 (디버깅용)
            frame_text = f"프레임: {frame_count}"
            image = put_korean_text(image, frame_text, (w-200, 30), 16, (255, 255, 255))
            
            # 사용법 안내
            guide_text = "ESC: 종료"
            image = put_korean_text(image, guide_text, (w-150, h-30), 16, (255, 255, 255))
            
            # 화면에 출력
            try:
                cv2.imshow(window_name, image)
                print(f"프레임 {frame_count} 표시 완료")
            except Exception as e:
                print(f"화면 표시 오류: {e}")
                break
            
            # ESC 키로 종료
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('q'):  # q 키
                break
    
    # 정리
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("시스템이 종료되었습니다.")

if __name__ == "__main__":
    main() 
_global_font = None
_font_loaded = False

def get_korean_font(font_size=20):
    """한글 폰트를 한 번만 로드하여 재사용"""
    global _global_font, _font_loaded
    
    if _font_loaded and _global_font:
        return _global_font
    
    # 한글 폰트 로드 (여러 폰트 경로 시도)
    font_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Arial.ttf",  # macOS
        "C:/Windows/Fonts/malgun.ttf",      # Windows
        "/home/woo/kuBig2025/opencv/data/NanumPenScript-Regular.ttf"  # 사용자 지정 경로
    ]
    
    for font_path in font_paths:
        try:
            _global_font = ImageFont.truetype(font_path, font_size)
            _font_loaded = True
            print(f"한글 폰트 로드 성공: {font_path}")
            break
        except Exception as e:
            continue
    
    if not _font_loaded:
        # 폰트 로드 실패 시 기본 폰트 사용
        _global_font = ImageFont.load_default()
        print("기본 폰트 사용")
        _font_loaded = True
    
    return _global_font

def put_korean_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """PIL을 사용한 한글 텍스트 렌더링 함수"""
    try:
        # PIL 이미지로 변환
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 폰트 가져오기
        font = get_korean_font(font_size)
        
        # 색상을 RGB로 변환 (PIL은 RGB 사용)
        color_rgb = (color[2], color[1], color[0])  # BGR to RGB
        
        # 텍스트 그리기
        draw.text(position, text, font=font, fill=color_rgb)
        
        # OpenCV 이미지로 변환
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_cv
        
    except Exception as e:
        print(f"PIL 한글 텍스트 렌더링 오류: {e}")
        # 오류 시 기본 OpenCV 텍스트 사용
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

class RealTimePoseClassifier:
    def __init__(self, model_path='pose_classifier_model.pkl'):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.load_model(model_path)
        
    def normalize_coordinates(self, landmarks):
        """어깨 중심 기준으로 좌표 정규화"""
        normalized = []
        
        # 어깨 중심점 계산 (랜드마크 11, 12)
        if landmarks[11] != -1 and landmarks[12] != -1:
            shoulder_center_x = (landmarks[11] + landmarks[12]) / 2
            shoulder_center_y = (landmarks[11+1] + landmarks[12+1]) / 2
        else:
            shoulder_center_x, shoulder_center_y = 0, 0
            
        # 어깨 너비로 정규화
        if landmarks[11] != -1 and landmarks[12] != -1:
            shoulder_width = abs(landmarks[11] - landmarks[12])
        else:
            shoulder_width = 1.0
            
        # 각 랜드마크를 어깨 중심 기준으로 정규화
        for i in range(0, len(landmarks), 2):
            if landmarks[i] != -1 and landmarks[i+1] != -1:
                norm_x = (landmarks[i] - shoulder_center_x) / max(shoulder_width, 0.001)
                norm_y = (landmarks[i+1] - shoulder_center_y) / max(shoulder_width, 0.001)
                normalized.extend([norm_x, norm_y])
            else:
                normalized.extend([0, 0])
                
        return normalized
    
    def calculate_angles(self, landmarks):
        """주요 각도 계산"""
        angles = []
        
        # 목-어깨-팔꿈치 각도 (랜드마크 0-11-12)
        if landmarks[0] != -1 and landmarks[11] != -1 and landmarks[12] != -1:
            dx1 = landmarks[11] - landmarks[0]
            dy1 = landmarks[11+1] - landmarks[0+1]
            dx2 = landmarks[12] - landmarks[11]
            dy2 = landmarks[12+1] - landmarks[11+1]
            
            dot_product = dx1*dx2 + dy1*dy2
            mag1 = np.sqrt(dx1*dx1 + dy1*dy1)
            mag2 = np.sqrt(dx2*dx2 + dy2*dy2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                angles.append(np.degrees(angle))
            else:
                angles.append(0)
        else:
            angles.append(0)
        
        return angles
    
    def extract_features(self, landmarks):
        """특징 추출"""
        features = []
        
        # 1. 정규화된 좌표 (중요한 부위만 선택)
        normalized_coords = self.normalize_coordinates(landmarks)
        
        # 머리, 목, 어깨 부위만 선택 (랜드마크 0-12)
        important_landmarks = []
        for i in range(0, 26, 2):  # 0-12번 랜드마크만
            important_landmarks.extend([normalized_coords[i], normalized_coords[i+1]])
        
        features.extend(important_landmarks)
        
        # 2. 각도 특징
        angles = self.calculate_angles(landmarks)
        features.extend(angles)
        
        # 3. 어깨 비율
        if landmarks[11] != -1 and landmarks[12] != -1:
            shoulder_width = abs(landmarks[11] - landmarks[12])
            shoulder_height = abs(landmarks[11+1] - landmarks[12+1])
            shoulder_ratio = shoulder_width / max(shoulder_height, 0.001)
            features.append(shoulder_ratio)
        else:
            features.append(1.0)
        
        # 4. 대칭성 특징
        if landmarks[11] != -1 and landmarks[12] != -1:
            shoulder_symmetry = abs(landmarks[11+1] - landmarks[12+1])
            features.append(shoulder_symmetry)
        else:
            features.append(0)
        
        return features
    
    def load_model(self, model_path):
        """모델 로드"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            
            print(f"모델이 {model_path}에서 로드되었습니다.")
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            print("먼저 모델을 학습시켜주세요.")
            self.is_trained = False
    
    def predict(self, landmarks):
        """실시간 예측"""
        if not self.is_trained or self.model is None or self.scaler is None:
            return None, None
        
        try:
            features = self.extract_features(landmarks)
            features_scaled = self.scaler.transform([features])
            
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            return prediction, probability
        except Exception as e:
            print(f"예측 오류: {e}")
            return None, None

def landmarks_to_array(landmarks):
    """MediaPipe landmarks를 배열로 변환"""
    landmarks_array = []
    for landmark in landmarks:
        landmarks_array.extend([landmark.x, landmark.y])
    return landmarks_array

def main():
    print("실시간 자세 분류 시스템 시작...")
    
    # 자세 분류기 초기화
    classifier = RealTimePoseClassifier()
    
    if not classifier.is_trained:
        print("모델이 로드되지 않았습니다. 먼저 모델을 학습시켜주세요.")
        return
    
    # 웹캠 열기 (여러 카메라 시도)
    cap = None
    for camera_id in [0, 1, 2]:
        try:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                print(f"카메라 {camera_id} 연결 성공")
                break
            else:
                cap.release()
        except Exception as e:
            print(f"카메라 {camera_id} 연결 실패: {e}")
            if cap:
                cap.release()
    
    if cap is None or not cap.isOpened():
        print("사용 가능한 카메라를 찾을 수 없습니다.")
        return
    
    # 카메라 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # 창 생성 및 설정
    window_name = '실시간 자세 분류 시스템'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    # MediaPipe Pose 초기화
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        print("웹캠이 시작되었습니다. ESC 키를 눌러 종료하세요.")
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            frame_count += 1
            
            # BGR → RGB 변환
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # 포즈 추론 실행
            results = pose.process(image)
            
            # 결과 표시를 위해 다시 BGR로 변환
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 좌우반전 (거울 모드) - 랜드마크 그리기 전에 수행
            image = cv2.flip(image, 1)
            
            # 화면에 기본 정보 표시
            h, w, _ = image.shape
            
            # 제목 표시
            title_text = "실시간 자세 분류 시스템"
            image = put_korean_text(image, title_text, (10, 30), 24, (255, 255, 255))
            
            # 관절 랜드마크 그리기
            if results.pose_landmarks:
                # 랜드마크를 배열로 변환
                landmarks_array = landmarks_to_array(results.pose_landmarks.landmark)
                
                # 자세 분류 예측
                prediction, probability = classifier.predict(landmarks_array)
                
                if prediction is not None and probability is not None:
                    # 예측 결과 표시
                    pose_type = "정면" if prediction == 1 else "측면"
                    confidence = max(probability)
                    
                    # 결과 텍스트
                    result_text = f"자세: {pose_type}"
                    confidence_text = f"신뢰도: {confidence:.1%}"
                    
                    # 색상 설정 (정면: 초록, 측면: 파랑)
                    color = (0, 255, 0) if prediction == 1 else (255, 0, 0)
                    
                    # 화면에 결과 표시
                    image = put_korean_text(image, result_text, (10, 70), 32, color)
                    image = put_korean_text(image, confidence_text, (10, 110), 20, (255, 255, 255))
                    
                    # 상세 확률 정보
                    front_prob = probability[0] if len(probability) > 0 else 0
                    side_prob = probability[1] if len(probability) > 1 else 0
                    
                    detail_text = f"정면 확률: {front_prob:.1%}"
                    image = put_korean_text(image, detail_text, (10, 140), 16, (0, 255, 0))
                    
                    detail_text2 = f"측면 확률: {side_prob:.1%}"
                    image = put_korean_text(image, detail_text2, (10, 160), 16, (255, 0, 0))
                    
                    # 콘솔에 결과 출력 (1초마다)
                    current_time = time.time()
                    if not hasattr(main, 'last_print_time'):
                        main.last_print_time = 0
                    
                    if current_time - main.last_print_time >= 1.0:  # 1초마다 출력
                        print(f"[{time.strftime('%H:%M:%S')}] 자세: {pose_type} (신뢰도: {confidence:.1%})")
                        print(f"  - 정면 확률: {front_prob:.1%}")
                        print(f"  - 측면 확률: {side_prob:.1%}")
                        
                        # 주요 랜드마크 좌표 출력
                        print("  - 주요 랜드마크 좌표:")
                        key_landmarks = [
                            (0, "NOSE", "코"),
                            (7, "LEFT_EAR", "왼쪽 귀"),
                            (8, "RIGHT_EAR", "오른쪽 귀"),
                            (11, "LEFT_SHOULDER", "왼쪽 어깨"),
                            (12, "RIGHT_SHOULDER", "오른쪽 어깨"),
                            (23, "LEFT_HIP", "왼쪽 엉덩이"),
                            (24, "RIGHT_HIP", "오른쪽 엉덩이")
                        ]
                        
                        for idx, name, korean in key_landmarks:
                            if idx * 2 < len(landmarks_array):
                                x = landmarks_array[idx * 2]
                                y = landmarks_array[idx * 2 + 1]
                                print(f"    {name:15s} ({korean:8s}): x={x:.3f}, y={y:.3f}")
                        
                        print("-" * 40)
                        main.last_print_time = current_time
                
                # 랜드마크 그리기 - 좌우반전된 이미지에 맞춰 좌표 변환
                # 좌우반전을 고려하여 x좌표를 반전
                flipped_landmarks = results.pose_landmarks
                for landmark in flipped_landmarks.landmark:
                    # x좌표를 반전 (1 - x)
                    landmark.x = 1.0 - landmark.x
                
                # 반전된 랜드마크로 그리기
                mp_drawing.draw_landmarks(
                    image, 
                    flipped_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # 어깨 거리 정보 표시 (디버깅용) - 반전된 좌표 사용
                left_shoulder = flipped_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = flipped_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                shoulder_distance = abs(left_shoulder.x - right_shoulder.x)
                distance_text = f"어깨 거리: {shoulder_distance:.3f}"
                image = put_korean_text(image, distance_text, (10, 200), 14, (255, 255, 255))
                
            else:
                # 사람이 감지되지 않을 때
                no_person_text = "사람이 감지되지 않습니다"
                image = put_korean_text(image, no_person_text, (10, 70), 24, (0, 0, 255))
                
                # 콘솔에 상태 출력 (1초마다)
                current_time = time.time()
                if not hasattr(main, 'last_print_time'):
                    main.last_print_time = 0
                
                if current_time - main.last_print_time >= 1.0:
                    print(f"[{time.strftime('%H:%M:%S')}] 사람이 감지되지 않습니다")
                    print("-" * 40)
                    main.last_print_time = current_time
            
            # 프레임 번호 표시 (디버깅용)
            frame_text = f"프레임: {frame_count}"
            image = put_korean_text(image, frame_text, (w-200, 30), 16, (255, 255, 255))
            
            # 사용법 안내
            guide_text = "ESC: 종료"
            image = put_korean_text(image, guide_text, (w-150, h-30), 16, (255, 255, 255))
            
            # 화면에 출력
            try:
                cv2.imshow(window_name, image)
                print(f"프레임 {frame_count} 표시 완료")
            except Exception as e:
                print(f"화면 표시 오류: {e}")
                break
            
            # ESC 키로 종료
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('q'):  # q 키
                break
    
    # 정리
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    print("시스템이 종료되었습니다.")

if __name__ == "__main__":
    main() 