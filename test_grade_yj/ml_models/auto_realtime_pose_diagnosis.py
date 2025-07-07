import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from PIL import Image, ImageDraw, ImageFont
import os

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 등급 정보
GRADE_INFO = {
    'a': ('A등급 (완벽)', (0, 255, 0), '완벽한 자세입니다!'),
    'b': ('B등급 (양호)', (0, 255, 255), '양호한 자세입니다.'),
    'c': ('C등급 (보통)', (0, 165, 255), '보통 자세입니다. 개선이 필요합니다.'),
    'd': ('D등급 (나쁨)', (0, 0, 255), '나쁜 자세입니다. 즉시 교정하세요!'),
    'e': ('특수 자세', (128, 0, 128), '특수한 자세입니다.'),
}

# 한글 폰트 설정
def get_korean_font(size=32):
    """한글 폰트 로드"""
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Arial Unicode MS.ttf"  # macOS
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
    
    # 폰트를 찾지 못한 경우 기본 폰트 사용
    return ImageFont.load_default()

def put_korean_text(img, text, position, font_size=32, color=(255, 255, 255), thickness=2):
    """한글 텍스트를 이미지에 그리기"""
    try:
        # PIL 이미지로 변환
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 폰트 로드
        font = get_korean_font(font_size)
        
        # 텍스트 그리기
        draw.text(position, text, font=font, fill=color[::-1])  # RGB to BGR
        
        # OpenCV 이미지로 다시 변환
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"한글 텍스트 렌더링 오류: {e}")
        # 폴백: 기본 OpenCV 텍스트 사용
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_size/32, color, thickness)
        return img

def load_pose_grade_model(model_path):
    """모델 로딩"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['feature_names']

def extract_features_from_landmarks(landmarks):
    """실시간 랜드마크에서 특성 추출 - 정면/측면 자동 감지"""
    try:
        # 좌표 추출
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

        # 각도 계산 함수
        def calc_angle(a, b, c):
            a = np.array([a.x, a.y])
            b = np.array([b.x, b.y])
            c = np.array([c.x, c.y])
            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            if angle > 180.0:
                angle = 360 - angle
            return angle

        # 시점 자동 감지 (어깨의 x좌표 차이로 판단)
        shoulder_x_diff = abs(left_shoulder.x - right_shoulder.x)
        is_front_view = shoulder_x_diff > 0.5  # 어깨가 충분히 분리되어 있으면 정면 (덜 민감하게)
        
        # 시점 정보 (원-핫 인코딩)
        view_features = [1 if is_front_view else 0,  # 정면
                        1 if not is_front_view else 0]  # 측면

        if is_front_view:
            # 정면 자세 분석
            # 목 각도 (측면에서 측정하는 것이 정확하므로 근사값 사용)
            neck_angle = 90  # 정면에서는 목 각도 측정이 어려움
            
            # 척추 각도 (측면에서 측정하는 것이 정확하므로 근사값 사용)
            spine_angle = 180  # 정면에서는 척추 각도 측정이 어려움
            
            # 어깨 비대칭성 (정면에서 정확히 측정 가능)
            shoulder_asymmetry = abs(left_shoulder.y - right_shoulder.y)
            
            # 골반 기울기 (정면에서 정확히 측정 가능)
            pelvic_tilt = abs(left_hip.y - right_hip.y)
            
        else:
            # 측면 자세 분석
            # 목 각도 (측면에서 정확히 측정 가능)
            neck_angle = calc_angle(left_ear, left_shoulder, left_elbow)
            
            # 척추 각도 (측면에서 정확히 측정 가능)
            spine_angle = calc_angle(left_shoulder, left_hip, left_knee)
            
            # 어깨 비대칭성 (측면에서는 측정 어려움)
            shoulder_asymmetry = 0
            
            # 골반 기울기 (측면에서는 측정 어려움)
            pelvic_tilt = 0

        # 분석 결과 시뮬레이션 (train_grade_model2.py와 동일한 구조)
        analysis = {
            'neck': {
                'vertical_deviation': abs(neck_angle - 90),
                'neck_angle': neck_angle - 90,
                'grade': 'A' if abs(neck_angle - 90) < 10 else 'C'
            },
            'spine': {
                'spine_angle': spine_angle - 180,
                'is_hunched': spine_angle < 160
            },
            'shoulder': {
                'height_difference': shoulder_asymmetry,
                'shoulder_angle': 0,
                'is_asymmetric': shoulder_asymmetry > 0.05
            },
            'pelvic': {
                'height_difference': pelvic_tilt,
                'pelvic_angle': 0,
                'is_tilted': pelvic_tilt > 0.05
            }
        }

        # 목 관련 추가 특성
        neck_features = [
            analysis['neck']['vertical_deviation'],
            analysis['neck']['neck_angle']
        ]

        # 척추 관련 추가 특성
        spine_features = [
            analysis['spine']['spine_angle'],
            1 if analysis['spine']['is_hunched'] else 0
        ]

        # 어깨 관련 추가 특성
        shoulder_features = [
            analysis['shoulder']['height_difference'],
            analysis['shoulder']['shoulder_angle'],
            1 if analysis['shoulder']['is_asymmetric'] else 0
        ]

        # 골반 관련 추가 특성
        pelvic_features = [
            analysis['pelvic']['height_difference'],
            analysis['pelvic']['pelvic_angle'],
            1 if analysis['pelvic']['is_tilted'] else 0
        ]

        # 파생 특성들 (train_grade_model2.py와 동일)
        derived_features = [
            abs(analysis['neck']['neck_angle']) * abs(analysis['spine']['spine_angle']),
            analysis['shoulder']['height_difference'] + analysis['pelvic']['height_difference'],
            100 - (abs(analysis['neck']['neck_angle']) * 2 +
                   abs(analysis['spine']['spine_angle']) * 2 +
                   analysis['shoulder']['height_difference'] * 1000 +
                   analysis['pelvic']['height_difference'] * 1000),
            analysis['neck']['neck_angle'] ** 2,
            analysis['spine']['spine_angle'] ** 2,
            analysis['shoulder']['height_difference'] ** 2,
            analysis['pelvic']['height_difference'] ** 2,
            (1 if analysis['spine']['is_hunched'] else 0) +
            (1 if analysis['shoulder']['is_asymmetric'] else 0) +
            (1 if analysis['pelvic']['is_tilted'] else 0) +
            (1 if analysis['neck']['grade'] in ['C', 'D'] else 0)
        ]

        # 모든 특성 결합 (train_grade_model2.py와 동일한 순서)
        all_features = (
            [neck_angle, spine_angle, shoulder_asymmetry, pelvic_tilt] +
            view_features + neck_features + spine_features +
            shoulder_features + pelvic_features + derived_features
        )

        return np.array(all_features), is_front_view
    except Exception as e:
        print(f"특성 추출 오류: {e}")
        return None, None

def main():
    print("=== 카메라 연결 시도 시작 ===")
    
    # 모델 로딩
    try:
        model, feature_names = load_pose_grade_model("ml_models/pose_grade_model.pkl")
        print("✅ 모델 로딩 성공")
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return
    
    # 카메라 연결 시도
    cap = None
    for camera_index in [0, 1, 2]:
        print(f"카메라 인덱스 {camera_index} 시도 중...")
        cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            print(f"✅ 카메라 {camera_index}에 연결 성공!")
            # 카메라 정보 출력
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"   해상도: {width}x{height}, FPS: {fps}")
            break
        else:
            print(f"❌ 카메라 {camera_index} 연결 실패")
            cap.release()
    
    if cap is None or not cap.isOpened():
        print("❌ 모든 카메라 연결 실패!")
        print("확인사항:")
        print("1. 카메라가 물리적으로 연결되어 있는지 확인")
        print("2. 다른 프로그램에서 카메라를 사용 중인지 확인")
        print("3. 카메라 권한이 있는지 확인")
        print("4. v4l2-ctl --list-devices 명령으로 카메라 확인")
        return
    
    # 해상도 설정
    print("해상도를 640x480으로 설정 중...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 설정된 해상도 확인
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"실제 설정된 해상도: {actual_width}x{actual_height}")
    
    print("=== MediaPipe 초기화 중 ===")
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        print("✅ MediaPipe 초기화 완료")
        print("실시간 자세 등급 인식 시작 (q: 종료)")
        
        # 윈도우 크기 설정
        cv2.namedWindow("Real-time Pose Grade Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Real-time Pose Grade Detection", 800, 600)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"❌ 프레임 읽기 실패 (프레임 {frame_count})")
                break
            
            frame_count += 1
            if frame_count % 30 == 0:  # 30프레임마다 상태 출력
                print(f"프레임 처리 중... ({frame_count}프레임)")
            
            # BGR → RGB 변환
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # 포즈 추론 실행
            results = pose.process(image)
            
            # 결과 표시를 위해 다시 BGR로 변환
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                # 포즈 랜드마크 그리기
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                # 특성 추출 및 등급 예측
                features, is_front_view = extract_features_from_landmarks(results.pose_landmarks.landmark)
                if features is not None:
                    try:
                        # 모델 예측
                        pred = model.predict([features])[0]
                        prob = model.predict_proba([features])[0]
                        grade_name, color, grade_description = GRADE_INFO.get(pred, ('Unknown', (128, 128, 128), '알 수 없는 자세'))
                        confidence = np.max(prob) * 100
                        
                        # 메인 등급 표시 박스 (크기 조정)
                        cv2.rectangle(image, (10, 10), (350, 120), (0, 0, 0), -1)
                        cv2.rectangle(image, (10, 10), (350, 120), color, 3)
                        
                        # 등급 표시 (폰트 크기 조정)
                        image = put_korean_text(image, f"자세 등급: {grade_name}", (20, 40), font_size=24, color=color)
                        image = put_korean_text(image, f"신뢰도: {confidence:.1f}%", (20, 70), font_size=18, color=(255,255,255))
                        image = put_korean_text(image, grade_description, (20, 95), font_size=14, color=(255,255,255))
                        
                        # 시점 정보 표시
                        view_text = "정면 자세" if is_front_view else "측면 자세"
                        view_color = (0, 255, 255) if is_front_view else (255, 165, 0)
                        image = put_korean_text(image, f"감지된 시점: {view_text}", (20, 175), font_size=16, color=view_color)
                        
                        # 추가 분석 정보 표시 (크기 조정)
                        cv2.rectangle(image, (10, 200), (280, 240), (50, 50, 50), -1)
                        image = put_korean_text(image, f"목 각도: {features[0]:.1f}°", (20, 220), font_size=14, color=(255,255,255))
                        image = put_korean_text(image, f"척추 각도: {features[1]:.1f}°", (20, 235), font_size=14, color=(255,255,255))
                        
                        # 등급별 색상으로 프레임 테두리 표시
                        cv2.rectangle(image, (0, 0), (image.shape[1]-1, image.shape[0]-1), color, 3)
                        
                    except Exception as e:
                        cv2.rectangle(image, (10, 10), (300, 80), (0, 0, 0), -1)
                        image = put_korean_text(image, f"예측 오류: {str(e)[:30]}", (20, 50), font_size=16, color=(0,0,255))
                else:
                    cv2.rectangle(image, (10, 10), (250, 60), (0, 0, 0), -1)
                    image = put_korean_text(image, "특성 추출 실패", (20, 40), font_size=16, color=(0,0,255))
            else:
                # 포즈가 감지되지 않을 때 안내 메시지
                cv2.rectangle(image, (10, 10), (400, 100), (0, 0, 0), -1)
                image = put_korean_text(image, "No pose detected", (20, 40), font_size=20, color=(0, 0, 255))
                image = put_korean_text(image, "Please stand in front of camera", (20, 70), font_size=16, color=(0, 0, 255))
            
            cv2.imshow("Real-time Pose Grade Detection", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    print("프로그램 종료 중...")
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 프로그램 종료 완료")

if __name__ == "__main__":
    main()