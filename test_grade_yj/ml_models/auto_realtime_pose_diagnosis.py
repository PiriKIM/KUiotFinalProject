import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 등급 정보
GRADE_INFO = {
    'a': ('A등급 (완벽)', (0, 255, 0)),
    'b': ('B등급 (양호)', (0, 255, 255)),
    'c': ('C등급 (보통)', (0, 165, 255)),
    'd': ('D등급 (나쁨)', (0, 0, 255)),
    'e': ('특수 자세', (128, 0, 128)),
}

def load_pose_grade_model(model_path):
    """모델 로딩"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['feature_names']

def extract_features_from_landmarks(landmarks):
    """실시간 랜드마크에서 특성 추출 - train_grade_model2.py와 동일한 방식"""
    try:
        # 좌표 추출
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]

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

        # 기본 특성 계산
        neck_angle = calc_angle(left_ear, left_shoulder, left_elbow)
        spine_angle = calc_angle(left_shoulder, left_hip, left_knee)
        shoulder_asymmetry = abs(left_shoulder.y - right_shoulder.y)
        pelvic_tilt = abs(left_hip.y - right_hip.y)

        # 시점 정보 (측면 가정)
        view_features = [0, 1]  # 측면

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

        return np.array(all_features)
    except Exception as e:
        print(f"특성 추출 오류: {e}")
        return None

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
    print("해상도를 1280x720으로 설정 중...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 설정된 해상도 확인
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"실제 설정된 해상도: {actual_width}x{actual_height}")
    
    print("=== MediaPipe 초기화 중 ===")
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        print("✅ MediaPipe 초기화 완료")
        print("실시간 자세 등급 인식 시작 (q: 종료)")
        
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
                features = extract_features_from_landmarks(results.pose_landmarks.landmark)
                if features is not None:
                    try:
                        # 모델 예측
                        pred = model.predict([features])[0]
                        prob = model.predict_proba([features])[0]
                        grade_name, color = GRADE_INFO.get(pred, ('Unknown', (128, 128, 128)))
                        confidence = np.max(prob) * 100
                        
                        # 결과 표시
                        cv2.rectangle(image, (10, 10), (400, 120), (0, 0, 0), -1)
                        cv2.putText(image, f"{grade_name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                        cv2.putText(image, f"신뢰도: {confidence:.1f}%", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                        
                        # 추가 정보 표시
                        cv2.putText(image, f"Landmarks: {len(results.pose_landmarks.landmark)}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        
                    except Exception as e:
                        cv2.putText(image, f"예측 오류: {str(e)[:30]}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                else:
                    cv2.putText(image, "특성 추출 실패", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                # 포즈가 감지되지 않을 때 안내 메시지
                cv2.putText(image, "No pose detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "Please stand in front of camera", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow("Real-time Pose Grade Detection", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    print("프로그램 종료 중...")
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 프로그램 종료 완료")

if __name__ == "__main__":
    main()