import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import os
import sys

# 상위 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.posture_analyzer import PostureAnalyzer

class RealtimePoseGradeTester:
    def __init__(self, model_path="pose_grade_model.pkl"):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.analyzer = PostureAnalyzer()
        self.model = None
        self.feature_names = []
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """훈련된 모델 로드"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.feature_names = model_data['feature_names']
            print(f"모델이 로드되었습니다: {model_path}")
        except FileNotFoundError:
            print(f"모델 파일을 찾을 수 없습니다: {model_path}")
            print("먼저 모델을 훈련해주세요.")
            return False
        except Exception as e:
            print(f"모델 로드 중 오류: {e}")
            return False
        return True
        
    def extract_features(self, landmarks, view_angle="1"):
        """특성 추출"""
        try:
            # 자세 분석 수행
            neck_analysis = self.analyzer.analyze_turtle_neck_detailed(landmarks)
            spine_analysis = self.analyzer.analyze_spine_curvature(landmarks)
            shoulder_analysis = self.analyzer.analyze_shoulder_asymmetry(landmarks)
            pelvic_analysis = self.analyzer.analyze_pelvic_tilt(landmarks)
            
            # 기본 특성
            basic_features = [
                neck_analysis['neck_angle'],
                spine_analysis['spine_angle'],
                shoulder_analysis['height_difference'],
                pelvic_analysis['height_difference']
            ]
            
            # 시점 정보 (원-핫 인코딩)
            view_features = [1 if view_angle == '1' else 0,  # 정면
                           1 if view_angle == '2' else 0]   # 측면
            
            # 목 관련 추가 특성
            neck_features = [
                neck_analysis['vertical_deviation'],
                neck_analysis['neck_angle']
            ]
            
            # 척추 관련 추가 특성
            spine_features = [
                spine_analysis['spine_angle'],
                1 if spine_analysis['is_hunched'] else 0
            ]
            
            # 어깨 관련 추가 특성
            shoulder_features = [
                shoulder_analysis['height_difference'],
                shoulder_analysis['shoulder_angle'],
                1 if shoulder_analysis['is_asymmetric'] else 0
            ]
            
            # 골반 관련 추가 특성
            pelvic_features = [
                pelvic_analysis['height_difference'],
                pelvic_analysis['pelvic_angle'],
                1 if pelvic_analysis['is_tilted'] else 0
            ]
            
            # 모든 특성 결합
            all_features = (basic_features + view_features + neck_features + 
                          spine_features + shoulder_features + pelvic_features)
            
            return np.array(all_features)
            
        except Exception as e:
            print(f"특성 추출 중 오류: {e}")
            return None
            
    def predict_grade(self, features):
        """등급 예측"""
        if self.model is None or features is None:
            return None, 0.0
            
        try:
            # 예측 수행
            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]
            confidence = max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            print(f"예측 중 오류: {e}")
            return None, 0.0
            
    def draw_pose_landmarks(self, frame, landmarks):
        """포즈 랜드마크 그리기"""
        h, w, _ = frame.shape
        
        # 주요 랜드마크 그리기
        key_points = [
            self.mp_pose.PoseLandmark.LEFT_EAR,
            self.mp_pose.PoseLandmark.RIGHT_EAR,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP
        ]
        
        for point in key_points:
            landmark = landmarks[point]
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
        # 목-어깨-골반 연결선 그리기
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # 귀 중앙
        ear_center = (int((left_ear.x + right_ear.x) * w / 2), 
                     int((left_ear.y + right_ear.y) * h / 2))
        # 어깨 중앙
        shoulder_center = (int((left_shoulder.x + right_shoulder.x) * w / 2), 
                          int((left_shoulder.y + right_shoulder.y) * h / 2))
        # 골반 중앙
        hip_center = (int((left_hip.x + right_hip.x) * w / 2), 
                     int((left_hip.y + right_hip.y) * h / 2))
        
        # 연결선 그리기
        cv2.line(frame, ear_center, shoulder_center, (255, 0, 0), 2)
        cv2.line(frame, shoulder_center, hip_center, (255, 0, 0), 2)
        
    def draw_grade_info(self, frame, grade, confidence, analysis, view_angle="1"):
        """등급 정보 화면에 표시"""
        # 배경 박스
        cv2.rectangle(frame, (10, 10), (450, 280), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, 280), (255, 255, 255), 2)
        
        # 시점 정보
        view_name = "정면" if view_angle == "1" else "측면"
        cv2.putText(frame, f"View: {view_name}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 등급별 색상
        grade_colors = {
            'a': (0, 255, 0),    # 녹색 - 완벽한 자세
            'b': (0, 255, 255),  # 노란색 - 양호한 자세
            'c': (0, 165, 255),  # 주황색 - 보통 자세
            'd': (0, 0, 255),    # 빨간색 - 나쁜 자세
            'e': (128, 0, 128)   # 보라색 - 특수 자세
        }
        
        color = grade_colors.get(grade, (255, 255, 255))
        
        # 등급 표시
        grade_names = {
            'a': 'A등급 (완벽)',
            'b': 'B등급 (양호)',
            'c': 'C등급 (보통)',
            'd': 'D등급 (나쁨)',
            'e': '특수 자세'
        }
        grade_name = grade_names.get(grade, grade)
        cv2.putText(frame, f"Grade: {grade_name}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # 신뢰도 표시
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 상세 분석 정보
        if analysis:
            cv2.putText(frame, f"Neck Angle: {analysis['neck']['neck_angle']:.1f}°", 
                       (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Spine Angle: {analysis['spine']['spine_angle']:.1f}°", 
                       (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Shoulder Diff: {analysis['shoulder']['height_difference']:.3f}", 
                       (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Pelvic Diff: {analysis['pelvic']['height_difference']:.3f}", 
                       (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 키보드 안내
        cv2.putText(frame, "Press '1'/'2' to change view, 'q' to quit", (20, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    def run_realtime_test(self):
        """실시간 테스트 실행"""
        if self.model is None:
            print("모델이 로드되지 않았습니다.")
            return
            
        cap = cv2.VideoCapture(0)
        current_view = "1"  # 기본값: 정면
        
        print("실시간 자세 등급 테스트를 시작합니다.")
        print("'1': 정면, '2': 측면, 'q': 종료")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # BGR to RGB 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 특성 추출
                features = self.extract_features(landmarks, current_view)
                
                if features is not None:
                    # 등급 예측
                    grade, confidence = self.predict_grade(features)
                    
                    # 자세 분석 (표시용)
                    analysis = {
                        'neck': self.analyzer.analyze_turtle_neck_detailed(landmarks),
                        'spine': self.analyzer.analyze_spine_curvature(landmarks),
                        'shoulder': self.analyzer.analyze_shoulder_asymmetry(landmarks),
                        'pelvic': self.analyzer.analyze_pelvic_tilt(landmarks)
                    }
                    
                    # 화면에 정보 표시
                    self.draw_grade_info(frame, grade, confidence, analysis, current_view)
                    
                    # 포즈 랜드마크 그리기
                    self.draw_pose_landmarks(frame, landmarks)
            
            cv2.imshow('Real-time Pose Grade Test', frame)
            
            # 키보드 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                current_view = "1"
                print("시점 변경: 정면")
            elif key == ord('2'):
                current_view = "2"
                print("시점 변경: 측면")
                
        cap.release()
        cv2.destroyAllWindows()

def main():
    tester = RealtimePoseGradeTester()
    tester.run_realtime_test()

if __name__ == "__main__":
    main() 