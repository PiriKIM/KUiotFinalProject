#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실시간 자세 등급 피드백 시스템 (4way 모델 통합)

python3 realtime_posture_feedback_4way.py --csv data/results/side_analysis_p1/side_angle_analysis.csv

웹캠으로 실시간 영상을 받아서 자세를 분석하고 등급(A/B/C)과 피드백을 화면에 표시합니다.
4way 모델을 사용하여 측면을 자동으로 감지하고, 자세 등급을 매깁니다.
"""

import os
import sys

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
from PIL import Image, ImageDraw, ImageFont
import pickle
import time
import math

# 자체 모듈 import
sys.path.append('side_angle_analysis_folder')
from posture_grade_classifier import PostureGradeClassifier


class RealTimePoseClassifier4Way:
    def __init__(self, model_path='pose_classifier_4way_model.pkl'):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.classes = ['정면', '좌측면', '우측면']
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
        """특징 추출 (4way 모델과 동일한 구조)"""
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
        
        # 5. 어깨 방향 특징 (좌측면/우측면 구분용) - 4way 모델과 동일
        if landmarks[11] != -1 and landmarks[12] != -1:
            # 왼쪽 어깨가 더 위에 있는지 (좌측면 특징)
            left_shoulder_higher = landmarks[11+1] - landmarks[12+1]
            features.append(left_shoulder_higher)
            
            # 어깨의 x축 차이 (측면 구분용)
            shoulder_x_diff = landmarks[11] - landmarks[12]
            features.append(shoulder_x_diff)
        else:
            features.extend([0, 0])
        
        return features
    
    def load_model(self, model_path):
        """4way 모델 로드"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.classes = model_data.get('classes', ['정면', '좌측면', '우측면'])
            
            print(f"4way 모델이 {model_path}에서 로드되었습니다.")
            print(f"클래스: {self.classes}")
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            print("먼저 4way 모델을 학습시켜주세요.")
            self.is_trained = False
    
    def predict(self, landmarks):
        """실시간 예측 (3클래스)"""
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


class PostureAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def calculate_angle(self, a, b, c):
        """세 점으로 각도 계산"""
        a_pt = np.array([a.x, a.y]) if not isinstance(a, np.ndarray) else a
        b_pt = np.array([b.x, b.y]) if not isinstance(b, np.ndarray) else b
        c_pt = np.array([c.x, c.y]) if not isinstance(c, np.ndarray) else c

        ba = a_pt - b_pt
        bc = c_pt - b_pt

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def analyze_turtle_neck_detailed(self, landmarks):
        """목 자세 상세 분석"""
        left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # 목 중심점 계산
        neck_top_x = (left_ear.x + right_ear.x) / 2
        neck_top_y = (left_ear.y + right_ear.y) / 2
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # 목 각도 계산
        ear_center = np.array([neck_top_x, neck_top_y])
        shoulder_center = np.array([shoulder_center_x, shoulder_center_y])
        vertical = np.array([shoulder_center[0], ear_center[1]])
        neck_angle = self.calculate_angle(ear_center, shoulder_center, vertical)
        
        # 등급 분류
        grade, desc = self.grade_neck_posture(neck_angle)
        
        # 수직 이탈도 계산
        vertical_deviation = abs(neck_top_x - shoulder_center_x)
        
        return {
            'neck_angle': neck_angle,
            'grade': grade,
            'grade_description': desc,
            'vertical_deviation': vertical_deviation,
            'neck_top': (neck_top_x, neck_top_y),
            'shoulder_center': (shoulder_center_x, shoulder_center_y)
        }

    def grade_neck_posture(self, neck_angle):
        """목 자세 등급 분류"""
        if neck_angle <= 5:
            return 'A', "완벽한 자세"
        elif neck_angle <= 10:
            return 'B', "양호한 자세"
        elif neck_angle <= 15:
            return 'C', "보통 자세"
        else:
            return 'D', "나쁜 자세"

    def get_comprehensive_grade(self, landmarks):
        """종합 자세 등급 계산"""
        neck_analysis = self.analyze_turtle_neck_detailed(landmarks)
        
        # 목 자세 점수 계산
        neck_score = self.get_neck_score(neck_analysis['neck_angle'])
        
        # 종합 점수 (목 자세에 집중)
        total_score = neck_score
        
        # 종합 등급 결정
        if total_score >= 90:
            grade = 'A'
        elif total_score >= 80:
            grade = 'B'
        elif total_score >= 70:
            grade = 'C'
        else:
            grade = 'D'
        
        return {
            'total_grade': grade,
            'total_score': total_score,
            'neck_score': neck_score,
            'details': {
                'neck': neck_analysis
            }
        }

    def get_neck_score(self, neck_angle):
        """목 자세 점수 계산"""
        if neck_angle <= 5:
            return 100
        elif neck_angle <= 10:
            return 90
        elif neck_angle <= 15:
            return 80
        elif neck_angle <= 20:
            return 70
        else:
            return 60


def landmarks_to_array(landmarks):
    """MediaPipe landmarks를 배열로 변환"""
    landmarks_array = []
    for landmark in landmarks:
        landmarks_array.extend([landmark.x, landmark.y])
    return landmarks_array


def put_korean_text(img, text, position, font_size=32, color=(255, 255, 255), thickness=2):
    """
    한글 텍스트를 이미지에 그리는 함수
    NanumGothic 폰트를 사용합니다.
    """
    try:
        # NanumGothic 폰트 경로 (시스템에 따라 다를 수 있음)
        font_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
            "/System/Library/Fonts/NanumGothic.ttf",  # macOS
            "C:/Windows/Fonts/malgun.ttf",  # Windows
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"  # 대체 폰트
        ]
        
        font = None
        for font_path in font_paths:
            if Path(font_path).exists():
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue
        
        if font is None:
            # 폰트를 찾을 수 없으면 기본 폰트 사용
            font = ImageFont.load_default()
        
        # PIL 이미지 생성
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 텍스트 그리기
        draw.text(position, text, font=font, fill=color[::-1])  # BGR to RGB
        
        # OpenCV 이미지로 변환
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    except Exception as e:
        print(f"한글 텍스트 렌더링 오류: {e}")
        # 오류 발생 시 기본 OpenCV 텍스트 사용
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_size/32, color, thickness)
        return img


def calculate_grade_thresholds(csv_path: str):
    """CSV 파일에서 등급 분류 기준값들을 계산"""
    try:
        df = pd.read_csv(csv_path)
        cva_angles = df['cva_angle'].dropna().values
        
        if len(cva_angles) == 0:
            return None
        
        # 절댓값 기준으로 계산
        abs_angles = np.abs(cva_angles)
        min_abs = abs_angles.min()
        max_abs = abs_angles.max()
        angle_range = max_abs - min_abs
        
        # 10단계로 나누기
        if angle_range == 0:
            stages = np.ones_like(abs_angles, dtype=int)
        else:
            stages = ((abs_angles - min_abs) / angle_range * 9 + 1).astype(int)
            stages = np.clip(stages, 1, 10)
        
        # 1단계에 해당하는 각도들
        stage1_angles = abs_angles[stages == 1]
        stage1_threshold = np.percentile(stage1_angles, 50) if len(stage1_angles) > 0 else min_abs
        
        return {
            'min_abs': min_abs,
            'max_abs': max_abs,
            'stage1_threshold': stage1_threshold
        }
    except Exception as e:
        print(f"❌ 기준값 계산 오류: {e}")
        return None


def calculate_cva_angle(landmarks, side='right'):
    """
    Mediapipe 랜드마크에서 CVA 각도 계산 (수정된 버전)
    
    CVA = Cervical Vertebral Angle
    목-어깨 선과 수직선 사이의 각도
    
    Args:
        landmarks: Mediapipe pose landmarks
        side: 측면 ('right' 또는 'left')
    
    Returns:
        CVA 각도 (도)
    """
    if side == 'right':
        # 오른쪽 측면: 귀(8), 어깨(12)
        ear = np.array([landmarks[8].x, landmarks[8].y])
        shoulder = np.array([landmarks[12].x, landmarks[12].y])
    else:
        # 왼쪽 측면: 귀(7), 어깨(11)
        ear = np.array([landmarks[7].x, landmarks[7].y])
        shoulder = np.array([landmarks[11].x, landmarks[11].y])
    
    # 목-어깨 벡터 (어깨에서 귀로)
    neck_vector = ear - shoulder
    
    # 수직 벡터 (위쪽 방향)
    vertical_vector = np.array([0, -1])  # y축 음의 방향 (화면에서 위쪽)
    
    # 각도 계산
    dot_product = np.dot(neck_vector, vertical_vector)
    norm_neck = np.linalg.norm(neck_vector)
    norm_vertical = np.linalg.norm(vertical_vector)
    
    if norm_neck == 0:
        return 0.0
    
    cos_angle = np.clip(dot_product / (norm_neck * norm_vertical), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    # 목이 앞으로 기울어지면 양수, 뒤로 기울어지면 음수
    # x 좌표로 방향 판단 (귀가 어깨보다 앞에 있으면 양수)
    if neck_vector[0] > 0:  # 귀가 어깨보다 오른쪽(앞쪽)에 있으면
        angle_deg = angle_deg
    else:
        angle_deg = -angle_deg
    
    return angle_deg


def get_feedback_message(grade: str, cva_angle: float) -> tuple:
    """등급에 따른 피드백 메시지와 색상 반환"""
    if grade == 'A':
        message = "최고! 바른 자세입니다. 👍"
        color = (0, 255, 0)  # 초록색
    elif grade == 'B':
        message = "보통 자세입니다. 조금만 더 신경써보세요! 💪"
        color = (0, 255, 255)  # 노란색
    else:  # C
        message = "자세가 많이 무너졌어요! 바로잡으세요! ⚠️"
        color = (0, 0, 255)  # 빨간색
    
    return message, color


def detect_side_with_4way_model(landmarks, classifier):
    """
    4way 모델을 사용한 측면 감지
    """
    try:
        # 랜드마크를 배열로 변환
        landmarks_array = landmarks_to_array(landmarks)
        
        # 4way 모델로 예측
        prediction, probability = classifier.predict(landmarks_array)
        
        if prediction is not None and probability is not None:
            # 예측 결과에 따른 측면 결정
            if prediction == 0:  # 정면
                return None  # 정면일 때는 None 반환
            elif prediction == 1:  # 좌측면
                return 'left'
            elif prediction == 2:  # 우측면
                return 'right'
            else:
                return 'right'  # 기본값
        else:
            # 예측 실패 시 기본값
            return 'right'
            
    except Exception as e:
        print(f"4way 모델 측면 감지 오류: {e}")
        # 오류 발생 시 기본 가시성 기반 감지로 fallback
        right_ear_visibility = landmarks[8].visibility
        left_ear_visibility = landmarks[7].visibility
        right_shoulder_visibility = landmarks[12].visibility
        left_shoulder_visibility = landmarks[11].visibility
        
        right_score = right_ear_visibility + right_shoulder_visibility
        left_score = left_ear_visibility + left_shoulder_visibility
        
        if right_score > left_score:
            return 'right'
        else:
            return 'left'


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='실시간 자세 등급 피드백 시스템 (4way 모델 통합)')
    parser.add_argument('--right-csv', '-r', 
                       help='오른쪽 측면 기준 데이터 CSV 파일 경로')
    parser.add_argument('--left-csv', '-l', 
                       help='왼쪽 측면 기준 데이터 CSV 파일 경로')
    parser.add_argument('--csv', '-c', 
                       help='기준 데이터 CSV 파일 경로 (측면별로 자동 선택됨)')
    parser.add_argument('--side', '-s', choices=['right', 'left'], default='right',
                       help='측면 (right 또는 left)')
    parser.add_argument('--camera', type=int, default=0,
                       help='카메라 인덱스 (기본값: 0)')
    parser.add_argument('--model', '-m', default='pose_classifier_4way_model.pkl',
                       help='4way 모델 파일 경로')
    
    args = parser.parse_args()
    
    # 4way 모델 초기화
    print(f"🎯 4way 모델을 로드하고 있습니다...")
    classifier = RealTimePoseClassifier4Way(args.model)
    
    if not classifier.is_trained:
        print("❌ 4way 모델이 로드되지 않았습니다. 먼저 모델을 학습시켜주세요.")
        return 1
    
    # 측면별 CSV 파일 경로 설정
    if args.right_csv and args.left_csv:
        # 두 CSV 파일이 직접 입력된 경우
        right_csv_paths = [args.right_csv]
        left_csv_paths = [args.left_csv]
        print(f"🎯 직접 입력된 측면별 CSV 파일:")
        print(f"   오른쪽: {args.right_csv}")
        print(f"   왼쪽: {args.left_csv}")
    elif args.csv:
        # 하나의 CSV 파일이 입력된 경우 (기존 방식)
        csv_base_path = Path(args.csv)
        
        # 측면별로 다른 CSV 파일 사용
        if 'right' in str(csv_base_path) or 'p1' in str(csv_base_path):
            # 오른쪽 측면용 CSV 파일들
            right_csv_paths = [
                "data/results/side_analysis_p1/side_angle_analysis.csv",
                "data/posture_grades/posture_grades_right.csv",
                "data/landmarks_p1/raw_landmarks.csv"
            ]
            left_csv_paths = [
                "data/results/side_analysis_p2/side_angle_analysis.csv", 
                "data/posture_grades/posture_grades_left.csv",
                "data/landmarks_p2/raw_landmarks.csv"
            ]
        else:
            # 기본 경로 사용
            right_csv_paths = [str(csv_base_path)]
            left_csv_paths = [str(csv_base_path)]
    else:
        print("❌ CSV 파일을 지정해주세요.")
        print("   방법 1: --right-csv와 --left-csv로 각각 지정")
        print("   방법 2: --csv로 하나만 지정 (자동으로 측면별 파일 찾기)")
        return 1
    
    # 사용 가능한 CSV 파일 찾기
    def find_available_csv(csv_paths):
        for path in csv_paths:
            if Path(path).exists():
                return path
        return None
    
    right_csv = find_available_csv(right_csv_paths)
    left_csv = find_available_csv(left_csv_paths)
    
    if not right_csv or not left_csv:
        print(f"❌ 측면별 CSV 파일을 찾을 수 없습니다.")
        print(f"   오른쪽: {right_csv}")
        print(f"   왼쪽: {left_csv}")
        return 1
    
    print(f"📊 측면별 기준 데이터:")
    print(f"   오른쪽: {right_csv}")
    print(f"   왼쪽: {left_csv}")
    
    # 측면별 기준값 계산
    print(f"\n📊 기준값을 계산하고 있습니다...")
    right_thresholds = calculate_grade_thresholds(right_csv)
    left_thresholds = calculate_grade_thresholds(left_csv)
    
    if right_thresholds is None or left_thresholds is None:
        print("❌ 기준값 계산 실패")
        return 1
    
    print(f"✅ 기준값 계산 완료!")
    print(f"  오른쪽 - min_abs: {right_thresholds['min_abs']:.2f}, max_abs: {right_thresholds['max_abs']:.2f}")
    print(f"  왼쪽 - min_abs: {left_thresholds['min_abs']:.2f}, max_abs: {left_thresholds['max_abs']:.2f}")
    
    # 자세 분석기 초기화
    posture_analyzer = PostureAnalyzer()
    
    # 등급 분류기 초기화
    posture_classifier = PostureGradeClassifier()
    
    # Mediapipe 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # 카메라 초기화
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ 카메라를 열 수 없습니다. 카메라 인덱스 {args.camera}를 확인해주세요.")
        return 1

    # OpenCV 창 명시적으로 생성
    cv2.namedWindow('Real-time Posture Feedback (4way 통합)', cv2.WINDOW_NORMAL)
    
    print(f"\n🎥 카메라가 시작되었습니다.")
    print(f"💡 ESC 키를 누르면 종료됩니다.")
    
    frame_count = 0
    grade_history = []
    pose_history = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임을 읽을 수 없습니다.")
            break
        
        # 프레임을 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        # 화면에 기본 정보 표시
        frame = put_korean_text(frame, f"Camera: {args.camera} | 4way 모델 통합", 
                               (10, 30), font_size=24, color=(255, 255, 255))
        
        if results.pose_landmarks:
            # 4way 모델을 사용한 자동 측면 감지 (매 프레임마다)
            detected_side = detect_side_with_4way_model(results.pose_landmarks.landmark, classifier)
            if frame_count == 0:
                print(f"🎯 4way 모델로 감지된 측면: {detected_side}")
            elif frame_count % 30 == 0:  # 30프레임마다 측면 정보 출력
                print(f"🎯 측면 감지: {detected_side}")
            
            # 측면에 따른 기준값 선택
            if detected_side is not None:
                current_thresholds = right_thresholds if detected_side == 'right' else left_thresholds
                
                # CVA 각도 계산 (측면일 때만)
                cva_angle = calculate_cva_angle(results.pose_landmarks.landmark, detected_side)
                
                # 각도 범위 제한 (비정상적인 값 필터링)
                if abs(cva_angle) > 90:
                    cva_angle = np.clip(cva_angle, -90, 90)
                
                # 등급 분류 (측면별 기준값 사용)
                grade = posture_classifier.get_grade_for_angle(
                    cva_angle, 
                    current_thresholds['min_abs'], 
                    current_thresholds['max_abs'], 
                    current_thresholds['stage1_threshold']
                )
            else:
                # 정면일 때는 CVA 각도 계산하지 않음
                cva_angle = None
                grade = None
            
            # 자세 등급 분석 (PostureAnalyzer 사용)
            comprehensive_grade = posture_analyzer.get_comprehensive_grade(results.pose_landmarks.landmark)
            posture_grade = comprehensive_grade['total_grade']
            posture_score = comprehensive_grade['total_score']
            
            # 디버깅 정보 (처음 5프레임에서만)
            if frame_count < 5:
                if detected_side == 'right':
                    ear = results.pose_landmarks.landmark[8]
                    shoulder = results.pose_landmarks.landmark[12]
                else:
                    ear = results.pose_landmarks.landmark[7]
                    shoulder = results.pose_landmarks.landmark[11]
                
                print(f"🔍 프레임 {frame_count}: 측면={detected_side}, 각도={cva_angle:.1f}°")
                print(f"   귀: ({ear.x:.3f}, {ear.y:.3f}), 가시성: {ear.visibility:.3f}")
                print(f"   어깨: ({shoulder.x:.3f}, {shoulder.y:.3f}), 가시성: {shoulder.visibility:.3f}")
                print(f"   사용 기준값: {current_thresholds['min_abs']:.2f} ~ {current_thresholds['max_abs']:.2f}")
                print(f"   자세 등급: {posture_grade} (점수: {posture_score:.1f})")
            
            # 등급 히스토리 업데이트 (최근 10프레임)
            if grade is not None:
                grade_history.append(grade)
                if len(grade_history) > 10:
                    grade_history.pop(0)
            
            # 자세 히스토리 업데이트 (최근 10프레임)
            pose_history.append(posture_grade)
            if len(pose_history) > 10:
                pose_history.pop(0)
            
            # 안정화된 등급 (최근 5프레임 중 가장 많은 등급)
            if len(grade_history) >= 5 and grade is not None:
                stable_grade = max(set(grade_history[-5:]), key=grade_history[-5:].count)
            else:
                stable_grade = grade
            
            # 안정화된 자세 등급
            if len(pose_history) >= 5:
                stable_posture_grade = max(set(pose_history[-5:]), key=pose_history[-5:].count)
            else:
                stable_posture_grade = posture_grade
            
            # 피드백 메시지와 색상 (측면일 때만)
            if stable_grade is not None and cva_angle is not None:
                message, color = get_feedback_message(stable_grade, cva_angle)
            else:
                # 정면일 때는 자세 등급만 표시
                message = "정면 자세 분석 중..."
                color = (128, 128, 128)  # 회색
            
            # 화면에 정보 표시
            # 등급 표시 (측면일 때만)
            if stable_grade is not None:
                frame = put_korean_text(frame, f"Grade: {stable_grade}", 
                                       (30, 80), font_size=48, color=color)
            else:
                frame = put_korean_text(frame, "Front View", 
                                       (30, 80), font_size=48, color=(128, 128, 128))
            
            # CVA 각도 표시 (측면일 때만)
            if cva_angle is not None:
                frame = put_korean_text(frame, f"CVA: {cva_angle:.1f}°", 
                                       (30, 130), font_size=32, color=(255, 255, 255))
            # 정면일 때는 CVA 각도를 표시하지 않음
            
            # 자세 등급 표시 (PostureAnalyzer 결과)
            # frame = put_korean_text(frame, f"Posture Grade: {stable_posture_grade} (점수: {posture_score:.1f})", 
            #                        (30, 190), font_size=24, color=posture_color)
            
            # 피드백 메시지 표시
            frame = put_korean_text(frame, message, 
                                   (30, 220), font_size=28, color=color)
            
            # 프레임 카운트 표시
            frame_count += 1
            frame = put_korean_text(frame, f"Frame: {frame_count}", 
                                   (30, 250), font_size=20, color=(255, 255, 255))
            
        else:
            # 랜드마크가 감지되지 않을 때
            frame = put_korean_text(frame, "No pose detected", 
                                   (30, 80), font_size=32, color=(0, 0, 255))
            frame = put_korean_text(frame, "Please stand in front of the camera", 
                                   (30, 120), font_size=24, color=(255, 255, 255))
        
        # 화면 표시
        cv2.imshow('Real-time Posture Feedback (4way 통합)', frame)
        
        # ESC 키로 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # 정리
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n🎉 실시간 자세 분석이 종료되었습니다.")
    print(f"📊 총 분석 프레임: {frame_count}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 