#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실시간 자세 분석 Flask 라우트
ESP32-CAM에서 전송된 영상을 받아서 실시간 자세 분석을 수행합니다.
"""

import cv2
import numpy as np
import mediapipe as mp
import pickle
import pandas as pd
from pathlib import Path
import threading
import time
from datetime import datetime
from flask import Blueprint, request, jsonify, render_template, Response, current_app
from flask_login import login_required, current_user
from apps.app import db
from apps.crud.models import RealtimePostureRecord
import warnings
import subprocess
import os
warnings.filterwarnings('ignore')

# 자체 모듈 import
import sys
sys.path.append('/home/piri/KUiotFinalProject/merge_mp/ml_models')
from posture_grade_classifier import PostureGradeClassifier

realtime_bp = Blueprint('realtime', __name__)

# 전역 변수
current_frame = None
analysis_results = {}
is_analyzing = False
frame_lock = threading.Lock()

# 음성 재생 관련 전역 변수
last_audio_play_time = 0
AUDIO_COOLDOWN = 30  # 30초 쿨다운
AUDIO_FILE_PATH = '/home/piri/KUiotFinalProject/flaskbook/output.wav'

def play_audio_if_needed(grade):
    """C등급 감지 시 음성 재생 (30초 쿨다운 적용)"""
    global last_audio_play_time
    
    if grade == 'C':
        current_time = time.time()
        
        # 30초 쿨다운 확인
        if current_time - last_audio_play_time >= AUDIO_COOLDOWN:
            if os.path.exists(AUDIO_FILE_PATH):
                try:
                    # 여러 음성 재생 방법 시도
                    audio_played = False
                    
                    # 방법 1: aplay 사용
                    try:
                        subprocess.Popen(['aplay', AUDIO_FILE_PATH], 
                                       stdout=subprocess.DEVNULL, 
                                       stderr=subprocess.DEVNULL)
                        audio_played = True
                        print(f"🔊 C등급 감지! aplay로 음성 재생됨 (시간: {datetime.now().strftime('%H:%M:%S')})")
                    except Exception as e:
                        print(f"❌ aplay 재생 실패: {e}")
                    
                    # 방법 2: paplay 사용 (PulseAudio)
                    if not audio_played:
                        try:
                            subprocess.Popen(['paplay', AUDIO_FILE_PATH], 
                                           stdout=subprocess.DEVNULL, 
                                           stderr=subprocess.DEVNULL)
                            audio_played = True
                            print(f"🔊 C등급 감지! paplay로 음성 재생됨 (시간: {datetime.now().strftime('%H:%M:%S')})")
                        except Exception as e:
                            print(f"❌ paplay 재생 실패: {e}")
                    
                    # 방법 3: ffplay 사용
                    if not audio_played:
                        try:
                            subprocess.Popen(['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', AUDIO_FILE_PATH], 
                                           stdout=subprocess.DEVNULL, 
                                           stderr=subprocess.DEVNULL)
                            audio_played = True
                            print(f"🔊 C등급 감지! ffplay로 음성 재생됨 (시간: {datetime.now().strftime('%H:%M:%S')})")
                        except Exception as e:
                            print(f"❌ ffplay 재생 실패: {e}")
                    
                    if audio_played:
                        last_audio_play_time = current_time
                    else:
                        print("❌ 모든 음성 재생 방법이 실패했습니다.")
                        
                except Exception as e:
                    print(f"❌ 음성 재생 중 오류: {e}")
            else:
                print(f"❌ 음성 파일을 찾을 수 없습니다: {AUDIO_FILE_PATH}")

class MLPoseClassifier:
    """ML 모델 기반 자세 분류기 (4way 모델 사용)"""
    
    def __init__(self, model_path='/home/piri/KUiotFinalProject/pose_classifier_4way_model.pkl'):
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
            
            print(f"✅ 4way 모델이 {model_path}에서 로드되었습니다.")
            print(f"📊 클래스: {self.classes}")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print("💡 먼저 4way 모델을 학습시켜주세요.")
            self.is_trained = False
    
    def predict_pose(self, landmarks):
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
            print(f"❌ 예측 오류: {e}")
            return None, None


def calculate_cva_angle(landmarks, side='right'):
    """
    Mediapipe 랜드마크에서 CVA 각도 계산
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
    if neck_vector[0] > 0:  # 귀가 어깨보다 오른쪽(앞쪽)에 있으면
        angle_deg = angle_deg
    else:
        angle_deg = -angle_deg
    
    return angle_deg


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


def landmarks_to_array(landmarks):
    """MediaPipe landmarks를 배열로 변환"""
    landmarks_array = []
    for landmark in landmarks:
        landmarks_array.extend([landmark.x, landmark.y])
    return landmarks_array


def detect_side_ml(pose_classifier, landmarks):
    """ML 모델을 사용한 측면 감지"""
    try:
        # 랜드마크를 배열로 변환
        landmarks_array = landmarks_to_array(landmarks)
        
        # ML 모델로 예측
        prediction, probability = pose_classifier.predict_pose(landmarks_array)
        
        if prediction is not None and probability is not None:
            # 예측 결과에 따른 측면 반환
            if prediction == 1:  # 정면
                return 'front', probability
            elif prediction == 2:  # 좌측면
                return 'left', probability
            elif prediction == 3:  # 우측면
                return 'right', probability
            else:
                return 'unknown', probability
        else:
            return 'unknown', None
            
    except Exception as e:
        print(f"❌ ML 측면 감지 오류: {e}")
        return 'unknown', None


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


# 전역 변수로 분석기 초기화
pose_classifier = None
grade_classifier = None
right_thresholds = None
left_thresholds = None
mp_pose = None
pose = None

def initialize_analyzers():
    """분석기들을 초기화"""
    global pose_classifier, grade_classifier, right_thresholds, left_thresholds, mp_pose, pose
    
    print("🤖 실시간 자세 분석기를 초기화하고 있습니다...")
    
    # ML 자세 분류기 초기화
    pose_classifier = MLPoseClassifier()
    
    if not pose_classifier.is_trained:
        print("❌ ML 모델이 로드되지 않았습니다.")
        return False
    
    # 등급 분류기 초기화
    grade_classifier = PostureGradeClassifier()
    
    # 측면별 기준값 계산
    right_csv = "/home/piri/KUiotFinalProject/merge_mp/data/right_side_angle_analysis.csv"
    left_csv = "/home/piri/KUiotFinalProject/merge_mp/data/left_side_angle_analysis.csv"
    
    right_thresholds = calculate_grade_thresholds(right_csv)
    left_thresholds = calculate_grade_thresholds(left_csv)
    
    if right_thresholds is None or left_thresholds is None:
        print("❌ 기준값 계산 실패")
        return False
    
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
    
    print("✅ 실시간 자세 분석기 초기화 완료!")
    return True


def analyze_frame(frame):
    """프레임을 분석하여 자세 결과를 반환"""
    global pose_classifier, grade_classifier, right_thresholds, left_thresholds, pose
    
    if pose_classifier is None or grade_classifier is None or pose is None:
        return None
    
    try:
        # 프레임을 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            # ML 모델을 사용한 측면 감지
            detected_side, probability = detect_side_ml(pose_classifier, results.pose_landmarks.landmark)
            
            # 측면에 따른 기준값 선택
            if detected_side == 'left':
                current_thresholds = left_thresholds
                current_side = 'left'
            elif detected_side == 'right':
                current_thresholds = right_thresholds
                current_side = 'right'
            else:  # front 또는 unknown
                current_thresholds = right_thresholds
                current_side = 'right'
            
            # CVA 각도 계산 (측면이 있을 때만)
            if detected_side in ['left', 'right']:
                cva_angle = calculate_cva_angle(results.pose_landmarks.landmark, current_side)
                
                # 각도 범위 제한
                if abs(cva_angle) > 90:
                    cva_angle = np.clip(cva_angle, -90, 90)
                
                # 등급 분류
                grade = grade_classifier.get_grade_for_angle(
                    cva_angle, 
                    current_thresholds['min_abs'], 
                    current_thresholds['max_abs'], 
                    current_thresholds['stage1_threshold']
                )
                
                # C등급 감지 시 음성 재생
                play_audio_if_needed(grade)
                
                # 피드백 메시지
                message, color = get_feedback_message(grade, cva_angle)
                
                # ML 신뢰도
                ml_confidence = max(probability) if probability is not None else 0.0
                
                return {
                    'detected_side': detected_side,
                    'ml_confidence': ml_confidence,
                    'cva_angle': cva_angle,
                    'posture_grade': grade,
                    'feedback_message': message,
                    'min_abs_threshold': current_thresholds['min_abs'],
                    'max_abs_threshold': current_thresholds['max_abs'],
                    'stage1_threshold': current_thresholds['stage1_threshold'],
                    'pose_detected': True
                }
            else:
                return {
                    'detected_side': detected_side,
                    'ml_confidence': 0.0,
                    'cva_angle': 0.0,
                    'posture_grade': 'N/A',
                    'feedback_message': 'Front View Detected',
                    'pose_detected': True
                }
        else:
            return {
                'pose_detected': False,
                'feedback_message': 'No pose detected'
            }
            
    except Exception as e:
        print(f"❌ 프레임 분석 오류: {e}")
        return None


@realtime_bp.route('/esp32_stream', methods=['POST'])
def esp32_stream():
    """ESP32-CAM에서 전송된 영상 데이터를 받아서 처리"""
    global current_frame
    
    try:
        # 바이너리 데이터로 받기
        image_data = request.get_data()
        
        # 바이트 배열을 numpy 배열로 변환
        nparr = np.frombuffer(image_data, np.uint8)
        
        # 이미지 디코딩
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            with frame_lock:
                current_frame = frame.copy()
            
            # 분석 결과 반환
            return jsonify({
                'status': 'success',
                'message': 'Frame received and processed',
                'frame_size': frame.shape
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to decode image'
            }), 400
            
    except Exception as e:
        print(f"❌ ESP32 스트림 처리 오류: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@realtime_bp.route('/video_feed')
def video_feed():
    """실시간 비디오 스트림을 제공하는 제너레이터"""
    def generate():
        global current_frame
        
        while True:
            with frame_lock:
                if current_frame is not None:
                    # JPEG로 인코딩
                    ret, buffer = cv2.imencode('.jpg', current_frame)
                    if ret:
                        frame_data = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            
            time.sleep(0.1)  # 10 FPS
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@realtime_bp.route('/realtime_analysis')
@login_required
def realtime_analysis_page():
    """실시간 자세 분석 페이지"""
    return render_template('crud/realtime_analysis.html')


@realtime_bp.route('/start_analysis', methods=['POST'])
@login_required
def start_analysis():
    """실시간 분석 시작"""
    global is_analyzing
    
    if not is_analyzing:
        is_analyzing = True
        
        # 분석기 초기화 (아직 초기화되지 않은 경우)
        if pose_classifier is None:
            if not initialize_analyzers():
                return jsonify({'status': 'error', 'message': '분석기 초기화 실패'}), 500
        
        return jsonify({'status': 'success', 'message': '분석이 시작되었습니다.'})
    else:
        return jsonify({'status': 'info', 'message': '이미 분석이 진행 중입니다.'})


@realtime_bp.route('/stop_analysis', methods=['POST'])
@login_required
def stop_analysis():
    """실시간 분석 중지"""
    global is_analyzing
    is_analyzing = False
    return jsonify({'status': 'success', 'message': '분석이 중지되었습니다.'})


@realtime_bp.route('/get_analysis_result')
@login_required
def get_analysis_result():
    """현재 분석 결과를 반환"""
    global current_frame, analysis_results, is_analyzing
    
    if not is_analyzing or current_frame is None:
        return jsonify({'status': 'no_analysis'})
    
    # 프레임 분석
    result = analyze_frame(current_frame)
    
    if result and result.get('pose_detected', False):
        # N/A 등급이 아닌 경우에만 데이터베이스에 저장
        if result.get('posture_grade') and result.get('posture_grade') != 'N/A':
            try:
                record = RealtimePostureRecord(
                    user_id=current_user.id,
                    detected_side=result.get('detected_side', 'unknown'),
                    ml_confidence=result.get('ml_confidence', 0.0),
                    cva_angle=result.get('cva_angle', 0.0),
                    posture_grade=result.get('posture_grade'),
                    feedback_message=result.get('feedback_message', ''),
                    min_abs_threshold=result.get('min_abs_threshold', 0.0),
                    max_abs_threshold=result.get('max_abs_threshold', 0.0),
                    stage1_threshold=result.get('stage1_threshold', 0.0),
                    frame_count=len(analysis_results) + 1
                )
                
                db.session.add(record)
                db.session.commit()
                
                # 분석 결과에 저장
                analysis_results[record.id] = result
                
            except Exception as e:
                print(f"❌ 데이터베이스 저장 오류: {e}")
        else:
            # N/A 등급인 경우 분석 결과에만 저장 (DB 저장 안함)
            analysis_results[f"temp_{len(analysis_results)}"] = result
    
    return jsonify({
        'status': 'success',
        'result': result
    })


@realtime_bp.route('/analysis_history')
@login_required
def analysis_history():
    """실시간 분석 히스토리 페이지"""
    # 최근 100개의 분석 기록 조회
    records = RealtimePostureRecord.query.filter_by(user_id=current_user.id)\
        .order_by(RealtimePostureRecord.timestamp.desc())\
        .limit(100).all()
    
    return render_template('crud/analysis_history.html', records=records)


# 분석기 초기화
initialize_analyzers() 