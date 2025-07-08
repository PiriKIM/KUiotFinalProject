#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실시간 자세 등급 피드백 시스템

웹캠으로 실시간 영상을 받아서 자세를 분석하고 등급(A/B/C)과 피드백을 화면에 표시합니다.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
from PIL import Image, ImageDraw, ImageFont

# 자체 모듈 import
sys.path.append('side_angle_analysis_folder')
from posture_grade_classifier import PostureGradeClassifier


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


def detect_side(landmarks):
    """
    정교한 측면 감지 알고리즘 (화면 좌우반전 고려)
    카메라에 더 잘 보이는 측면을 감지합니다.
    화면이 좌우반전되어 있으므로 left/right를 반대로 처리합니다.
    """
    try:
        # 어깨와 골반의 x좌표를 이용한 측면 감지
        left_shoulder_x = landmarks[11].x  # LEFT_SHOULDER
        right_shoulder_x = landmarks[12].x  # RIGHT_SHOULDER
        left_hip_x = landmarks[23].x        # LEFT_HIP
        right_hip_x = landmarks[24].x       # RIGHT_HIP
        
        # 귀 위치 정보
        left_ear_x = landmarks[7].x         # LEFT_EAR
        right_ear_x = landmarks[8].x        # RIGHT_EAR
        
        # 어깨와 골반의 평균 x좌표
        shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2
        hip_center_x = (left_hip_x + right_hip_x) / 2
        
        # 어깨 비대칭 계산
        shoulder_diff = abs(left_shoulder_x - right_shoulder_x)
        
        # 가중치 기반 점수 계산 시스템 (화면 좌우반전 고려)
        left_score = 0
        right_score = 0
        
        # 1. 어깨 중심점 기준 (가중치: 3) - 화면 반전 고려
        if shoulder_center_x > 0.51:
            right_score += 3  # 화면에서 오른쪽에 보이면 실제로는 왼쪽 측면
        elif shoulder_center_x < 0.49:
            left_score += 3   # 화면에서 왼쪽에 보이면 실제로는 오른쪽 측면
        elif shoulder_center_x > 0.5:
            right_score += 1
        else:
            left_score += 1
        
        # 2. 어깨 비대칭 기준 (가중치: 4) - 화면 반전 고려
        if shoulder_diff > 0.03:
            if left_shoulder_x < right_shoulder_x:
                right_score += 4  # 화면에서 왼쪽 어깨가 더 왼쪽에 있으면 실제로는 오른쪽 측면
            else:
                left_score += 4   # 화면에서 오른쪽 어깨가 더 왼쪽에 있으면 실제로는 왼쪽 측면
        elif shoulder_diff > 0.01:
            if left_shoulder_x < right_shoulder_x:
                right_score += 2
            else:
                left_score += 2
        
        # 3. 골반 중심점 기준 (가중치: 3) - 화면 반전 고려
        if hip_center_x > 0.51:
            right_score += 3
        elif hip_center_x < 0.49:
            left_score += 3
        elif hip_center_x > 0.5:
            right_score += 1
        else:
            left_score += 1
        
        # 4. 귀 위치 기준 (가중치: 2) - 화면 반전 고려
        ear_diff = abs(left_ear_x - right_ear_x)
        if ear_diff > 0.02:
            if left_ear_x < right_ear_x:
                right_score += 2
            else:
                left_score += 2
        elif ear_diff > 0.01:
            if left_ear_x < right_ear_x:
                right_score += 1
            else:
                left_score += 1
        
        # 5. 어깨와 골반의 상대적 위치 (가중치: 3) - 화면 반전 고려
        shoulder_hip_relative = shoulder_center_x - hip_center_x
        if abs(shoulder_hip_relative) > 0.005:
            if shoulder_hip_relative > 0:
                right_score += 3
            else:
                left_score += 3
        elif abs(shoulder_hip_relative) > 0.002:
            if shoulder_hip_relative > 0:
                right_score += 1
            else:
                left_score += 1
        
        # 6. 전체 랜드마크 평균 위치 (가중치: 2) - 화면 반전 고려
        all_landmarks_x = [left_shoulder_x, right_shoulder_x, left_hip_x, right_hip_x, left_ear_x, right_ear_x]
        avg_x = sum(all_landmarks_x) / len(all_landmarks_x)
        if avg_x > 0.505:
            right_score += 2
        elif avg_x < 0.495:
            left_score += 2
        elif avg_x > 0.5:
            right_score += 1
        else:
            left_score += 1
        
        # 7. 특별한 패턴 검증 - 화면 반전 고려
        if left_shoulder_x > 0.5 and right_shoulder_x < 0.5:
            right_score += 2  # 화면에서 왼쪽 어깨가 오른쪽에 있으면 실제로는 오른쪽 측면
        elif right_shoulder_x > 0.5 and left_shoulder_x < 0.5:
            left_score += 2   # 화면에서 오른쪽 어깨가 오른쪽에 있으면 실제로는 왼쪽 측면
        
        # 8. 극단적 위치 검증 - 화면 반전 고려
        if left_shoulder_x > 0.55 or left_hip_x > 0.55:
            right_score += 3
        elif right_shoulder_x < 0.45 or right_hip_x < 0.45:
            left_score += 3
        
        # 최종 판정 (화면 반전 고려)
        if left_score > right_score:
            return 'right'  # 화면에서 왼쪽으로 보이면 실제로는 오른쪽 측면
        elif right_score > left_score:
            return 'left'   # 화면에서 오른쪽으로 보이면 실제로는 왼쪽 측면
        else:
            # 동점인 경우 기본값
            return 'right'
                
    except Exception as e:
        print(f"측면 감지 오류: {e}")
        # 오류 발생 시 기본 가시성 기반 감지로 fallback (화면 반전 고려)
        right_ear_visibility = landmarks[8].visibility
        left_ear_visibility = landmarks[7].visibility
        right_shoulder_visibility = landmarks[12].visibility
        left_shoulder_visibility = landmarks[11].visibility
        
        right_score = right_ear_visibility + right_shoulder_visibility
        left_score = left_ear_visibility + left_shoulder_visibility
        
        if right_score > left_score:
            return 'left'   # 화면 반전 고려
        else:
            return 'right'  # 화면 반전 고려


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='실시간 자세 등급 피드백 시스템')
    parser.add_argument('--csv', '-c', required=True,
                       help='기준 데이터 CSV 파일 경로 (측면별로 자동 선택됨)')
    parser.add_argument('--side', '-s', choices=['right', 'left'], default='right',
                       help='측면 (right 또는 left)')
    parser.add_argument('--camera', type=int, default=0,
                       help='카메라 인덱스 (기본값: 0)')
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not Path(args.csv).exists():
        print(f"❌ CSV 파일을 찾을 수 없습니다: {args.csv}")
        return 1
    
    print(f"🎯 실시간 자세 등급 피드백 시스템을 시작합니다...")
    print(f"📁 기준 데이터: {args.csv}")
    print(f"📐 측면: {args.side}")
    print(f"📷 카메라: {args.camera}")
    
    # 측면별 CSV 파일 경로 설정
    csv_base_path = Path(args.csv)
    right_csv_path = csv_base_path
    left_csv_path = csv_base_path
    
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
    
    # 등급 분류기 초기화
    classifier = PostureGradeClassifier()
    
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
    
    print(f"\n🎥 카메라가 시작되었습니다.")
    print(f"💡 ESC 키를 누르면 종료됩니다.")
    
    frame_count = 0
    grade_history = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임을 읽을 수 없습니다.")
            break
        
        # 프레임을 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        # 화면에 기본 정보 표시
        frame = put_korean_text(frame, f"Camera: {args.camera} | Side: {args.side}", 
                               (10, 30), font_size=24, color=(255, 255, 255))
        
        if results.pose_landmarks:
            # 자동 측면 감지 (매 프레임마다)
            detected_side = detect_side(results.pose_landmarks.landmark)
            if frame_count == 0:
                print(f"🎯 자동 감지된 측면: {detected_side}")
            current_side = detected_side
            
            # 측면에 따른 기준값 선택
            current_thresholds = right_thresholds if current_side == 'right' else left_thresholds
            
            # CVA 각도 계산
            cva_angle = calculate_cva_angle(results.pose_landmarks.landmark, current_side)
            
            # 디버깅 정보 (처음 5프레임에서만)
            if frame_count < 5:
                if current_side == 'right':
                    ear = results.pose_landmarks.landmark[8]
                    shoulder = results.pose_landmarks.landmark[12]
                else:
                    ear = results.pose_landmarks.landmark[7]
                    shoulder = results.pose_landmarks.landmark[11]
                
                # 측면 감지 디버깅 정보 추가
                left_shoulder_x = results.pose_landmarks.landmark[11].x
                right_shoulder_x = results.pose_landmarks.landmark[12].x
                left_hip_x = results.pose_landmarks.landmark[23].x
                right_hip_x = results.pose_landmarks.landmark[24].x
                left_ear_x = results.pose_landmarks.landmark[7].x
                right_ear_x = results.pose_landmarks.landmark[8].x
                
                shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2
                hip_center_x = (left_hip_x + right_hip_x) / 2
                shoulder_diff = abs(left_shoulder_x - right_shoulder_x)
                
                print(f"🔍 프레임 {frame_count}: 측면={current_side}, 각도={cva_angle:.1f}°")
                print(f"   귀: ({ear.x:.3f}, {ear.y:.3f}), 가시성: {ear.visibility:.3f}")
                print(f"   어깨: ({shoulder.x:.3f}, {shoulder.y:.3f}), 가시성: {shoulder.visibility:.3f}")
                print(f"   사용 기준값: {current_thresholds['min_abs']:.2f} ~ {current_thresholds['max_abs']:.2f}")
                print(f"   측면 감지 정보:")
                print(f"     - 어깨 중심: {shoulder_center_x:.3f}, 골반 중심: {hip_center_x:.3f}")
                print(f"     - 어깨 차이: {shoulder_diff:.3f}")
                print(f"     - 왼쪽 어깨: {left_shoulder_x:.3f}, 오른쪽 어깨: {right_shoulder_x:.3f}")
                print(f"     - 왼쪽 귀: {left_ear_x:.3f}, 오른쪽 귀: {right_ear_x:.3f}")
            
            # 각도 범위 제한 (비정상적인 값 필터링)
            if abs(cva_angle) > 90:
                cva_angle = np.clip(cva_angle, -90, 90)
            
            # 등급 분류 (측면별 기준값 사용)
            grade = classifier.get_grade_for_angle(
                cva_angle, 
                current_thresholds['min_abs'], 
                current_thresholds['max_abs'], 
                current_thresholds['stage1_threshold']
            )
            
            # 등급 히스토리 업데이트 (최근 10프레임)
            grade_history.append(grade)
            if len(grade_history) > 10:
                grade_history.pop(0)
            
            # 안정화된 등급 (최근 5프레임 중 가장 많은 등급)
            if len(grade_history) >= 5:
                stable_grade = max(set(grade_history[-5:]), key=grade_history[-5:].count)
            else:
                stable_grade = grade
            
            # 피드백 메시지와 색상
            message, color = get_feedback_message(stable_grade, cva_angle)
            
            # 화면에 정보 표시
            # 등급 표시
            frame = put_korean_text(frame, f"Grade: {stable_grade}", 
                                   (30, 80), font_size=48, color=color)
            
            # CVA 각도 표시
            frame = put_korean_text(frame, f"CVA: {cva_angle:.1f}°", 
                                   (30, 130), font_size=32, color=(255, 255, 255))
            
            # 자동 감지된 측면 정보 표시 (색상으로 구분)
            side_color = (0, 255, 0) if current_side == 'right' else (255, 0, 0)  # 초록색(오른쪽) vs 빨간색(왼쪽)
            frame = put_korean_text(frame, f"Auto Side: {current_side.upper()}", 
                                   (30, 160), font_size=24, color=side_color)
            
            # 피드백 메시지 표시
            frame = put_korean_text(frame, message, 
                                   (30, 200), font_size=28, color=color)
            
            # 프레임 카운트 표시
            frame_count += 1
            frame = put_korean_text(frame, f"Frame: {frame_count}", 
                                   (30, 230), font_size=20, color=(255, 255, 255))
            
        else:
            # 랜드마크가 감지되지 않을 때
            frame = put_korean_text(frame, "No pose detected", 
                                   (30, 80), font_size=32, color=(0, 0, 255))
            frame = put_korean_text(frame, "Please stand in front of the camera", 
                                   (30, 120), font_size=24, color=(255, 255, 255))
        
        # 화면 표시
        cv2.imshow('Real-time Posture Feedback', frame)
        
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