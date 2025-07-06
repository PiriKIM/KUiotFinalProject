#!/usr/bin/env python3
# ===============================================
# 📌 스마트 각도 분석 스크립트 (방향 감지 포함)
#
# ✅ 기능:
# - raw_landmarks.csv 읽기
# - 3클래스 분류로 방향 구분 (정면/좌측면/우측면)
# - 방향별 최적화된 랜드마크 선택
# - CVA 1, CVA 2 각도 계산
# - 최종 CSV: 피사체ID, 프레임명, CVA 1, CVA 2
# ===============================================

import sys
import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class CSVBasedPoseClassifier:
    """CSV 파일용 3클래스 자세 분류기 (4way 모델 사용)"""
    
    def __init__(self, model_path='pose_classifier_4way_model.pkl'):
        """분류기 초기화"""
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.classes = ['정면', '좌측면', '우측면']
        self.load_model(model_path)
        
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
    
    def predict_direction(self, landmarks_df: pd.DataFrame) -> str:
        """4way 모델을 사용한 방향 예측"""
        if not self.is_trained or self.model is None or self.scaler is None:
            print("4way 모델이 로드되지 않았습니다. 기본값 'front' 사용")
            return 'front'
        
        try:
            # 랜드마크를 배열로 변환 (실시간과 동일하게 x좌표 반전)
            landmarks_array = []
            for i in range(33):
                landmark_data = landmarks_df[landmarks_df['landmark_id'] == i]
                if not landmark_data.empty:
                    # 실시간과 동일하게 x좌표 반전 (1.0 - x)
                    x = 1.0 - landmark_data.iloc[0]['x']
                    y = landmark_data.iloc[0]['y']
                    landmarks_array.extend([x, y])
                else:
                    landmarks_array.extend([-1, -1])
            
            # 특징 추출
            features = self.extract_features(landmarks_array)
            features_scaled = self.scaler.transform([features])
            
            # 예측
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            # 예측 결과를 방향으로 변환 (realtime_pose_classifier_4way.py와 동일한 방식)
            # prediction: 1(정면), 2(좌측면), 3(우측면)
            # classes: ['정면', '좌측면', '우측면']
            pose_type = self.classes[prediction - 1] if 1 <= prediction <= 3 else "정면"
            
            # pose_type을 direction으로 변환
            direction_map = {'정면': 'front', '좌측면': 'left', '우측면': 'right'}
            direction = direction_map.get(pose_type, 'front')
            
            confidence = max(probability)
            print(f"방향 예측: {direction} (신뢰도: {confidence:.1%})")
            
            return direction
            
        except Exception as e:
            print(f"방향 예측 오류: {e}")
            return 'front'  # 기본값

class SmartAngleCalculator:
    """스마트 각도 계산기"""
    
    def __init__(self):
        """각도 계산기 초기화"""
        # 방향별 랜드마크 선택 전략
        self.direction_landmarks = {
            'front': {
                'ear': ['LEFT_EAR', 'RIGHT_EAR'],  # 양쪽 귀 모두 사용
                'shoulder': ['LEFT_SHOULDER', 'RIGHT_SHOULDER'],  # 양쪽 어깨 모두 사용
                'hip': ['LEFT_HIP', 'RIGHT_HIP'],  # 양쪽 엉덩이 모두 사용
                'description': '정면: 양쪽 랜드마크 평균값 사용'
            },
            'left': {
                'ear': ['LEFT_EAR'],  # 왼쪽 귀만 사용
                'shoulder': ['LEFT_SHOULDER'],  # 왼쪽 어깨만 사용
                'hip': ['LEFT_HIP'],  # 왼쪽 엉덩이만 사용
                'description': '좌측면: 왼쪽 랜드마크만 사용'
            },
            'right': {
                'ear': ['RIGHT_EAR'],  # 오른쪽 귀만 사용
                'shoulder': ['RIGHT_SHOULDER'],  # 오른쪽 어깨만 사용
                'hip': ['RIGHT_HIP'],  # 오른쪽 엉덩이만 사용
                'description': '우측면: 오른쪽 랜드마크만 사용'
            }
        }
    
    def get_landmark_coordinates(self, landmarks_data: Dict, landmark_names: List[str]) -> List[Tuple[float, float]]:
        """랜드마크 이름 리스트로부터 좌표 추출"""
        coordinates = []
        
        for name in landmark_names:
            if name in landmarks_data:
                x = landmarks_data[name]['x']
                y = landmarks_data[name]['y']
                coordinates.append((x, y))
            else:
                coordinates.append(None)
        
        return coordinates
    
    def calculate_average_coordinates(self, coordinates: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """좌표들의 평균값 계산 (None 제외)"""
        valid_coords = [coord for coord in coordinates if coord is not None]
        
        if not valid_coords:
            return None
        
        if len(valid_coords) == 1:
            return valid_coords[0]
        
        # 평균 계산
        avg_x = sum(coord[0] for coord in valid_coords) / len(valid_coords)
        avg_y = sum(coord[1] for coord in valid_coords) / len(valid_coords)
        
        return (avg_x, avg_y)
    
    def calculate_angle(self, point1: Tuple[float, float], point2: Tuple[float, float], point3: Tuple[float, float]) -> float:
        """세 점으로부터 각도 계산 (도 단위)"""
        if None in [point1, point2, point3]:
            return 0.0
        
        # 벡터 계산
        vec1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        vec2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        # 벡터 크기 계산
        mag1 = np.linalg.norm(vec1)
        mag2 = np.linalg.norm(vec2)
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        # 내적 계산
        dot_product = np.dot(vec1, vec2)
        
        # 코사인 각도 계산
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # 각도 계산 (라디안 → 도)
        angle = np.arccos(cos_angle)
        angle_degrees = np.degrees(angle)
        
        return angle_degrees
    
    def calculate_cva_angle(self, landmarks_data: Dict, direction: str) -> float:
        """목각도(CVA) 계산 - 카메라 방향별 최적화"""
        if direction not in self.direction_landmarks:
            return 0.0
        
        # 방향별 랜드마크 선택
        direction_config = self.direction_landmarks[direction]
        
        # 필요한 랜드마크 좌표 추출
        ear_coords = self.get_landmark_coordinates(landmarks_data, direction_config['ear'])
        shoulder_coords = self.get_landmark_coordinates(landmarks_data, direction_config['shoulder'])
        
        # 평균 좌표 계산
        ear_avg = self.calculate_average_coordinates(ear_coords)
        shoulder_avg = self.calculate_average_coordinates(shoulder_coords)
        
        # CVA 각도 계산: 귀-어깨-수직선
        if ear_avg and shoulder_avg:
            # 수직선 상의 점 (어깨와 같은 x좌표, 더 위쪽 y좌표)
            vertical_point = (shoulder_avg[0], shoulder_avg[1] - 0.1)  # 어깨보다 0.1 위
            
            cva_angle = self.calculate_angle(ear_avg, shoulder_avg, vertical_point)
            return cva_angle
        else:
            return 0.0
    
    def calculate_spine_angle(self, landmarks_data: Dict, direction: str) -> float:
        """척추각도 계산 - 카메라 방향별 최적화"""
        if direction not in self.direction_landmarks:
            return 0.0
        
        # 방향별 랜드마크 선택
        direction_config = self.direction_landmarks[direction]
        
        # 필요한 랜드마크 좌표 추출
        shoulder_coords = self.get_landmark_coordinates(landmarks_data, direction_config['shoulder'])
        hip_coords = self.get_landmark_coordinates(landmarks_data, direction_config['hip'])
        
        # 평균 좌표 계산
        shoulder_avg = self.calculate_average_coordinates(shoulder_coords)
        hip_avg = self.calculate_average_coordinates(hip_coords)
        
        # 척추각도 계산: 어깨-엉덩이-수직선
        if shoulder_avg and hip_avg:
            # 수직선 상의 점 (어깨와 같은 x좌표, 더 위쪽 y좌표)
            vertical_point = (shoulder_avg[0], shoulder_avg[1] - 0.1)  # 어깨보다 0.1 위
            
            spine_angle = self.calculate_angle(shoulder_avg, hip_avg, vertical_point)
            return spine_angle
        else:
            return 0.0

def landmarks_to_dict(landmarks_df: pd.DataFrame) -> Dict:
    """DataFrame을 딕셔너리 형태로 변환"""
    landmarks_dict = {}
    
    for _, row in landmarks_df.iterrows():
        landmark_name = row['landmark_name']
        landmarks_dict[landmark_name] = {
            'x': row['x'],
            'y': row['y'],
            'z': row['z'],
            'visibility': row['visibility']
        }
    
    return landmarks_dict

def main():
    """메인 실행 함수"""
    print("스마트 각도 분석 시작 (방향 감지 포함)...")
    
    # 파일 경로
    input_csv = "data/landmarks/raw_landmarks.csv"
    output_csv = "data/angles/smart_angles_with_direction.csv"
    
    # 출력 디렉토리 생성
    output_dir = Path(output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 파일 존재 확인
    if not os.path.exists(input_csv):
        print(f"입력 파일이 존재하지 않습니다: {input_csv}")
        return
    
    # 분류기와 계산기 초기화
    classifier = CSVBasedPoseClassifier()
    calculator = SmartAngleCalculator()
    
    # CSV 파일 읽기
    print(f"CSV 파일 읽기: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"총 데이터 수: {len(df)}")
    
    # 프레임별로 그룹화
    frame_groups = df.groupby(['subject_id', 'frame_name'])
    print(f"총 프레임 수: {len(frame_groups)}")
    
    # 결과 저장용 리스트
    results = []
    
    for (subject_id, frame_name), frame_data in frame_groups:
        print(f"분석 중: {subject_id} - {frame_name}")
        
        # 1. 방향 예측
        direction = classifier.predict_direction(frame_data)
        
        # 2. 랜드마크를 딕셔너리로 변환
        landmarks_dict = landmarks_to_dict(frame_data)
        
        # 3. 각도 계산
        cva_angle = calculator.calculate_cva_angle(landmarks_dict, direction)
        spine_angle = calculator.calculate_spine_angle(landmarks_dict, direction)
        
        # 4. 결과 저장
        result_row = {
            'subject_id': subject_id,
            'frame_name': frame_name,
            'direction': direction,
            'CVA_1': cva_angle,  # 목각도
            'CVA_2': spine_angle  # 척추각도
        }
        
        results.append(result_row)
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(results)
    
    # CSV 파일로 저장
    results_df.to_csv(output_csv, index=False)
    print(f"결과가 {output_csv}에 저장되었습니다.")
    
    # 통계 출력
    print("\n=== 분석 결과 통계 ===")
    print(f"총 프레임 수: {len(results_df)}")
    print("\n방향별 분포:")
    direction_counts = results_df['direction'].value_counts()
    for direction, count in direction_counts.items():
        direction_name = {'front': '정면', 'left': '좌측면', 'right': '우측면'}[direction]
        print(f"  {direction_name}: {count}개")
    
    print(f"\n각도 통계:")
    print(f"  CVA 1 (목각도) 평균: {results_df['CVA_1'].mean():.2f}°")
    print(f"  CVA 1 (목각도) 표준편차: {results_df['CVA_1'].std():.2f}°")
    print(f"  CVA 2 (척추각도) 평균: {results_df['CVA_2'].mean():.2f}°")
    print(f"  CVA 2 (척추각도) 표준편차: {results_df['CVA_2'].std():.2f}°")
    
    print(f"\n각도 범위:")
    print(f"  CVA 1 최소값: {results_df['CVA_1'].min():.2f}°")
    print(f"  CVA 1 최대값: {results_df['CVA_1'].max():.2f}°")
    print(f"  CVA 2 최소값: {results_df['CVA_2'].min():.2f}°")
    print(f"  CVA 2 최대값: {results_df['CVA_2'].max():.2f}°")
    
    print(f"\n분석 완료!")
    print(f"입력 파일: {input_csv}")
    print(f"출력 파일: {output_csv}")

if __name__ == "__main__":
    main() 