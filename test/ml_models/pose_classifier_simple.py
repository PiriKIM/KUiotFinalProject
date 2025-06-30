import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def normalize_coordinates(landmarks):
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

def calculate_angles(landmarks):
    """주요 각도 계산"""
    angles = []
    
    # 목-어깨-팔꿈치 각도 (랜드마크 0-11-12)
    if landmarks[0] != -1 and landmarks[11] != -1 and landmarks[12] != -1:
        # 간단한 각도 계산
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

def extract_features(landmarks):
    """특징 추출"""
    features = []
    
    # 1. 정규화된 좌표
    normalized_coords = normalize_coordinates(landmarks)
    features.extend(normalized_coords)
    
    # 2. 각도 특징
    angles = calculate_angles(landmarks)
    features.extend(angles)
    
    # 3. 어깨 비율
    if landmarks[11] != -1 and landmarks[12] != -1:
        shoulder_width = abs(landmarks[11] - landmarks[12])
        shoulder_height = abs(landmarks[11+1] - landmarks[12+1])
        shoulder_ratio = shoulder_width / max(shoulder_height, 0.001)
        features.append(shoulder_ratio)
    else:
        features.append(1.0)
    
    return features

def main():
    print("자세 분류 모델 학습 시작...")
    
    # 데이터 로딩
    df = pd.read_csv('test/data/P1_0628/pose_landmarks_P1.csv')
    
    # 라벨 1(정면)과 2(측면)만 사용
    df_filtered = df[df['label'].isin([1, 2])].copy()
    
    print(f"총 데이터: {len(df)}")
    print(f"필터링된 데이터: {len(df_filtered)}")
    print(f"라벨 분포:\n{df_filtered['label'].value_counts()}")
    
    # 특징 추출
    features_list = []
    labels = []
    
    for idx, row in df_filtered.iterrows():
        landmarks = []
        for i in range(33):
            x = row[f'landmark_{i}_x']
            y = row[f'landmark_{i}_y']
            landmarks.extend([x, y])
        
        features = extract_features(landmarks)
        features_list.append(features)
        labels.append(row['label'])
    
    X = np.array(features_list)
    y = np.array(labels)
    
    print(f"추출된 특징 수: {X.shape[1]}")
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 모델 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n모델 정확도: {accuracy:.3f}")
    print("\n분류 보고서:")
    print(classification_report(y_test, y_pred, target_names=['정면', '측면']))
    
    # 특징 중요도
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n특징 중요도 (상위 10개):")
    for i in range(min(10, len(indices))):
        if i < 66:  # 정규화된 좌표
            landmark_idx = i // 2
            coord_type = 'x' if i % 2 == 0 else 'y'
            print(f"{i+1:2d}. norm_landmark_{landmark_idx}_{coord_type}: {importances[indices[i]]:.4f}")
        elif i == 66:  # 각도
            print(f"{i+1:2d}. angle_neck_shoulder_elbow: {importances[indices[i]]:.4f}")
        else:  # 어깨 비율
            print(f"{i+1:2d}. shoulder_ratio: {importances[indices[i]]:.4f}")
    
    print("\n모델 학습 완료!")

if __name__ == "__main__":
    main()
