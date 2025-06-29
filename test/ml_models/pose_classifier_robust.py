import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle

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
    """특징 추출 (과적합 방지를 위해 특징 수 제한)"""
    features = []
    
    # 1. 정규화된 좌표 (중요한 부위만 선택)
    normalized_coords = normalize_coordinates(landmarks)
    
    # 머리, 목, 어깨 부위만 선택 (랜드마크 0-12)
    important_landmarks = []
    for i in range(0, 26, 2):  # 0-12번 랜드마크만
        important_landmarks.extend([normalized_coords[i], normalized_coords[i+1]])
    
    features.extend(important_landmarks)
    
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
    
    # 4. 대칭성 특징
    if landmarks[11] != -1 and landmarks[12] != -1:
        shoulder_symmetry = abs(landmarks[11+1] - landmarks[12+1])
        features.append(shoulder_symmetry)
    else:
        features.append(0)
    
    return features

def prepare_data(csv_file):
    """데이터 준비 및 특징 추출"""
    print("데이터 로딩 중...")
    df = pd.read_csv(csv_file)
    
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
    return X, y

def train_robust_model(X, y):
    """과적합을 방지한 강화된 모델 학습"""
    print("강화된 모델 학습 중...")
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 특성 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 과적합 방지를 위한 모델 설정
    model = RandomForestClassifier(
        n_estimators=50,  # 트리 수 제한
        max_depth=10,     # 깊이 제한
        min_samples_split=5,  # 분할 최소 샘플 수
        min_samples_leaf=2,   # 리프 최소 샘플 수
        random_state=42
    )
    
    # 교차 검증으로 성능 평가
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv)
    
    print(f"교차 검증 점수: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # 최종 모델 학습
    model.fit(X_train_scaled, y_train)
    
    # 테스트 세트에서 평가
    y_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"테스트 정확도: {test_accuracy:.3f}")
    
    # 상세 평가
    print("\n분류 보고서:")
    print(classification_report(y_test, y_pred, target_names=['정면', '측면']))
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['정면', '측면'], 
               yticklabels=['정면', '측면'])
    plt.title('혼동 행렬 (강화된 모델)')
    plt.ylabel('실제')
    plt.xlabel('예측')
    plt.savefig('confusion_matrix_robust.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 특징 중요도 분석
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n특징 중요도 (상위 10개):")
        feature_names = []
        for i in range(33):
            feature_names.extend([f'norm_landmark_{i}_x', f'norm_landmark_{i}_y'])
        feature_names.extend(['angle_neck_shoulder_elbow', 'shoulder_ratio', 'shoulder_symmetry'])
        
        for i in range(min(10, len(indices))):
            print(f"{i+1:2d}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        # 중요도 시각화
        plt.figure(figsize=(12, 8))
        plt.title("특징 중요도 (강화된 모델)")
        plt.bar(range(min(10, len(indices))), 
               importances[indices[:10]])
        plt.xticks(range(min(10, len(indices))), 
                  [feature_names[i] for i in indices[:10]], 
                  rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('feature_importance_robust.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return model, scaler, test_accuracy

def main():
    """메인 실행 함수"""
    # 데이터 준비
    X, y = prepare_data('test/data/P1_0628/pose_landmarks_P1.csv')
    
    # 강화된 모델 학습
    model, scaler, accuracy = train_robust_model(X, y)
    
    print(f"\n강화된 모델 최종 정확도: {accuracy:.3f}")
    print("과적합 방지 조치:")
    print("- 특징 수 제한 (28개)")
    print("- 모델 복잡도 제한 (max_depth=10)")
    print("- 교차 검증 적용")
    print("- 특성 스케일링")
    print("모델 학습 완료!")

    with open('pose_classifier_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'is_trained': True
        }, f)
    print("모델이 pose_classifier_model.pkl로 저장되었습니다.")

if __name__ == "__main__":
    main() 