# ===============================================
# 📌 3클래스 자세 분류 모듈 (4way 모델)
#
# ✅ 특징:
# - 정면, 좌측면, 우측면 3클래스 분류
# - MediaPipe 랜드마크 기반 특징 추출
# - RandomForest 분류기 사용
# - 파이프라인 통합용 모듈
# ===============================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PoseClassifier4Way:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.classes = ['정면', '좌측면', '우측면']
        
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
        
        # 5. 어깨 방향 특징 (좌측면/우측면 구분용)
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
    
    def prepare_data(self, csv_file):
        """CSV 파일에서 데이터 준비"""
        print(f"데이터 로드 중: {csv_file}")
        
        # CSV 파일 읽기
        df = pd.read_csv(csv_file)
        print(f"총 데이터 수: {len(df)}")
        
        # 라벨 분포 확인
        label_counts = df['label'].value_counts().sort_index()
        print("라벨 분포:")
        for label, count in label_counts.items():
            label_name = ['기타', '정면', '좌측면', '우측면'][label]
            print(f"  라벨 {label} ({label_name}): {count}개")
        
        # 라벨 0 (기타) 제외하고 3클래스만 사용
        df_filtered = df[df['label'] > 0].copy()
        print(f"필터링된 데이터 수: {len(df_filtered)}")
        
        # 특징 추출
        X = []
        y = []
        
        for _, row in df_filtered.iterrows():
            # 랜드마크 좌표 추출 (33개 랜드마크 * 2좌표 = 66개)
            landmarks = []
            for i in range(33):
                x_col = f'landmark_{i}_x'
                y_col = f'landmark_{i}_y'
                if x_col in row and y_col in row:
                    landmarks.extend([row[x_col], row[y_col]])
                else:
                    landmarks.extend([-1, -1])  # 랜드마크가 없으면 -1
            
            # 특징 추출
            features = self.extract_features(landmarks)
            X.append(features)
            
            # 라벨 조정 (1,2,3 -> 0,1,2)
            y.append(row['label'] - 1)
        
        return np.array(X), np.array(y)
    
    def train_model(self, X, y):
        """모델 훈련"""
        print("모델 훈련 시작...")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 특성 스케일링
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # RandomForest 분류기 훈련
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # 성능 평가
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 교차 검증
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        print(f"테스트 정확도: {accuracy:.3f}")
        print(f"교차 검증 점수: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # 분류 리포트
        print("\n분류 리포트:")
        print(classification_report(y_test, y_pred, target_names=self.classes))
        
        # 혼동 행렬
        cm = confusion_matrix(y_test, y_pred)
        print("\n혼동 행렬:")
        print(cm)
        
        # 특징 중요도
        feature_importance = self.model.feature_importances_
        print(f"\n총 특징 수: {len(feature_importance)}")
        print("상위 10개 특징 중요도:")
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. 특징 {idx}: {feature_importance[idx]:.4f}")
        
        self.is_trained = True
        return accuracy, cv_scores.mean()
    
    def save_model(self, model_path):
        """모델 저장"""
        if not self.is_trained:
            print("모델이 훈련되지 않았습니다.")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'classes': self.classes
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"모델이 {model_path}에 저장되었습니다.")
    
    def load_model(self, model_path):
        """모델 로드"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.classes = model_data.get('classes', ['정면', '좌측면', '우측면'])
            
            print(f"모델이 {model_path}에서 로드되었습니다.")
            return True
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return False
    
    def predict(self, landmarks):
        """단일 예측"""
        if not self.is_trained:
            return None, None
        
        features = self.extract_features(landmarks)
        features_scaled = self.scaler.transform([features])
        
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return prediction, probability
    
    def create_visualizations(self, X, y, output_dir):
        """시각화 생성"""
        if not self.is_trained:
            print("모델이 훈련되지 않았습니다.")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 혼동 행렬 시각화
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        cm = confusion_matrix(y, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title('3클래스 자세 분류 혼동 행렬')
        plt.ylabel('실제')
        plt.xlabel('예측')
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrix_4way.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 특징 중요도 시각화
        feature_importance = self.model.feature_importances_
        top_indices = np.argsort(feature_importance)[-15:][::-1]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_indices)), feature_importance[top_indices])
        plt.yticks(range(len(top_indices)), [f'특징 {i}' for i in top_indices])
        plt.xlabel('중요도')
        plt.title('상위 15개 특징 중요도')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_path / 'feature_importance_4way.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"시각화가 {output_dir}에 저장되었습니다.")

def main():
    """메인 실행 함수"""
    # 데이터 준비
    csv_file = '../../test/data/P1_0706/pose_landmarks_P1_merged.csv'
    X, y = prepare_data(csv_file)
    
    # 모델 훈련
    classifier = PoseClassifier4Way()
    accuracy, cv_score = classifier.train_model(X, y)
    
    # 모델 저장
    model_path = 'pose_classifier_4way_model.pkl'
    classifier.save_model(model_path)
    
    # 시각화 생성
    output_dir = '../../data/results/visualization'
    classifier.create_visualizations(X, y, output_dir)
    
    print(f"\n훈련 완료!")
    print(f"최종 정확도: {accuracy:.3f}")
    print(f"교차 검증 점수: {cv_score:.3f}")

if __name__ == "__main__":
    main() 