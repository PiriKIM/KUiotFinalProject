import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class PoseClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def calculate_angles(self, landmarks):
        """랜드마크 간의 각도 계산"""
        angles = []
        
        # 목-어깨-팔꿈치 각도 (랜드마크 0-11-12)
        if landmarks[0] != -1 and landmarks[11] != -1 and landmarks[12] != -1:
            angle = self._calculate_angle(
                landmarks[0:2], landmarks[11:13], landmarks[12:14]
            )
            angles.append(angle)
        else:
            angles.append(0)
            
        # 어깨-팔꿈치-손목 각도 (랜드마크 11-13-15)
        if landmarks[11] != -1 and landmarks[13] != -1 and landmarks[15] != -1:
            angle = self._calculate_angle(
                landmarks[11:13], landmarks[13:15], landmarks[15:17]
            )
            angles.append(angle)
        else:
            angles.append(0)
            
        # 어깨-팔꿈치-손목 각도 (랜드마크 12-14-16)
        if landmarks[12] != -1 and landmarks[14] != -1 and landmarks[16] != -1:
            angle = self._calculate_angle(
                landmarks[12:14], landmarks[14:16], landmarks[16:18]
            )
            angles.append(angle)
        else:
            angles.append(0)
            
        return angles
    
    def _calculate_angle(self, point1, point2, point3):
        """세 점으로 이루어진 각도 계산"""
        a = np.array(point1)
        b = np.array(point2)
        c = np.array(point3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def calculate_ratios(self, landmarks):
        """신체 비율 계산"""
        ratios = []
        
        # 어깨 너비 계산 (랜드마크 11, 12)
        if landmarks[11] != -1 and landmarks[12] != -1:
            shoulder_width = abs(landmarks[11] - landmarks[12])
        else:
            shoulder_width = 1.0
            
        # 어깨 높이 계산
        if landmarks[11] != -1 and landmarks[12] != -1:
            shoulder_height = abs(landmarks[11+1] - landmarks[12+1])
        else:
            shoulder_height = 1.0
            
        # 어깨 비율
        ratios.append(shoulder_width / max(shoulder_height, 0.001))
        
        # 머리 크기 대비 어깨 너비
        if landmarks[0] != -1 and landmarks[1] != -1:
            head_size = abs(landmarks[0] - landmarks[1])
            ratios.append(shoulder_width / max(head_size, 0.001))
        else:
            ratios.append(1.0)
            
        return ratios
    
    def normalize_coordinates(self, landmarks):
        """상대 좌표로 정규화"""
        normalized = []
        
        # 어깨 중심점 계산
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
    
    def extract_features(self, landmarks):
        """모든 특징 추출"""
        features = []
        
        # 1. 정규화된 좌표
        normalized_coords = self.normalize_coordinates(landmarks)
        features.extend(normalized_coords)
        
        # 2. 각도 특징
        angles = self.calculate_angles(landmarks)
        features.extend(angles)
        
        # 3. 비율 특징
        ratios = self.calculate_ratios(landmarks)
        features.extend(ratios)
        
        # 4. 대칭성 특징
        symmetry_features = self.calculate_symmetry(landmarks)
        features.extend(symmetry_features)
        
        return features
    
    def calculate_symmetry(self, landmarks):
        """신체 대칭성 계산"""
        symmetry_features = []
        
        # 좌우 어깨 대칭성
        if landmarks[11] != -1 and landmarks[12] != -1:
            shoulder_symmetry = abs(landmarks[11+1] - landmarks[12+1])
            symmetry_features.append(shoulder_symmetry)
        else:
            symmetry_features.append(0)
            
        # 좌우 팔꿈치 대칭성
        if landmarks[13] != -1 and landmarks[14] != -1:
            elbow_symmetry = abs(landmarks[13+1] - landmarks[14+1])
            symmetry_features.append(elbow_symmetry)
        else:
            symmetry_features.append(0)
            
        return symmetry_features
    
    def prepare_data(self, csv_file):
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
            
            features = self.extract_features(landmarks)
            features_list.append(features)
            labels.append(row['label'])
        
        X = np.array(features_list)
        y = np.array(labels)
        
        # 특징 이름 생성
        self.feature_names = []
        for i in range(33):
            self.feature_names.extend([f'norm_landmark_{i}_x', f'norm_landmark_{i}_y'])
        self.feature_names.extend(['angle_neck_shoulder_elbow', 'angle_shoulder_elbow_wrist_l', 'angle_shoulder_elbow_wrist_r'])
        self.feature_names.extend(['shoulder_ratio', 'head_shoulder_ratio'])
        self.feature_names.extend(['shoulder_symmetry', 'elbow_symmetry'])
        
        print(f"추출된 특징 수: {X.shape[1]}")
        return X, y
    
    def train(self, X, y):
        """모델 학습"""
        print("모델 학습 중...")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 특성 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 여러 모델 시도
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42)
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            print(f"\n{name} 학습 중...")
            
            # 교차 검증
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            print(f"교차 검증 점수: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # 테스트 세트에서 평가
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            test_score = accuracy_score(y_test, y_pred)
            print(f"테스트 정확도: {test_score:.3f}")
            
            if test_score > best_score:
                best_score = test_score
                best_model = model
        
        self.model = best_model
        print(f"\n최종 모델 정확도: {best_score:.3f}")
        
        # 상세 평가
        y_pred = self.model.predict(X_test_scaled)
        print("\n분류 보고서:")
        print(classification_report(y_test, y_pred, target_names=['정면', '측면']))
        
        # 혼동 행렬
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['정면', '측면'], 
                   yticklabels=['정면', '측면'])
        plt.title('혼동 행렬')
        plt.ylabel('실제')
        plt.xlabel('예측')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return best_score
    
    def predict(self, landmarks):
        """새로운 데이터 예측"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        features = self.extract_features(landmarks)
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return prediction, probability
    
    def feature_importance(self):
        """특징 중요도 분석"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            print("Random Forest 모델이 아니거나 학습되지 않았습니다.")
            return
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("특징 중요도 (상위 20개):")
        for i in range(min(20, len(indices))):
            print(f"{i+1:2d}. {self.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        # 중요도 시각화
        plt.figure(figsize=(12, 8))
        plt.title("특징 중요도")
        plt.bar(range(min(20, len(indices))), 
               importances[indices[:20]])
        plt.xticks(range(min(20, len(indices))), 
                  [self.feature_names[i] for i in indices[:20]], 
                  rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """메인 실행 함수"""
    # 모델 생성
    classifier = PoseClassifier()
    
    # 데이터 준비
    X, y = classifier.prepare_data('test/data/P1_0628/pose_landmarks_P1.csv')
    
    # 모델 학습
    accuracy = classifier.train(X, y)
    
    # 특징 중요도 분석
    classifier.feature_importance()
    
    print(f"\n최종 모델 정확도: {accuracy:.3f}")
    print("모델 학습 완료!")

if __name__ == "__main__":
    main() 