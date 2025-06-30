import sqlite3
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys

# 상위 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PoseGradeModelTrainer:
    def __init__(self, db_path="pose_grade_data.db"):
        self.db_path = db_path
        self.model = None
        self.feature_names = []
        
    def load_data(self):
        """데이터베이스에서 데이터 로드"""
        conn = sqlite3.connect(self.db_path)
        
        # 기본 특성 데이터 로드 (모든 등급 포함)
        query = '''
            SELECT participant_id, view_angle, pose_grade, neck_angle, spine_angle, 
                   shoulder_asymmetry, pelvic_tilt, analysis_results
            FROM pose_grade_data
            WHERE analysis_results IS NOT NULL
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"로드된 데이터: {len(df)}개 샘플")
        print(f"시점별 분포:")
        view_counts = df['view_angle'].value_counts()
        for view, count in view_counts.items():
            view_name = "정면" if view == "1" else "측면"
            print(f"  {view_name}: {count}개")
        
        print(f"등급별 분포:")
        grade_counts = df['pose_grade'].value_counts()
        grade_names = {
            'a': 'A등급 (완벽)', 
            'b': 'B등급 (양호)', 
            'c': 'C등급 (보통)', 
            'd': 'D등급 (나쁨)',
            'e': '특수 자세'
        }
        for grade, count in grade_counts.items():
            print(f"  {grade_names.get(grade, grade)}: {count}개")
        
        return df
        
    def extract_features(self, df):
        """특성 추출 및 전처리"""
        features = []
        labels = []
        
        for _, row in df.iterrows():
            try:
                # 기본 특성
                basic_features = [
                    row['neck_angle'],
                    row['spine_angle'],
                    row['shoulder_asymmetry'],
                    row['pelvic_tilt']
                ]
                
                # 시점 정보 (원-핫 인코딩)
                view_features = [1 if row['view_angle'] == '1' else 0,  # 정면
                               1 if row['view_angle'] == '2' else 0]   # 측면
                
                # 분석 결과에서 추가 특성 추출
                analysis = json.loads(row['analysis_results'])
                
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
                
                # 모든 특성 결합
                all_features = (basic_features + view_features + neck_features + 
                              spine_features + shoulder_features + pelvic_features)
                
                features.append(all_features)
                labels.append(row['pose_grade'])
                
            except Exception as e:
                print(f"데이터 처리 중 오류: {e}")
                continue
        
        # 특성 이름 정의
        self.feature_names = [
            'neck_angle', 'spine_angle', 'shoulder_asymmetry', 'pelvic_tilt',
            'view_front', 'view_side',
            'neck_vertical_deviation', 'neck_angle_detailed',
            'spine_angle_detailed', 'spine_hunched',
            'shoulder_height_diff', 'shoulder_angle', 'shoulder_asymmetric',
            'pelvic_height_diff', 'pelvic_angle', 'pelvic_tilted'
        ]
        
        return np.array(features), np.array(labels)
        
    def train_model(self, X, y):
        """모델 훈련"""
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Random Forest 모델 훈련
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # 예측 및 평가
        y_pred = self.model.predict(X_test)
        
        # 성능 평가
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n모델 정확도: {accuracy:.4f}")
        print(f"\n분류 보고서:\n{classification_report(y_test, y_pred)}")
        
        return X_test, y_test, y_pred
        
    def plot_confusion_matrix(self, y_test, y_pred, save_path="confusion_matrix_grade.png"):
        """혼동 행렬 시각화"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['a', 'b', 'c', 'd', 'e'],
                   yticklabels=['a', 'b', 'c', 'd', 'e'])
        plt.title('자세 등급 분류 혼동 행렬 (특수 자세 포함)')
        plt.xlabel('예측 등급')
        plt.ylabel('실제 등급')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_feature_importance(self, save_path="feature_importance_grade.png"):
        """특성 중요도 시각화"""
        if self.model is None:
            print("모델이 훈련되지 않았습니다.")
            return
            
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('특성 중요도')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), 
                  [self.feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('특성')
        plt.ylabel('중요도')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_model(self, model_path="pose_grade_model.pkl"):
        """모델 저장"""
        if self.model is None:
            print("모델이 훈련되지 않았습니다.")
            return
            
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"모델이 {model_path}에 저장되었습니다.")
        
    def train_and_evaluate(self):
        """전체 훈련 및 평가 과정"""
        print("=== 자세 등급 분류 모델 훈련 (특수 자세 포함) ===")
        
        # 데이터 로드
        df = self.load_data()
        if len(df) == 0:
            print("훈련할 데이터가 없습니다.")
            return
            
        # 특성 추출
        X, y = self.extract_features(df)
        print(f"추출된 특성 수: {X.shape[1]}")
        
        # 모델 훈련
        X_test, y_test, y_pred = self.train_model(X, y)
        
        # 결과 시각화
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_feature_importance()
        
        # 모델 저장
        self.save_model()
        
        print("\n모델 훈련이 완료되었습니다!")

def main():
    trainer = PoseGradeModelTrainer()
    trainer.train_and_evaluate()

if __name__ == "__main__":
    main() 