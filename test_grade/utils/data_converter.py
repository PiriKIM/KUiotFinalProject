import json
import pandas as pd
import numpy as np
import sqlite3

def json_landmarks_to_columns(landmarks_json):
    """JSON 형태의 랜드마크를 개별 컬럼으로 변환"""
    landmarks_data = json.loads(landmarks_json)
    
    # 33개 랜드마크의 x, y 좌표만 추출 (기존 모델과 호환)
    columns = {}
    for i, landmark in enumerate(landmarks_data):
        columns[f'landmark_{i}_x'] = landmark['x']
        columns[f'landmark_{i}_y'] = landmark['y']
    
    return columns

def convert_pose_grade_db_to_csv(db_path, output_csv_path):
    """pose_grade_data.db를 CSV로 변환 (기존 모델과 호환)"""
    conn = sqlite3.connect(db_path)
    
    # 모든 데이터 로드 (이미 66개 개별 컬럼으로 저장됨)
    df = pd.read_sql_query("""
        SELECT * FROM pose_grade_data
    """, conn)
    
    # 라벨 매핑 (기존 모델과 호환)
    view_to_label = {'1': 1, '2': 2, '3': 3}  # 1:정면, 2:측면, 3:기타
    df['label'] = df['view_angle'].map(view_to_label)
    
    # 필요한 컬럼만 선택하여 CSV로 저장
    landmark_columns = []
    for i in range(33):
        landmark_columns.extend([f'landmark_{i}_x', f'landmark_{i}_y'])
    
    result_columns = ['timestamp', 'participant_id', 'label', 'pose_grade'] + landmark_columns
    result_df = df[result_columns]
    
    # CSV로 저장
    result_df.to_csv(output_csv_path, index=False)
    print(f"변환 완료: {output_csv_path}")
    print(f"총 {len(result_df)}개 행, {len(result_df.columns)}개 컬럼")
    
    conn.close()
    return result_df

def convert_pose_grade_db_to_ml_ready(db_path, output_csv_path):
    """pose_grade_data.db를 머신러닝 모델용 CSV로 변환"""
    conn = sqlite3.connect(db_path)
    
    # 자세 등급별 데이터 로드 (A, B, C, D 등급만)
    df = pd.read_sql_query("""
        SELECT * FROM pose_grade_data
        WHERE pose_grade IN ('a', 'b', 'c', 'd')
    """, conn)
    
    # 자세 등급을 라벨로 변환
    grade_to_label = {'a': 1, 'b': 2, 'c': 3, 'd': 4}  # A=1, B=2, C=3, D=4
    df['label'] = df['pose_grade'].map(grade_to_label)
    
    # 필요한 컬럼만 선택
    landmark_columns = []
    for i in range(33):
        landmark_columns.extend([f'landmark_{i}_x', f'landmark_{i}_y'])
    
    result_columns = ['timestamp', 'participant_id', 'label', 'pose_grade'] + landmark_columns
    result_df = df[result_columns]
    
    # CSV로 저장
    result_df.to_csv(output_csv_path, index=False)
    print(f"머신러닝용 변환 완료: {output_csv_path}")
    print(f"총 {len(result_df)}개 행, {len(result_df.columns)}개 컬럼")
    print(f"라벨 분포:\n{result_df['label'].value_counts()}")
    
    conn.close()
    return result_df

def analyze_pose_grade_data(db_path):
    """pose_grade_data.db 분석"""
    conn = sqlite3.connect(db_path)
    
    # 기본 통계
    df = pd.read_sql_query("""
        SELECT pose_grade, view_angle, participant_id, COUNT(*) as count
        FROM pose_grade_data
        GROUP BY pose_grade, view_angle, participant_id
    """, conn)
    
    print("=== pose_grade_data.db 분석 결과 ===")
    print(f"총 데이터 수: {df['count'].sum()}")
    print("\n자세 등급별 분포:")
    grade_dist = df.groupby('pose_grade')['count'].sum()
    for grade, count in grade_dist.items():
        print(f"  {grade}: {count}개")
    
    print("\n시점별 분포:")
    view_dist = df.groupby('view_angle')['count'].sum()
    view_names = {'1': '정면', '2': '측면', '3': '기타 상태'}
    for view, count in view_dist.items():
        view_name = view_names.get(str(view), f'알 수 없음({view})')
        print(f"  {view_name}({view}): {count}개")
    
    print("\n참가자별 분포:")
    participant_dist = df.groupby('participant_id')['count'].sum()
    for participant, count in participant_dist.items():
        print(f"  {participant}: {count}개")
    
    # 랜드마크 데이터 품질 분석
    print("\n랜드마크 데이터 품질 분석:")
    landmark_df = pd.read_sql_query("""
        SELECT * FROM pose_grade_data LIMIT 1000
    """, conn)
    
    # -1 값 개수 확인 (누락된 랜드마크)
    landmark_columns = []
    for i in range(33):
        landmark_columns.extend([f'landmark_{i}_x', f'landmark_{i}_y'])
    
    missing_count = (landmark_df[landmark_columns] == -1).sum().sum()
    total_landmarks = len(landmark_df) * 66  # 33개 랜드마크 × 2개 좌표
    missing_rate = missing_count / total_landmarks * 100
    
    print(f"  총 랜드마크 수: {total_landmarks}")
    print(f"  누락된 랜드마크 수: {missing_count}")
    print(f"  누락률: {missing_rate:.2f}%")
    
    conn.close()
    return df

def export_landmark_statistics(db_path, output_csv_path):
    """랜드마크 통계 정보를 CSV로 내보내기"""
    conn = sqlite3.connect(db_path)
    
    # 랜드마크 컬럼명 생성
    landmark_columns = []
    for i in range(33):
        landmark_columns.extend([f'landmark_{i}_x', f'landmark_{i}_y'])
    
    # 기본 통계 계산
    df = pd.read_sql_query(f"""
        SELECT pose_grade, view_angle, participant_id, 
               COUNT(*) as frame_count,
               AVG(neck_angle) as avg_neck_angle,
               AVG(spine_angle) as avg_spine_angle,
               AVG(shoulder_asymmetry) as avg_shoulder_asymmetry,
               AVG(pelvic_tilt) as avg_pelvic_tilt
        FROM pose_grade_data
        GROUP BY pose_grade, view_angle, participant_id
    """, conn)
    
    # CSV로 저장
    df.to_csv(output_csv_path, index=False)
    print(f"랜드마크 통계 내보내기 완료: {output_csv_path}")
    
    conn.close()
    return df

if __name__ == "__main__":
    # 사용 예시
    db_path = "pose_grade_data.db"
    
    # 데이터 분석
    analyze_pose_grade_data(db_path)
    
    # 기존 모델과 호환되는 CSV로 변환
    convert_pose_grade_db_to_csv(db_path, "pose_grade_converted.csv")
    
    # 머신러닝 모델용 CSV로 변환 (자세 등급 분류)
    convert_pose_grade_db_to_ml_ready(db_path, "pose_grade_ml_ready.csv")
    
    # 랜드마크 통계 내보내기
    export_landmark_statistics(db_path, "pose_grade_statistics.csv") 