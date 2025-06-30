import sqlite3
import pandas as pd
import json
from datetime import datetime

def check_database(db_path="pose_grade_data.db"):
    """데이터베이스 상태 확인"""
    try:
        conn = sqlite3.connect(db_path)
        
        # 테이블 존재 여부 확인
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("=== 데이터베이스 상태 확인 ===")
        print(f"데이터베이스 파일: {db_path}")
        print(f"테이블 목록: {[table[0] for table in tables]}")
        
        if 'pose_grade_data' in [table[0] for table in tables]:
            # 데이터 통계
            df = pd.read_sql_query("SELECT * FROM pose_grade_data", conn)
            
            print(f"\n총 데이터 수: {len(df)}개")
            print(f"수집 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
            
            # 시점별 분포
            print(f"\n시점별 분포:")
            view_counts = df['view_angle'].value_counts()
            for view, count in view_counts.items():
                view_name = "정면" if view == "1" else "측면" if view == "2" else view
                print(f"  {view_name} ({view}): {count}개 ({count/len(df)*100:.1f}%)")
            
            # 등급별 분포
            print(f"\n등급별 분포:")
            grade_counts = df['pose_grade'].value_counts()
            grade_names = {
                'a': 'A등급 (완벽)',
                'b': 'B등급 (양호)',
                'c': 'C등급 (보통)',
                'd': 'D등급 (나쁨)',
                'e': '특수 자세'
            }
            for grade, count in grade_counts.items():
                grade_name = grade_names.get(grade, grade)
                print(f"  {grade_name}: {count}개 ({count/len(df)*100:.1f}%)")
            
            # 참가자별 분포
            print(f"\n참가자별 분포:")
            participant_counts = df['participant_id'].value_counts()
            for participant, count in participant_counts.items():
                print(f"  {participant}: {count}개")
            
            # 특성 통계
            print(f"\n특성 통계:")
            numeric_columns = ['neck_angle', 'spine_angle', 'shoulder_asymmetry', 'pelvic_tilt']
            for col in numeric_columns:
                if col in df.columns:
                    print(f"  {col}: 평균={df[col].mean():.3f}, 표준편차={df[col].std():.3f}")
            
            # 시점별 등급 분포
            print(f"\n시점별 등급 분포:")
            pivot_table = pd.crosstab(df['view_angle'], df['pose_grade'], margins=True)
            print(pivot_table)
            
        else:
            print("pose_grade_data 테이블이 존재하지 않습니다.")
            
        conn.close()
        
    except Exception as e:
        print(f"데이터베이스 확인 중 오류: {e}")

def export_to_csv(db_path="pose_grade_data.db", output_path="pose_grade_data.csv"):
    """데이터베이스를 CSV로 내보내기"""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM pose_grade_data", conn)
        
        # JSON 컬럼들을 파싱하여 개별 컬럼으로 분리
        if 'analysis_results' in df.columns:
            analysis_data = []
            for _, row in df.iterrows():
                try:
                    analysis = json.loads(row['analysis_results'])
                    analysis_data.append({
                        'neck_angle_detailed': analysis['neck']['neck_angle'],
                        'neck_vertical_deviation': analysis['neck']['vertical_deviation'],
                        'spine_angle_detailed': analysis['spine']['spine_angle'],
                        'spine_hunched': analysis['spine']['is_hunched'],
                        'shoulder_height_diff': analysis['shoulder']['height_difference'],
                        'shoulder_angle': analysis['shoulder']['shoulder_angle'],
                        'shoulder_asymmetric': analysis['shoulder']['is_asymmetric'],
                        'pelvic_height_diff': analysis['pelvic']['height_difference'],
                        'pelvic_angle': analysis['pelvic']['pelvic_angle'],
                        'pelvic_tilted': analysis['pelvic']['is_tilted']
                    })
                except:
                    analysis_data.append({
                        'neck_angle_detailed': None,
                        'neck_vertical_deviation': None,
                        'spine_angle_detailed': None,
                        'spine_hunched': None,
                        'shoulder_height_diff': None,
                        'shoulder_angle': None,
                        'shoulder_asymmetric': None,
                        'pelvic_height_diff': None,
                        'pelvic_angle': None,
                        'pelvic_tilted': None
                    })
            
            analysis_df = pd.DataFrame(analysis_data)
            df = pd.concat([df, analysis_df], axis=1)
        
        # 시점과 등급 이름 변환
        df['view_angle_name'] = df['view_angle'].map({'1': '정면', '2': '측면'})
        grade_names = {
            'a': 'A등급 (완벽)',
            'b': 'B등급 (양호)',
            'c': 'C등급 (보통)',
            'd': 'D등급 (나쁨)',
            'e': '특수 자세'
        }
        df['pose_grade_name'] = df['pose_grade'].map(grade_names)
        
        # CSV로 저장
        df.to_csv(output_path, index=False)
        print(f"데이터가 {output_path}에 저장되었습니다.")
        print(f"총 {len(df)}개 행, {len(df.columns)}개 컬럼")
        
        conn.close()
        
    except Exception as e:
        print(f"CSV 내보내기 중 오류: {e}")

def clean_database(db_path="pose_grade_data.db"):
    """데이터베이스 정리 (중복 데이터 제거 등)"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 원본 데이터 수
        cursor.execute("SELECT COUNT(*) FROM pose_grade_data")
        original_count = cursor.fetchone()[0]
        
        # 중복 데이터 제거 (동일한 timestamp와 participant_id)
        cursor.execute('''
            DELETE FROM pose_grade_data 
            WHERE id NOT IN (
                SELECT MIN(id) 
                FROM pose_grade_data 
                GROUP BY timestamp, participant_id, view_angle, pose_grade
            )
        ''')
        
        # NULL 값이 있는 행 제거
        cursor.execute('''
            DELETE FROM pose_grade_data 
            WHERE neck_angle IS NULL 
            OR spine_angle IS NULL 
            OR shoulder_asymmetry IS NULL 
            OR pelvic_tilt IS NULL
        ''')
        
        conn.commit()
        
        # 정리 후 데이터 수
        cursor.execute("SELECT COUNT(*) FROM pose_grade_data")
        cleaned_count = cursor.fetchone()[0]
        
        print(f"데이터베이스 정리 완료:")
        print(f"  원본 데이터: {original_count}개")
        print(f"  정리 후 데이터: {cleaned_count}개")
        print(f"  제거된 데이터: {original_count - cleaned_count}개")
        
        conn.close()
        
    except Exception as e:
        print(f"데이터베이스 정리 중 오류: {e}")

def analyze_data_quality(db_path="pose_grade_data.db"):
    """데이터 품질 분석"""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM pose_grade_data", conn)
        
        print("=== 데이터 품질 분석 ===")
        
        # 각 시점별 등급 분포
        print("\n1. 시점별 등급 분포:")
        for view in ['1', '2']:
            view_name = "정면" if view == "1" else "측면"
            view_data = df[df['view_angle'] == view]
            if len(view_data) > 0:
                print(f"\n{view_name} ({len(view_data)}개):")
                grade_dist = view_data['pose_grade'].value_counts()
                for grade, count in grade_dist.items():
                    grade_names = {'a': 'A등급', 'b': 'B등급', 'c': 'C등급', 'd': 'D등급', 'e': '특수 자세'}
                    print(f"  {grade_names.get(grade, grade)}: {count}개")
        
        # 특성값 분포 분석
        print("\n2. 특성값 분포:")
        numeric_cols = ['neck_angle', 'spine_angle', 'shoulder_asymmetry', 'pelvic_tilt']
        for col in numeric_cols:
            if col in df.columns:
                print(f"\n{col}:")
                print(f"  범위: {df[col].min():.3f} ~ {df[col].max():.3f}")
                print(f"  평균: {df[col].mean():.3f}")
                print(f"  표준편차: {df[col].std():.3f}")
        
        # 데이터 불균형 확인
        print("\n3. 데이터 불균형 분석:")
        grade_counts = df['pose_grade'].value_counts()
        total = len(df)
        for grade, count in grade_counts.items():
            percentage = count / total * 100
            grade_names = {'a': 'A등급', 'b': 'B등급', 'c': 'C등급', 'd': 'D등급', 'e': '특수 자세'}
            grade_name = grade_names.get(grade, grade)
            print(f"  {grade_name}: {count}개 ({percentage:.1f}%)")
        
        # 특수 자세(e등급) 상세 분석
        if 'e' in df['pose_grade'].values:
            print("\n4. 특수 자세 상세 분석:")
            e_data = df[df['pose_grade'] == 'e']
            print(f"  특수 자세 데이터: {len(e_data)}개")
            
            # 특수 자세의 특성값 분포
            for col in numeric_cols:
                if col in e_data.columns:
                    print(f"  {col} (특수 자세): 평균={e_data[col].mean():.3f}, 표준편차={e_data[col].std():.3f}")
        
        conn.close()
        
    except Exception as e:
        print(f"데이터 품질 분석 중 오류: {e}")

def main():
    print("=== 자세 등급 데이터베이스 관리 도구 ===")
    
    while True:
        print("\n1. 데이터베이스 상태 확인")
        print("2. CSV로 내보내기")
        print("3. 데이터베이스 정리")
        print("4. 데이터 품질 분석")
        print("5. 종료")
        
        choice = input("\n선택하세요 (1-5): ")
        
        if choice == '1':
            check_database()
        elif choice == '2':
            output_path = input("출력 파일명을 입력하세요 (기본: pose_grade_data.csv): ")
            if not output_path:
                output_path = "pose_grade_data.csv"
            export_to_csv(output_path=output_path)
        elif choice == '3':
            confirm = input("데이터베이스를 정리하시겠습니까? (y/N): ")
            if confirm.lower() == 'y':
                clean_database()
        elif choice == '4':
            analyze_data_quality()
        elif choice == '5':
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다.")

if __name__ == "__main__":
    main() 