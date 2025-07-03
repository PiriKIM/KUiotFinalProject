import sqlite3
import pandas as pd
import os

def export_p5_to_csv():
    """p5 데이터베이스를 CSV로 내보내기"""
    p5_db_path = "test_grade/collector/db/p5/pose_grade_data.db"
    p5_csv_path = "test_grade/collector/db/p5/p5_data.csv"
    
    # p5 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(p5_csv_path), exist_ok=True)
    
    try:
        # p5 데이터베이스 연결
        conn = sqlite3.connect(p5_db_path)
        
        # 테이블 목록 확인
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        print(f"p5 데이터베이스의 테이블들: {tables['name'].tolist()}")
        
        # pose_grade_data 테이블 찾기
        target_table = None
        for table_name in tables['name']:
            if table_name == 'pose_grade_data':
                target_table = table_name
                break
        
        if target_table:
            print(f"테이블 '{target_table}'을 CSV로 내보내는 중...")
            
            # 테이블 데이터 읽기
            df = pd.read_sql_query(f"SELECT * FROM {target_table}", conn)
            print(f"p5 데이터 행 수: {len(df)}")
            
            # CSV로 저장
            df.to_csv(p5_csv_path, index=False)
            print(f"p5 데이터가 {p5_csv_path}에 저장되었습니다.")
            
            return df
        else:
            print("pose_grade_data 테이블을 찾을 수 없습니다.")
            return None
            
    except Exception as e:
        print(f"p5 데이터베이스 처리 중 오류: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def merge_with_existing():
    """기존 merge.csv에 p5 데이터 추가"""
    merge_csv_path = "test_grade/collector/db/merge/merged.csv"
    p5_csv_path = "test_grade/collector/db/p5/p5_data.csv"
    new_merge_path = "test_grade/collector/db/merge/merged_with_p5.csv"
    
    try:
        # 기존 merge.csv 읽기
        if os.path.exists(merge_csv_path):
            existing_df = pd.read_csv(merge_csv_path)
            print(f"기존 merge.csv 행 수: {len(existing_df)}")
        else:
            print("기존 merge.csv 파일을 찾을 수 없습니다.")
            return
        
        # p5 데이터 읽기
        if os.path.exists(p5_csv_path):
            p5_df = pd.read_csv(p5_csv_path)
            print(f"p5 데이터 행 수: {len(p5_df)}")
        else:
            print("p5 CSV 파일을 찾을 수 없습니다. 먼저 export_p5_to_csv()를 실행하세요.")
            return
        
        # 데이터 병합
        merged_df = pd.concat([existing_df, p5_df], ignore_index=True)
        print(f"병합된 데이터 행 수: {len(merged_df)}")
        
        # 새로운 파일로 저장
        merged_df.to_csv(new_merge_path, index=False)
        print(f"병합된 데이터가 {new_merge_path}에 저장되었습니다.")
        
        # 기존 파일 백업 및 교체
        backup_path = merge_csv_path + ".backup"
        os.rename(merge_csv_path, backup_path)
        os.rename(new_merge_path, merge_csv_path)
        print(f"기존 파일이 {backup_path}로 백업되고 새로운 데이터로 교체되었습니다.")
        
    except Exception as e:
        print(f"병합 중 오류: {e}")

if __name__ == "__main__":
    print("=== p5 데이터 처리 시작 ===")
    
    # 1단계: p5 데이터베이스를 CSV로 내보내기
    print("\n1단계: p5 데이터베이스를 CSV로 내보내기")
    p5_df = export_p5_to_csv()
    
    if p5_df is not None:
        # 2단계: 기존 merge.csv에 추가
        print("\n2단계: 기존 merge.csv에 p5 데이터 추가")
        merge_with_existing()
    
    print("\n=== 처리 완료 ===") 