import pandas as pd
import os

def merge_csv_files():
    """두 개의 CSV 파일을 합치는 함수"""
    
    # 파일 경로
    file1 = "test/data/P1_0706/pose_landmarks_P1.csv"
    file2 = "test/data/P1_0706/pose_landmarks_P1_additional.csv"
    output_file = "test/data/P1_0706/pose_landmarks_P1_merged.csv"
    
    print("CSV 파일 합치기 시작...")
    
    # 파일 존재 확인
    if not os.path.exists(file1):
        print(f"파일이 존재하지 않습니다: {file1}")
        return
    
    if not os.path.exists(file2):
        print(f"파일이 존재하지 않습니다: {file2}")
        return
    
    # CSV 파일 읽기
    print(f"첫 번째 파일 읽기: {file1}")
    df1 = pd.read_csv(file1)
    print(f"첫 번째 파일 데이터 수: {len(df1)}")
    
    print(f"두 번째 파일 읽기: {file2}")
    df2 = pd.read_csv(file2)
    print(f"두 번째 파일 데이터 수: {len(df2)}")
    
    # 데이터 합치기
    print("데이터 합치기...")
    merged_df = pd.concat([df1, df2], ignore_index=True)
    
    # ID 재설정 (1부터 시작)
    merged_df['id'] = range(1, len(merged_df) + 1)
    
    print(f"합쳐진 데이터 수: {len(merged_df)}")
    
    # 라벨 분포 확인
    print("\n라벨 분포:")
    print(merged_df['label'].value_counts().sort_index())
    
    # 사람별 분포 확인
    print("\n사람별 분포:")
    print(merged_df['person_id'].value_counts().sort_index())
    
    # 합쳐진 파일 저장
    print(f"\n합쳐진 파일 저장: {output_file}")
    merged_df.to_csv(output_file, index=False)
    
    print("파일 합치기 완료!")
    print(f"저장 위치: {output_file}")
    
    return output_file

if __name__ == "__main__":
    merge_csv_files() 