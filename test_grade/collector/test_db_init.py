import sqlite3
import os

def test_database_init():
    """데이터베이스 초기화 테스트"""
    db_path = "pose_grade_data.db"
    
    print("=== 데이터베이스 초기화 테스트 ===")
    print(f"데이터베이스 파일: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 기존 테이블 삭제 (스키마 변경을 위해)
        cursor.execute("DROP TABLE IF EXISTS pose_grade_data")
        print("기존 테이블 삭제 완료")
        
        # 33개 랜드마크의 x, y 좌표를 개별 컬럼으로 생성
        landmark_columns = []
        for i in range(33):
            landmark_columns.extend([f'landmark_{i}_x REAL', f'landmark_{i}_y REAL'])
        
        landmark_columns_str = ', '.join(landmark_columns)
        
        # 테이블 생성
        create_query = f'''
            CREATE TABLE IF NOT EXISTS pose_grade_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                participant_id TEXT,
                view_angle TEXT,
                pose_grade TEXT,
                auto_grade TEXT,
                neck_angle REAL,
                spine_angle REAL,
                shoulder_asymmetry REAL,
                pelvic_tilt REAL,
                total_score REAL,
                analysis_results TEXT,
                {landmark_columns_str}
            )
        '''
        
        cursor.execute(create_query)
        print("테이블 생성 완료")
        
        # 테이블 구조 확인
        cursor.execute("PRAGMA table_info(pose_grade_data)")
        columns = cursor.fetchall()
        print(f"총 컬럼 수: {len(columns)}개")
        
        # 컬럼 목록 출력 (처음 10개만)
        print("컬럼 목록 (처음 10개):")
        for i, col in enumerate(columns[:10]):
            print(f"  {i+1}. {col[1]} ({col[2]})")
        
        if len(columns) > 10:
            print(f"  ... 외 {len(columns)-10}개 컬럼")
        
        conn.commit()
        conn.close()
        
        # 파일 크기 확인
        if os.path.exists(db_path):
            file_size = os.path.getsize(db_path)
            print(f"데이터베이스 파일 크기: {file_size} bytes")
        
        print("데이터베이스 초기화 성공!")
        
    except Exception as e:
        print(f"데이터베이스 초기화 실패: {e}")

if __name__ == "__main__":
    test_database_init() 