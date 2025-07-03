import sqlite3
import os

def debug_database():
    db_path = "pose_grade_data.db"
    
    print(f"데이터베이스 파일 크기: {os.path.getsize(db_path)} 바이트")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 테이블 구조 확인
    cursor.execute("PRAGMA table_info(pose_grade_data)")
    columns = cursor.fetchall()
    print(f"\n테이블 구조:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")
    
    # 데이터 수 확인
    cursor.execute("SELECT COUNT(*) FROM pose_grade_data")
    count = cursor.fetchone()[0]
    print(f"\n총 데이터 수: {count}개")
    
    if count > 0:
        # 최근 데이터 3개 확인
        cursor.execute("SELECT id, participant_id, view_angle, pose_grade, timestamp FROM pose_grade_data ORDER BY id DESC LIMIT 3")
        recent_data = cursor.fetchall()
        print(f"\n최근 데이터 3개:")
        for row in recent_data:
            print(f"  ID: {row[0]}, 참가자: {row[1]}, 시점: {row[2]}, 등급: {row[3]}, 시간: {row[4]}")
        
        # 등급별 분포
        cursor.execute("SELECT pose_grade, COUNT(*) FROM pose_grade_data GROUP BY pose_grade")
        grade_dist = cursor.fetchall()
        print(f"\n등급별 분포:")
        for grade, count in grade_dist:
            print(f"  {grade}등급: {count}개")
    else:
        # 테이블이 비어있는지 확인
        cursor.execute("SELECT * FROM pose_grade_data LIMIT 1")
        empty_check = cursor.fetchone()
        print(f"테이블이 비어있음: {empty_check is None}")
        
        # 테이블 삭제 후 재생성 테스트
        print(f"\n테이블 재생성 테스트...")
        cursor.execute("DROP TABLE IF EXISTS pose_grade_data")
        cursor.execute('''
            CREATE TABLE pose_grade_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                participant_id TEXT,
                view_angle TEXT,
                pose_grade TEXT,
                neck_angle REAL,
                spine_angle REAL,
                shoulder_asymmetry REAL,
                pelvic_tilt REAL,
                landmarks TEXT,
                analysis_results TEXT
            )
        ''')
        conn.commit()
        print(f"테이블 재생성 완료")
    
    conn.close()

if __name__ == "__main__":
    debug_database() 