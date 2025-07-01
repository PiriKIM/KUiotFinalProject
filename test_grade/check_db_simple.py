import sqlite3
import os

def check_db():
    db_path = "pose_grade_data.db"
    
    if not os.path.exists(db_path):
        print(f"데이터베이스 파일이 존재하지 않습니다: {db_path}")
        return
    
    file_size = os.path.getsize(db_path)
    print(f"데이터베이스 파일 크기: {file_size} 바이트")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 테이블 목록 확인
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"테이블 목록: {tables}")
        
        if 'pose_grade_data' in [table[0] for table in tables]:
            # 데이터 수 확인
            cursor.execute("SELECT COUNT(*) FROM pose_grade_data")
            count = cursor.fetchone()[0]
            print(f"총 데이터 수: {count}개")
            
            if count > 0:
                # 최근 데이터 몇 개 확인
                cursor.execute("SELECT * FROM pose_grade_data ORDER BY id DESC LIMIT 5")
                recent_data = cursor.fetchall()
                print("\n최근 데이터 5개:")
                for row in recent_data:
                    print(f"  ID: {row[0]}, 참가자: {row[2]}, 시점: {row[3]}, 등급: {row[4]}")
        else:
            print("pose_grade_data 테이블이 존재하지 않습니다.")
            
        conn.close()
        
    except Exception as e:
        print(f"오류: {e}")

if __name__ == "__main__":
    check_db() 