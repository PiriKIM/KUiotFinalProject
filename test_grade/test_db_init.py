import sqlite3
from datetime import datetime
import json

def test_db_init():
    """데이터베이스 초기화 테스트"""
    db_path = "pose_grade_data.db"
    
    try:
        # 데이터베이스 연결
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 테이블 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pose_grade_data (
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
        
        # 테스트 데이터 삽입
        test_data = {
            'timestamp': datetime.now().isoformat(),
            'participant_id': 'test_user',
            'view_angle': '1',
            'pose_grade': 'a',
            'neck_angle': 5.0,
            'spine_angle': 2.0,
            'shoulder_asymmetry': 0.01,
            'pelvic_tilt': 0.005,
            'landmarks': json.dumps([{'x': 0.1, 'y': 0.2, 'z': 0.3, 'visibility': 0.9}]),
            'analysis_results': json.dumps({'test': 'data'})
        }
        
        cursor.execute('''
            INSERT INTO pose_grade_data 
            (timestamp, participant_id, view_angle, pose_grade, neck_angle, spine_angle, 
             shoulder_asymmetry, pelvic_tilt, landmarks, analysis_results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            test_data['timestamp'],
            test_data['participant_id'],
            test_data['view_angle'],
            test_data['pose_grade'],
            test_data['neck_angle'],
            test_data['spine_angle'],
            test_data['shoulder_asymmetry'],
            test_data['pelvic_tilt'],
            test_data['landmarks'],
            test_data['analysis_results']
        ))
        
        conn.commit()
        
        # 데이터 확인
        cursor.execute("SELECT COUNT(*) FROM pose_grade_data")
        count = cursor.fetchone()[0]
        print(f"테이블 생성 및 테스트 데이터 삽입 완료!")
        print(f"총 데이터 수: {count}개")
        
        # 테이블 구조 확인
        cursor.execute("PRAGMA table_info(pose_grade_data)")
        columns = cursor.fetchall()
        print("\n테이블 구조:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
        
        conn.close()
        
    except Exception as e:
        print(f"오류: {e}")

if __name__ == "__main__":
    test_db_init() 