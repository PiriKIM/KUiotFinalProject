#!/usr/bin/env python3
"""
데이터베이스 저장 문제 진단 스크립트
"""

import sqlite3
import os
import time
from datetime import datetime
import json

def test_database_connection():
    """데이터베이스 연결 테스트"""
    print("=== 데이터베이스 연결 테스트 ===")
    
    db_path = "pose_grade_data.db"
    
    # 파일 존재 여부 확인
    if os.path.exists(db_path):
        file_size = os.path.getsize(db_path)
        print(f"데이터베이스 파일 존재: {db_path}")
        print(f"파일 크기: {file_size} bytes")
    else:
        print(f"데이터베이스 파일이 존재하지 않음: {db_path}")
        return False
    
    try:
        # 연결 테스트
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 테이블 존재 여부 확인
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"테이블 목록: {[table[0] for table in tables]}")
        
        if 'pose_grade_data' in [table[0] for table in tables]:
            # 테이블 구조 확인
            cursor.execute("PRAGMA table_info(pose_grade_data)")
            columns = cursor.fetchall()
            print(f"테이블 컬럼 수: {len(columns)}")
            
            # 데이터 수 확인
            cursor.execute("SELECT COUNT(*) FROM pose_grade_data")
            count = cursor.fetchone()[0]
            print(f"현재 데이터 수: {count}")
            
            conn.close()
            return True
        else:
            print("pose_grade_data 테이블이 존재하지 않음")
            conn.close()
            return False
            
    except Exception as e:
        print(f"데이터베이스 연결 오류: {e}")
        return False

def test_data_insertion():
    """데이터 삽입 테스트"""
    print("\n=== 데이터 삽입 테스트 ===")
    
    db_path = "pose_grade_data.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 테스트 데이터 준비
        test_landmarks = []
        for i in range(33):
            test_landmarks.extend([0.1 + i*0.01, 0.2 + i*0.01])  # 66개 값
        
        # 컬럼명 생성
        landmark_columns = []
        for i in range(33):
            landmark_columns.extend([f'landmark_{i}_x', f'landmark_{i}_y'])
        
        # SQL 쿼리 생성
        columns = ['timestamp', 'participant_id', 'view_angle', 'pose_grade', 'auto_grade',
                  'neck_angle', 'spine_angle', 'shoulder_asymmetry', 'pelvic_tilt', 
                  'total_score', 'analysis_results'] + landmark_columns
        
        placeholders = ['?'] * len(columns)
        
        # 테스트 값
        test_analysis = {
            'neck': {'neck_angle': 5.0, 'grade': 'A'},
            'spine': {'spine_angle': 2.0, 'is_hunched': False},
            'shoulder': {'height_difference': 0.01, 'is_asymmetric': False},
            'pelvic': {'height_difference': 0.005, 'is_tilted': False}
        }
        
        values = [
            datetime.now().isoformat(),
            'test_user',
            '1',
            'a',
            None,
            5.0,
            2.0,
            0.01,
            0.005,
            85.0,
            json.dumps(test_analysis)
        ] + test_landmarks
        
        # 삽입 실행
        insert_query = f'''
            INSERT INTO pose_grade_data 
            ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        '''
        
        print(f"삽입 쿼리 길이: {len(insert_query)}")
        print(f"값 개수: {len(values)}")
        
        cursor.execute(insert_query, values)
        conn.commit()
        
        # 삽입 확인
        cursor.execute("SELECT COUNT(*) FROM pose_grade_data")
        new_count = cursor.fetchone()[0]
        print(f"삽입 후 데이터 수: {new_count}")
        
        # 최근 삽입된 데이터 확인
        cursor.execute("SELECT * FROM pose_grade_data ORDER BY id DESC LIMIT 1")
        latest = cursor.fetchone()
        if latest:
            print(f"최근 삽입된 데이터 ID: {latest[0]}")
            print(f"참가자 ID: {latest[2]}")
            print(f"시점: {latest[3]}")
            print(f"등급: {latest[4]}")
        
        conn.close()
        print("데이터 삽입 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"데이터 삽입 테스트 실패: {e}")
        return False

def test_concurrent_access():
    """동시 접근 테스트"""
    print("\n=== 동시 접근 테스트 ===")
    
    db_path = "pose_grade_data.db"
    
    def insert_data(thread_id):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 간단한 테스트 데이터
            cursor.execute('''
                INSERT INTO pose_grade_data 
                (timestamp, participant_id, view_angle, pose_grade, neck_angle, spine_angle, shoulder_asymmetry, pelvic_tilt, analysis_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                f'thread_{thread_id}',
                '1',
                'a',
                5.0,
                2.0,
                0.01,
                0.005,
                '{"test": "data"}'
            ))
            
            conn.commit()
            conn.close()
            print(f"스레드 {thread_id}: 삽입 성공")
            return True
            
        except Exception as e:
            print(f"스레드 {thread_id}: 삽입 실패 - {e}")
            return False
    
    import threading
    
    # 5개 스레드로 동시 삽입 테스트
    threads = []
    results = []
    
    for i in range(5):
        thread = threading.Thread(target=lambda i=i: results.append(insert_data(i)))
        threads.append(thread)
        thread.start()
    
    # 모든 스레드 완료 대기
    for thread in threads:
        thread.join()
    
    success_count = sum(results)
    print(f"동시 접근 테스트 결과: {success_count}/5 성공")

def check_file_permissions():
    """파일 권한 확인"""
    print("\n=== 파일 권한 확인 ===")
    
    db_path = "pose_grade_data.db"
    
    if os.path.exists(db_path):
        # 파일 권한 확인
        stat_info = os.stat(db_path)
        print(f"파일 권한: {oct(stat_info.st_mode)[-3:]}")
        print(f"소유자: {stat_info.st_uid}")
        print(f"그룹: {stat_info.st_gid}")
        
        # 쓰기 권한 확인
        if os.access(db_path, os.W_OK):
            print("파일 쓰기 권한: 있음")
        else:
            print("파일 쓰기 권한: 없음")
            
        # 디렉토리 권한 확인
        dir_path = os.path.dirname(os.path.abspath(db_path))
        if os.access(dir_path, os.W_OK):
            print("디렉토리 쓰기 권한: 있음")
        else:
            print("디렉토리 쓰기 권한: 없음")
    else:
        print("데이터베이스 파일이 존재하지 않음")

def main():
    print("데이터베이스 저장 문제 진단 시작")
    print("=" * 50)
    
    # 1. 파일 권한 확인
    check_file_permissions()
    
    # 2. 데이터베이스 연결 테스트
    if not test_database_connection():
        print("데이터베이스 연결 실패. 문제를 해결한 후 다시 시도하세요.")
        return
    
    # 3. 데이터 삽입 테스트
    if not test_data_insertion():
        print("데이터 삽입 실패. 데이터베이스 구조를 확인하세요.")
        return
    
    # 4. 동시 접근 테스트
    test_concurrent_access()
    
    print("\n" + "=" * 50)
    print("진단 완료!")

if __name__ == "__main__":
    main() 