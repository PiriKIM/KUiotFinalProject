#!/usr/bin/env python3
"""
테스트 데이터 생성 스크립트
사용법: python3 create_test_data.py
"""

from apps.app import create_app, db
from apps.crud.models import User, PostureRecord
from datetime import datetime, timedelta
import random

def create_test_data():
    """테스트 데이터 생성"""
    app = create_app()
    
    with app.app_context():
        # 기존 테스트 사용자 확인
        test_user = User.query.filter_by(username='test_user').first()
        
        if not test_user:
            # 새 테스트 사용자 생성
            test_user = User(
                username='test_user',
                email='test@example.com'
            )
            test_user.password = 'password123'
            db.session.add(test_user)
            db.session.commit()
            print("✅ 테스트 사용자가 생성되었습니다.")
        else:
            print("✅ 기존 테스트 사용자를 사용합니다.")
        
        # 기존 기록 삭제 (선택사항)
        existing_records = PostureRecord.query.filter_by(user_id=test_user.id).all()
        if existing_records:
            delete_old = input(f"기존 분석 기록 {len(existing_records)}개를 삭제하시겠습니까? (y/n): ").lower().strip()
            if delete_old == 'y':
                for record in existing_records:
                    db.session.delete(record)
                db.session.commit()
                print("✅ 기존 기록이 삭제되었습니다.")
        
        # 테스트 분석 기록 생성
        num_records = int(input("생성할 테스트 기록 수를 입력하세요 (기본값: 20): ") or "20")
        
        print(f"📊 {num_records}개의 테스트 분석 기록을 생성합니다...")
        
        for i in range(num_records):
            # 랜덤 날짜 (최근 30일 내)
            days_ago = random.randint(0, 30)
            analysis_date = datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23), minutes=random.randint(0, 59))
            
            # 랜덤 자세 분석 결과
            neck_angle = random.uniform(15, 45)
            neck_grade = random.choice(['A', 'B', 'C', 'D'])
            neck_descriptions = {
                'A': '정상',
                'B': '약간 굽음',
                'C': '굽음',
                'D': '심한 굽음'
            }
            
            spine_is_hunched = random.choice([True, False])
            spine_angle = random.uniform(0, 30) if spine_is_hunched else random.uniform(0, 10)
            
            shoulder_is_asymmetric = random.choice([True, False])
            shoulder_height_difference = random.uniform(0, 5) if shoulder_is_asymmetric else 0
            
            pelvic_is_tilted = random.choice([True, False])
            pelvic_angle = random.uniform(0, 15) if pelvic_is_tilted else random.uniform(0, 5)
            
            spine_is_twisted = random.choice([True, False])
            spine_alignment = random.uniform(0, 20) if spine_is_twisted else random.uniform(0, 5)
            
            # 분석 기록 생성
            record = PostureRecord(
                user_id=test_user.id,
                analysis_date=analysis_date,
                neck_angle=neck_angle,
                neck_grade=neck_grade,
                neck_description=neck_descriptions[neck_grade],
                spine_is_hunched=spine_is_hunched,
                spine_angle=spine_angle,
                shoulder_is_asymmetric=shoulder_is_asymmetric,
                shoulder_height_difference=shoulder_height_difference,
                pelvic_is_tilted=pelvic_is_tilted,
                pelvic_angle=pelvic_angle,
                spine_is_twisted=spine_is_twisted,
                spine_alignment=spine_alignment
            )
            
            # 종합 점수 계산
            record.overall_score = record.calculate_overall_score()
            record.overall_grade = record.calculate_overall_grade()
            
            db.session.add(record)
        
        try:
            db.session.commit()
            print(f"✅ {num_records}개의 테스트 분석 기록이 생성되었습니다!")
            print(f"📱 이제 http://localhost:5000/auth/login 에서 로그인하여 확인하세요")
            print(f"   사용자명: test_user")
            print(f"   비밀번호: password123")
        except Exception as e:
            db.session.rollback()
            print(f"❌ 테스트 데이터 생성 실패: {e}")

if __name__ == '__main__':
    create_test_data() 