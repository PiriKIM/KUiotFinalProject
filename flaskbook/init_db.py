#!/usr/bin/env python3
"""
데이터베이스 초기화 스크립트
사용법: python init_db.py
"""

from apps.app import create_app, db
from apps.crud.models import User, PostureRecord

def init_database():
    """데이터베이스 초기화"""
    app = create_app()
    
    with app.app_context():
        # 데이터베이스 테이블 생성
        db.create_all()
        print("✅ 데이터베이스 테이블이 생성되었습니다.")
        
        # 테스트 사용자 생성 (선택사항)
        create_test_user = input("테스트 사용자를 생성하시겠습니까? (y/n): ").lower().strip()
        
        if create_test_user == 'y':
            test_user = User(
                username='test_user',
                email='test@example.com'
            )
            test_user.password = 'password123'
            
            try:
                db.session.add(test_user)
                db.session.commit()
                print("✅ 테스트 사용자가 생성되었습니다.")
                print("   사용자명: test_user")
                print("   비밀번호: password123")
            except Exception as e:
                db.session.rollback()
                print(f"❌ 테스트 사용자 생성 실패: {e}")
        
        print("\n🎉 데이터베이스 초기화가 완료되었습니다!")
        print("이제 Flask 애플리케이션을 실행할 수 있습니다.")

if __name__ == '__main__':
    init_database() 