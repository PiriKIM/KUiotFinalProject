#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실시간 자세 분석 데이터베이스 마이그레이션 스크립트
"""

from apps.app import create_app, db
from apps.crud.models import RealtimePostureRecord

def migrate_realtime_tables():
    """실시간 분석 테이블 생성"""
    app = create_app()
    
    with app.app_context():
        try:
            # 새로운 테이블 생성
            db.create_all()
            print("✅ 실시간 분석 테이블이 성공적으로 생성되었습니다.")
            
            # 테이블 확인
            tables = db.engine.table_names()
            print(f"📊 현재 데이터베이스 테이블: {tables}")
            
            if 'realtime_posture_record' in tables:
                print("✅ realtime_posture_record 테이블이 존재합니다.")
            else:
                print("❌ realtime_posture_record 테이블이 생성되지 않았습니다.")
                
        except Exception as e:
            print(f"❌ 마이그레이션 오류: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("🔄 실시간 자세 분석 데이터베이스 마이그레이션을 시작합니다...")
    success = migrate_realtime_tables()
    
    if success:
        print("🎉 마이그레이션이 완료되었습니다!")
    else:
        print("💥 마이그레이션에 실패했습니다.") 