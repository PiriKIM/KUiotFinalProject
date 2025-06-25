#!/usr/bin/env python3
"""
Flask 애플리케이션 실행 스크립트
사용법: python run.py
"""

from apps.app import create_app

app = create_app()

if __name__ == '__main__':
    print("🚀 자세 분석 시스템을 시작합니다...")
    print("📱 웹 브라우저에서 http://localhost:5000/auth/login 으로 접속하세요")
    print("⏹️  종료하려면 Ctrl+C를 누르세요")
    print("-" * 50)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    ) 