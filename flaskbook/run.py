from flask import redirect, url_for, session, Response
import webbrowser
import time
import threading
import requests
from io import BytesIO

# Flask 앱 생성
from apps.app import create_app
app = create_app()

@app.route('/')
def root():
    # 로그인하지 않은 사용자는 로그인 페이지로 리다이렉트
    if 'user_id' not in session:
        return redirect(url_for('auth.login'))
    # 로그인한 사용자는 메인 페이지로 리다이렉트
    return redirect(url_for('crud.index'))

@app.route('/esp32-stream')
def esp32_proxy():
    """ESP32-CAM 스트림을 프록시하는 엔드포인트"""
    print(f"ESP32 프록시 요청 받음: {time.strftime('%H:%M:%S')}")
    
    try:
        # ESP32-CAM에서 스트림 가져오기 (GET 요청 사용)
        print("ESP32-CAM에 연결 시도 중...")
        response = requests.get('http://192.168.0.102:81/stream', 
                              timeout=30,  # 타임아웃 30초로 증가
                              stream=True,
                              headers={
                                  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                                  'Accept': 'image/jpeg,image/*,*/*;q=0.8',
                                  'Accept-Encoding': 'identity',
                                  'Connection': 'keep-alive',
                                  'Cache-Control': 'no-cache'
                              })
        
        print(f"ESP32-CAM 응답 상태: {response.status_code}")
        print(f"ESP32-CAM 응답 헤더: {dict(response.headers)}")
        
        response.raise_for_status()
        
        # 스트리밍 응답 생성
        def generate():
            try:
                chunk_count = 0
                for chunk in response.iter_content(chunk_size=4096):  # 청크 크기 감소
                    if chunk:
                        chunk_count += 1
                        if chunk_count % 50 == 0:  # 로그 빈도 감소
                            print(f"스트리밍 청크 전송: {chunk_count}")
                        yield chunk
            except Exception as e:
                print(f"스트리밍 생성 오류: {e}")
        
        content_type = response.headers.get('content-type', 'image/jpeg')
        print(f"응답 Content-Type: {content_type}")
        
        return Response(
            generate(),
            content_type=content_type,
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Cache-Control',
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0',
                'Connection': 'keep-alive',
                'Transfer-Encoding': 'chunked'
            }
        )
    except requests.exceptions.Timeout:
        print(f"ESP32-CAM 연결 시간 초과: {time.strftime('%H:%M:%S')}")
        return "ESP32-CAM 연결 시간 초과", 504
    except requests.exceptions.ConnectionError as e:
        print(f"ESP32-CAM 연결 실패: {e} - {time.strftime('%H:%M:%S')}")
        return "ESP32-CAM 연결 실패", 503
    except Exception as e:
        print(f"ESP32 프록시 오류: {e} - {time.strftime('%H:%M:%S')}")
        return f"ESP32-CAM 오류: {str(e)}", 500

@app.route('/esp32-stream', methods=['OPTIONS'])
def esp32_proxy_options():
    """CORS preflight 요청 처리"""
    return Response(
        status=200,
        headers={
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Cache-Control'
        }
    )

@app.route('/test-esp32')
def test_esp32():
    """ESP32-CAM 스트림 테스트 페이지"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ESP32-CAM 테스트</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .test-container { max-width: 800px; margin: 0 auto; }
            img { max-width: 100%; border: 2px solid #ddd; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .success { background: #d4edda; color: #155724; }
            .error { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="test-container">
            <h1>ESP32-CAM 스트림 테스트</h1>
            <div id="status" class="status">로딩 중...</div>
            <img id="esp32-img" src="/esp32-stream" 
                 onload="document.getElementById('status').innerHTML='✅ ESP32-CAM 연결 성공!'; document.getElementById('status').className='status success';" 
                 onerror="document.getElementById('status').innerHTML='❌ ESP32-CAM 연결 실패'; document.getElementById('status').className='status error';" />
            <p><a href="/">메인 페이지로 돌아가기</a></p>
        </div>
    </body>
    </html>
    '''

def open_browser():
    time.sleep(1)
    webbrowser.open("http://localhost:5000")

if __name__ == '__main__':
    print("🚀 ESP32-CAM 자세 분석 시스템 시작")
    print("📡 ESP32-CAM IP 주소: 192.168.0.102")
    print("🌐 브라우저에서 http://localhost:5000 접속")
    print("🔄 프록시 엔드포인트: http://localhost:5000/esp32-stream")
    threading.Thread(target=open_browser).start()
    app.run(debug=True)
