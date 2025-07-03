document.addEventListener("DOMContentLoaded", () => {
  // DOM 요소들
  const esp32Stream = document.getElementById("esp32-stream");
  const streamPlaceholder = document.getElementById("stream-placeholder");
  const startBtn = document.getElementById("start-btn");
  const stopBtn = document.getElementById("stop-btn");
  const connectionStatus = document.getElementById("connection-status");
  const currentStateElement = document.getElementById("current-state");
  const resultBox = document.getElementById("result");
  const overallScore = document.getElementById("overall-score");
  const scoreNumber = document.getElementById("score-number");
  const scoreGrade = document.getElementById("score-grade");

  // ESP32-CAM 설정
  // 실제 ESP32-CAM IP 주소로 변경하세요!
  // 예: "http://192.168.1.100:81/stream"
  const ESP32_STREAM_URL = "http://localhost:5000/esp32-stream";  // 프록시 서버 사용
  const ANALYSIS_INTERVAL = 2000; // 2초마다 분석
  
  // 상태 변수들
  let isAnalyzing = false;
  let analysisInterval = null;
  let latestLandmarks = null;
  let connectionRetryCount = 0;
  const MAX_RETRY_COUNT = 5;

  // 전역 상태 관리
  window.esp32State = {
    isConnected: false,
    isTestingConnection: false,
    isLoadingStream: false,
    lastRequestTime: 0,
    connectionRetryCount: 0,
    maxRetries: 3,
    requestCount: 0
  };

  const MIN_REQUEST_INTERVAL = 10000; // 10초 간격으로 제한

  // ESP32 연결 테스트 - 영상 스트림은 HTML에서만 띄움
  function testESP32Connection() {
    // 연결 상태만 표시 (영상 스트림은 HTML에서 바로 띄움)
    updateConnectionStatus("connected", "ESP32-CAM 연결됨");
    startBtn.disabled = false;
    currentStateElement.textContent = "ESP32-CAM 연결됨 - 분석 시작 버튼을 클릭하세요";
  }

  // ESP32 스트림 표시 - 테스트 페이지처럼 단순하게
  // function showESP32Stream() { ... } // 삭제

  // 스트림 로드 에러 처리
  function handleStreamLoadError(errorType) {
    window.esp32State.connectionRetryCount++;
    
    if (window.esp32State.connectionRetryCount < MAX_RETRY_COUNT) {
      updateConnectionStatus("disconnected", `ESP32-CAM 연결 실패 (${window.esp32State.connectionRetryCount}/${MAX_RETRY_COUNT}) - ${errorType}`);
      console.log(`${errorType} - 5초 후 재시도 (${window.esp32State.connectionRetryCount}/${MAX_RETRY_COUNT})`);
      setTimeout(testESP32Connection, 5000); // 5초 후 재시도
    } else {
      window.esp32State.isConnected = false;
      updateConnectionStatus("disconnected", "ESP32-CAM 연결 실패 - IP 주소를 확인하세요");
      startBtn.disabled = true;
      
      // 사용자에게 IP 주소 변경 안내
      resultBox.innerHTML = `
        <div style="text-align: center; padding: 20px; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">
          <h3 style="color: #721c24;">ESP32-CAM 연결 실패</h3>
          <p>현재 IP 주소: <strong>${ESP32_STREAM_URL}</strong></p>
          <p>에러 타입: <strong>${errorType}</strong></p>
          <p>다음 사항을 확인해주세요:</p>
          <ul style="text-align: left; display: inline-block;">
            <li>ESP32-CAM이 켜져 있는지 확인</li>
            <li>ESP32-CAM IP 주소가 올바른지 확인</li>
            <li>같은 Wi-Fi 네트워크에 연결되어 있는지 확인</li>
            <li>브라우저에서 직접 <a href="${ESP32_STREAM_URL}" target="_blank">${ESP32_STREAM_URL}</a> 접속 테스트</li>
            <li>Flask 서버가 실행 중인지 확인</li>
          </ul>
          <p><small>IP 주소 변경이 필요한 경우 script.js 파일의 ESP32_STREAM_URL을 수정하세요.</small></p>
        </div>`;
    }
  }

  // 연결 상태 업데이트
  function updateConnectionStatus(status, message) {
    connectionStatus.className = `connection-status status-${status}`;
    connectionStatus.textContent = message;
  }

  // ESP32 분석 시작
  window.startESP32Analysis = function() {
    if (isAnalyzing) return;
    
    console.log("ESP32-CAM 분석 시작");
    isAnalyzing = true;
    startBtn.style.display = "none";
    stopBtn.style.display = "inline-block";
    
    currentStateElement.textContent = "ESP32-CAM 분석 중...";
    
    // 분석 인터벌 시작
    analysisInterval = setInterval(() => {
      if (!isAnalyzing) return;
      
      captureAndAnalyzeFrame();
    }, ANALYSIS_INTERVAL);
    
    // 첫 번째 분석 즉시 실행
    captureAndAnalyzeFrame();
  };

  // ESP32 분석 중지
  window.stopESP32Analysis = function() {
    console.log("ESP32-CAM 분석 중지");
    isAnalyzing = false;
    startBtn.style.display = "inline-block";
    stopBtn.style.display = "none";
    
    if (analysisInterval) {
      clearInterval(analysisInterval);
      analysisInterval = null;
    }
    
    currentStateElement.textContent = "분석 중지됨";
    overallScore.style.display = "none";
    resultBox.innerHTML = "";
  };

  // 프레임 캡처 및 분석
  function captureAndAnalyzeFrame() {
    try {
      // 캔버스 생성 및 ESP32 스트림에서 프레임 캡처
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      // ESP32 스트림 크기 설정
      canvas.width = esp32Stream.naturalWidth || 640;
      canvas.height = esp32Stream.naturalHeight || 480;
      
      // 스트림이 로드되지 않았으면 스킵
      if (esp32Stream.naturalWidth === 0) {
        console.log("ESP32 스트림이 아직 로드되지 않음");
        return;
      }
      
      // 프레임 캡처
      ctx.drawImage(esp32Stream, 0, 0, canvas.width, canvas.height);
      
      // Blob으로 변환하여 서버로 전송
      canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append("frame", blob, "frame.jpg");
        
        // 분석 요청
        fetch("/crud/analyze", {
          method: "POST",
          body: formData
        })
        .then(res => {
          if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
          }
          return res.json();
        })
        .then(data => {
          console.log("ESP32 분석 결과:", data);
          updateUI(data);
        })
        .catch(err => {
          console.error("ESP32 분석 오류:", err);
          currentStateElement.textContent = "분석 오류 발생";
        });
      }, "image/jpeg", 0.8);
      
    } catch (err) {
      console.error("프레임 캡처 오류:", err);
    }
  }

  // UI 업데이트
  function updateUI(data) {
    console.log("서버 응답 데이터:", data);
    
    if (data.landmarks) {
      latestLandmarks = data.landmarks;
    }
    
    if (data.state) {
      currentStateElement.textContent = data.state_message || data.state;
    }

    // 점수 표시
    if (data.overall_score !== undefined && data.overall_score !== null) {
      console.log("점수 표시:", data.overall_score, data.overall_grade);
      scoreNumber.textContent = data.overall_score;
      scoreGrade.textContent = `등급: ${data.overall_grade}`;
      overallScore.style.display = "block";

      // 상세 결과 표시
      let resultHTML = `
        <div style="text-align: center; padding: 20px;">
          <h3 style="color: #96ceb4;">자세 분석 완료</h3>
          <p><strong>종합 점수: ${data.overall_score}점 (${data.overall_grade}등급)</strong></p>`;
      
      if (data.neck) {
        resultHTML += `<p>목: ${data.neck.grade_description || '분석됨'}</p>`;
      }
      if (data.spine) {
        resultHTML += `<p>척추: ${data.spine.is_hunched ? "굽음" : "정상"}</p>`;
      }
      if (data.shoulder) {
        resultHTML += `<p>어깨: ${data.shoulder.is_asymmetric ? "비대칭" : "정상"}</p>`;
      }
      if (data.pelvic) {
        resultHTML += `<p>골반: ${data.pelvic.is_tilted ? "기울어짐" : "정상"}</p>`;
      }
      
      resultHTML += `</div>`;
      resultBox.innerHTML = resultHTML;
      
    } else if (data.state === "analyzing_side_pose") {
      // 분석 중일 때
      overallScore.style.display = "none";
      resultBox.innerHTML = `
        <div style="text-align: center; padding: 20px;">
          <h3 style="color: #ffa500;">자세 분석 중...</h3>
          <p>측면 자세를 유지해주세요</p>
        </div>`;
    } else {
      // 다른 상태일 때
      overallScore.style.display = "none";
      if (data.state_message) {
        resultBox.innerHTML = `
          <div style="text-align: center; padding: 20px;">
            <h3 style="color: #667eea;">${data.state_message}</h3>
          </div>`;
      }
    }
  }

  // 페이지 로드 시 ESP32 연결 테스트 시작
  console.log("ESP32-CAM 시스템 초기화 중...");
  
  // 즉시 연결 테스트 시작
  testESP32Connection();
});
