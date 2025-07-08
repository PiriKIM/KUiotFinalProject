document.addEventListener("DOMContentLoaded", () => {
  // DOM 요소들
  const esp32Stream = document.getElementById("esp32-stream");
  const streamPlaceholder = document.getElementById("stream-placeholder");
  const connectionStatus = document.getElementById("connection-status");
  const currentStateElement = document.getElementById("current-state");
  const resultBox = document.getElementById("result");
  const overallScore = document.getElementById("overall-score");
  const scoreNumber = document.getElementById("score-number");
  const scoreGrade = document.getElementById("score-grade");

  // ESP32-CAM 설정
  const ESP32_STREAM_URL = "http://localhost:5000/esp32-stream";
  const ESP32_ANALYZE_STREAM_URL = "http://localhost:5000/esp32-stream-analyze";
  const ANALYSIS_INTERVAL = 1000; // 1초마다 자동 분석
  
  // 상태 변수들
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

  // ESP32 연결 테스트
  function testESP32Connection() {
    updateConnectionStatus("connected", "ESP32-CAM 연결됨");
    currentStateElement.textContent = "ESP32-CAM 연결됨 - 자동 분석 시작";
    
    // 부저 상태 초기화
    updateBuzzerStatus();
    
    // 수동 분석 버튼 표시
    const manualBtn = document.getElementById('manual-analyze-btn');
    if (manualBtn) {
      manualBtn.style.display = 'inline-block';
    }
    
    // 자동 분석 다시 활성화
    startAutoAnalysis();
  }

  // 자동 분석 시작
  function startAutoAnalysis() {
    console.log("자동 분석 시작");
    
    if (currentStateElement) {
      currentStateElement.textContent = "자동 분석 시작 중...";
    }
    
    // 결과 초기화
    if (resultBox) {
      resultBox.innerHTML = `
        <div style="text-align: center; padding: 20px;">
          <h3 style="color: #ffa500;">자동 자세 분석 시작</h3>
          <p>ESP32-CAM에서 1초마다 자동으로 분석합니다...</p>
        </div>`;
    }
    if (overallScore) {
      overallScore.style.display = "none";
    }
    
    // 분석 인터벌 시작 (1초마다)
    analysisInterval = setInterval(() => {
      captureAndAnalyzeFrame();
    }, ANALYSIS_INTERVAL);
    
    // 첫 번째 분석 즉시 실행
    setTimeout(() => {
      captureAndAnalyzeFrame();
    }, 500);
  }

  // 스트림 로드 에러 처리
  function handleStreamLoadError(errorType) {
    window.esp32State.connectionRetryCount++;
    
    if (window.esp32State.connectionRetryCount < MAX_RETRY_COUNT) {
      updateConnectionStatus("disconnected", `ESP32-CAM 연결 실패 (${window.esp32State.connectionRetryCount}/${MAX_RETRY_COUNT}) - ${errorType}`);
      console.log(`${errorType} - 5초 후 재시도 (${window.esp32State.connectionRetryCount}/${MAX_RETRY_COUNT})`);
      setTimeout(testESP32Connection, 5000);
    } else {
      window.esp32State.isConnected = false;
      updateConnectionStatus("disconnected", "ESP32-CAM 연결 실패 - IP 주소를 확인하세요");
      
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
            <li>FastAPI 서버가 실행 중인지 확인</li>
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

  // 프레임 캡처 및 분석
  function captureAndAnalyzeFrame() {
    try {
      // 분석 중 상태 표시
      if (currentStateElement) {
        currentStateElement.textContent = "ESP32-CAM에서 이미지 분석 중...";
      }
      
      // FastAPI 서버에서 직접 분석 요청 (캡처 없이)
      fetch("/crud/analyze", {
        method: "POST",
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({})  // 빈 객체로 간단하게
      })
      .then(res => {
        if (!res.ok) {
          throw new Error(`HTTP error! status: ${res.status}`);
        }
        return res.json();
      })
      .then(data => {
        console.log("ESP32 분석 결과:", data);
        
        // 에러가 있으면 처리
        if (data.error) {
          console.error("분석 에러:", data.error);
          if (currentStateElement) {
            currentStateElement.textContent = `분석 오류: ${data.error}`;
          }
          if (resultBox) {
            resultBox.innerHTML = `
              <div style="text-align: center; padding: 20px; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;">
                <h3 style="color: #721c24;">분석 오류</h3>
                <p>${data.error}</p>
              </div>`;
          }
          return;
        }
        
        // 분석 성공 시 UI 업데이트
        if (data.state === "analyzed") {
          updateUI(data);
          if (currentStateElement) {
            currentStateElement.textContent = `분석 완료: ${data.overall_score}점 (${data.overall_grade}등급)`;
          }
        }
      })
      .catch(error => {
        console.error("분석 요청 실패:", error);
        if (currentStateElement) {
          currentStateElement.textContent = `분석 요청 실패: ${error.message}`;
        }
      });
    } catch (error) {
      console.error("분석 중 오류:", error);
      if (currentStateElement) {
        currentStateElement.textContent = `분석 중 오류: ${error.message}`;
      }
    }
  }

  // 부저 제어 함수들
  window.setVolume = function(volume) {
    const volumeValue = document.getElementById('volumeValue');
    
    if (volume <= 33) {
      volumeValue.textContent = '낮음';
      volumeValue.className = 'badge bg-success fs-6';
    } else if (volume <= 66) {
      volumeValue.textContent = '보통';
      volumeValue.className = 'badge bg-primary fs-6';
    } else {
      volumeValue.textContent = '높음';
      volumeValue.className = 'badge bg-danger fs-6';
    }ㅗ
    
    // 볼륨 설정 API 호출
    fetch('/api/buzzer/volume', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        volume: volume
      })
    })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        console.log(`볼륨 설정: ${volumeValue.textContent} (${volume}%)`);
      } else {
        console.error('볼륨 설정 실패');
      }
    })
    .catch(error => {
      console.error(`볼륨 설정 오류: ${error.message}`);
    });
  };

  window.testBuzzer = function() {
    fetch('/api/buzzer/test')
      .then(response => response.json())
      .then(data => {
        if (data.success) {
          console.log('테스트 비프 전송 성공');
        } else {
          console.error('테스트 비프 전송 실패');
        }
      })
      .catch(error => {
        console.error(`테스트 비프 오류: ${error.message}`);
      });
  };

  window.triggerBuzzer = function(duration) {
    // 현재 선택된 볼륨 값 가져오기
    const volumeValue = document.getElementById('volumeValue');
    let volume = 50; // 기본값
    
    if (volumeValue.textContent === '낮음') {
      volume = 25;
    } else if (volumeValue.textContent === '보통') {
      volume = 50;
    } else if (volumeValue.textContent === '높음') {
      volume = 75;
    }
    
    console.log(`${duration}ms buzzer 트리거 (볼륨: ${volumeValue.textContent})`);
    
    fetch('/api/buzzer/trigger', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        duration: duration,
        volume: volume
      })
    })
    .then(response => response.json())
    .then(data => {
      if (data.success) {
        console.log(`Buzzer 트리거 성공 (${duration}ms)`);
      } else {
        console.error('Buzzer 트리거 실패');
      }
    })
    .catch(error => {
      console.error(`Buzzer 트리거 오류: ${error.message}`);
    });
  };

  // 수동 분석 시작
  window.manualAnalyze = function() {
    console.log("수동 분석 시작");
    
    const manualBtn = document.getElementById('manual-analyze-btn');
    const stopBtn = document.getElementById('stop-analyze-btn');
    
    if (manualBtn) manualBtn.style.display = 'none';
    if (stopBtn) stopBtn.style.display = 'inline-block';
    
    if (currentStateElement) {
      currentStateElement.textContent = "수동 분석 시작 중...";
    }
    
    // 결과 초기화
    if (resultBox) {
      resultBox.innerHTML = `
        <div style="text-align: center; padding: 20px;">
          <h3 style="color: #ffa500;">수동 자세 분석 시작</h3>
          <p>ESP32-CAM에서 이미지를 분석합니다...</p>
        </div>`;
    }
    if (overallScore) {
      overallScore.style.display = "none";
    }
    
    // 분석 실행
    captureAndAnalyzeFrame();
  };

  // 분석 중지
  window.stopAnalyze = function() {
    console.log("분석 중지");
    
    const manualBtn = document.getElementById('manual-analyze-btn');
    const stopBtn = document.getElementById('stop-analyze-btn');
    
    if (manualBtn) manualBtn.style.display = 'inline-block';
    if (stopBtn) stopBtn.style.display = 'none';
    
    if (currentStateElement) {
      currentStateElement.textContent = "분석 중지됨 - 스트림만 표시";
    }
    
    if (resultBox) {
      resultBox.innerHTML = `
        <div style="text-align: center; padding: 20px;">
          <h3 style="color: #667eea;">분석 중지됨</h3>
          <p>수동 분석이 중지되었습니다. 스트림만 표시됩니다.</p>
        </div>`;
    }
    if (overallScore) {
      overallScore.style.display = "none";
    }
  };

  // 부저 상태 업데이트 (기존 함수 수정)
  function updateBuzzerStatus() {
    fetch('/api/buzzer/status')
      .then(res => res.json())
      .then(data => {
        console.log('부저 상태:', data);
        // 필요한 경우 부저 상태 표시 업데이트
      })
      .catch(err => {
        console.error('부저 상태 업데이트 오류:', err);
      });
  }

  // UI 업데이트
  function updateUI(data) {
    console.log("서버 응답 데이터:", data);
    
    if (data.landmarks) {
      latestLandmarks = data.landmarks;
    }
    
    if (data.state && currentStateElement) {
      currentStateElement.textContent = data.state_message || data.state;
    }

    // 부저 상태 업데이트
    if (data.buzzer_triggered !== undefined) {
      updateBuzzerStatus();
      
      // 부저가 트리거되었을 때 시각적 피드백
      if (data.buzzer_triggered) {
        const buzzerSection = document.querySelector('.buzzer-section');
        buzzerSection.style.animation = 'pulse 0.5s ease-in-out';
        setTimeout(() => {
          buzzerSection.style.animation = '';
        }, 500);
        
        // 알림 메시지 표시
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-warning alert-dismissible fade show mt-2';
        alertDiv.innerHTML = `
          <i class="fas fa-exclamation-triangle"></i>
          <strong>나쁜 자세 감지!</strong> 부저가 울렸습니다. 자세를 교정해주세요.
          <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        resultBox.appendChild(alertDiv);
        
        // 3초 후 자동으로 알림 제거
        setTimeout(() => {
          if (alertDiv.parentNode) {
            alertDiv.remove();
          }
        }, 3000);
      }
    }

    // 점수 표시 (단순화)
    if (data.overall_score !== undefined && data.overall_score !== null) {
      console.log("점수 표시:", data.overall_score, data.overall_grade);
      
      if (scoreNumber) scoreNumber.textContent = data.overall_score;
      if (scoreGrade) scoreGrade.textContent = `등급: ${data.overall_grade}`;
      if (overallScore) overallScore.style.display = "block";

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
      if (resultBox) resultBox.innerHTML = resultHTML;
      
    } else if (data.state === "analyzing_side_pose") {
      // 분석 중일 때
      if (overallScore) overallScore.style.display = "none";
      if (resultBox) {
        resultBox.innerHTML = `
          <div style="text-align: center; padding: 20px;">
            <h3 style="color: #ffa500;">자세 분석 중...</h3>
            <p>측면 자세를 유지해주세요</p>
          </div>`;
      }
    } else {
      // 다른 상태일 때
      if (overallScore) overallScore.style.display = "none";
      if (data.state_message && resultBox) {
        resultBox.innerHTML = `
          <div style="text-align: center; padding: 20px;">
            <h3 style="color: #667eea;">${data.state_message}</h3>
          </div>`;
      }
    }
  }

  // 스트림 에러 처리 및 자동 재연결
  window.handleStreamError = function() {
    console.log("ESP32 스트림 에러 발생 - 자동 재연결 시도");
    const streamError = document.getElementById('stream-error');
    const streamPlaceholder = document.getElementById('stream-placeholder');
    const esp32Stream = document.getElementById('esp32-stream');
    
    streamError.style.display = 'block';
    streamPlaceholder.style.display = 'none';
    
    // 3초 후 자동 재연결
    setTimeout(() => {
      console.log("스트림 재연결 시도...");
      esp32Stream.src = '/esp32-stream?' + new Date().getTime(); // 캐시 방지
      streamError.style.display = 'none';
      streamPlaceholder.style.display = 'block';
    }, 3000);
  };

  window.handleStreamLoad = function() {
    console.log("ESP32 스트림 로드 성공");
    const streamError = document.getElementById('stream-error');
    const streamPlaceholder = document.getElementById('stream-placeholder');
    
    streamError.style.display = 'none';
    streamPlaceholder.style.display = 'none';
  };

  // 페이지 로드 시 ESP32 연결 테스트 시작
  console.log("ESP32-CAM 시스템 초기화 중...");
  
  // 즉시 연결 테스트 시작
  testESP32Connection();
});
