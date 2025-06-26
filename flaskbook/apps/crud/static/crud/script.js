document.addEventListener("DOMContentLoaded", () => {
  const video = document.getElementById('webcam');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const resultBox = document.getElementById('result');
  const overallScore = document.getElementById('overall-score');
  const scoreNumber = document.getElementById('score-number');
  const scoreGrade = document.getElementById('score-grade');
  const currentStateElement = document.getElementById('current-state');
  const stateProgressElement = document.getElementById('state-progress');
  const progressFillElement = document.getElementById('progress-fill');
  const cameraBtnText = document.getElementById('camera-btn-text');

  let stream = null;
  let intervalId = null;
  let currentState = "no_human_detected";
  let stateMessage = "카메라 앞에 앉아주세요";

  // ▶︎ 권한 테스트용 콘솔 로그
  console.log("[DEBUG] DOMContentLoaded: 스크립트 로드됨.");

  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((testStream) => {
        console.log("[SUCCESS] 카메라 권한 요청 성공: 사전 테스트 통과.");
        // 테스트용 스트림은 바로 종료
        testStream.getTracks().forEach(track => track.stop());
      })
      .catch((err) => {
        console.error("[FAIL] 카메라 권한 요청 실패:", err);
      });
  } else {
    console.error("[ERROR] 이 브라우저는 getUserMedia를 지원하지 않습니다.");
  }

  // ▶︎ HTML 버튼에서 사용할 수 있도록 전역 함수로 노출
  window.toggleCamera = async function () {
    if (!stream) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        console.log("[INFO] 카메라 스트림 시작됨.");
        video.srcObject = stream;
        video.play();
        cameraBtnText.textContent = "카메라 종료";
        startAnalyzing();
      } catch (err) {
        alert("카메라 접근이 차단되었습니다.\n브라우저 주소창 왼쪽 자물쇠 아이콘을 눌러 권한을 허용해주세요.");
        console.error("[ERROR] toggleCamera: 카메라 접근 오류:", err);
      }
    } else {
      stopCamera();
    }
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
      console.log("[INFO] 카메라 스트림 종료됨.");
    }
    video.srcObject = null;
    clearInterval(intervalId);
    cameraBtnText.textContent = "카메라 시작";
    updateStateDisplay("no_human_detected", "카메라가 종료되었습니다.");
    overallScore.style.display = "none";
  }

  function updateStateDisplay(state, message, stableTime = null) {
    currentState = state;
    stateMessage = message;
    
    // 현재 상태 요소 업데이트
    if (currentStateElement) {
      currentStateElement.textContent = getStateKoreanName(state);
    }
    
    // 진행률 표시 업데이트
    if (stableTime !== null && state === "detecting_front_pose") {
      const progress = Math.min((stableTime / 2.0) * 100, 100);
      if (stateProgressElement) {
        stateProgressElement.style.display = "block";
      }
      if (progressFillElement) {
        progressFillElement.style.width = `${progress}%`;
      }
    } else {
      if (stateProgressElement) {
        stateProgressElement.style.display = "none";
      }
      if (progressFillElement) {
        progressFillElement.style.width = "0%";
      }
    }
    
    // 상태에 따른 색상 설정
    let color = "#ffffff"; // 기본 흰색
    let bgColor = "#333333"; // 기본 배경색
    
    switch(state) {
      case "no_human_detected":
        color = "#ff6b6b"; // 빨간색
        bgColor = "#2d1b1b";
        break;
      case "detecting_front_pose":
        color = "#4ecdc4"; // 청록색
        bgColor = "#1b2d2b";
        break;
      case "waiting_side_pose":
        color = "#45b7d1"; // 파란색
        bgColor = "#1b2b2d";
        break;
      case "analyzing_side_pose":
        color = "#96ceb4"; // 초록색
        bgColor = "#1b2d1b";
        break;
    }
    
    // 상태 메시지 표시
    let displayMessage = message;
    if (stableTime !== null && state === "detecting_front_pose") {
      displayMessage += ` (안정 시간: ${stableTime.toFixed(1)}초)`;
    }
    
    resultBox.innerHTML = `
      <div style="text-align: center; padding: 20px;">
        <h3 style="color: ${color}; margin-bottom: 10px;">현재 상태: ${getStateKoreanName(state)}</h3>
        <p style="color: ${color}; font-size: 16px; line-height: 1.5;">${displayMessage}</p>
        ${stableTime !== null && state === "detecting_front_pose" ? 
          `<div style="margin-top: 15px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 5px;">
            <small style="color: #ffd93d;">안정화 진행률: ${Math.min((stableTime / 2.0) * 100, 100).toFixed(0)}%</small>
          </div>` : ''
        }
      </div>
    `;
    
    resultBox.style.backgroundColor = bgColor;
    resultBox.style.border = `2px solid ${color}`;
    resultBox.className = "status";
  }

  function getStateKoreanName(state) {
    const stateNames = {
      "no_human_detected": "사람 미감지",
      "detecting_front_pose": "정면 자세 감지 중",
      "waiting_side_pose": "측면 자세 대기 중",
      "analyzing_side_pose": "자세 분석 중"
    };
    return stateNames[state] || state;
  }

  function startAnalyzing() {
    intervalId = setInterval(() => {
      if (!stream || video.videoWidth === 0 || video.videoHeight === 0) return;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('frame', blob, 'frame.jpg');

        fetch('/crud/analyze', {
          method: 'POST',
          body: formData
        })
        .then(res => res.json())
        .then(data => {
          if (data.error) {
            updateStateDisplay("no_human_detected", "사람이 감지되지 않았습니다.");
            overallScore.style.display = "none";
          } else {
            // 상태 정보 업데이트
            updateStateDisplay(data.state, data.state_message, data.stable_time);
            
            // 분석 중일 때만 결과 표시
            if (data.state === "analyzing_side_pose" && data.overall_score !== undefined) {
              // 종합 점수 표시
              scoreNumber.textContent = data.overall_score;
              scoreGrade.textContent = `등급: ${data.overall_grade}`;
              overallScore.style.display = "block";
              
              // 상세 분석 결과 표시
              resultBox.innerHTML = `
                <div style="text-align: center; padding: 20px;">
                  <h3 style="color: #96ceb4; margin-bottom: 15px;">자세 분석 완료!</h3>
                  <div style="background: rgba(150, 206, 180, 0.1); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <h4 style="color: #96ceb4; margin-bottom: 10px;">종합 점수: ${data.overall_score}점 (${data.overall_grade}등급)</h4>
                  </div>
                  <div style="text-align: left; background: rgba(255,255,255,0.05); padding: 15px; border-radius: 8px;">
                    <h5 style="color: #96ceb4; margin-bottom: 10px;">상세 분석 결과:</h5>
                    <p style="color: #ffffff; margin: 5px 0;">📱 <strong>목:</strong> ${data.neck.grade_description} (${data.neck.neck_angle.toFixed(1)}°)</p>
                    <p style="color: #ffffff; margin: 5px 0;">🦴 <strong>척추:</strong> ${data.spine.is_hunched ? "🔴 굽음" : "🟢 정상"}</p>
                    <p style="color: #ffffff; margin: 5px 0;">💪 <strong>어깨:</strong> ${data.shoulder.is_asymmetric ? "🔴 비대칭" : "🟢 정상"}</p>
                    <p style="color: #ffffff; margin: 5px 0;">🦵 <strong>골반:</strong> ${data.pelvic.is_tilted ? "🔴 기울어짐" : "🟢 정상"}</p>
                    <p style="color: #ffffff; margin: 5px 0;">🔄 <strong>척추 틀어짐:</strong> ${data.twist.is_twisted ? "🔴 있음" : "🟢 정상"}</p>
                  </div>
                  <div style="margin-top: 15px; font-size: 12px; color: #888;">
                    분석 시간: ${new Date().toLocaleString()}
                  </div>
                </div>
              `.trim();
              
              resultBox.style.backgroundColor = "#1b2d1b";
              resultBox.style.border = "2px solid #96ceb4";
              resultBox.className = "success";
            }
          }
        })
        .catch(err => {
          console.error("[ERROR] 분석 요청 실패:", err);
          updateStateDisplay("no_human_detected", "서버 오류로 분석 실패.");
          overallScore.style.display = "none";
        });
      }, 'image/jpeg');
    }, 1000);
  }
});
