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
  let latestLandmarks = null; // 최신 landmarks 저장

  // [AI 추가] FPS 측정 변수
  let frameCount = 0;
  let lastFpsUpdate = Date.now();
  let currentFps = 0;
  const fpsIndicator = document.getElementById('fps-indicator');

  // [AI 추가] 스켈레톤 연결 정보 (mediapipe pose 기준 index)
  const SKELETON_CONNECTIONS = [
    [11, 12], // 어깨선
    [11, 13], [13, 15], // 왼팔
    [12, 14], [14, 16], // 오른팔
    [11, 23], [12, 24], // 어깨-골반
    [23, 24], // 골반선
    [23, 25], [25, 27], // 왼다리
    [24, 26], [26, 28], // 오른다리
    [7, 11], [8, 12],   // 귀-어깨
  ];

  function updateFps() {
    frameCount++;
    const now = Date.now();
    if (now - lastFpsUpdate >= 1000) {
      currentFps = frameCount;
      frameCount = 0;
      lastFpsUpdate = now;
      if (fpsIndicator) {
        fpsIndicator.textContent = `FPS: ${currentFps}`;
      }
    }
  }

  // [AI 추가] landmarks와 스켈레톤을 canvas에 그림
  function drawLandmarksAndSkeleton(landmarks) {
    if (!landmarks) return;
    ctx.save();
    // 1. 스켈레톤(선) 그리기
    ctx.strokeStyle = 'lime';
    ctx.lineWidth = 2;
    SKELETON_CONNECTIONS.forEach(([startIdx, endIdx]) => {
      const start = landmarks.find(lm => lm.index === startIdx);
      const end = landmarks.find(lm => lm.index === endIdx);
      if (start && end) {
        ctx.beginPath();
        ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
        ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
        ctx.stroke();
      }
    });
    // 2. 점(landmarks) 그리기
    ctx.fillStyle = 'red';
    landmarks.forEach(lm => {
      ctx.beginPath();
      ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 5, 0, 2 * Math.PI);
      ctx.fill();
    });

    // [AI 추가] 척추, 목, 골반 등 중간점 계산 및 표시
    const getLm = idx => landmarks.find(lm => lm.index === idx);
    // 목 중심점 (귀의 중간)
    const leftEar = getLm(7), rightEar = getLm(8);
    if (leftEar && rightEar) {
      const neckX = (leftEar.x + rightEar.x) / 2;
      const neckY = (leftEar.y + rightEar.y) / 2;
      ctx.beginPath();
      ctx.arc(neckX * canvas.width, neckY * canvas.height, 7, 0, 2 * Math.PI);
      ctx.fillStyle = 'blue';
      ctx.fill();
      ctx.fillStyle = 'white';
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText('Neck', neckX * canvas.width + 8, neckY * canvas.height);
    }
    // 어깨 중심점
    const leftShoulder = getLm(11), rightShoulder = getLm(12);
    if (leftShoulder && rightShoulder) {
      const shoulderX = (leftShoulder.x + rightShoulder.x) / 2;
      const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
      ctx.beginPath();
      ctx.arc(shoulderX * canvas.width, shoulderY * canvas.height, 7, 0, 2 * Math.PI);
      ctx.fillStyle = 'orange';
      ctx.fill();
      ctx.fillStyle = 'white';
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText('Shoulder', shoulderX * canvas.width + 8, shoulderY * canvas.height);
    }
    // 골반 중심점
    const leftHip = getLm(23), rightHip = getLm(24);
    if (leftHip && rightHip) {
      const hipX = (leftHip.x + rightHip.x) / 2;
      const hipY = (leftHip.y + rightHip.y) / 2;
      ctx.beginPath();
      ctx.arc(hipX * canvas.width, hipY * canvas.height, 7, 0, 2 * Math.PI);
      ctx.fillStyle = 'purple';
      ctx.fill();
      ctx.fillStyle = 'white';
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText('Hip', hipX * canvas.width + 8, hipY * canvas.height);
    }
    // 척추 중심점 (어깨-골반 중간)
    if (leftShoulder && rightShoulder && leftHip && rightHip) {
      const shoulderX = (leftShoulder.x + rightShoulder.x) / 2;
      const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
      const hipX = (leftHip.x + rightHip.x) / 2;
      const hipY = (leftHip.y + rightHip.y) / 2;
      const spineX = (shoulderX + hipX) / 2;
      const spineY = (shoulderY + hipY) / 2;
      ctx.beginPath();
      ctx.arc(spineX * canvas.width, spineY * canvas.height, 7, 0, 2 * Math.PI);
      ctx.fillStyle = 'cyan';
      ctx.fill();
      ctx.fillStyle = 'black';
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText('Spine', spineX * canvas.width + 8, spineY * canvas.height);
    }

    // [AI 추가] 척추 중심선(목-어깨-골반) 표시
    if (leftEar && rightEar && leftShoulder && rightShoulder && leftHip && rightHip) {
      const neckX = (leftEar.x + rightEar.x) / 2;
      const neckY = (leftEar.y + rightEar.y) / 2;
      const shoulderX = (leftShoulder.x + rightShoulder.x) / 2;
      const shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
      const hipX = (leftHip.x + rightHip.x) / 2;
      const hipY = (leftHip.y + rightHip.y) / 2;
      ctx.save();
      ctx.strokeStyle = 'deepskyblue';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(neckX * canvas.width, neckY * canvas.height);
      ctx.lineTo(shoulderX * canvas.width, shoulderY * canvas.height);
      ctx.lineTo(hipX * canvas.width, hipY * canvas.height);
      ctx.stroke();
      ctx.restore();
    }
    ctx.restore();
  }

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
    latestLandmarks = null; // landmarks 초기화
    ctx.clearRect(0, 0, canvas.width, canvas.height); // canvas 초기화
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

      // [AI 추가] FPS 업데이트
      updateFps();

      // [AI 수정] canvas 크기를 비디오와 동일하게 설정
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // [AI 수정] canvas 스타일도 비디오와 동일하게 설정
      canvas.style.width = video.style.width || '100%';
      canvas.style.maxWidth = video.style.maxWidth || '500px';
      
      // [AI 수정] 비디오 프레임을 canvas에 그리기 (서버 전송용)
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
            latestLandmarks = null; // landmarks 초기화
            ctx.clearRect(0, 0, canvas.width, canvas.height); // canvas 초기화
          } else {
            // [AI 추가] landmarks 데이터 저장 및 표시
            if (data.landmarks) {
              latestLandmarks = data.landmarks;
              // [AI 수정] 비디오 프레임을 다시 그린 후 스켈레톤 오버레이
              ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
              drawLandmarksAndSkeleton(latestLandmarks);
            }

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
          latestLandmarks = null; // landmarks 초기화
          ctx.clearRect(0, 0, canvas.width, canvas.height); // canvas 초기화
        });
      }, 'image/jpeg');
    }, 1000);
  }
});
