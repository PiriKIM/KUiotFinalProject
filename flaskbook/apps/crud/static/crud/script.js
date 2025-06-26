document.addEventListener("DOMContentLoaded", () => {
  const video = document.getElementById('webcam');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const resultBox = document.getElementById('result');
  const overallScore = document.getElementById('overall-score');
  const scoreNumber = document.getElementById('score-number');
  const scoreGrade = document.getElementById('score-grade');
  
  let stream = null;
  let intervalId = null;
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

  // [AI 수정] landmarks와 스켈레톤을 canvas에 그림
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
        renderLoop();
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
    resultBox.innerText = "분석 대기 중...";
    resultBox.className = "status";
    overallScore.style.display = "none";
    latestLandmarks = null; // 25-06-26 변경: landmarks 초기화
    ctx.clearRect(0, 0, canvas.width, canvas.height); // 25-06-26 변경: canvas 초기화
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
          if (!data.error && data.landmarks) {
            latestLandmarks = data.landmarks; // 최신 landmarks만 갱신
          }
          if (data.error) {
            resultBox.innerText = "사람이 감지되지 않았습니다.";
            resultBox.className = "error";
            overallScore.style.display = "none";
            latestLandmarks = null; // 25-06-26 변경: 감지 안될 때 landmarks 초기화
            ctx.clearRect(0, 0, canvas.width, canvas.height); // [AI 추가] canvas 초기화
          } else {
            // 종합 점수 표시
            scoreNumber.textContent = data.overall_score;
            scoreGrade.textContent = `등급: ${data.overall_grade}`;
            overallScore.style.display = "block";
            // 상세 분석 결과 표시
            resultBox.innerHTML = `
<strong>상세 분석 결과:</strong>

[목] ${data.neck.grade_description} (${data.neck.neck_angle.toFixed(1)}°)
[척추] ${data.spine.is_hunched ? "굽음" : "정상"}
[어깨] ${data.shoulder.is_asymmetric ? "비대칭" : "정상"}
[골반] ${data.pelvic.is_tilted ? "기울어짐" : "정상"}
[척추 틀어짐] ${data.twist.is_twisted ? "있음" : "정상"}

<small>분석 시간: ${new Date().toLocaleString()}</small>
            `.trim();
            resultBox.className = "success";
            if (data.landmarks) {
              console.log("landmarks:", data.landmarks);
              drawLandmarksAndSkeleton(data.landmarks);
              latestLandmarks = data.landmarks;
            }
          }
        })
        .catch(err => {
          console.error("[ERROR] 분석 요청 실패:", err);
          resultBox.innerText = "서버 오류로 분석 실패.";
          resultBox.className = "error";
          overallScore.style.display = "none";
          latestLandmarks = null; // 25-06-26 변경: 서버 오류 시 landmarks 초기화
          ctx.clearRect(0, 0, canvas.width, canvas.height); // [AI 추가] canvas 초기화
        });
      }, 'image/jpeg');
    }, 1000); // 1초마다 분석
  }

  function renderLoop() {
    // 실시간 웹캠 프레임 그리기
    if (video.videoWidth > 0 && video.videoHeight > 0) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      // [AI 수정] landmarks와 스켈레톤 함께 그리기
      if (latestLandmarks) {
        drawLandmarksAndSkeleton(latestLandmarks);
      }
    }
    // FPS 카운트
    updateFps();
    requestAnimationFrame(renderLoop);
  }
});
