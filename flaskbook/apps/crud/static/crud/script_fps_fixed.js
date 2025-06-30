document.addEventListener("DOMContentLoaded", () => {
  const video = document.getElementById("webcam");
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const resultBox = document.getElementById("result");
  const overallScore = document.getElementById("overall-score");
  const scoreNumber = document.getElementById("score-number");
  const scoreGrade = document.getElementById("score-grade");
  const currentStateElement = document.getElementById("current-state");
  const stateProgressElement = document.getElementById("state-progress");
  const progressFillElement = document.getElementById("progress-fill");
  const cameraBtnText = document.getElementById("camera-btn-text");
  const fpsIndicator = document.getElementById("fps-indicator");

  let stream = null;
  let analyzing = false;
  let latestLandmarks = null;

  let frameCount = 0;
  let lastFpsUpdate = Date.now();

  const SKELETON_CONNECTIONS = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28], [7, 11], [8, 12]];

  window.toggleCamera = async function () {
    if (!stream) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.play();
        cameraBtnText.textContent = "카메라 종료";
        startAnalyzing();
      } catch (err) {
        alert("카메라 접근 오류");
      }
    } else {
      stopCamera();
    }
  };

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }
    video.srcObject = null;
    cameraBtnText.textContent = "카메라 시작";
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function updateFps() {
    frameCount++;
    const now = Date.now();
    if (now - lastFpsUpdate >= 1000) {
      fpsIndicator.textContent = `FPS: ${frameCount}`;
      frameCount = 0;
      lastFpsUpdate = now;
    }
  }

  function drawLandmarksAndSkeleton(landmarks) {
    if (!landmarks) return;
    ctx.save();
    ctx.strokeStyle = "lime";
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
    ctx.fillStyle = "red";
    landmarks.forEach(lm => {
      ctx.beginPath();
      ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 5, 0, 2 * Math.PI);
      ctx.fill();
    });
    ctx.restore();
  }

  function updateUI(data) {
    if (data.landmarks) latestLandmarks = data.landmarks;
    if (data.state) currentStateElement.textContent = data.state_message;

    if (data.state === "analyzing_side_pose" && data.overall_score !== undefined) {
      scoreNumber.textContent = data.overall_score;
      scoreGrade.textContent = `등급: ${data.overall_grade}`;
      overallScore.style.display = "block";

      resultBox.innerHTML = `
        <div style="text-align: center; padding: 20px;">
          <h3 style="color: #96ceb4;">자세 분석 완료</h3>
          <p>목: ${data.neck.grade_description}</p>
          <p>척추: ${data.spine.is_hunched ? "굽음" : "정상"}</p>
          <p>어깨: ${data.shoulder.is_asymmetric ? "비대칭" : "정상"}</p>
          <p>골반: ${data.pelvic.is_tilted ? "기울어짐" : "정상"}</p>
        </div>`;
    }
  }

  function startAnalyzing() {
    setInterval(() => {
      if (!stream || analyzing) return;
      analyzing = true;
      canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append("frame", blob, "frame.jpg");
        fetch("/crud/analyze", {
          method: "POST",
          body: formData
        })
          .then(res => res.json())
          .then(data => updateUI(data))
          .finally(() => analyzing = false);
      }, "image/jpeg");
    }, 300);

    function drawLoop() {
      if (!video.paused && !video.ended) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        if (latestLandmarks) drawLandmarksAndSkeleton(latestLandmarks);
        updateFps();
      }
      requestAnimationFrame(drawLoop);
    }
    drawLoop();
  }
});