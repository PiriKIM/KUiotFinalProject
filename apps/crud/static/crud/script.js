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
  let latestLandmarks = null; // 25-06-26 변경: 최신 landmarks 저장

  // 25-06-26 변경: landmarks를 canvas에 그림
  function drawLandmarks(landmarks) {
    if (!landmarks) return;
    ctx.save();
    ctx.fillStyle = 'red';
    landmarks.forEach(lm => {
      ctx.beginPath();
      ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 5, 0, 2 * Math.PI);
      ctx.fill();
    });
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
      // 25-06-26 변경: 임의의 파란 점 그리기 코드 제거

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
            resultBox.innerText = "사람이 감지되지 않았습니다.";
            resultBox.className = "error";
            overallScore.style.display = "none";
            latestLandmarks = null; // 25-06-26 변경: 감지 안될 때 landmarks 초기화
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
            // 25-06-26 변경: landmarks가 있으면 drawLandmarks 호출
            if (data.landmarks) {
              console.log("landmarks:", data.landmarks);
              drawLandmarks(data.landmarks);
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
        });
      }, 'image/jpeg');
    }, 1000);
  }
}); 