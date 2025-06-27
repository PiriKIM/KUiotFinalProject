
let video = document.getElementById("webcam");
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let streaming = false;
let analysisInterval = null;
let lastFrameTime = performance.now();
let frameCounter = 0;
let fps = 0;

function updateFPS() {
    const now = performance.now();
    const delta = (now - lastFrameTime) / 1000;
    fps = Math.round(1 / delta);
    lastFrameTime = now;
    document.getElementById("fps-indicator").innerText = `FPS: ${fps}`;
}

function toggleCamera() {
    if (!streaming) {
        navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
            video.srcObject = stream;
            video.play();
            streaming = true;

            analysisInterval = setInterval(() => {
                analyzeFrame();
            }, 1000); // 1초마다 분석

            requestAnimationFrame(drawLoop);
            document.getElementById("camera-btn-text").innerText = "카메라 정지";
        }).catch(function(err) {
            console.error("카메라 접근 오류:", err);
        });
    } else {
        let tracks = video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
        streaming = false;
        clearInterval(analysisInterval);
        document.getElementById("camera-btn-text").innerText = "카메라 시작";
    }
}

function drawLoop() {
    if (streaming) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        updateFPS();
        requestAnimationFrame(drawLoop);
    }
}

function analyzeFrame() {
    canvas.toBlob(function(blob) {
        const formData = new FormData();
        formData.append("frame", blob);
        fetch("/analyze", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            // 결과 처리 (예시)
            if (data.state_message) {
                document.getElementById("current-state").innerText = data.state_message;
            }
        })
        .catch(err => console.error("분석 에러:", err));
    }, "image/jpeg");
}
