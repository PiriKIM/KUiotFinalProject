const video = document.getElementById('webcam');
const canvas = document.getElementById('output-canvas');
const feedback = document.getElementById('feedback');
const ctx = canvas.getContext('2d');

let analyzing = false;

// 웹캠 스트림 시작
async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            video.play();
            drawLoop();
            setInterval(captureAndSendFrame, 1000); // 1초마다 프레임 전송
        };
    } catch (err) {
        feedback.textContent = '웹캠 접근이 불가합니다: ' + err.message;
    }
}

// 프레임 캡처 및 서버 전송
async function captureAndSendFrame() {
    if (analyzing) return; // 중복 요청 방지
    analyzing = true;

    // 캔버스에 현재 프레임 그리기
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(async (blob) => {
        if (!blob) {
            analyzing = false;
            return;
        }
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            if (!response.ok) {
                feedback.textContent = '분석 실패: 서버 오류';
                analyzing = false;
                return;
            }
            const result = await response.json();
            showAnalysisResult(result);
        } catch (err) {
            feedback.textContent = '분석 요청 중 오류 발생: ' + err.message;
        }
        analyzing = false;
    }, 'image/jpeg');
}

// 분석 결과 표시
function showAnalysisResult(result) {
    if (!result.detected) {
        feedback.textContent = result.message || '사람이 감지되지 않았습니다.';
        return;
    }
    let msg = '';
    if (result.neck) {
        msg += `목 등급: ${result.neck.grade} (${result.neck.feedback})\n`;
    }
    if (result.spine) {
        msg += `척추: ${result.spine.feedback}\n`;
    }
    if (result.shoulder) {
        msg += `어깨: ${result.shoulder.feedback}\n`;
    }
    if (result.pelvic) {
        msg += `골반: ${result.pelvic.feedback}\n`;
    }
    if (result.spine_twisting) {
        msg += `척추 틀어짐: ${result.spine_twisting.feedback}\n`;
    }
    feedback.textContent = msg.trim();
}

// 캔버스에 웹캠 프레임 그리기 (분석 결과 오버레이는 필요시 추가)
function drawLoop() {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    requestAnimationFrame(drawLoop);
}

window.onload = () => {
    canvas.width = 640;
    canvas.height = 480;
    startWebcam();
};