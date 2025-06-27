const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const resultDiv = document.getElementById('result');

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => { video.srcObject = stream; });

function captureAndSend() {
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL('image/jpeg');
    fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataURL })
    })
    .then(res => res.json())
    .then(data => {
        resultDiv.innerText = `등급: ${data.grade}, 설명: ${data.grade_description}, 각도: ${data.neck_angle}`;
    });
}

// 1초마다 캡처 및 분석 요청
setInterval(captureAndSend, 1000);