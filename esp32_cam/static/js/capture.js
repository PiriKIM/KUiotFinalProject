// DOM이 완전히 로드된 후에 실행
window.addEventListener("DOMContentLoaded", () => {
    const video = document.getElementById('video');
    const resultText = document.getElementById('result');
    const startBtn = document.getElementById('startBtn');

    let lastMessage = '';

    if (typeof navigator.mediaDevices === 'undefined') {
        alert("navigator.mediaDevices가 아직 준비되지 않았습니다.");
    }

    if (typeof navigator.mediaDevices.getUserMedia !== 'function') {
        alert("이 브라우저는 getUserMedia를 지원하지 않습니다.");
    }

    // 버튼 활성화
    startBtn.disabled = false;

    // 버튼 클릭 시 카메라 접근 시도
    startBtn.addEventListener("click", () => {
        // 카메라 접근
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                // console.log("✅ 카메라 접근 성공");
                alert("✅ 카메라 접근 성공");

                video.srcObject = stream;

                // video metadata가 로드된 이후에만 frame 캡처 시작
                video.onloadedmetadata = () => {
                    video.play();
                    // console.log("▶️ video metadata 로드 완료, 전송 시작");
                    alert("▶️ video metadata 로드 완료, 전송 시작");

                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');

                    // 초당 1프레임으로 서버 전송
                    setInterval(() => {
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;

                        // 만약 video 크기가 아직 설정되지 않았다면 skip
                        if (canvas.width === 0 || canvas.height === 0) {
                            console.warn("⚠️ 영상 크기 미설정, 전송 건너뜀");
                            return;
                        }

                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                        const imageData = canvas.toDataURL('image/jpeg');
                        fetch('/upload', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ image: imageData })
                        })
                            .then(res => res.json())
                            .then(data => {
                                // console.log("📨 서버 응답:", data.message);
                                resultText.textContent = `결과: ${data.message}`;

                                // 같은 메시지가 반복되면 음성 출력 생략
                                if (data.message !== lastMessage) {
                                    speak(data.message);
                                    lastMessage = data.message;
                                }
                            })
                            .catch(err => {
                                console.error('❌ 업로드 오류:', err);
                            });
                    }, 1000); // 1초 간격
                };
            })
            .catch(err => {
                console.error('❌ 카메라 접근 실패:', err);
                alert("카메라 접근 실패: " + err.name + " / " + err.message);
            });
    });

    // 텍스트를 음성으로 출력
    function speak(text) {
        const utterance = new SpeechSynthesisUtterance(text);
        speechSynthesis.speak(utterance);
    }
});
