import React, { useState, useRef, useEffect } from 'react';
import toast from 'react-hot-toast';

const PostureCorrection: React.FC = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentScore, setCurrentScore] = useState<number | null>(null);
  const [feedback, setFeedback] = useState<string[]>([]);
  const [sessionTime, setSessionTime] = useState(0);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sessionTimerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      stopStream();
      if (sessionTimerRef.current) {
        clearInterval(sessionTimerRef.current);
      }
    };
  }, []);

  const startStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsStreaming(true);
        
        sessionTimerRef.current = window.setInterval(() => {
          setSessionTime(prev => prev + 1);
        }, 1000);

        toast.success('웹캠이 시작되었습니다.');
      }
    } catch (error) {
      console.error('웹캠 접근 오류:', error);
      toast.error('웹캠에 접근할 수 없습니다. 권한을 확인해주세요.');
    }
  };

  const stopStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    if (sessionTimerRef.current) {
      clearInterval(sessionTimerRef.current);
      sessionTimerRef.current = null;
    }
    
    setIsStreaming(false);
    setIsAnalyzing(false);
    setCurrentScore(null);
    setFeedback([]);
    setSessionTime(0);
    
    toast.success('웹캠이 중지되었습니다.');
  };

  const startAnalysis = () => {
    if (!isStreaming) {
      toast.error('먼저 웹캠을 시작해주세요.');
      return;
    }

    setIsAnalyzing(true);
    setCurrentScore(null);
    setFeedback([]);
    
    const analysisInterval = window.setInterval(() => {
      const score = Math.floor(Math.random() * 26) + 70;
      setCurrentScore(score);
      
      const feedbacks: string[] = [];
      if (score < 80) {
        feedbacks.push('어깨를 펴주세요');
      }
      if (score < 85) {
        feedbacks.push('목을 곧게 펴주세요');
      }
      if (score < 90) {
        feedbacks.push('허리를 곧게 펴주세요');
      }
      
      setFeedback(feedbacks);
    }, 2000);

    setTimeout(() => {
      clearInterval(analysisInterval);
      setIsAnalyzing(false);
      toast.success('자세 분석이 완료되었습니다!');
    }, 10000);
  };

  const stopAnalysis = () => {
    setIsAnalyzing(false);
    setCurrentScore(null);
    setFeedback([]);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="container-fluid">
      <div className="row">
        <div className="col-12">
          <div className="card shadow">
            <div className="card-header">
              <h4 className="mb-0">
                <i className="bi bi-webcam me-2"></i>
                실시간 자세 교정
              </h4>
              <p className="text-muted mb-0">
                웹캠을 통해 실시간으로 자세를 감지하고 교정해드립니다.
              </p>
            </div>
            <div className="card-body">
              <div className="row">
                <div className="col-lg-8">
                  <div className="card">
                    <div className="card-header d-flex justify-content-between align-items-center">
                      <h6 className="mb-0">웹캠 영상</h6>
                      {isStreaming && (
                        <span className="badge bg-success">
                          <i className="bi bi-record-circle me-1"></i>
                          {formatTime(sessionTime)}
                        </span>
                      )}
                    </div>
                    <div className="card-body text-center">
                      <div className="position-relative">
                        <video
                          ref={videoRef}
                          autoPlay
                          playsInline
                          muted
                          className="img-fluid rounded"
                          style={{ maxHeight: '500px' }}
                        />
                        {!isStreaming && (
                          <div className="position-absolute top-50 start-50 translate-middle text-center">
                            <i className="bi bi-camera-video-off fa-3x text-muted mb-3"></i>
                            <p className="text-muted">웹캠을 시작해주세요</p>
                          </div>
                        )}
                        <canvas
                          ref={canvasRef}
                          className="position-absolute top-0 start-0"
                          style={{ display: 'none' }}
                        />
                      </div>
                      
                      <div className="mt-3">
                        {!isStreaming ? (
                          <button
                            className="btn btn-primary btn-lg"
                            onClick={startStream}
                          >
                            <i className="bi bi-camera-video me-2"></i>
                            웹캠 시작
                          </button>
                        ) : (
                          <div className="d-flex gap-2 justify-content-center">
                            {!isAnalyzing ? (
                              <button
                                className="btn btn-success btn-lg"
                                onClick={startAnalysis}
                              >
                                <i className="bi bi-play-circle me-2"></i>
                                자세 분석 시작
                              </button>
                            ) : (
                              <button
                                className="btn btn-warning btn-lg"
                                onClick={stopAnalysis}
                              >
                                <i className="bi bi-stop-circle me-2"></i>
                                분석 중지
                              </button>
                            )}
                            <button
                              className="btn btn-danger btn-lg"
                              onClick={stopStream}
                            >
                              <i className="bi bi-stop-circle me-2"></i>
                              웹캠 중지
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="col-lg-4">
                  <div className="card">
                    <div className="card-header">
                      <h6 className="mb-0">실시간 피드백</h6>
                    </div>
                    <div className="card-body">
                      {isAnalyzing ? (
                        <>
                          {currentScore && (
                            <div className="text-center mb-4">
                              <div className="display-4 fw-bold text-primary mb-2">
                                {currentScore}점
                              </div>
                              <div className="progress mb-2" style={{ height: '10px' }}>
                                <div
                                  className={`progress-bar ${
                                    currentScore >= 90 ? 'bg-success' :
                                    currentScore >= 80 ? 'bg-warning' : 'bg-danger'
                                  }`}
                                  style={{ width: `${currentScore}%` }}
                                ></div>
                              </div>
                              <small className="text-muted">
                                {currentScore >= 90 ? '훌륭한 자세입니다!' :
                                 currentScore >= 80 ? '좋은 자세입니다.' :
                                 '자세를 개선해주세요.'}
                              </small>
                            </div>
                          )}

                          {feedback.length > 0 && (
                            <div>
                              <h6 className="mb-3">개선 사항:</h6>
                              <ul className="list-group list-group-flush">
                                {feedback.map((item, index) => (
                                  <li key={index} className="list-group-item d-flex align-items-center">
                                    <i className="bi bi-exclamation-triangle text-warning me-2"></i>
                                    {item}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}

                          <div className="text-center mt-3">
                            <div className="spinner-border text-primary" role="status">
                              <span className="visually-hidden">분석 중...</span>
                            </div>
                            <p className="text-muted mt-2">자세를 분석하고 있습니다...</p>
                          </div>
                        </>
                      ) : (
                        <div className="text-center text-muted">
                          <i className="bi bi-info-circle fa-2x mb-3"></i>
                          <p>자세 분석을 시작하면 실시간 피드백을 받을 수 있습니다.</p>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="card mt-3">
                    <div className="card-header">
                      <h6 className="mb-0">
                        <i className="bi bi-lightbulb me-2"></i>
                        사용 팁
                      </h6>
                    </div>
                    <div className="card-body">
                      <ul className="list-unstyled mb-0">
                        <li className="mb-2">
                          <i className="bi bi-check-circle text-success me-2"></i>
                          웹캠 앞에서 자연스럽게 서주세요
                        </li>
                        <li className="mb-2">
                          <i className="bi bi-check-circle text-success me-2"></i>
                          전체 몸이 화면에 들어오도록 해주세요
                        </li>
                        <li className="mb-2">
                          <i className="bi bi-check-circle text-success me-2"></i>
                          밝은 곳에서 사용해주세요
                        </li>
                        <li>
                          <i className="bi bi-check-circle text-success me-2"></i>
                          움직이지 않고 고정된 자세를 유지해주세요
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PostureCorrection; 