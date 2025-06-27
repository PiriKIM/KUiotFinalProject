import React, { useState, useRef } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import './BodyAnalysis.css';

interface BodyAnalysisResult {
  front_analysis: {
    posture_score: number;
    angles: {
      neck_angle?: number;
      shoulder_angle?: number;
      spine_angle?: number;
      pelvis_angle?: number;
    };
    recommendations: string[];
    feedback: string;
    landmarks: Array<{x: number, y: number}>;
  };
  side_analysis: {
    posture_score: number;
    angles: {
      neck_angle?: number;
      shoulder_angle?: number;
      spine_angle?: number;
      pelvis_angle?: number;
    };
    recommendations: string[];
    feedback: string;
    landmarks: Array<{x: number, y: number}>;
  };
  overall_score: number;
  overall_feedback: string;
  improvement_suggestions: string[];
}

const BodyAnalysis: React.FC = () => {
  const { user } = useAuth();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<BodyAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const [frontImage, setFrontImage] = useState<File | null>(null);
  const [sideImage, setSideImage] = useState<File | null>(null);
  const [frontPreview, setFrontPreview] = useState<string | null>(null);
  const [sidePreview, setSidePreview] = useState<string | null>(null);

  // 이미지 업로드 처리
  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>, type: 'front' | 'side') => {
    const file = event.target.files?.[0];
    if (!file) return;

    // 이미지 타입 검증
    if (!file.type.startsWith('image/')) {
      setError('이미지 파일만 업로드 가능합니다.');
      return;
    }

    // 파일 크기 검증 (5MB 이하)
    if (file.size > 5 * 1024 * 1024) {
      setError('파일 크기는 5MB 이하여야 합니다.');
      return;
    }

    // 미리보기 생성
    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      if (type === 'front') {
        setFrontImage(file);
        setFrontPreview(result);
      } else {
        setSideImage(file);
        setSidePreview(result);
      }
    };
    reader.readAsDataURL(file);
    setError(null);
  };

  // 체형 분석 수행
  const analyzeBody = async () => {
    if (!user) {
      setError('로그인이 필요합니다.');
      return;
    }

    if (!frontImage || !sideImage) {
      setError('전면과 측면 사진을 모두 업로드해주세요.');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      console.log('체형 분석 시작...');
      
      // 이미지를 base64로 변환
      const frontBase64 = await fileToBase64(frontImage);
      const sideBase64 = await fileToBase64(sideImage);
      
      console.log('이미지 변환 완료');
      console.log('전면 이미지 크기:', frontBase64.length);
      console.log('측면 이미지 크기:', sideBase64.length);

      // API 요청 데이터 준비
      const requestData = {
        user_id: user.id,
        front_image: frontBase64,
        side_image: sideBase64,
        analysis_type: 'body_posture'
      };
      
      console.log('API 요청 데이터 준비 완료');
      console.log('사용자 ID:', user.id);

      // API 호출
      console.log('API 호출 시작...');
      const response = await fetch('http://localhost:8000/analyze/body', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(requestData)
      });

      console.log('API 응답 받음');
      console.log('응답 상태:', response.status);
      console.log('응답 헤더:', Object.fromEntries(response.headers.entries()));

      if (!response.ok) {
        const errorText = await response.text();
        console.error('API 오류 응답:', errorText);
        throw new Error(`분석 중 오류가 발생했습니다. (${response.status}: ${errorText})`);
      }

      const result = await response.json();
      console.log('분석 결과:', result);
      
      setAnalysisResult(result);
    } catch (err) {
      console.error('체형 분석 오류:', err);
      setError(`체형 분석 중 오류가 발생했습니다: ${err instanceof Error ? err.message : '알 수 없는 오류'}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // 파일을 base64로 변환
  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const result = reader.result as string;
        // data:image/jpeg;base64, 제거
        const base64 = result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = error => reject(error);
    });
  };

  // 이미지 초기화
  const resetImages = () => {
    setFrontImage(null);
    setSideImage(null);
    setFrontPreview(null);
    setSidePreview(null);
    setAnalysisResult(null);
    setError(null);
  };

  return (
    <div className="body-analysis-container">
      <div className="body-analysis-header">
        <h2>체형 분석</h2>
        <p>전면과 측면 사진을 업로드하여 체형과 자세를 분석해보세요.</p>
      </div>

      <div className="body-analysis-content">
        <div className="upload-section">
          <div className="upload-grid">
            {/* 전면 사진 업로드 */}
            <div className="upload-card">
              <h3>전면 사진</h3>
              <div className="upload-area">
                {frontPreview ? (
                  <div className="image-preview">
                    <img src={frontPreview} alt="전면 사진" />
                    <button 
                      className="btn btn-sm btn-outline-danger remove-btn"
                      onClick={() => {
                        setFrontImage(null);
                        setFrontPreview(null);
                      }}
                    >
                      <i className="fas fa-times"></i>
                    </button>
                  </div>
                ) : (
                  <div className="upload-placeholder">
                    <i className="fas fa-camera"></i>
                    <p>전면 사진을 업로드하세요</p>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={(e) => handleImageUpload(e, 'front')}
                      className="file-input"
                    />
                  </div>
                )}
              </div>
            </div>

            {/* 측면 사진 업로드 */}
            <div className="upload-card">
              <h3>측면 사진</h3>
              <div className="upload-area">
                {sidePreview ? (
                  <div className="image-preview">
                    <img src={sidePreview} alt="측면 사진" />
                    <button 
                      className="btn btn-sm btn-outline-danger remove-btn"
                      onClick={() => {
                        setSideImage(null);
                        setSidePreview(null);
                      }}
                    >
                      <i className="fas fa-times"></i>
                    </button>
                  </div>
                ) : (
                  <div className="upload-placeholder">
                    <i className="fas fa-camera"></i>
                    <p>측면 사진을 업로드하세요</p>
                    <input
                      type="file"
                      accept="image/*"
                      onChange={(e) => handleImageUpload(e, 'side')}
                      className="file-input"
                    />
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="upload-controls">
            {!isAnalyzing ? (
              <button
                className="btn btn-primary"
                onClick={analyzeBody}
                disabled={!frontImage || !sideImage || !user}
              >
                <i className="fas fa-play"></i>
                체형 분석 시작
              </button>
            ) : (
              <button className="btn btn-secondary" disabled>
                <i className="fas fa-spinner fa-spin"></i>
                분석 중...
              </button>
            )}
            
            <button
              className="btn btn-outline-secondary"
              onClick={resetImages}
            >
              <i className="fas fa-refresh"></i>
              초기화
            </button>
          </div>
        </div>

        <div className="analysis-section">
          {error && (
            <div className="alert alert-danger">
              <i className="fas fa-exclamation-triangle"></i>
              {error}
            </div>
          )}

          {analysisResult && (
            <div className="analysis-results">
              {/* 전체 점수 */}
              <div className="overall-score-section">
                <h3>전체 자세 점수</h3>
                <div className="score-display">
                  <div className="overall-score">
                    <span className="score-value">{analysisResult.overall_score}</span>
                    <span className="score-label">점</span>
                  </div>
                </div>
                <div className="score-bar">
                  <div 
                    className="score-fill"
                    style={{ width: `${analysisResult.overall_score}%` }}
                  ></div>
                </div>
                <p className="overall-feedback">{analysisResult.overall_feedback}</p>
              </div>

              {/* 전면 분석 결과 */}
              <div className="analysis-detail">
                <h3>전면 분석 결과</h3>
                <div className="analysis-grid">
                  <div className="score-item">
                    <span className="score-label">전면 점수</span>
                    <span className="score-value">{analysisResult.front_analysis.posture_score}점</span>
                  </div>
                  <div className="feedback-item">
                    <span className="feedback-label">피드백</span>
                    <p className="feedback-text">{analysisResult.front_analysis.feedback}</p>
                  </div>
                </div>
                <div className="recommendations">
                  <h4>개선 권장사항</h4>
                  <ul>
                    {analysisResult.front_analysis.recommendations.map((rec, index) => (
                      <li key={index}>{rec}</li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* 측면 분석 결과 */}
              <div className="analysis-detail">
                <h3>측면 분석 결과</h3>
                <div className="analysis-grid">
                  <div className="score-item">
                    <span className="score-label">측면 점수</span>
                    <span className="score-value">{analysisResult.side_analysis.posture_score}점</span>
                  </div>
                  <div className="feedback-item">
                    <span className="feedback-label">피드백</span>
                    <p className="feedback-text">{analysisResult.side_analysis.feedback}</p>
                  </div>
                </div>
                <div className="recommendations">
                  <h4>개선 권장사항</h4>
                  <ul>
                    {analysisResult.side_analysis.recommendations.map((rec, index) => (
                      <li key={index}>{rec}</li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* 종합 개선 제안 */}
              <div className="improvement-suggestions">
                <h3>종합 개선 제안</h3>
                <ul>
                  {analysisResult.improvement_suggestions.map((suggestion, index) => (
                    <li key={index}>{suggestion}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {!analysisResult && !error && (
            <div className="analysis-placeholder">
              <i className="fas fa-chart-line"></i>
              <p>사진을 업로드하고 분석을 시작하면 결과가 여기에 표시됩니다.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default BodyAnalysis; 