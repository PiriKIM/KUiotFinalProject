import React, { useState } from 'react';

const Statistics: React.FC = () => {
  const [selectedPeriod, setSelectedPeriod] = useState<'week' | 'month' | 'year'>('month');

  // 더미 데이터
  const statsData = {
    week: {
      totalSessions: 7,
      averageScore: 82,
      improvement: 5,
      sessions: [
        { date: '1월 15일', score: 85, type: '체형 분석' },
        { date: '1월 14일', score: 78, type: '자세 교정' },
        { date: '1월 13일', score: 88, type: '체형 분석' },
        { date: '1월 12일', score: 80, type: '자세 교정' },
        { date: '1월 11일', score: 83, type: '체형 분석' },
        { date: '1월 10일', score: 79, type: '자세 교정' },
        { date: '1월 9일', score: 87, type: '체형 분석' }
      ]
    },
    month: {
      totalSessions: 28,
      averageScore: 84,
      improvement: 12,
      sessions: [
        { date: '1월 15일', score: 85, type: '체형 분석' },
        { date: '1월 14일', score: 78, type: '자세 교정' },
        { date: '1월 13일', score: 88, type: '체형 분석' },
        { date: '1월 12일', score: 80, type: '자세 교정' },
        { date: '1월 11일', score: 83, type: '체형 분석' },
        { date: '1월 10일', score: 79, type: '자세 교정' },
        { date: '1월 9일', score: 87, type: '체형 분석' },
        { date: '1월 8일', score: 82, type: '자세 교정' },
        { date: '1월 7일', score: 86, type: '체형 분석' },
        { date: '1월 6일', score: 81, type: '자세 교정' }
      ]
    },
    year: {
      totalSessions: 156,
      averageScore: 81,
      improvement: 18,
      sessions: [
        { date: '1월', score: 84, type: '월평균' },
        { date: '12월', score: 82, type: '월평균' },
        { date: '11월', score: 79, type: '월평균' },
        { date: '10월', score: 77, type: '월평균' },
        { date: '9월', score: 75, type: '월평균' },
        { date: '8월', score: 73, type: '월평균' }
      ]
    }
  };

  const currentData = statsData[selectedPeriod];

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-success';
    if (score >= 80) return 'text-primary';
    if (score >= 70) return 'text-warning';
    return 'text-danger';
  };

  const getScoreBadge = (score: number) => {
    if (score >= 90) return 'bg-success';
    if (score >= 80) return 'bg-primary';
    if (score >= 70) return 'bg-warning';
    return 'bg-danger';
  };

  return (
    <div className="container-fluid">
      {/* 헤더 */}
      <div className="row mb-4">
        <div className="col-12">
          <div className="card bg-gradient-primary text-white">
            <div className="card-body">
              <h2 className="card-title mb-2">
                <i className="bi bi-graph-up me-2"></i>
                통계 및 분석
              </h2>
              <p className="card-text mb-0">
                체형 분석 및 자세 교정 기록을 확인해보세요.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* 기간 선택 */}
      <div className="row mb-4">
        <div className="col-12">
          <div className="card">
            <div className="card-body">
              <div className="btn-group" role="group">
                <input
                  type="radio"
                  className="btn-check"
                  name="period"
                  id="week"
                  checked={selectedPeriod === 'week'}
                  onChange={() => setSelectedPeriod('week')}
                />
                <label className="btn btn-outline-primary" htmlFor="week">
                  이번 주
                </label>

                <input
                  type="radio"
                  className="btn-check"
                  name="period"
                  id="month"
                  checked={selectedPeriod === 'month'}
                  onChange={() => setSelectedPeriod('month')}
                />
                <label className="btn btn-outline-primary" htmlFor="month">
                  이번 달
                </label>

                <input
                  type="radio"
                  className="btn-check"
                  name="period"
                  id="year"
                  checked={selectedPeriod === 'year'}
                  onChange={() => setSelectedPeriod('year')}
                />
                <label className="btn btn-outline-primary" htmlFor="year">
                  올해
                </label>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 통계 카드 */}
      <div className="row mb-4">
        <div className="col-xl-3 col-md-6 mb-4">
          <div className="card border-left-primary shadow h-100 py-2">
            <div className="card-body">
              <div className="row no-gutters align-items-center">
                <div className="col mr-2">
                  <div className="text-xs font-weight-bold text-primary text-uppercase mb-1">
                    총 세션
                  </div>
                  <div className="h5 mb-0 font-weight-bold text-gray-800">
                    {currentData.totalSessions}회
                  </div>
                </div>
                <div className="col-auto">
                  <i className="bi bi-calendar-check fa-2x text-gray-300"></i>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="col-xl-3 col-md-6 mb-4">
          <div className="card border-left-success shadow h-100 py-2">
            <div className="card-body">
              <div className="row no-gutters align-items-center">
                <div className="col mr-2">
                  <div className="text-xs font-weight-bold text-success text-uppercase mb-1">
                    평균 점수
                  </div>
                  <div className="h5 mb-0 font-weight-bold text-gray-800">
                    {currentData.averageScore}점
                  </div>
                </div>
                <div className="col-auto">
                  <i className="bi bi-star fa-2x text-gray-300"></i>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="col-xl-3 col-md-6 mb-4">
          <div className="card border-left-info shadow h-100 py-2">
            <div className="card-body">
              <div className="row no-gutters align-items-center">
                <div className="col mr-2">
                  <div className="text-xs font-weight-bold text-info text-uppercase mb-1">
                    개선률
                  </div>
                  <div className="h5 mb-0 font-weight-bold text-gray-800">
                    +{currentData.improvement}%
                  </div>
                </div>
                <div className="col-auto">
                  <i className="bi bi-arrow-up-right fa-2x text-gray-300"></i>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="col-xl-3 col-md-6 mb-4">
          <div className="card border-left-warning shadow h-100 py-2">
            <div className="card-body">
              <div className="row no-gutters align-items-center">
                <div className="col mr-2">
                  <div className="text-xs font-weight-bold text-warning text-uppercase mb-1">
                    연속 기록
                  </div>
                  <div className="h5 mb-0 font-weight-bold text-gray-800">
                    {selectedPeriod === 'week' ? '7일' : selectedPeriod === 'month' ? '28일' : '6개월'}
                  </div>
                </div>
                <div className="col-auto">
                  <i className="bi bi-fire fa-2x text-gray-300"></i>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 차트 및 기록 */}
      <div className="row">
        {/* 점수 추이 차트 */}
        <div className="col-lg-8 mb-4">
          <div className="card shadow">
            <div className="card-header py-3">
              <h6 className="m-0 font-weight-bold text-primary">점수 추이</h6>
            </div>
            <div className="card-body">
              <div className="chart-container" style={{ position: 'relative', height: '300px' }}>
                <div className="d-flex align-items-end justify-content-between h-100">
                  {currentData.sessions.map((session, index) => (
                    <div key={index} className="text-center">
                      <div className="mb-2">
                        <span className={`badge ${getScoreBadge(session.score)}`}>
                          {session.score}점
                        </span>
                      </div>
                      <div
                        className="bg-primary rounded"
                        style={{
                          width: '30px',
                          height: `${(session.score / 100) * 200}px`,
                          minHeight: '20px'
                        }}
                      ></div>
                      <div className="mt-2">
                        <small className="text-muted">{session.date}</small>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 최근 기록 */}
        <div className="col-lg-4 mb-4">
          <div className="card shadow">
            <div className="card-header py-3">
              <h6 className="m-0 font-weight-bold text-primary">최근 기록</h6>
            </div>
            <div className="card-body">
              <div className="list-group list-group-flush">
                {currentData.sessions.slice(0, 5).map((session, index) => (
                  <div key={index} className="list-group-item d-flex justify-content-between align-items-center">
                    <div>
                      <div className="fw-bold">{session.date}</div>
                      <small className="text-muted">{session.type}</small>
                    </div>
                    <span className={`badge ${getScoreBadge(session.score)}`}>
                      {session.score}점
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 상세 통계 */}
      <div className="row">
        <div className="col-12">
          <div className="card shadow">
            <div className="card-header py-3">
              <h6 className="m-0 font-weight-bold text-primary">상세 통계</h6>
            </div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-hover">
                  <thead>
                    <tr>
                      <th>날짜</th>
                      <th>유형</th>
                      <th>점수</th>
                      <th>상태</th>
                      <th>개선사항</th>
                    </tr>
                  </thead>
                  <tbody>
                    {currentData.sessions.map((session, index) => (
                      <tr key={index}>
                        <td>{session.date}</td>
                        <td>
                          <i className={`bi ${session.type === '체형 분석' ? 'bi-camera-video' : 'bi-webcam'} me-2`}></i>
                          {session.type}
                        </td>
                        <td>
                          <span className={`fw-bold ${getScoreColor(session.score)}`}>
                            {session.score}점
                          </span>
                        </td>
                        <td>
                          <span className="text-success">
                            <i className="bi bi-check-circle me-1"></i>
                            완료
                          </span>
                        </td>
                        <td>
                          <small className="text-muted">
                            {session.score >= 90 ? '훌륭함' :
                             session.score >= 80 ? '양호' :
                             session.score >= 70 ? '보통' : '개선 필요'}
                          </small>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Statistics; 