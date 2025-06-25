import React from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';

const Dashboard: React.FC = () => {
  const { user } = useAuth();

  // 더미 통계 데이터
  const stats = {
    totalAnalyses: 15,
    thisWeek: 3,
    averageScore: 85,
    improvementRate: 12
  };

  const recentActivities = [
    { id: 1, type: '체형 분석', date: '2024-01-15', score: 87 },
    { id: 2, type: '자세 교정', date: '2024-01-14', score: 82 },
    { id: 3, type: '체형 분석', date: '2024-01-12', score: 89 },
    { id: 4, type: '자세 교정', date: '2024-01-10', score: 78 }
  ];

  return (
    <div className="container-fluid">
      {/* 환영 메시지 */}
      <div className="row mb-4">
        <div className="col-12">
          <div className="card bg-gradient-primary text-white">
            <div className="card-body">
              <h2 className="card-title mb-2">
                안녕하세요, {user?.full_name || user?.username || '사용자'}님! 👋
              </h2>
              <p className="card-text mb-0">
                오늘도 건강한 자세로 하루를 시작해보세요.
              </p>
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
                    총 분석 횟수
                  </div>
                  <div className="h5 mb-0 font-weight-bold text-gray-800">
                    {stats.totalAnalyses}회
                  </div>
                </div>
                <div className="col-auto">
                  <i className="bi bi-graph-up fa-2x text-gray-300"></i>
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
                    이번 주 분석
                  </div>
                  <div className="h5 mb-0 font-weight-bold text-gray-800">
                    {stats.thisWeek}회
                  </div>
                </div>
                <div className="col-auto">
                  <i className="bi bi-calendar-week fa-2x text-gray-300"></i>
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
                    평균 점수
                  </div>
                  <div className="h5 mb-0 font-weight-bold text-gray-800">
                    {stats.averageScore}점
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
          <div className="card border-left-warning shadow h-100 py-2">
            <div className="card-body">
              <div className="row no-gutters align-items-center">
                <div className="col mr-2">
                  <div className="text-xs font-weight-bold text-warning text-uppercase mb-1">
                    개선률
                  </div>
                  <div className="h5 mb-0 font-weight-bold text-gray-800">
                    +{stats.improvementRate}%
                  </div>
                </div>
                <div className="col-auto">
                  <i className="bi bi-arrow-up-right fa-2x text-gray-300"></i>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 빠른 액션 */}
      <div className="row mb-4">
        <div className="col-12">
          <div className="card shadow">
            <div className="card-header py-3">
              <h6 className="m-0 font-weight-bold text-primary">빠른 시작</h6>
            </div>
            <div className="card-body">
              <div className="row">
                <div className="col-md-6 mb-3">
                  <Link to="/body-analysis" className="text-decoration-none">
                    <div className="card h-100 border-0 shadow-sm hover-lift">
                      <div className="card-body text-center">
                        <i className="bi bi-camera-video fa-3x text-primary mb-3"></i>
                        <h5 className="card-title">체형 분석</h5>
                        <p className="card-text text-muted">
                          사진을 업로드하여 체형을 분석해보세요
                        </p>
                      </div>
                    </div>
                  </Link>
                </div>
                <div className="col-md-6 mb-3">
                  <Link to="/posture-correction" className="text-decoration-none">
                    <div className="card h-100 border-0 shadow-sm hover-lift">
                      <div className="card-body text-center">
                        <i className="bi bi-webcam fa-3x text-success mb-3"></i>
                        <h5 className="card-title">자세 교정</h5>
                        <p className="card-text text-muted">
                          실시간으로 자세를 감지하고 교정해보세요
                        </p>
                      </div>
                    </div>
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 최근 활동 */}
      <div className="row">
        <div className="col-12">
          <div className="card shadow">
            <div className="card-header py-3 d-flex flex-row align-items-center justify-content-between">
              <h6 className="m-0 font-weight-bold text-primary">최근 활동</h6>
              <Link to="/statistics" className="btn btn-sm btn-primary">
                전체 보기
              </Link>
            </div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-hover">
                  <thead>
                    <tr>
                      <th>활동</th>
                      <th>날짜</th>
                      <th>점수</th>
                      <th>상태</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recentActivities.map((activity) => (
                      <tr key={activity.id}>
                        <td>
                          <i className={`bi ${activity.type === '체형 분석' ? 'bi-camera-video' : 'bi-webcam'} me-2`}></i>
                          {activity.type}
                        </td>
                        <td>{activity.date}</td>
                        <td>
                          <span className={`badge ${activity.score >= 80 ? 'bg-success' : activity.score >= 60 ? 'bg-warning' : 'bg-danger'}`}>
                            {activity.score}점
                          </span>
                        </td>
                        <td>
                          <span className="text-success">
                            <i className="bi bi-check-circle me-1"></i>
                            완료
                          </span>
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

export default Dashboard; 