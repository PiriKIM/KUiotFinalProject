import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';

const Header: React.FC = () => {
  const { user, logout } = useAuth();
  const location = useLocation();

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  const handleLogout = () => {
    logout();
  };

  return (
    <nav className="navbar navbar-expand-lg navbar-light bg-white shadow-sm">
      <div className="container-fluid">
        {/* 브랜드 로고 */}
        <Link className="navbar-brand fw-bold text-gradient" to="/dashboard">
          자세요정
        </Link>

        {/* 모바일 토글 버튼 */}
        <button
          className="navbar-toggler"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#navbarNav"
          aria-controls="navbarNav"
          aria-expanded="false"
          aria-label="Toggle navigation"
        >
          <span className="navbar-toggler-icon"></span>
        </button>

        {/* 네비게이션 메뉴 */}
        <div className="collapse navbar-collapse" id="navbarNav">
          <ul className="navbar-nav me-auto">
            <li className="nav-item">
              <Link
                className={`nav-link ${isActive('/dashboard') ? 'active' : ''}`}
                to="/dashboard"
              >
                대시보드
              </Link>
            </li>
            <li className="nav-item">
              <Link
                className={`nav-link ${isActive('/body-analysis') ? 'active' : ''}`}
                to="/body-analysis"
              >
                체형 분석
              </Link>
            </li>
            <li className="nav-item">
              <Link
                className={`nav-link ${isActive('/posture-correction') ? 'active' : ''}`}
                to="/posture-correction"
              >
                자세 교정
              </Link>
            </li>
            <li className="nav-item">
              <Link
                className={`nav-link ${isActive('/statistics') ? 'active' : ''}`}
                to="/statistics"
              >
                통계
              </Link>
            </li>
          </ul>

          {/* 사용자 메뉴 */}
          <ul className="navbar-nav">
            <li className="nav-item dropdown">
              <a
                className="nav-link dropdown-toggle d-flex align-items-center"
                href="#"
                id="navbarDropdown"
                role="button"
                data-bs-toggle="dropdown"
                aria-expanded="false"
              >
                <i className="bi bi-person-circle me-1"></i>
                {user?.full_name || user?.username || '사용자'}
              </a>
              <ul className="dropdown-menu dropdown-menu-end" aria-labelledby="navbarDropdown">
                <li>
                  <Link className="dropdown-item" to="/profile">
                    <i className="bi bi-person me-2"></i>
                    프로필
                  </Link>
                </li>
                <li><hr className="dropdown-divider" /></li>
                <li>
                  <button className="dropdown-item text-danger" onClick={handleLogout}>
                    <i className="bi bi-box-arrow-right me-2"></i>
                    로그아웃
                  </button>
                </li>
              </ul>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default Header; 