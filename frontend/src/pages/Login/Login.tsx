import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import toast from 'react-hot-toast';

const Login: React.FC = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.email || !formData.password) {
      toast.error('이메일과 비밀번호를 모두 입력해주세요.');
      return;
    }

    setIsLoading(true);
    
    try {
      await login(formData.email, formData.password);
      toast.success('로그인에 성공했습니다!');
      navigate('/dashboard');
    } catch (error) {
      toast.error('로그인에 실패했습니다. 다시 시도해주세요.');
      console.error('Login error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-vh-100 d-flex align-items-center justify-content-center bg-light">
      <div className="container">
        <div className="row justify-content-center">
          <div className="col-md-6 col-lg-4">
            <div className="card shadow">
              <div className="card-body p-5">
                {/* 헤더 */}
                <div className="text-center mb-4">
                  <h2 className="fw-bold text-gradient mb-2">자세요정</h2>
                  <p className="text-muted">AI 기반 체형 분석 및 자세 교정</p>
                </div>

                {/* 로그인 폼 */}
                <form onSubmit={handleSubmit}>
                  <div className="mb-3">
                    <label htmlFor="email" className="form-label">
                      이메일
                    </label>
                    <input
                      type="email"
                      className="form-control"
                      id="email"
                      name="email"
                      value={formData.email}
                      onChange={handleChange}
                      placeholder="example@email.com"
                      required
                    />
                  </div>

                  <div className="mb-4">
                    <label htmlFor="password" className="form-label">
                      비밀번호
                    </label>
                    <input
                      type="password"
                      className="form-control"
                      id="password"
                      name="password"
                      value={formData.password}
                      onChange={handleChange}
                      placeholder="비밀번호를 입력하세요"
                      required
                    />
                  </div>

                  <button
                    type="submit"
                    className="btn btn-primary w-100 mb-3"
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <>
                        <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                        로그인 중...
                      </>
                    ) : (
                      '로그인'
                    )}
                  </button>
                </form>

                {/* 회원가입 링크 */}
                <div className="text-center">
                  <p className="mb-0">
                    계정이 없으신가요?{' '}
                    <Link to="/register" className="text-decoration-none">
                      회원가입
                    </Link>
                  </p>
                </div>

                {/* 테스트 계정 안내 */}
                <div className="mt-4 p-3 bg-light rounded">
                  <small className="text-muted">
                    <strong>테스트 계정:</strong><br />
                    이메일: test@example.com<br />
                    비밀번호: password123
                  </small>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login; 