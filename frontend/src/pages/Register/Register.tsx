import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import toast from 'react-hot-toast';

const Register: React.FC = () => {
  const [formData, setFormData] = useState({
    email: '',
    username: '',
    password: '',
    confirmPassword: '',
    full_name: '',
    age: '',
    height: '',
    weight: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  
  const { register } = useAuth();
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
    
    // 필수 필드 검증
    if (!formData.email || !formData.username || !formData.password || !formData.confirmPassword) {
      toast.error('필수 항목을 모두 입력해주세요.');
      return;
    }

    // 비밀번호 확인
    if (formData.password !== formData.confirmPassword) {
      toast.error('비밀번호가 일치하지 않습니다.');
      return;
    }

    // 비밀번호 길이 검증
    if (formData.password.length < 6) {
      toast.error('비밀번호는 최소 6자 이상이어야 합니다.');
      return;
    }

    setIsLoading(true);
    
    try {
      const userData = {
        email: formData.email,
        username: formData.username,
        password: formData.password,
        full_name: formData.full_name || undefined,
        age: formData.age ? parseInt(formData.age) : undefined,
        height: formData.height ? parseInt(formData.height) : undefined,
        weight: formData.weight ? parseInt(formData.weight) : undefined
      };

      await register(userData.username, userData.email, userData.password);
      toast.success('회원가입에 성공했습니다!');
      navigate('/dashboard');
    } catch (error) {
      toast.error('회원가입에 실패했습니다. 다시 시도해주세요.');
      console.error('Register error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-vh-100 d-flex align-items-center justify-content-center bg-light py-5">
      <div className="container">
        <div className="row justify-content-center">
          <div className="col-md-8 col-lg-6">
            <div className="card shadow">
              <div className="card-body p-5">
                {/* 헤더 */}
                <div className="text-center mb-4">
                  <h2 className="fw-bold text-gradient mb-2">회원가입</h2>
                  <p className="text-muted">자세요정과 함께 건강한 자세를 만들어보세요</p>
                </div>

                {/* 회원가입 폼 */}
                <form onSubmit={handleSubmit}>
                  <div className="row">
                    {/* 필수 정보 */}
                    <div className="col-md-6">
                      <h5 className="mb-3">필수 정보</h5>
                      
                      <div className="mb-3">
                        <label htmlFor="email" className="form-label">
                          이메일 <span className="text-danger">*</span>
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

                      <div className="mb-3">
                        <label htmlFor="username" className="form-label">
                          사용자명 <span className="text-danger">*</span>
                        </label>
                        <input
                          type="text"
                          className="form-control"
                          id="username"
                          name="username"
                          value={formData.username}
                          onChange={handleChange}
                          placeholder="사용자명을 입력하세요"
                          required
                        />
                      </div>

                      <div className="mb-3">
                        <label htmlFor="password" className="form-label">
                          비밀번호 <span className="text-danger">*</span>
                        </label>
                        <input
                          type="password"
                          className="form-control"
                          id="password"
                          name="password"
                          value={formData.password}
                          onChange={handleChange}
                          placeholder="최소 6자 이상"
                          required
                        />
                      </div>

                      <div className="mb-3">
                        <label htmlFor="confirmPassword" className="form-label">
                          비밀번호 확인 <span className="text-danger">*</span>
                        </label>
                        <input
                          type="password"
                          className="form-control"
                          id="confirmPassword"
                          name="confirmPassword"
                          value={formData.confirmPassword}
                          onChange={handleChange}
                          placeholder="비밀번호를 다시 입력하세요"
                          required
                        />
                      </div>
                    </div>

                    {/* 선택 정보 */}
                    <div className="col-md-6">
                      <h5 className="mb-3">선택 정보</h5>
                      
                      <div className="mb-3">
                        <label htmlFor="full_name" className="form-label">
                          실명
                        </label>
                        <input
                          type="text"
                          className="form-control"
                          id="full_name"
                          name="full_name"
                          value={formData.full_name}
                          onChange={handleChange}
                          placeholder="실명을 입력하세요"
                        />
                      </div>

                      <div className="mb-3">
                        <label htmlFor="age" className="form-label">
                          나이
                        </label>
                        <input
                          type="number"
                          className="form-control"
                          id="age"
                          name="age"
                          value={formData.age}
                          onChange={handleChange}
                          placeholder="나이"
                          min="1"
                          max="120"
                        />
                      </div>

                      <div className="mb-3">
                        <label htmlFor="height" className="form-label">
                          키 (cm)
                        </label>
                        <input
                          type="number"
                          className="form-control"
                          id="height"
                          name="height"
                          value={formData.height}
                          onChange={handleChange}
                          placeholder="키"
                          min="100"
                          max="250"
                        />
                      </div>

                      <div className="mb-3">
                        <label htmlFor="weight" className="form-label">
                          몸무게 (kg)
                        </label>
                        <input
                          type="number"
                          className="form-control"
                          id="weight"
                          name="weight"
                          value={formData.weight}
                          onChange={handleChange}
                          placeholder="몸무게"
                          min="30"
                          max="200"
                        />
                      </div>
                    </div>
                  </div>

                  <button
                    type="submit"
                    className="btn btn-primary w-100 mt-4"
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <>
                        <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                        회원가입 중...
                      </>
                    ) : (
                      '회원가입'
                    )}
                  </button>
                </form>

                {/* 로그인 링크 */}
                <div className="text-center mt-4">
                  <p className="mb-0">
                    이미 계정이 있으신가요?{' '}
                    <Link to="/login" className="text-decoration-none">
                      로그인
                    </Link>
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Register; 