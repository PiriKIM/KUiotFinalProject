import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// 사용자 타입 정의
interface User {
  id: number;
  username: string;
  email: string;
  full_name?: string;
}

// 인증 컨텍스트 타입 정의
interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<boolean>;
  register: (username: string, email: string, password: string) => Promise<boolean>;
  logout: () => void;
  loading: boolean;
  updateUser: (userData: Partial<User>) => void;
}

// 컨텍스트 생성
const AuthContext = createContext<AuthContextType | undefined>(undefined);

// AuthProvider 컴포넌트
interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // 로컬 스토리지에서 사용자 정보 복원
  useEffect(() => {
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      try {
        setUser(JSON.parse(savedUser));
      } catch (error) {
        console.error('사용자 정보 파싱 오류:', error);
        localStorage.removeItem('user');
      }
    }
    setLoading(false);
  }, []);

  // 로그인 함수
  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      setLoading(true);
      
      // 실제 API 호출 대신 임시 로그인 로직
      const response = await fetch('http://localhost:8000/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      if (response.ok) {
        const data = await response.json();
        const userData = {
          id: data.user_id || 1,
          username: data.username || email.split('@')[0],
          email: email,
        };
        
        setUser(userData);
        localStorage.setItem('user', JSON.stringify(userData));
        localStorage.setItem('token', data.access_token || 'dummy-token');
        return true;
      } else {
        // 임시 로그인 (개발용)
        const userData = {
          id: 1,
          username: email.split('@')[0],
          email: email,
        };
        
        setUser(userData);
        localStorage.setItem('user', JSON.stringify(userData));
        localStorage.setItem('token', 'dummy-token');
        return true;
      }
    } catch (error) {
      console.error('로그인 오류:', error);
      
      // 개발용 임시 로그인
      const userData = {
        id: 1,
        username: email.split('@')[0],
        email: email,
      };
      
      setUser(userData);
      localStorage.setItem('user', JSON.stringify(userData));
      localStorage.setItem('token', 'dummy-token');
      return true;
    } finally {
      setLoading(false);
    }
  };

  // 회원가입 함수
  const register = async (username: string, email: string, password: string): Promise<boolean> => {
    try {
      setLoading(true);
      
      // 실제 API 호출 대신 임시 회원가입 로직
      const response = await fetch('http://localhost:8000/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, email, password }),
      });

      if (response.ok) {
        const data = await response.json();
        const userData = {
          id: data.user_id || 1,
          username: username,
          email: email,
        };
        
        setUser(userData);
        localStorage.setItem('user', JSON.stringify(userData));
        localStorage.setItem('token', data.access_token || 'dummy-token');
        return true;
      } else {
        // 임시 회원가입 (개발용)
        const userData = {
          id: 1,
          username: username,
          email: email,
        };
        
        setUser(userData);
        localStorage.setItem('user', JSON.stringify(userData));
        localStorage.setItem('token', 'dummy-token');
        return true;
      }
    } catch (error) {
      console.error('회원가입 오류:', error);
      
      // 개발용 임시 회원가입
      const userData = {
        id: 1,
        username: username,
        email: email,
      };
      
      setUser(userData);
      localStorage.setItem('user', JSON.stringify(userData));
      localStorage.setItem('token', 'dummy-token');
      return true;
    } finally {
      setLoading(false);
    }
  };

  // 로그아웃 함수
  const logout = () => {
    setUser(null);
    localStorage.removeItem('user');
    localStorage.removeItem('token');
  };

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user,
    login,
    register,
    logout,
    loading,
    updateUser: (userData: Partial<User>) => {
      if (user) {
        const updatedUser = { ...user, ...userData };
        setUser(updatedUser);
        localStorage.setItem('user', JSON.stringify(updatedUser));
      }
    },
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

// useAuth 훅
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}; 