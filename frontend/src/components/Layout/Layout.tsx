import React from 'react';
import { Outlet } from 'react-router-dom';
import Header from './Header';

const Layout: React.FC = () => {
  return (
    <div className="d-flex flex-column min-vh-100">
      <Header />
      <main className="flex-grow-1">
        <div className="container-fluid py-4">
          <Outlet />
        </div>
      </main>
    </div>
  );
};

export default Layout; 