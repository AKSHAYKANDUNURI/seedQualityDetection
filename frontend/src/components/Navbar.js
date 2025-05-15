import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';

function Navbar() {
  const location = useLocation();
  const [isOpen, setIsOpen] = useState(false);

  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  const closeMenu = () => {
    setIsOpen(false);
  };
  
  return (
    <nav className="navbar">
      <div className="navbar-logo">
        <Link to="/" onClick={closeMenu}>🌾 Paddy Seed Quality</Link>
      </div>
      
      <button className="navbar-mobile-toggle" onClick={toggleMenu}>
        {isOpen ? '✕' : '☰'}
      </button>
      
      <ul className={`navbar-links ${isOpen ? 'open' : ''}`}>
        <li className={location.pathname === '/' ? 'active' : ''}>
          <Link to="/" onClick={closeMenu}>Home</Link>
        </li>
        <li className={location.pathname === '/test' ? 'active' : ''}>
          <Link to="/test" onClick={closeMenu}>Test Seed Quality</Link>
        </li>
        <li className={location.pathname === '/history' ? 'active' : ''}>
          <Link to="/history" onClick={closeMenu}>History</Link>
        </li>
      </ul>
    </nav>
  );
}

export default Navbar;