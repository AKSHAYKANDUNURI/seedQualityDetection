import React from 'react';
import { Link } from 'react-router-dom';
import './HomePage.css';

function HomePage() {
  return (
    <div className="home-page">
      <div className="hero-section">
        <div className="hero-content">
          <h1>Paddy Seed Quality Detection</h1>
          <p>
            Our AI-powered tool helps farmers and agricultural experts assess the quality of paddy seeds
            to improve crop yields and reduce waste. Upload seed images to get instant quality analysis.
          </p>
          <Link to="/test" className="cta-button">Test Your Seeds Now</Link>
        </div>
        <div className="hero-image">
          <img src="/images/seed-hero.jpg" alt="Paddy seeds" onError={(e) => e.target.style.display = 'none'} />
        </div>
      </div>
      
      <div className="info-section">
        <h2>Why Seed Quality Matters</h2>
        <div className="info-cards">
          <div className="info-card">
            <div className="info-icon">🌱</div>
            <h3>Better Germination</h3>
            <p>High-quality seeds have higher germination rates, leading to better crop establishment.</p>
          </div>
          <div className="info-card">
            <div className="info-icon">📈</div>
            <h3>Higher Yield</h3>
            <p>Quality seeds produce stronger plants with improved resistance to diseases and pests.</p>
          </div>
          <div className="info-card">
            <div className="info-icon">💰</div>
            <h3>Cost Effective</h3>
            <p>Selecting quality seeds helps reduce waste and improves return on investment.</p>
          </div>
        </div>
      </div>
      
      <div className="how-it-works">
        <h2>How It Works</h2>
        <div className="steps">
          <div className="step">
            <div className="step-number">1</div>
            <h3>Upload Image</h3>
            <p>Take a clear photo of your paddy seeds and upload it to our system.</p>
          </div>
          <div className="step">
            <div className="step-number">2</div>
            <h3>AI Analysis</h3>
            <p>Our advanced AI model analyzes the image to assess seed quality.</p>
          </div>
          <div className="step">
            <div className="step-number">3</div>
            <h3>Get Results</h3>
            <p>Receive detailed quality assessment with visualization and recommendations.</p>
          </div>
        </div>
        <Link to="/test" className="secondary-cta">Try It Now</Link>
      </div>
    </div>
  );
}

export default HomePage;