import React from 'react';
import SeedAnalyzer from '../components/SeedAnalyzer';
import { saveToHistory } from '../utils/localStorage';
import './TestPage.css';

function TestPage() {
  const handleSaveResult = (result) => {
    saveToHistory(result);
  };

  return (
    <div className="test-page">
      <h1>Test Seed Quality</h1>
      <p className="test-intro">
        Upload an image of paddy seeds to analyze their quality. Our AI will assess 
        the seeds and provide a detailed quality report.
      </p>
      <SeedAnalyzer onSaveResult={handleSaveResult} />
    </div>
  );
}

export default TestPage;