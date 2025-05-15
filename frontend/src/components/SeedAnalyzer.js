import React, { useState, useRef } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import jsPDF from 'jspdf';
import './SeedAnalyzer.css';

function SeedAnalyzer({ onSaveResult }) {
  const [selectedImage, setSelectedImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const getChartData = () => {
    if (!prediction) return [];
    return [
      { name: 'Good Seed', value: prediction.good || prediction.confidence_good / 100 || 0 },
      { name: 'Bad Seed', value: prediction.bad || prediction.confidence_bad / 100 || 0 }
    ];
  };

  const handleImageSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Validate file type
      const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
      if (!validTypes.includes(file.type)) {
        setError('Please select a valid image file (JPG, JPEG, PNG)');
        return;
      }
      
      // Check file size (limit to 10MB)
      if (file.size > 10 * 1024 * 1024) {
        setError('File size too large. Please select an image under 10MB');
        return;
      }
      
      setSelectedImage(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const handlePrediction = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000, // 30 second timeout
      });

      setPrediction(response.data);
      
      // Save to history if callback provided
      if (onSaveResult) {
        onSaveResult({
          id: Date.now(),
          timestamp: new Date().toISOString(),
          imageName: selectedImage.name,
          imageUrl: preview,
          prediction: response.data
        });
      }
    } catch (err) {
      console.error('Prediction error:', err);
      if (err.response && err.response.data && err.response.data.error) {
        setError(err.response.data.error);
      } else if (err.code === 'ECONNABORTED') {
        setError('Request timeout. Please try again.');
      } else if (err.code === 'ERR_NETWORK') {
        setError('Cannot connect to server. Make sure the backend is running on port 5000.');
      } else {
        setError('Error making prediction. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = () => {
    if (!prediction || !selectedImage) return;

    // Create PDF document
    const pdf = new jsPDF();
    const reportDate = new Date().toLocaleString();
    const fileName = selectedImage.name;
    const fileSize = (selectedImage.size / 1024).toFixed(2) + ' KB';
    
    // Add header with logo-like styling
    pdf.setFillColor(51, 51, 153);
    pdf.rect(0, 0, 210, 30, 'F');
    pdf.setTextColor(255, 255, 255);
    pdf.setFontSize(20);
    pdf.text('Seed Quality Analysis Report', 105, 15, { align: 'center' });
    
    // Reset color for main content
    pdf.setTextColor(0, 0, 0);
    pdf.setFontSize(12);
    
    // Add report content
    let yPos = 40;
    
    // Add file info section
    pdf.setFontSize(14);
    pdf.text('Analysis Information', 20, yPos);
    pdf.setFontSize(12);
    yPos += 10;
    pdf.setDrawColor(200, 200, 200);
    pdf.line(20, yPos, 190, yPos);
    yPos += 10;
    pdf.text(`Date: ${reportDate}`, 20, yPos);
    yPos += 8;
    pdf.text(`File: ${fileName} (${fileSize})`, 20, yPos);
    
    // Add results section
    yPos += 15;
    pdf.setFontSize(14);
    pdf.text('Analysis Results', 20, yPos);
    pdf.setFontSize(12);
    yPos += 10;
    pdf.setDrawColor(200, 200, 200);
    pdf.line(20, yPos, 190, yPos);
    yPos += 10;
    
    // Result label with colored box
    const resultColor = prediction.label.includes('Good') ? [72, 187, 120] : [229, 62, 62];
    pdf.setFillColor(...resultColor);
    pdf.rect(20, yPos - 5, 8, 8, 'F');
    pdf.text(`Classification: ${prediction.label}`, 35, yPos);
    yPos += 10;
    
    // Main metrics
    pdf.text(`Confidence: ${prediction.confidence.toFixed(1)}%`, 35, yPos);
    yPos += 8;
    pdf.text(`Good Seed Probability: ${(prediction.good * 100).toFixed(1)}%`, 35, yPos);
    yPos += 8;
    pdf.text(`Bad Seed Probability: ${(prediction.bad * 100).toFixed(1)}%`, 35, yPos);
    yPos += 8;
    pdf.text(`Score: ${prediction.score.toFixed(4)}`, 35, yPos);
    
    // Add recommendation section
    yPos += 15;
    pdf.setFontSize(14);
    pdf.text('Recommendation', 20, yPos);
    pdf.setFontSize(12);
    yPos += 10;
    pdf.setDrawColor(200, 200, 200);
    pdf.line(20, yPos, 190, yPos);
    yPos += 10;
    
    // Add recommendation text with wrapping
    const recommendationText = prediction.label.includes('Good') 
      ? 'These seeds appear to be of good quality and suitable for planting. Store them in a cool, dry place for best results.'
      : 'These seeds appear to be of poor quality. Consider using different seeds for planting or improving seed storage conditions.';
    
    const splitText = pdf.splitTextToSize(recommendationText, 150);
    pdf.text(splitText, 35, yPos);
    
    // Add footer
    pdf.setFontSize(10);
    pdf.setTextColor(100, 100, 100);
    pdf.text('Generated by Seed Quality Analysis System', 105, 280, { align: 'center' });
    
    // Save the PDF
    const baseFileName = selectedImage.name.split('.')[0];
    pdf.save(`${baseFileName}-seed-quality-report.pdf`);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      
      // Validate file type
      const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
      if (!validTypes.includes(file.type)) {
        setError('Please select a valid image file (JPG, JPEG, PNG)');
        return;
      }
      
      // Check file size (limit to 10MB)
      if (file.size > 10 * 1024 * 1024) {
        setError('File size too large. Please select an image under 10MB');
        return;
      }
      
      setSelectedImage(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const resetAnalyzer = () => {
    setSelectedImage(null);
    setPreview(null);
    setPrediction(null);
    setError(null);
    setDragging(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleFileInputClick = () => {
    fileInputRef.current?.click();
  };
  
  const getSuggestion = () => {
    if (!prediction) return null;
    
    if (prediction.label.includes('Good')) {
      return (
        <div className="suggestion good">
          <h4>✅ Recommendation</h4>
          <p>These seeds appear to be of good quality and suitable for planting. Store them in a cool, dry place for best results.</p>
        </div>
      );
    } else {
      return (
        <div className="suggestion bad">
          <h4>⚠️ Recommendation</h4>
          <p>These seeds appear to be of poor quality. Consider using different seeds for planting or improving seed storage conditions.</p>
        </div>
      );
    }
  };

  return (
    <div className="seed-analyzer">
      <div className="analyzer-container">
        <div className="upload-section">
          <div
            className={`upload-area ${selectedImage ? 'has-image' : ''} ${dragging ? 'dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={handleFileInputClick}
          >
            {preview ? (
              <div className="preview-container">
                <img src={preview} alt="Preview" className="preview-image" />
                <div className="preview-overlay">
                  <span>Click to change image</span>
                </div>
              </div>
            ) : (
              <div className="upload-placeholder">
                <div className="upload-icon">📁</div>
                <p>Drag and drop an image here, or click to select</p>
                <p className="upload-hint">Supports JPG, PNG, JPEG (max 10MB)</p>
              </div>
            )}
          </div>

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleImageSelect}
            accept="image/*"
            className="file-input"
          />

          <div className="button-group">
            <button
              onClick={handlePrediction}
              disabled={!selectedImage || loading}
              className="predict-button"
            >
              {loading ? (
                <>
                  <span className="loading-spinner"></span>
                  Analyzing...
                </>
              ) : (
                '🔍 Analyze Quality'
              )}
            </button>
            
            {selectedImage && (
              <button onClick={resetAnalyzer} className="reset-button">
                🔄 Reset
              </button>
            )}
          </div>
        </div>

        {error && (
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            {error}
          </div>
        )}
      </div>
      
      <div className="results-container">
        {prediction && (
          <div className="prediction-section">
            <h2>Analysis Result</h2>
            <div className={`result-card ${prediction.label.includes('Good') ? 'good' : 'bad'}`}>
              <div className="result-label">
                <span className="result-icon">
                  {prediction.label.includes('Good') ? '✅' : '❌'}
                </span>
                <span className="result-text">{prediction.label}</span>
              </div>
              
              <div className="quality-metrics">
                <div className="quality-metric-item">
                  <span className="metric-label">Good Seed:</span>
                  <span className="metric-value">{prediction.good 
                    ? (prediction.good * 100).toFixed(1) 
                    : prediction.confidence_good
                    ? prediction.confidence_good.toFixed(1) 
                    : 0}%</span>
                </div>
                <div className="quality-metric-item">
                  <span className="metric-label">Bad Seed:</span>
                  <span className="metric-value">{prediction.bad 
                    ? (prediction.bad * 100).toFixed(1) 
                    : prediction.confidence_bad
                    ? prediction.confidence_bad.toFixed(1)
                    : 0}%</span>
                </div>
              </div>
              
              <div className="confidence-meter">
                <div className="confidence-label">
                  Confidence: {prediction.confidence
                    ? prediction.confidence.toFixed(1)
                    : Math.max(
                        prediction.confidence_good || 0,
                        prediction.confidence_bad || 0
                      ).toFixed(1)}%
                </div>
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{ width: `${prediction.confidence || Math.max(
                      prediction.confidence_good || 0,
                      prediction.confidence_bad || 0
                    )}%` }}
                  ></div>
                </div>
              </div>
              
              <div className="score-details">
                <small>Score: {prediction.score.toFixed(4)}</small>
              </div>
              
              <div className="chart-container" style={{ width: '100%', height: 200, marginTop: 20 }}>
                <ResponsiveContainer>
                  <BarChart data={getChartData()}>
                    <XAxis dataKey="name" />
                    <YAxis domain={[0, 1]} tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
                    <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
                    <Bar dataKey="value" fill={(data, index) => index === 0 ? '#4caf50' : '#f44336'} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              {getSuggestion()}
              
              <div className="action-buttons">
                <button
                  onClick={downloadReport}
                  className="download-button"
                >
                  📥 Download Report
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default SeedAnalyzer;