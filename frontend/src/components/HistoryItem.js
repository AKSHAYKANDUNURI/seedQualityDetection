import React from 'react';
import './HistoryItem.css';

function HistoryItem({ item, onDelete }) {
  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  return (
    <div className="history-item">
      <div className="history-item-image">
        <img src={item.imageUrl} alt={item.imageName} />
      </div>
      <div className="history-item-details">
        <h3>{item.imageName}</h3>
        <p className="history-date">{formatDate(item.timestamp)}</p>
        <div className={`history-result ${item.prediction.label.includes('Good') ? 'good' : 'bad'}`}>
          {item.prediction.label}
        </div>
        <div className="history-metrics">
          <span className="good">Good: {item.prediction.good 
            ? (item.prediction.good * 100).toFixed(1) 
            : item.prediction.confidence_good
            ? item.prediction.confidence_good.toFixed(1) 
            : 0}%</span>
          <span className="bad">Bad: {item.prediction.bad 
            ? (item.prediction.bad * 100).toFixed(1) 
            : item.prediction.confidence_bad
            ? item.prediction.confidence_bad.toFixed(1)
            : 0}%</span>
        </div>
      </div>
      <button className="delete-btn" onClick={() => onDelete(item.id)}>
        🗑️
      </button>
    </div>
  );
}

export default HistoryItem;