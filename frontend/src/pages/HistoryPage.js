import React, { useState, useEffect } from 'react';
import HistoryItem from '../components/HistoryItem';
import { getHistoryItems, deleteHistoryItem, clearHistory } from '../utils/localStorage';
import './HistoryPage.css';

function HistoryPage() {
  const [historyItems, setHistoryItems] = useState([]);

  useEffect(() => {
    // Load history items on component mount
    setHistoryItems(getHistoryItems());
  }, []);

  const handleDeleteItem = (id) => {
    deleteHistoryItem(id);
    setHistoryItems(getHistoryItems());
  };

  const handleClearHistory = () => {
    if (window.confirm('Are you sure you want to clear all history items?')) {
      clearHistory();
      setHistoryItems([]);
    }
  };

  return (
    <div className="history-page">
      <h1>Analysis History</h1>
      
      {historyItems.length > 0 ? (
        <>
          <div className="history-actions">
            <button className="clear-all-btn" onClick={handleClearHistory}>
              Clear All History
            </button>
          </div>
          
          <div className="history-list">
            {historyItems.map(item => (
              <HistoryItem 
                key={item.id} 
                item={item} 
                onDelete={handleDeleteItem} 
              />
            ))}
          </div>
        </>
      ) : (
        <div className="empty-history">
          <div className="empty-icon">📂</div>
          <h2>No History Found</h2>
          <p>Your analysis history will appear here after you test seed images.</p>
        </div>
      )}
    </div>
  );
}

export default HistoryPage;