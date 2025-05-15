// Maximum number of history items to store
const MAX_HISTORY_ITEMS = 50;

// Get all history items
export const getHistoryItems = () => {
  try {
    const historyItems = localStorage.getItem('seedQualityHistory');
    return historyItems ? JSON.parse(historyItems) : [];
  } catch (error) {
    console.error('Error retrieving history:', error);
    return [];
  }
};

// Save a new result to history
export const saveToHistory = (result) => {
  try {
    const history = getHistoryItems();
    
    // Add new item at the beginning
    const updatedHistory = [result, ...history];
    
    // Limit the number of items stored
    if (updatedHistory.length > MAX_HISTORY_ITEMS) {
      updatedHistory.splice(MAX_HISTORY_ITEMS);
    }
    
    localStorage.setItem('seedQualityHistory', JSON.stringify(updatedHistory));
    return true;
  } catch (error) {
    console.error('Error saving to history:', error);
    return false;
  }
};

// Delete a specific history item
export const deleteHistoryItem = (id) => {
  try {
    const history = getHistoryItems();
    const updatedHistory = history.filter(item => item.id !== id);
    localStorage.setItem('seedQualityHistory', JSON.stringify(updatedHistory));
    return true;
  } catch (error) {
    console.error('Error deleting history item:', error);
    return false;
  }
};

// Clear all history
export const clearHistory = () => {
  try {
    localStorage.removeItem('seedQualityHistory');
    return true;
  } catch (error) {
    console.error('Error clearing history:', error);
    return false;
  }
}; 