#!/bin/bash

echo "Setting up Seed Quality Analysis System dependencies..."

# Install Python dependencies
echo "Installing Python dependencies..."
pip install tensorflow scikit-learn matplotlib seaborn pandas pillow flask flask-cors

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install --save jspdf

echo "Dependencies installation complete!"
echo ""
echo "Run the application with:"
echo "1. Start backend: python backend/app.py"
echo "2. Start frontend: cd frontend && npm start"
echo ""
echo "Run testing with:"
echo "- Testing with custom images: python testing/test_model_accuracy.py"
echo "- Testing with random samples: python testing/random_test.py" 