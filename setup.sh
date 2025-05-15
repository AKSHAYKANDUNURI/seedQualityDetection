#!/bin/bash

# Create project structure
mkdir -p model backend frontend

# Install Python dependencies
pip install tensorflow keras flask flask-cors pillow numpy scikit-learn matplotlib

# Setup React frontend
cd frontend
npm install
cd ..

# Create requirements file
echo "tensorflow>=2.12.0
keras
flask
flask-cors
pillow
numpy
scikit-learn
matplotlib
opencv-python" > requirements.txt

echo "Setup complete!"
echo "1. Train model: python model/train_model.py"
echo "2. Start backend: python backend/app.py"
echo "3. Start frontend: cd frontend && npm start" 