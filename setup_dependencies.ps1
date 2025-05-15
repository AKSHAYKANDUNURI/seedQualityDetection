# PowerShell script to set up Seed Quality Analysis System dependencies

Write-Host "Setting up Seed Quality Analysis System dependencies..." -ForegroundColor Green

# Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
pip install tensorflow scikit-learn matplotlib seaborn pandas pillow flask flask-cors

# Install frontend dependencies
Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
Set-Location -Path .\frontend
npm install --save jspdf
Set-Location -Path ..

Write-Host "Dependencies installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Run the application with:" -ForegroundColor Yellow
Write-Host "1. Start backend: python backend/app.py" -ForegroundColor White
Write-Host "2. Start frontend: cd frontend && npm start" -ForegroundColor White
Write-Host ""
Write-Host "Run testing with:" -ForegroundColor Yellow
Write-Host "- Testing with custom images: python testing/test_model_accuracy.py" -ForegroundColor White
Write-Host "- Testing with random samples: python testing/random_test.py" -ForegroundColor White 