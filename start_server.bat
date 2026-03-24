@echo off
echo Setting up AI.RiskPredictor Website...
echo.

echo 1. Creating Python virtual environment (optional but recommended)...
echo 2. Installing requirements (Django, scikit-learn, joblib, pandas)...
pip install django scikit-learn joblib pandas numpy

echo.
echo 3. Training and exporting the Linear Regression Piecewise Models...
cd /d "%~dp0"
python export_models.py

echo.
echo 4. Starting the Django Server...
python manage.py runserver

echo.
pause
