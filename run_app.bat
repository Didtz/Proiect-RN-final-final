@echo off
REM Plant Identifier - Quick Start Script for Windows
REM This script handles the setup and running of the plant identification app

echo.
echo ===============================================
echo   PLANT IDENTIFIER - Application Launcher
echo ===============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo [OK] Python found
echo.

REM Check if virtual environment exists, create if not
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
) else (
    echo [OK] Virtual environment found
    call venv\Scripts\activate.bat
)

echo.
echo [INFO] Checking if model is trained...

REM Check if model file exists
if not exist "models\plant_model.h5" (
    echo.
    echo [WARNING] Trained model not found!
    echo.
    echo The app needs a trained model to work. 
    echo Training will take approximately 10-15 minutes.
    echo.
    set /p train="Do you want to train the model now? (y/n): "
    
    if /i "%train%"=="y" (
        echo.
        echo [INFO] Starting model training...
        python train_model.py
        if errorlevel 1 (
            echo [ERROR] Training failed. Please check the error messages above.
            pause
            exit /b 1
        )
        echo [OK] Training completed successfully!
    ) else (
        echo [ERROR] Cannot run app without trained model.
        pause
        exit /b 1
    )
) else (
    echo [OK] Model found
)

echo.
echo ===============================================
echo   Starting Flask Web Application
echo ===============================================
echo.
echo [INFO] The app will open at: http://localhost:5000
echo [INFO] Press Ctrl+C to stop the server
echo.

REM Run the Flask app
python app_improved.py

pause
