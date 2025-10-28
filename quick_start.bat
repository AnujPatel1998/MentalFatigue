@echo off
REM Quick Start Script for Mental Fatigue Estimation System (Windows)

echo ================================================
echo Mental Fatigue Estimation - Quick Start
echo ================================================

REM Create directories
echo Creating project structure...
mkdir backend\models backend\utils data trained_models\text_models trained_models\voice_models reports uploads 2>nul

REM Create __init__.py files
type nul > backend\__init__.py
type nul > backend\models\__init__.py
type nul > backend\utils\__init__.py

REM Setup virtual environment
echo Setting up Python virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing Python packages...
python -m pip install --upgrade pip
pip install flask flask-cors numpy pandas scikit-learn
pip install nltk textblob vaderSentiment
pip install librosa soundfile pydub
pip install shap lime
pip install langdetect
pip install statsmodels
pip install fpdf reportlab matplotlib seaborn plotly
pip install sqlalchemy requests python-dotenv

REM Download NLTK data
echo Downloading NLTK data...
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

echo.
echo ================================================
echo Setup Complete!
echo ================================================
echo.
echo Next Steps:
echo 1. Copy all backend Python files to backend\ directory
echo 2. Extract frontend ZIP to frontend\ directory
echo 3. Start backend: cd backend ^&^& python app.py
echo 4. Start frontend: cd frontend ^&^& python -m http.server 8080
echo 5. Open browser: http://localhost:8080
echo.
echo ================================================
pause
