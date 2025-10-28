
# Create comprehensive backend app.py file
backend_app = """
Mental Fatigue Estimation - Flask Backend Application
Main API server for handling text and voice-based fatigue detection
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Import model modules
from models.text_fatigue_model_UPDATED import TextFatigueAnalyzer
from models.voice_fatigue_model_UPDATED import VoiceFatigueAnalyzer
from models.trend_forecasting import TrendForecaster
from utils.text_processor import TextProcessor
from utils.voice_processor import VoiceProcessor
from utils.explainability import ExplainabilityModule
from utils.translator import MultilingualTranslator

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['REPORTS_FOLDER'] = '../reports'

# Initialize components
text_analyzer = TextFatigueAnalyzer()
voice_analyzer = VoiceFatigueAnalyzer()
trend_forecaster = TrendForecaster()
text_processor = TextProcessor()
voice_processor = VoiceProcessor()
xai_module = ExplainabilityModule()
translator = MultilingualTranslator()

# In-memory storage (replace with database in production)
athlete_data = []

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'Mental Fatigue Estimation API',
        'version': '1.0.0',
        'status': 'active',
        'endpoints': {
            'text_analysis': '/api/analyze/text',
            'voice_analysis': '/api/analyze/voice',
            'trend_forecast': '/api/forecast',
            'dashboard_data': '/api/dashboard',
            'generate_report': '/api/report'
        }
    })

@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    """
    Analyze text input for mental fatigue detection
    Expected JSON: {"text": "athlete's text input", "athlete_id": "optional"}
    """
    try:
        data = request.get_json()
        text = data.get('text', '')
        athlete_id = data.get('athlete_id', 'anonymous')
        language = data.get('language', 'en')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Translate if not English
        original_text = text
        if language != 'en':
            text = translator.translate_to_english(text, language)
        
        # Process text
        processed_features = text_processor.extract_features(text)
        
        # Predict fatigue score
        fatigue_score = text_analyzer.predict_fatigue(processed_features)
        
        # Get sentiment analysis
        sentiment = text_processor.get_sentiment(text)
        
        # Generate explanations using XAI
        explanations = xai_module.explain_text_prediction(
            text, processed_features, fatigue_score
        )
        
        # Generate recommendations
        recommendations = generate_recommendations(fatigue_score, 'text')
        
        # Store data
        entry = {
            'timestamp': datetime.now().isoformat(),
            'athlete_id': athlete_id,
            'type': 'text',
            'input': original_text,
            'fatigue_score': float(fatigue_score),
            'sentiment': sentiment,
            'language': language
        }
        athlete_data.append(entry)
        
        return jsonify({
            'success': True,
            'fatigue_score': float(fatigue_score),
            'fatigue_level': get_fatigue_level(fatigue_score),
            'sentiment': sentiment,
            'explanations': explanations,
            'recommendations': recommendations,
            'features': processed_features,
            'timestamp': entry['timestamp']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/voice', methods=['POST'])
def analyze_voice():
    """
    Analyze voice recording for mental fatigue detection
    Expected: Audio file upload (WAV, MP3)
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        athlete_id = request.form.get('athlete_id', 'anonymous')
        
        # Save audio file temporarily
        filename = f"{athlete_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        audio_file.save(filepath)
        
        # Process audio
        audio_features = voice_processor.extract_audio_features(filepath)
        
        # Predict fatigue score
        fatigue_score = voice_analyzer.predict_fatigue(audio_features)
        
        # Generate explanations
        explanations = xai_module.explain_voice_prediction(
            audio_features, fatigue_score
        )
        
        # Generate recommendations
        recommendations = generate_recommendations(fatigue_score, 'voice')
        
        # Store data
        entry = {
            'timestamp': datetime.now().isoformat(),
            'athlete_id': athlete_id,
            'type': 'voice',
            'fatigue_score': float(fatigue_score),
            'audio_features': audio_features
        }
        athlete_data.append(entry)
        
        # Clean up audio file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'fatigue_score': float(fatigue_score),
            'fatigue_level': get_fatigue_level(fatigue_score),
            'audio_features': audio_features,
            'explanations': explanations,
            'recommendations': recommendations,
            'timestamp': entry['timestamp']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast', methods=['POST'])
def forecast_trend():
    """
    Generate fatigue trend forecast using time-series analysis
    Expected JSON: {"athlete_id": "id", "days_ahead": 7}
    """
    try:
        data = request.get_json()
        athlete_id = data.get('athlete_id', 'anonymous')
        days_ahead = data.get('days_ahead', 7)
        
        # Get historical data for athlete
        historical_data = [
            entry for entry in athlete_data 
            if entry['athlete_id'] == athlete_id
        ]
        
        if len(historical_data) < 5:
            return jsonify({
                'error': 'Insufficient historical data for forecasting',
                'message': 'At least 5 data points required'
            }), 400
        
        # Prepare time series data
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Generate forecast
        forecast = trend_forecaster.forecast(df, days_ahead)
        
        return jsonify({
            'success': True,
            'forecast': forecast,
            'days_ahead': days_ahead,
            'historical_points': len(historical_data)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    """
    Get dashboard data for visualization
    Query params: athlete_id, start_date, end_date, type
    """
    try:
        athlete_id = request.args.get('athlete_id', 'anonymous')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        data_type = request.args.get('type', 'all')
        
        # Filter data
        filtered_data = [
            entry for entry in athlete_data 
            if entry['athlete_id'] == athlete_id
        ]
        
        if data_type != 'all':
            filtered_data = [
                entry for entry in filtered_data 
                if entry['type'] == data_type
            ]
        
        if start_date:
            start = datetime.fromisoformat(start_date)
            filtered_data = [
                entry for entry in filtered_data
                if datetime.fromisoformat(entry['timestamp']) >= start
            ]
        
        if end_date:
            end = datetime.fromisoformat(end_date)
            filtered_data = [
                entry for entry in filtered_data
                if datetime.fromisoformat(entry['timestamp']) <= end
            ]
        
        # Calculate statistics
        if filtered_data:
            scores = [entry['fatigue_score'] for entry in filtered_data]
            stats = {
                'average_fatigue': float(np.mean(scores)),
                'max_fatigue': float(np.max(scores)),
                'min_fatigue': float(np.min(scores)),
                'std_fatigue': float(np.std(scores)),
                'total_entries': len(filtered_data),
                'text_entries': len([e for e in filtered_data if e['type'] == 'text']),
                'voice_entries': len([e for e in filtered_data if e['type'] == 'voice'])
            }
        else:
            stats = {
                'average_fatigue': 0,
                'max_fatigue': 0,
                'min_fatigue': 0,
                'std_fatigue': 0,
                'total_entries': 0,
                'text_entries': 0,
                'voice_entries': 0
            }
        
        return jsonify({
            'success': True,
            'data': filtered_data,
            'statistics': stats,
            'athlete_id': athlete_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/report', methods=['POST'])
def generate_report():
    """
    Generate downloadable wellness report (PDF/CSV)
    Expected JSON: {"athlete_id": "id", "format": "pdf/csv"}
    """
    try:
        data = request.get_json()
        athlete_id = data.get('athlete_id', 'anonymous')
        report_format = data.get('format', 'pdf')
        
        # Get athlete data
        athlete_entries = [
            entry for entry in athlete_data 
            if entry['athlete_id'] == athlete_id
        ]
        
        if not athlete_entries:
            return jsonify({'error': 'No data found for this athlete'}), 404
        
        # Generate report
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
        
        if report_format == 'pdf':
            filename = f"wellness_report_{athlete_id}_{datetime.now().strftime('%Y%m%d')}.pdf"
            filepath = os.path.join(app.config['REPORTS_FOLDER'], filename)
            
            c = canvas.Canvas(filepath, pagesize=letter)
            width, height = letter
            
            # Title
            c.setFont("Helvetica-Bold", 20)
            c.drawString(1*inch, height - 1*inch, "Mental Fatigue Wellness Report")
            
            # Athlete info
            c.setFont("Helvetica", 12)
            c.drawString(1*inch, height - 1.5*inch, f"Athlete ID: {athlete_id}")
            c.drawString(1*inch, height - 1.8*inch, f"Report Date: {datetime.now().strftime('%Y-%m-%d')}")
            c.drawString(1*inch, height - 2.1*inch, f"Total Assessments: {len(athlete_entries)}")
            
            # Statistics
            scores = [e['fatigue_score'] for e in athlete_entries]
            c.drawString(1*inch, height - 2.7*inch, "Summary Statistics:")
            c.drawString(1.5*inch, height - 3.0*inch, f"Average Fatigue Score: {np.mean(scores):.2f}/10")
            c.drawString(1.5*inch, height - 3.3*inch, f"Highest Score: {np.max(scores):.2f}")
            c.drawString(1.5*inch, height - 3.6*inch, f"Lowest Score: {np.min(scores):.2f}")
            
            c.save()
            
            return send_file(filepath, as_attachment=True)
            
        elif report_format == 'csv':
            df = pd.DataFrame(athlete_entries)
            filename = f"wellness_report_{athlete_id}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(app.config['REPORTS_FOLDER'], filename)
            df.to_csv(filepath, index=False)
            
            return send_file(filepath, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_fatigue_level(score):
    """Convert numeric score to categorical level"""
    if score < 3:
        return "Low"
    elif score < 5:
        return "Mild"
    elif score < 7:
        return "Moderate"
    elif score < 8.5:
        return "High"
    else:
        return "Severe"

def generate_recommendations(fatigue_score, input_type):
    """Generate personalized wellness recommendations"""
    recommendations = []
    
    if fatigue_score < 3:
        recommendations = [
            "Maintain current wellness routines",
            "Continue regular sleep schedule",
            "Keep up balanced nutrition"
        ]
    elif fatigue_score < 5:
        recommendations = [
            "Consider light recovery exercises",
            "Ensure 7-9 hours of quality sleep",
            "Practice mindfulness for 10 minutes daily"
        ]
    elif fatigue_score < 7:
        recommendations = [
            "Schedule a rest day or light training",
            "Consult with sports psychologist",
            "Focus on stress-reduction techniques",
            "Monitor sleep quality closely"
        ]
    else:
        recommendations = [
            "URGENT: Consult with medical team immediately",
            "Take 2-3 days complete rest",
            "Avoid high-intensity training",
            "Consider professional mental health support",
            "Review training load with coach"
        ]
    
    return recommendations

if __name__ == '__main__':
    print("=" * 70)
    print("Mental Fatigue Estimation API Server")
    print("=" * 70)
    print("Server starting on http://localhost:5000")
    print("Press CTRL+C to stop the server")
    print("=" * 70)
    app.run(debug=True, host='0.0.0.0', port=5000)

print("BACKEND APP.PY CREATED")
print("=" * 80)
print(f"Lines of code: {len(backend_app.split(chr(10)))}")
print(f"Total characters: {len(backend_app)}")
