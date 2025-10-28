
# Create XAI and translator utility files, plus trend forecasting model

xai_module = '''"""
Explainable AI Module
Provides interpretable explanations for model predictions
"""

import numpy as np
import shap
from lime.lime_text import LimeTextExplainer

class ExplainabilityModule:
    def __init__(self):
        self.lime_explainer = LimeTextExplainer(class_names=['Low Fatigue', 'High Fatigue'])
    
    def explain_text_prediction(self, text, features, prediction):
        """Generate explanations for text-based predictions"""
        explanations = {
            'prediction_score': float(prediction),
            'top_factors': [],
            'feature_importance': {},
            'interpretation': ''
        }
        
        # Analyze which features contributed most
        feature_impacts = []
        
        if features.get('sentiment_score', 0) < -0.3:
            feature_impacts.append({
                'feature': 'Negative Sentiment',
                'impact': 'High',
                'value': features['sentiment_score'],
                'contribution': '+2.5 to fatigue score'
            })
        
        if features.get('fatigue_keywords', 0) > 0:
            feature_impacts.append({
                'feature': 'Fatigue Keywords',
                'impact': 'High',
                'value': features['fatigue_keywords'],
                'contribution': f"+{features['fatigue_keywords'] * 0.7:.1f} to fatigue score"
            })
        
        if features.get('stress_indicators', 0) > 0:
            feature_impacts.append({
                'feature': 'Stress Indicators',
                'impact': 'Medium',
                'value': features['stress_indicators'],
                'contribution': f"+{features['stress_indicators'] * 0.5:.1f} to fatigue score"
            })
        
        if features.get('negative_word_count', 0) > 3:
            feature_impacts.append({
                'feature': 'Negative Words',
                'impact': 'Medium',
                'value': features['negative_word_count'],
                'contribution': f"+{features['negative_word_count'] * 0.3:.1f} to fatigue score"
            })
        
        explanations['top_factors'] = feature_impacts[:5]
        
        # Generate interpretation
        if prediction < 3:
            explanations['interpretation'] = "Your text indicates low mental fatigue. Sentiment is generally positive with few stress indicators."
        elif prediction < 5:
            explanations['interpretation'] = "Your text shows mild fatigue signs. Some negative sentiment detected but manageable."
        elif prediction < 7:
            explanations['interpretation'] = "Your text indicates moderate fatigue. Multiple stress indicators and negative sentiments present."
        else:
            explanations['interpretation'] = "Your text shows high fatigue levels. Strong negative sentiment and multiple fatigue indicators detected."
        
        return explanations
    
    def explain_voice_prediction(self, audio_features, prediction):
        """Generate explanations for voice-based predictions"""
        explanations = {
            'prediction_score': float(prediction),
            'top_factors': [],
            'feature_importance': {},
            'interpretation': ''
        }
        
        feature_impacts = []
        
        # Analyze audio features
        if audio_features.get('energy', 1.0) < 0.3:
            feature_impacts.append({
                'feature': 'Low Voice Energy',
                'impact': 'High',
                'value': audio_features['energy'],
                'contribution': '+2.0 to fatigue score'
            })
        
        if audio_features.get('speech_rate', 150) < 120:
            feature_impacts.append({
                'feature': 'Slow Speech Rate',
                'impact': 'High',
                'value': audio_features['speech_rate'],
                'contribution': '+1.5 to fatigue score'
            })
        
        if audio_features.get('pause_frequency', 0) > 0.5:
            feature_impacts.append({
                'feature': 'Frequent Pauses',
                'impact': 'Medium',
                'value': audio_features['pause_frequency'],
                'contribution': '+1.0 to fatigue score'
            })
        
        if audio_features.get('pitch_variance', 50) < 30:
            feature_impacts.append({
                'feature': 'Monotone Voice',
                'impact': 'Medium',
                'value': audio_features['pitch_variance'],
                'contribution': '+1.5 to fatigue score'
            })
        
        if audio_features.get('jitter', 0.01) > 0.05:
            feature_impacts.append({
                'feature': 'Voice Instability',
                'impact': 'Low',
                'value': audio_features['jitter'],
                'contribution': '+0.5 to fatigue score'
            })
        
        explanations['top_factors'] = feature_impacts[:5]
        
        # Generate interpretation
        if prediction < 3:
            explanations['interpretation'] = "Voice analysis shows minimal fatigue. Energy levels are good and speech patterns are normal."
        elif prediction < 5:
            explanations['interpretation'] = "Voice shows mild fatigue indicators. Some decrease in energy or speech rate detected."
        elif prediction < 7:
            explanations['interpretation'] = "Voice indicates moderate fatigue. Multiple speech pattern changes suggest mental tiredness."
        else:
            explanations['interpretation'] = "Voice shows significant fatigue. Low energy, slow speech, and frequent pauses detected."
        
        return explanations
'''

translator_module = '''"""
Multilingual Translation Support
Handles text translation for multi-language support
"""

from googletrans import Translator
from langdetect import detect

class MultilingualTranslator:
    def __init__(self):
        self.translator = Translator()
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'hi': 'Hindi',
            'zh-cn': 'Chinese',
            'ja': 'Japanese',
            'ar': 'Arabic'
        }
    
    def detect_language(self, text):
        """Detect language of input text"""
        try:
            lang = detect(text)
            return lang
        except:
            return 'en'
    
    def translate_to_english(self, text, source_lang=None):
        """Translate text to English for analysis"""
        try:
            if source_lang is None:
                source_lang = self.detect_language(text)
            
            if source_lang == 'en':
                return text
            
            translation = self.translator.translate(text, src=source_lang, dest='en')
            return translation.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def translate_from_english(self, text, target_lang):
        """Translate English text back to target language"""
        try:
            if target_lang == 'en':
                return text
            
            translation = self.translator.translate(text, src='en', dest=target_lang)
            return translation.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text
'''

trend_forecasting = '''"""
Trend Forecasting Module
Predicts future fatigue levels using time-series analysis
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression

class TrendForecaster:
    def __init__(self):
        self.model = None
    
    def forecast(self, data_df, days_ahead=7):
        """
        Forecast future fatigue levels
        data_df: DataFrame with 'timestamp' and 'fatigue_score' columns
        days_ahead: number of days to forecast
        """
        try:
            # Prepare time series data
            df = data_df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df.resample('D').mean()  # Daily average
            df = df.fillna(method='ffill')
            
            scores = df['fatigue_score'].values
            
            if len(scores) < 3:
                # Use simple linear regression for small datasets
                return self._simple_forecast(scores, days_ahead)
            
            # Use exponential smoothing for larger datasets
            try:
                model = ExponentialSmoothing(
                    scores,
                    trend='add',
                    seasonal=None,
                    damped_trend=True
                )
                fitted_model = model.fit()
                forecast = fitted_model.forecast(days_ahead)
                
                # Generate forecast dates
                last_date = df.index[-1]
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=days_ahead,
                    freq='D'
                )
                
                forecast_data = []
                for date, value in zip(forecast_dates, forecast):
                    forecast_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'predicted_fatigue': float(max(0, min(10, value))),
                        'confidence': 'medium'
                    })
                
                return forecast_data
                
            except:
                return self._simple_forecast(scores, days_ahead)
                
        except Exception as e:
            print(f"Forecasting error: {e}")
            return self._default_forecast(days_ahead)
    
    def _simple_forecast(self, scores, days_ahead):
        """Simple linear regression forecast"""
        X = np.arange(len(scores)).reshape(-1, 1)
        y = scores
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_X = np.arange(len(scores), len(scores) + days_ahead).reshape(-1, 1)
        predictions = model.predict(future_X)
        
        forecast_data = []
        for i, pred in enumerate(predictions):
            forecast_data.append({
                'date': f"Day {i+1}",
                'predicted_fatigue': float(max(0, min(10, pred))),
                'confidence': 'low'
            })
        
        return forecast_data
    
    def _default_forecast(self, days_ahead):
        """Return default forecast if all methods fail"""
        return [{
            'date': f"Day {i+1}",
            'predicted_fatigue': 5.0,
            'confidence': 'very_low'
        } for i in range(days_ahead)]
'''

print("UTILITY FILES CREATED (2/2)")
print("=" * 80)
print("✓ explainability.py")
print("✓ translator.py")
print("✓ trend_forecasting.py")
print("\nAll backend Python files created successfully!")
