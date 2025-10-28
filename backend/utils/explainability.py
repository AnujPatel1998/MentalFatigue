xai_module = """
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