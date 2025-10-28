voice_model = """
Voice-Based Fatigue Detection Model
Analyzes speech features to predict mental fatigue
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os

class VoiceFatigueAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load pre-trained model or create new one"""
        model_path = '../trained_models/voice_models/voice_fatigue_model.pkl'
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.is_trained = True
        else:
            # Create new model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    
    def predict_fatigue(self, audio_features):
        """
        Predict fatigue score from voice features
        audio_features: dict with pitch, energy, tempo, etc.
        returns: fatigue score (0-10)
        """
        try:
            # Extract feature values
            feature_vector = [
                audio_features.get('mean_pitch', 0),
                audio_features.get('pitch_variance', 0),
                audio_features.get('energy', 0),
                audio_features.get('tempo', 0),
                audio_features.get('speech_rate', 0),
                audio_features.get('pause_frequency', 0),
                audio_features.get('jitter', 0),
                audio_features.get('shimmer', 0),
                audio_features.get('mfcc_mean', 0),
                audio_features.get('spectral_centroid', 0)
            ]
            
            if self.is_trained:
                feature_array = np.array(feature_vector).reshape(1, -1)
                scaled_features = self.scaler.transform(feature_array)
                prediction = self.model.predict(scaled_features)[0]
            else:
                prediction = self._rule_based_prediction(audio_features)
            
            fatigue_score = max(0, min(10, prediction))
            return fatigue_score
            
        except Exception as e:
            print(f"Voice prediction error: {e}")
            return self._rule_based_prediction(audio_features)
    
    def _rule_based_prediction(self, audio_features):
        """Fallback rule-based prediction for voice"""
        base_score = 5.0
        
        # Low energy indicates fatigue
        energy = audio_features.get('energy', 0.5)
        if energy < 0.3:
            base_score += 2
        elif energy < 0.5:
            base_score += 1
        
        # Slow speech rate indicates fatigue
        speech_rate = audio_features.get('speech_rate', 150)
        if speech_rate < 120:
            base_score += 1.5
        
        # High pause frequency indicates fatigue
        pause_freq = audio_features.get('pause_frequency', 0)
        base_score += pause_freq * 0.5
        
        # Low pitch variance indicates monotone (fatigue)
        pitch_var = audio_features.get('pitch_variance', 50)
        if pitch_var < 30:
            base_score += 1.5
        
        return max(0, min(10, base_score))
    
    def train(self, X_train, y_train):
        """Train the model with new data"""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
    
    def save_model(self, path='../trained_models/voice_models/voice_fatigue_model.pkl'):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)