"""
Voice-Based Fatigue Detection Model
Analyzes speech features to predict mental fatigue
UPDATED: Now loads pre-trained model from file
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
        self.feature_names = []
        self.load_or_create_model()

    def load_or_create_model(self):
        """Load pre-trained model or create new one"""
        model_path = '../trained_models/voice_models/voice_fatigue_model.pkl'

        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']
                self.is_trained = True

                print(f"✓ Loaded pre-trained voice model (v{model_data['version']})")
                print(f"  Trained on {model_data['trained_samples']} samples")

            except Exception as e:
                print(f"⚠ Error loading model: {e}")
                print("  Creating new model...")
                self._create_default_model()
        else:
            print("⚠ No pre-trained model found at:", model_path)
            print("  Creating new model with default parameters...")
            self._create_default_model()

    def _create_default_model(self):
        """Create default model if pre-trained one not available"""
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.feature_names = [
            'mean_pitch', 'pitch_variance', 'energy', 'tempo',
            'speech_rate', 'pause_frequency', 'jitter', 'shimmer',
            'mfcc_mean', 'spectral_centroid'
        ]
        self.is_trained = False

    def predict_fatigue(self, audio_features):
        """
        Predict fatigue score from voice features
        audio_features: dict with pitch, energy, tempo, etc.
        returns: fatigue score (0-10)
        """
        try:
            # Extract feature values in correct order
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
                # Use trained model
                feature_array = np.array(feature_vector).reshape(1, -1)
                scaled_features = self.scaler.transform(feature_array)
                prediction = self.model.predict(scaled_features)[0]
                print(f"  → Using ML model prediction: {prediction:.2f}")
            else:
                # Use rule-based prediction if model not trained
                prediction = self._rule_based_prediction(audio_features)
                print(f"  → Using rule-based prediction: {prediction:.2f}")

            fatigue_score = max(0, min(10, prediction))
            return fatigue_score

        except Exception as e:
            print(f"⚠ Voice prediction error: {e}")
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
        print("✓ Model trained successfully")

    def save_model(self, path='../trained_models/voice_models/voice_fatigue_model.pkl'):
        """Save trained model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'version': '1.0.0',
            'trained_samples': 'custom'
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✓ Model saved to: {path}")
