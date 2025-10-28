text_model = """
Text-Based Fatigue Detection Model
Uses NLP and sentiment analysis to predict mental fatigue from text
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os

class TextFatigueAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load pre-trained model or create new one"""
        model_path = '../trained_models/text_models/fatigue_model.pkl'
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.is_trained = True
        else:
            # Create new model with default parameters
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
    
    def predict_fatigue(self, features):
        """
        Predict fatigue score from text features
        features: dict with sentiment, complexity, emotion scores
        returns: fatigue score (0-10)
        """
        try:
            # Extract feature values
            feature_vector = [
                features.get('sentiment_score', 0),
                features.get('negative_word_count', 0),
                features.get('stress_indicators', 0),
                features.get('fatigue_keywords', 0),
                features.get('sentence_complexity', 0),
                features.get('emotional_intensity', 0),
                features.get('text_length', 0),
                features.get('punctuation_ratio', 0)
            ]
            
            if self.is_trained:
                # Use trained model
                feature_array = np.array(feature_vector).reshape(1, -1)
                scaled_features = self.scaler.transform(feature_array)
                prediction = self.model.predict(scaled_features)[0]
            else:
                # Use rule-based prediction if model not trained
                prediction = self._rule_based_prediction(features)
            
            # Ensure score is in range [0, 10]
            fatigue_score = max(0, min(10, prediction))
            
            return fatigue_score
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._rule_based_prediction(features)
    
    def _rule_based_prediction(self, features):
        """Fallback rule-based prediction"""
        base_score = 5.0
        
        # Adjust based on sentiment
        sentiment = features.get('sentiment_score', 0)
        if sentiment < -0.5:
            base_score += 3
        elif sentiment < -0.2:
            base_score += 1.5
        elif sentiment > 0.5:
            base_score -= 2
        
        # Adjust based on stress indicators
        stress = features.get('stress_indicators', 0)
        base_score += stress * 0.5
        
        # Adjust based on fatigue keywords
        fatigue_kw = features.get('fatigue_keywords', 0)
        base_score += fatigue_kw * 0.7
        
        # Adjust based on negative words
        neg_words = features.get('negative_word_count', 0)
        base_score += neg_words * 0.3
        
        return max(0, min(10, base_score))
    
    def train(self, X_train, y_train):
        """Train the model with new data"""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
    
    def save_model(self, path='../trained_models/text_models/fatigue_model.pkl'):
        """Save trained model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

# Save to CSV for download
import pandas as pd

files_data = {
    'filename': ['text_fatigue_model.py', 'voice_fatigue_model.py'],
    'location': ['backend/models/', 'backend/models/'],
    'lines': [len(text_model.split('\n')), len(voice_model.split('\n'))],
    'description': ['Text-based fatigue detection using NLP', 'Voice-based fatigue detection from audio']
}

df = pd.DataFrame(files_data)
df.to_csv('model_files_summary.csv', index=False)
print("\n✓ Model files summary saved to CSV")
