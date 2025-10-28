
import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Create directories if they don't exist
os.makedirs('trained_models/text_models', exist_ok=True)
os.makedirs('trained_models/voice_models', exist_ok=True)

print("="*80)
print("CREATING PRE-TRAINED MACHINE LEARNING MODELS")
print("="*80)

# ============================================================================
# 1. TEXT FATIGUE MODEL - Training and Saving
# ============================================================================

print("\n[1/4] Creating Text Fatigue Detection Model...")

# Generate synthetic training data for text analysis
# Features: sentiment_score, negative_word_count, stress_indicators, fatigue_keywords,
#           sentence_complexity, emotional_intensity, text_length, punctuation_ratio
np.random.seed(42)

# Create 1000 training samples
n_samples = 1000

# Generate features with realistic distributions
X_text_train = np.column_stack([
    np.random.uniform(-1, 1, n_samples),      # sentiment_score
    np.random.poisson(3, n_samples),          # negative_word_count
    np.random.poisson(2, n_samples),          # stress_indicators
    np.random.poisson(2, n_samples),          # fatigue_keywords
    np.random.uniform(5, 25, n_samples),      # sentence_complexity
    np.random.uniform(0, 1, n_samples),       # emotional_intensity
    np.random.uniform(50, 500, n_samples),    # text_length
    np.random.uniform(0, 0.1, n_samples)      # punctuation_ratio
])

# Generate target fatigue scores (0-10) based on features
y_text_train = (
    5.0 +  # base score
    (-2.5 * X_text_train[:, 0]) +  # negative sentiment increases fatigue
    (0.3 * X_text_train[:, 1]) +   # negative words
    (0.5 * X_text_train[:, 2]) +   # stress indicators
    (0.7 * X_text_train[:, 3]) +   # fatigue keywords
    (0.05 * X_text_train[:, 4]) +  # complexity
    (0.5 * X_text_train[:, 5]) +   # emotional intensity
    np.random.normal(0, 0.5, n_samples)  # noise
)

# Clip to valid range [0, 10]
y_text_train = np.clip(y_text_train, 0, 10)

# Train the model
text_scaler = StandardScaler()
X_text_scaled = text_scaler.fit_transform(X_text_train)

text_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
text_model.fit(X_text_scaled, y_text_train)

# Calculate training score
train_score = text_model.score(X_text_scaled, y_text_train)
print(f"   ✓ Model trained with R² score: {train_score:.4f}")

# Save the model and scaler
text_model_data = {
    'model': text_model,
    'scaler': text_scaler,
    'feature_names': [
        'sentiment_score', 'negative_word_count', 'stress_indicators',
        'fatigue_keywords', 'sentence_complexity', 'emotional_intensity',
        'text_length', 'punctuation_ratio'
    ],
    'version': '1.0.0',
    'trained_samples': n_samples
}

with open('trained_models/text_models/fatigue_model.pkl', 'wb') as f:
    pickle.dump(text_model_data, f)

print(f"   ✓ Saved to: trained_models/text_models/fatigue_model.pkl")
print(f"   ✓ Model size: {os.path.getsize('trained_models/text_models/fatigue_model.pkl')} bytes")

# ============================================================================
# 2. VOICE FATIGUE MODEL - Training and Saving
# ============================================================================

print("\n[2/4] Creating Voice Fatigue Detection Model...")

# Generate synthetic training data for voice analysis
# Features: mean_pitch, pitch_variance, energy, tempo, speech_rate,
#           pause_frequency, jitter, shimmer, mfcc_mean, spectral_centroid

# Create 1000 training samples
X_voice_train = np.column_stack([
    np.random.uniform(100, 250, n_samples),   # mean_pitch (Hz)
    np.random.uniform(20, 80, n_samples),     # pitch_variance
    np.random.uniform(0.2, 0.8, n_samples),   # energy
    np.random.uniform(80, 160, n_samples),    # tempo (BPM)
    np.random.uniform(100, 180, n_samples),   # speech_rate (words/min)
    np.random.uniform(0.1, 0.8, n_samples),   # pause_frequency
    np.random.uniform(0.005, 0.05, n_samples),# jitter
    np.random.uniform(0.02, 0.1, n_samples),  # shimmer
    np.random.uniform(-5, 5, n_samples),      # mfcc_mean
    np.random.uniform(1500, 3000, n_samples)  # spectral_centroid (Hz)
])

# Generate target fatigue scores based on voice features
y_voice_train = (
    5.0 +  # base score
    (-0.01 * X_voice_train[:, 2] * 10) +  # lower energy = more fatigue
    (-0.015 * (X_voice_train[:, 4] - 140)) +  # slower speech = more fatigue
    (2.0 * X_voice_train[:, 5]) +  # more pauses = more fatigue
    (-0.02 * X_voice_train[:, 1]) +  # less pitch variance = more fatigue
    (20 * X_voice_train[:, 6]) +  # higher jitter = more fatigue
    np.random.normal(0, 0.5, n_samples)  # noise
)

# Clip to valid range [0, 10]
y_voice_train = np.clip(y_voice_train, 0, 10)

# Train the model
voice_scaler = StandardScaler()
X_voice_scaled = voice_scaler.fit_transform(X_voice_train)

voice_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
voice_model.fit(X_voice_scaled, y_voice_train)

# Calculate training score
train_score_voice = voice_model.score(X_voice_scaled, y_voice_train)
print(f"   ✓ Model trained with R² score: {train_score_voice:.4f}")

# Save the model and scaler
voice_model_data = {
    'model': voice_model,
    'scaler': voice_scaler,
    'feature_names': [
        'mean_pitch', 'pitch_variance', 'energy', 'tempo',
        'speech_rate', 'pause_frequency', 'jitter', 'shimmer',
        'mfcc_mean', 'spectral_centroid'
    ],
    'version': '1.0.0',
    'trained_samples': n_samples
}

with open('trained_models/voice_models/voice_fatigue_model.pkl', 'wb') as f:
    pickle.dump(voice_model_data, f)

print(f"   ✓ Saved to: trained_models/voice_models/voice_fatigue_model.pkl")
print(f"   ✓ Model size: {os.path.getsize('trained_models/voice_models/voice_fatigue_model.pkl')} bytes")

print("\n" + "="*80)
print("MODELS CREATED SUCCESSFULLY!")
print("="*80)
print(f"\nText Model Performance: R² = {train_score:.4f}")
print(f"Voice Model Performance: R² = {train_score_voice:.4f}")
print("\nBoth models are now ready to use with the application!")
