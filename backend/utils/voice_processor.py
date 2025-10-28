voice_processor ="""
Voice Processing Utilities
Extract audio features for fatigue analysis
"""

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

class VoiceProcessor:
    def __init__(self):
        self.sample_rate = 22050
    
    def extract_audio_features(self, audio_path):
        """Extract comprehensive features from audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Pitch (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            mean_pitch = np.mean(pitch_values) if pitch_values else 0
            pitch_variance = np.var(pitch_values) if pitch_values else 0
            
            # Energy/Amplitude
            rms = librosa.feature.rms(y=y)[0]
            energy = np.mean(rms)
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # Zero crossing rate (speech rate indicator)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            speech_rate = np.mean(zcr) * sr / 2
            
            # Pauses (silence detection)
            threshold = 0.02
            is_silence = rms < threshold
            silence_segments = np.diff(is_silence.astype(int))
            pause_frequency = np.sum(silence_segments == 1) / (len(y) / sr)
            
            # Jitter and Shimmer (voice quality)
            jitter = self._calculate_jitter(y, sr)
            shimmer = self._calculate_shimmer(rms)
            
            # MFCCs (spectral features)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)[0]
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            
            features = {
                'mean_pitch': float(mean_pitch),
                'pitch_variance': float(pitch_variance),
                'energy': float(energy),
                'tempo': float(tempo),
                'speech_rate': float(speech_rate),
                'pause_frequency': float(pause_frequency),
                'jitter': float(jitter),
                'shimmer': float(shimmer),
                'mfcc_mean': float(mfcc_mean),
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff': float(spectral_rolloff),
                'duration': float(len(y) / sr)
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return self._default_features()
    
    def _calculate_jitter(self, y, sr):
        """Calculate pitch period perturbation (jitter)"""
        try:
            # Simple jitter estimation
            autocorr = librosa.autocorrelate(y)
            peaks = signal.find_peaks(autocorr)[0]
            if len(peaks) > 1:
                periods = np.diff(peaks)
                jitter = np.std(periods) / np.mean(periods) if np.mean(periods) > 0 else 0
                return jitter
            return 0
        except:
            return 0
    
    def _calculate_shimmer(self, rms):
        """Calculate amplitude perturbation (shimmer)"""
        try:
            if len(rms) > 1:
                shimmer = np.std(np.diff(rms)) / np.mean(rms) if np.mean(rms) > 0 else 0
                return shimmer
            return 0
        except:
            return 0
    
    def _default_features(self):
        """Return default features if extraction fails"""
        return {
            'mean_pitch': 150.0,
            'pitch_variance': 50.0,
            'energy': 0.5,
            'tempo': 120.0,
            'speech_rate': 150.0,
            'pause_frequency': 0.5,
            'jitter': 0.01,
            'shimmer': 0.05,
            'mfcc_mean': 0.0,
            'spectral_centroid': 2000.0,
            'spectral_rolloff': 4000.0,
            'duration': 5.0
        }