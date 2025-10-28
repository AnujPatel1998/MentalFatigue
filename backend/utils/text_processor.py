text_processor = """
Text Processing Utilities
Extract features from text for fatigue analysis
"""

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

class TextProcessor:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.fatigue_keywords = [
            'tired', 'exhausted', 'fatigue', 'weary', 'drained',
            'burnout', 'stressed', 'overwhelmed', 'anxious', 'worry',
            'cant focus', 'distracted', 'sleepy', 'weak', 'depressed'
        ]
        self.stress_indicators = [
            'pressure', 'stress', 'anxiety', 'nervous', 'tense',
            'worried', 'afraid', 'scared', 'panic', 'frustrat'
        ]
    
    def extract_features(self, text):
        """Extract comprehensive features from text"""
        text_lower = text.lower()
        
        # Sentiment analysis
        vader_scores = self.vader.polarity_scores(text)
        blob = TextBlob(text)
        
        # Tokenize
        words = word_tokenize(text_lower)
        sentences = sent_tokenize(text)
        
        # Count keywords
        fatigue_count = sum(1 for kw in self.fatigue_keywords if kw in text_lower)
        stress_count = sum(1 for kw in self.stress_indicators if kw in text_lower)
        
        # Negative word count
        negative_words = [word for word in words if self.vader.polarity_scores(word)['compound'] < -0.5]
        
        # Complexity metrics
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Punctuation analysis
        exclamation_count = text.count('!')
        question_count = text.count('?')
        punctuation_ratio = (exclamation_count + question_count) / len(text) if len(text) > 0 else 0
        
        features = {
            'sentiment_score': vader_scores['compound'],
            'positive_score': vader_scores['pos'],
            'negative_score': vader_scores['neg'],
            'neutral_score': vader_scores['neu'],
            'subjectivity': blob.sentiment.polarity,
            'fatigue_keywords': fatigue_count,
            'stress_indicators': stress_count,
            'negative_word_count': len(negative_words),
            'text_length': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': avg_word_length,
            'sentence_complexity': avg_sentence_length,
            'punctuation_ratio': punctuation_ratio,
            'emotional_intensity': abs(vader_scores['compound'])
        }
        
        return features
    
    def get_sentiment(self, text):
        """Get simple sentiment classification"""
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            return {'label': 'Positive', 'score': compound}
        elif compound <= -0.05:
            return {'label': 'Negative', 'score': compound}
        else:
            return {'label': 'Neutral', 'score': compound}

import numpy as np