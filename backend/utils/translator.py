translator_module = """
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