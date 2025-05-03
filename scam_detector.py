import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import logging

logging.basicConfig(level=logging.INFO)

class ScamDetector:
    def __init__(self):
        """Initialize scam detector with RandomForest classifier"""
        self.model = None
        self.vectorizer = None
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model and vectorizer"""
        try:
            self.model = joblib.load("scam_detector.pkl")
            self.vectorizer = joblib.load("scam_vectorizer.pkl")
            logging.info("Scam detection model loaded successfully")
        except FileNotFoundError:
            logging.warning("No pre-trained model found. Initializing new model.")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.vectorizer = TfidfVectorizer(max_features=1000)
    
    def extract_features(self, text: str) -> dict:
        """Extract features from job description"""
        features = {}
        
        # Text-based features
        text = text.lower()
        features['length'] = len(text)
        features['has_email'] = bool(re.search(r'[\w\.-]+@[\w\.-]+', text))
        features['has_phone'] = bool(re.search(r'\b\d{10}\b', text))
        features['has_url'] = bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        
        # Suspicious keywords
        suspicious_words = [
            'urgent', 'immediate', 'quick money', 'work from home',
            'no experience needed', 'earn money fast', 'get rich quick',
            'investment', 'bitcoin', 'crypto', 'lottery', 'inheritance',
            'foreign prince', 'bank transfer', 'wire transfer'
        ]
        features['suspicious_word_count'] = sum(1 for word in suspicious_words if word in text)
        
        # Contact information completeness
        features['has_company_name'] = bool(re.search(r'(company|corporation|inc\.|ltd\.|llc)', text))
        features['has_location'] = bool(re.search(r'(address|location|city|state|zip)', text))
        
        return features
    
    def predict(self, text: str) -> dict:
        """Predict if a job posting is likely a scam"""
        try:
            # Extract features
            features = self.extract_features(text)
            
            # Convert features to vector
            feature_vector = self.vectorizer.transform([text])
            
            # Get prediction and probability
            is_scam = bool(self.model.predict(feature_vector)[0])
            scam_probability = max(self.model.predict_proba(feature_vector)[0])
            
            return {
                "is_scam": is_scam,
                "scam_probability": scam_probability,
                "features": features
            }
        except Exception as e:
            logging.error(f"Error in scam prediction: {str(e)}")
            return {
                "is_scam": False,
                "scam_probability": 0.0,
                "features": {}
            }
    
    def predict_batch(self, texts: list) -> list:
        """Predict scam probability for multiple job descriptions"""
        return [self.predict(text) for text in texts] 