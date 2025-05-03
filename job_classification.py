import re
from typing import List, Dict, Tuple
from joblib import load
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class JobClassifier:
    def __init__(self):
        """Initialize job classifier"""
        self.model = None
        self.vectorizer = None
        self.labels = [
            "plumber", "electrician", "carpenter", "painter", "welder", "mason", "mechanic",
            "driver", "delivery boy", "loader",
            "sweeper", "housekeeping", "garbage collector", "pest control",
            "security guard", "watchman", "parking attendant",
            "tailor", "embroidery worker", "ironing staff",
            "chef", "kitchen helper", "dishwasher",
            "salesman", "store helper", "cashier",
            "barber", "ac technician", "gardener", "lift operator",
            "warehouse worker", "office boy", "computer operator"
        ]
        self._load_model()
    
    def _load_model(self):
        """Load joblib model and vectorizer"""
        try:
            self.model = load("job_classifier.pkl")
            self.vectorizer = load("vectorizer.pkl")
            logging.info("Model and vectorizer loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        return re.sub(r'\s+', ' ', text).lower().strip()
    
    def predict(self, text: str) -> Dict[str, any]:
        """Predict job category with confidence score"""
        try:
            clean_text = self.clean_text(text)
            X = self.vectorizer.transform([clean_text])
            prediction = self.model.predict(X)[0]
            confidence = max(self.model.predict_proba(X)[0])
            
            return {
                "text": text,
                "category": prediction,
                "confidence": confidence
            }
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            return {
                "text": text,
                "category": "unknown",
                "confidence": 0.0
            }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """Predict categories for multiple job descriptions"""
        return [self.predict(text) for text in texts]

def main():
    # Test job descriptions
    test_jobs = [
        "Required experienced driver for office cab service.",
        "Hiring a painter for house painting work.",
        "Need a welder to fix metal gate in workshop.",
        "Looking for a tailor to stitch school uniforms.",
        "Opening for chef in a restaurant near station.",
        "We need an AC technician for installation and service.",
        "Warehouse helper required to load/unload goods.",
        "Hiring a barber for men's salon in main market.",
        "Watchman needed for apartment night shift.",
        "Dishwasher required in hotel kitchen."
    ]
    
    try:
        print("\nTesting Job Classifier")
        print("=" * 50)
        
        classifier = JobClassifier()
        predictions = classifier.predict_batch(test_jobs)
        
        for pred in predictions:
            print(f"\n📝 Job: {pred['text']}")
            print(f"🔍 Predicted Category: {pred['category']}")
            print(f"📊 Confidence: {pred['confidence']:.2f}")
        
    except Exception as e:
        print(f"Error testing classifier: {str(e)}")

if __name__ == "__main__":
    main() 