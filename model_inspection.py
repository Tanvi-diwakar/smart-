import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def inspect_model():
    """Inspect the model and its capabilities"""
    try:
        # Load the model and vectorizer
        model = joblib.load('job_classifier.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        
        print("\nModel Inspection Results")
        print("=" * 50)
        
        # 1. Model Information
        print("\n1. Model Information:")
        print(f"Model Type: {type(model).__name__}")
        print(f"Model Parameters: {model.get_params()}")
        
        # 2. Available Methods
        print("\n2. Available Methods:")
        available_methods = [method for method in dir(model) if not method.startswith('_')]
        print("Methods:", available_methods)
        
        # 3. Feature Information
        print("\n3. Feature Information:")
        print(f"Number of Features: {len(vectorizer.get_feature_names_out())}")
        print("Sample Features:", vectorizer.get_feature_names_out()[:10])
        
        # 4. Class Information
        print("\n4. Class Information:")
        if hasattr(model, 'classes_'):
            print(f"Number of Classes: {len(model.classes_)}")
            print("Classes:", model.classes_)
        
        # 5. Model Capabilities
        print("\n5. Model Capabilities:")
        capabilities = {
            'predict': hasattr(model, 'predict'),
            'predict_proba': hasattr(model, 'predict_proba'),
            'score': hasattr(model, 'score'),
            'feature_importances': hasattr(model, 'feature_importances_')
        }
        for capability, available in capabilities.items():
            print(f"{capability}: {'Available' if available else 'Not Available'}")
        
        # 6. Model Performance Metrics
        print("\n6. Model Performance Metrics:")
        if hasattr(model, 'oob_score_'):
            print(f"Out-of-bag Score: {model.oob_score_:.4f}")
        if hasattr(model, 'feature_importances_'):
            print("Feature Importance Available")
        
    except Exception as e:
        logging.error(f"Error inspecting model: {str(e)}")

if __name__ == "__main__":
    inspect_model() 