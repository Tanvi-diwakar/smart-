import joblib
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_optimized_model():
    """Test the optimized model with new, unseen examples"""
    try:
        # Load optimized model and vectorizer
        model = joblib.load('optimized_job_classifier.pkl')
        vectorizer = joblib.load('optimized_vectorizer.pkl')
        
        # New test cases (different from training data)
        test_cases = [
            "Need someone to fix electrical wiring in my new apartment, must have experience with residential work",
            "Looking for a skilled painter who can do both interior and exterior painting, experience with modern techniques required",
            "Wanted: Professional driver for school bus route, must have valid license and clean driving record",
            "Required: Experienced AC technician for regular maintenance of split and window ACs in office building",
            "Seeking computer operator with knowledge of MS Office and data entry, typing speed 40wpm required",
            "Need a security guard for night shift at commercial complex, ex-servicemen preferred",
            "Looking for chef specializing in Chinese cuisine for new restaurant",
            "Required: Carpenter for custom furniture and cabinet work, must have 2+ years experience",
            "Need professional cleaner for daily office cleaning and maintenance",
            "Hiring experienced plumber for new construction project, knowledge of modern plumbing systems required"
        ]
        
        print("\nTesting Optimized Model with New Examples")
        print("=" * 50)
        
        for job_desc in test_cases:
            # Transform text
            vec_text = vectorizer.transform([job_desc])
            
            # Get prediction and probabilities
            prediction = model.predict(vec_text)[0]
            probabilities = model.predict_proba(vec_text)[0]
            
            # Get confidence score for the prediction
            confidence = np.max(probabilities)
            
            # Get top 3 predictions with probabilities
            class_probs = list(zip(model.classes_, probabilities))
            top_3 = sorted(class_probs, key=lambda x: x[1], reverse=True)[:3]
            
            print(f"\nJob Description: {job_desc}")
            print(f"Predicted Category: {prediction}")
            print(f"Confidence: {confidence:.2%}")
            print("Top 3 Predictions:")
            for category, prob in top_3:
                print(f"  - {category}: {prob:.2%}")
            
    except Exception as e:
        logging.error(f"Error testing model: {str(e)}")

if __name__ == "__main__":
    test_optimized_model() 