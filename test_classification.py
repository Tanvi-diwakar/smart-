from transformers import pipeline
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def classify_job_description(classifier, description: str, categories: list):
    """Classify a single job description"""
    try:
        result = classifier(
            description,
            candidate_labels=categories,
            hypothesis_template="This is a job description for a {}."
        )
        
        # Sort predictions by score
        predictions = list(zip(result['labels'], result['scores']))
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions
    except Exception as e:
        logging.error(f"Error classifying description: {str(e)}")
        return None

def main():
    print("\nJob Classification System")
    print("=" * 50)
    
    try:
        # Initialize the classifier
        print("\nInitializing BART-large-MNLI classifier...")
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # Define job categories
        categories = [
            "software engineer",
            "electrician",
            "sweeper",
            "data scientist",
            "plumber",
            "mechanic",
            "driver",
            "teacher",
            "nurse",
            "sales"
        ]
        
        print("\nAvailable job categories:")
        for i, category in enumerate(categories, 1):
            print(f"{i}. {category}")
        
        while True:
            print("\nEnter a job description (or 'quit' to exit):")
            print("Example: Looking for someone to sweep office floors")
            
            try:
                # Get input using raw_input for better compatibility
                description = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                break
            
            if description.lower() == 'quit':
                print("Goodbye!")
                break
            
            if not description:
                print("Please enter a valid job description.")
                continue
            
            print("\nClassifying job description...")
            predictions = classify_job_description(classifier, description, categories)
            
            if predictions:
                print("\nClassification Results:")
                print("-" * 30)
                for label, score in predictions:
                    print(f"{label:20s}: {score:.2f}")
                print("-" * 30)
            
            # Add small delay to prevent rate limiting
            time.sleep(0.5)
            
    except Exception as e:
        logging.error(f"Error in main program: {str(e)}")
        return

if __name__ == "__main__":
    main() 