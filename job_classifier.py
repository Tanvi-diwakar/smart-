import pandas as pd
import torch
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JobClassifier:
    def __init__(self, model_path: str):
        """Initialize the job classifier with a fine-tuned model."""
        self.labels = [
            "plumber", "driver", "sweeper", "electrician", "chef", 
            "carpenter", "painter", "guard", "cleaner", "computer operator", 
            "technician", "security"
        ]
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.eval()  # Set to evaluation mode
            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict_category(self, text: str) -> Tuple[str, float]:
        """Predict job category and confidence score for given text."""
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()
            conf = torch.softmax(logits, dim=1)[0][pred].item()
            
            return self.labels[pred], round(conf, 2)
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def process_jobs(self, jobs: List[Dict]) -> pd.DataFrame:
        """Process a list of jobs and return a DataFrame with predictions."""
        try:
            # Add predictions to data
            for job in jobs:
                combined_text = f"{job['title']} {job['description']}"
                category, confidence = self.predict_category(combined_text)
                job["category"] = category
                job["confidence"] = confidence

            # Convert to DataFrame
            df = pd.DataFrame(jobs)
            
            # Perform clustering
            if len(df) >= 3:  # Only cluster if we have enough samples
                X = df[["salary", "experience"]]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                kmeans = KMeans(n_clusters=3, random_state=42)
                df["level_cluster"] = kmeans.fit_predict(X_scaled)
                
                # Map cluster to readable level
                cluster_map = {
                    0: "entry-level",
                    1: "mid-level",
                    2: "senior-level"
                }
                df["job_level"] = df["level_cluster"].map(cluster_map)
            
            return df
        except Exception as e:
            logger.error(f"Error processing jobs: {str(e)}")
            raise

def main():
    # Example usage
    model_path = "./your-finetuned-model"
    
    # Sample data
    sample_jobs = [
        {
            "title": "Urgent electrician needed",
            "description": "Fix wiring and install fuse boxes",
            "salary": 18000,
            "experience": 2
        },
        {
            "title": "Chef required for hotel",
            "description": "Prepare Indian and Chinese dishes",
            "salary": 28000,
            "experience": 5
        },
        {
            "title": "Driver needed",
            "description": "Drive company truck for delivery",
            "salary": 15000,
            "experience": 1
        }
    ]
    
    try:
        # Initialize classifier
        classifier = JobClassifier(model_path)
        
        # Process jobs
        df = classifier.process_jobs(sample_jobs)
        
        # Save to Excel
        output_path = "classified_jobs.xlsx"
        df.to_excel(output_path, index=False)
        logger.info(f"✅ Excel exported: {output_path}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 