import logging
import time
from typing import List, Dict, Optional, Union
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from bs4 import BeautifulSoup
import requests
import re
from datetime import datetime
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import load, dump
import numpy as np
import pymongo
import psycopg2
from psycopg2.extras import Json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    try:
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Convert to lowercase and strip
        text = text.lower().strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Remove extra spaces around punctuation
        text = re.sub(r'\s+([.,!?-])\s+', r'\1 ', text)
        
        return text.strip()
    except Exception as e:
        logging.error(f"Error cleaning text: {str(e)}")
        return text

class MLJobClassifier:
    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        self.model = None
        self.vectorizer = None
        self.categories = [
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
        
        if model_path and vectorizer_path:
            self.load_model(model_path, vectorizer_path)
        else:
            self.initialize_model()
    
    def initialize_model(self):
        """Initialize the ML model and vectorizer"""
        try:
            logging.info("Initializing ML classifier...")
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.model = LogisticRegression(
                max_iter=1000,
                multi_class='multinomial',
                solver='lbfgs'
            )
            logging.info("ML classifier initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing ML classifier: {str(e)}")
            raise
    
    def load_model(self, model_path: str, vectorizer_path: str):
        """Load pre-trained model and vectorizer"""
        try:
            logging.info(f"Loading model from {model_path}")
            logging.info(f"Loading vectorizer from {vectorizer_path}")
            
            self.model = load(model_path)
            self.vectorizer = load(vectorizer_path)
            
            logging.info("Model and vectorizer loaded successfully")
            return True
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return False
    
    def save_model(self, model_path: str, vectorizer_path: str):
        """Save the trained model and vectorizer"""
        try:
            logging.info(f"Saving model to {model_path}")
            logging.info(f"Saving vectorizer to {vectorizer_path}")
            
            dump(self.model, model_path)
            dump(self.vectorizer, vectorizer_path)
            
            logging.info("Model and vectorizer saved successfully")
            return True
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            return False
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess a single text input"""
        try:
            return text.lower().strip()
        except Exception as e:
            logging.error(f"Error preprocessing text: {str(e)}")
            return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of text inputs"""
        try:
            return [self.preprocess_text(text) for text in texts]
        except Exception as e:
            logging.error(f"Error preprocessing batch: {str(e)}")
            return texts
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict categories for a batch of job descriptions"""
        try:
            if not self.model or not self.vectorizer:
                raise ValueError("Model or vectorizer not initialized")
            
            # Preprocess texts
            cleaned_texts = self.preprocess_batch(texts)
            
            # Vectorize texts
            X = self.vectorizer.transform(cleaned_texts)
            
            # Get predictions and probabilities
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            results = []
            for text, pred, probs in zip(texts, predictions, probabilities):
                # Get confidence scores for all classes
                scores = list(zip(self.model.classes_, probs))
                scores.sort(key=lambda x: x[1], reverse=True)
                
                results.append({
                    'description': text,
                    'category': pred,
                    'confidence': float(probs.max()),
                    'metadata': {
                        'all_predictions': {label: float(score) for label, score in scores},
                        'timestamp': datetime.now().isoformat()
                    }
                })
            
            return results
        except Exception as e:
            logging.error(f"Error predicting batch: {str(e)}")
            return []
    
    def predict(self, text: str) -> Dict:
        """Predict category for a single job description"""
        try:
            results = self.predict_batch([text])
            return results[0] if results else None
        except Exception as e:
            logging.error(f"Error predicting: {str(e)}")
            return None
    
    def categorize_job(self, text: str) -> str:
        """Simple categorization of a single job description"""
        try:
            result = self.predict(text)
            return result['category'] if result else "unknown"
        except Exception as e:
            logging.error(f"Error categorizing job: {str(e)}")
            return "unknown"
    
    def categorize_jobs(self, jobs: List[str]) -> List[Dict[str, str]]:
        """Categorize multiple job descriptions"""
        try:
            results = self.predict_batch(jobs)
            return [{
                'text': result['description'],
                'category': result['category']
            } for result in results]
        except Exception as e:
            logging.error(f"Error categorizing jobs: {str(e)}")
            return []

class BERTJobClassifier:
    def __init__(self, model_path: str = None, labels: List[str] = None):
        self.model = None
        self.tokenizer = None
        self.labels = labels or [
            "plumber",
            "electrician",
            "driver",
            "sweeper",
            "software engineer",
            "data scientist",
            "teacher",
            "nurse",
            "sales"
        ]
        
        if model_path:
            self.load_model(model_path)
        else:
            self.initialize_model()
    
    def initialize_model(self):
        """Initialize the BERT model and tokenizer"""
        try:
            logging.info("Initializing BERT classifier...")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased",
                num_labels=len(self.labels)
            )
            logging.info("BERT classifier initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing BERT classifier: {str(e)}")
            raise
    
    def load_model(self, model_path: str):
        """Load a pre-trained BERT model"""
        try:
            logging.info(f"Loading BERT model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            logging.info("BERT model loaded successfully")
            return True
        except Exception as e:
            logging.error(f"Error loading BERT model: {str(e)}")
            return False
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict categories for a batch of job descriptions"""
        try:
            if not self.model or not self.tokenizer:
                raise ValueError("Model or tokenizer not initialized")
            
            # Tokenize and prepare inputs
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
            
            results = []
            for text, pred_class, probs in zip(texts, predicted_classes, probabilities):
                # Get confidence scores for all classes
                scores = probs.tolist()
                predictions = list(zip(self.labels, scores))
                predictions.sort(key=lambda x: x[1], reverse=True)
                
                results.append({
                    'description': text,
                    'category': self.labels[pred_class],
                    'confidence': float(scores[pred_class]),
                    'metadata': {
                        'all_predictions': {label: float(score) for label, score in predictions},
                        'timestamp': datetime.now().isoformat()
                    }
                })
            
            return results
        except Exception as e:
            logging.error(f"Error predicting batch: {str(e)}")
            return []
    
    def predict(self, text: str) -> Dict:
        """Predict category for a single job description"""
        try:
            results = self.predict_batch([text])
            return results[0] if results else None
        except Exception as e:
            logging.error(f"Error predicting: {str(e)}")
            return None
    
    def categorize_job(self, text: str) -> str:
        """Simple categorization of a single job description"""
        try:
            result = self.predict(text)
            return result['category'] if result else "unknown"
        except Exception as e:
            logging.error(f"Error categorizing job: {str(e)}")
            return "unknown"
    
    def categorize_jobs(self, jobs: List[str]) -> List[Dict[str, str]]:
        """Categorize multiple job descriptions"""
        try:
            results = self.predict_batch(jobs)
            return [{
                'text': result['description'],
                'category': result['category']
            } for result in results]
        except Exception as e:
            logging.error(f"Error categorizing jobs: {str(e)}")
            return []

class JobPipeline:
    def __init__(self, classifier_type: str = "ml", storage_type: str = "json"):
        self.classifier = None
        self.ml_classifier = None
        self.bert_classifier = None
        self.storage_type = storage_type
        self.storage_config = {}
        self.categories = [
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
        
        if classifier_type == "ml":
            self.ml_classifier = MLJobClassifier()
        elif classifier_type == "bert":
            self.bert_classifier = BERTJobClassifier()
        else:
            self.initialize_classifier()
    
    def initialize_classifier(self):
        """Initialize the BERT-based classifier"""
        try:
            logging.info("Initializing BART-large-MNLI classifier...")
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            logging.info("Classifier initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing classifier: {str(e)}")
            raise

    def scrape_job_description(self, url: str) -> Optional[str]:
        """Scrape job description from a given URL"""
        try:
            logging.info(f"Scraping job description from {url}")
            
            # Add headers to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            logging.info("Job description scraped successfully")
            return text
            
        except Exception as e:
            logging.error(f"Error scraping job description: {str(e)}")
            return None

    def preprocess_text(self, text: str) -> str:
        """Preprocess and normalize the text"""
        try:
            # Clean the text using the clean_text function
            text = clean_text(text)
            
            # Remove common stop words (basic implementation)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = text.split()
            filtered_words = [word for word in words if word not in stop_words]
            
            return ' '.join(filtered_words)
            
        except Exception as e:
            logging.error(f"Error preprocessing text: {str(e)}")
            return text

    def classify_job(self, description: str) -> Dict:
        """Classify the job description using the selected classifier"""
        try:
            if self.ml_classifier:
                return self.ml_classifier.predict(description)
            elif self.bert_classifier:
                return self.bert_classifier.predict(description)
            elif self.classifier:
                # Preprocess the description
                processed_description = self.preprocess_text(description)
                
                # Get classification results
                result = self.classifier(
                    processed_description,
                    candidate_labels=self.categories,
                    hypothesis_template="This is a job description for a {}."
                )
                
                # Sort predictions by score
                predictions = list(zip(result['labels'], result['scores']))
                predictions.sort(key=lambda x: x[1], reverse=True)
                
                return {
                    'top_category': predictions[0][0],
                    'confidence': float(predictions[0][1]),
                    'all_predictions': {label: float(score) for label, score in predictions}
                }
            else:
                raise ValueError("No classifier initialized")
            
        except Exception as e:
            logging.error(f"Error classifying job: {str(e)}")
            return None

    def configure_storage(self, **kwargs):
        """Configure storage settings"""
        self.storage_config = kwargs
    
    def save_to_json(self, jobs: List[Dict]) -> bool:
        """Save classified jobs to JSON file"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/jobs_{timestamp}.json"
            
            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(jobs, f, indent=2)
            
            logging.info(f"Jobs saved to {filename}")
            return True
        except Exception as e:
            logging.error(f"Error saving to JSON: {str(e)}")
            return False
    
    def save_to_mongodb(self, jobs: List[Dict]) -> bool:
        """Save classified jobs to MongoDB"""
        try:
            if not self.storage_config.get('mongodb_uri'):
                raise ValueError("MongoDB URI not configured")
            
            client = pymongo.MongoClient(self.storage_config['mongodb_uri'])
            db = client[self.storage_config.get('db_name', 'job_classifier')]
            collection = db[self.storage_config.get('collection_name', 'classified_jobs')]
            
            # Add timestamp to each job
            timestamp = datetime.now()
            for job in jobs:
                job['timestamp'] = timestamp
            
            # Insert jobs
            result = collection.insert_many(jobs)
            
            logging.info(f"Jobs saved to MongoDB: {len(result.inserted_ids)} records")
            return True
        except Exception as e:
            logging.error(f"Error saving to MongoDB: {str(e)}")
            return False
    
    def save_to_postgres(self, jobs: List[Dict]) -> bool:
        """Save classified jobs to PostgreSQL"""
        try:
            if not self.storage_config.get('postgres_uri'):
                raise ValueError("PostgreSQL URI not configured")
            
            conn = psycopg2.connect(self.storage_config['postgres_uri'])
            cur = conn.cursor()
            
            # Create table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS classified_jobs (
                    id SERIAL PRIMARY KEY,
                    description TEXT,
                    category TEXT,
                    confidence FLOAT,
                    timestamp TIMESTAMP,
                    metadata JSONB
                )
            """)
            
            # Insert jobs
            timestamp = datetime.now()
            for job in jobs:
                cur.execute("""
                    INSERT INTO classified_jobs 
                    (description, category, confidence, timestamp, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    job.get('description', ''),
                    job.get('category', ''),
                    job.get('confidence', 0.0),
                    timestamp,
                    Json(job.get('metadata', {}))
                ))
            
            conn.commit()
            logging.info(f"Jobs saved to PostgreSQL: {len(jobs)} records")
            return True
        except Exception as e:
            logging.error(f"Error saving to PostgreSQL: {str(e)}")
            return False
    
    def save_jobs(self, jobs: List[Dict]) -> bool:
        """Save classified jobs using the configured storage type"""
        try:
            if self.storage_type == "json":
                return self.save_to_json(jobs)
            elif self.storage_type == "mongodb":
                return self.save_to_mongodb(jobs)
            elif self.storage_type == "postgres":
                return self.save_to_postgres(jobs)
            else:
                raise ValueError(f"Unsupported storage type: {self.storage_type}")
        except Exception as e:
            logging.error(f"Error saving jobs: {str(e)}")
            return False
    
    def process_jobs(self, jobs: List[str]) -> List[Dict]:
        """Process and classify multiple job descriptions"""
        try:
            classified_jobs = []
            
            for job_text in jobs:
                # Classify the job
                result = self.classify_job(job_text)
                
                if result:
                    classified_jobs.append({
                        "description": job_text,
                        "category": result['top_category'],
                        "confidence": result['confidence'],
                        "metadata": {
                            "all_predictions": result['all_predictions'],
                            "timestamp": datetime.now().isoformat()
                        }
                    })
            
            # Save the classified jobs
            if classified_jobs:
                self.save_jobs(classified_jobs)
            
            return classified_jobs
        except Exception as e:
            logging.error(f"Error processing jobs: {str(e)}")
            return []

def main():
    # Example usage
    pipeline = JobPipeline()
    
    # Example job URLs (replace with actual job posting URLs)
    job_urls = [
        "https://example.com/job1",
        "https://example.com/job2"
    ]
    
    for url in job_urls:
        print(f"\nProcessing job from: {url}")
        result = pipeline.process_job(url)
        
        if result:
            print("\nClassification Results:")
            print("-" * 50)
            print(f"Top Category: {result['classification']['top_category']}")
            print(f"Confidence: {result['classification']['confidence']:.2f}")
            print("\nAll Predictions:")
            for category, score in result['classification']['all_predictions'].items():
                print(f"{category:20s}: {score:.2f}")
            print("-" * 50)
        else:
            print("Failed to process job")

if __name__ == "__main__":
    main() 