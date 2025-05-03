import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- API Key Security ---
API_KEY = "your-secret-api-key"  # Change this to a strong secret

def verify_api_key(request: Request):
    key = request.headers.get("x-api-key")
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

# --- FastAPI App ---
app = FastAPI()

class JobItem(BaseModel):
    title: str
    description: str
    salary: float
    experience: float

class JobRequest(BaseModel):
    jobs: List[JobItem]

@app.post("/classify_jobs", dependencies=[Depends(verify_api_key)])
def classify_jobs(request: JobRequest):
    # Dummy prediction logic for demonstration
    labels = ["plumber", "driver", "sweeper", "electrician", "chef", "carpenter", "painter", "guard", "cleaner", "computer operator", "technician", "security"]
    data = []
    for i, job in enumerate(request.jobs):
        data.append({
            "title": job.title,
            "description": job.description,
            "salary": job.salary,
            "experience": job.experience,
            "category": labels[i % len(labels)],
            "confidence": 0.9
        })
    df = pd.DataFrame(data)
    if len(df) >= 3:
        X = df[["salary", "experience"]]
        kmeans = KMeans(n_clusters=3, random_state=42)
        df["level_cluster"] = kmeans.fit_predict(X)
        cluster_map = {0: "entry-level", 1: "mid-level", 2: "senior-level"}
        df["job_level"] = df["level_cluster"].map(cluster_map)
    else:
        df["job_level"] = "unknown"
    # Save to CSV and Excel
    df.to_csv("classified_jobs.csv", index=False)
    df.to_excel("classified_jobs.xlsx", index=False)
    return df.to_dict(orient="records")

@app.get("/download_csv", dependencies=[Depends(verify_api_key)])
def download_csv():
    file_path = "classified_jobs.csv"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="CSV file not found")
    return FileResponse(file_path, media_type='text/csv', filename="classified_jobs.csv")

@app.get("/download_excel", dependencies=[Depends(verify_api_key)])
def download_excel():
    file_path = "classified_jobs.xlsx"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Excel file not found")
    return FileResponse(file_path, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename="classified_jobs.xlsx")

def test_bert_classifier():
    # Define labels
    labels = ["electrician", "plumber", "sweeper", "driver"]
    
    # Initialize model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(labels)
        )
        print("\nModel and tokenizer initialized successfully")
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return
    
    # Test cases
    test_jobs = [
        {
            "description": "Expert needed to handle wiring and circuit installation.",
            "expected": "electrician"
        },
        {
            "description": "Need someone to drive a 4-wheeler for daily office commute.",
            "expected": "driver"
        },
        {
            "description": "Looking for someone to clean the office floors and restrooms.",
            "expected": "sweeper"
        },
        {
            "description": "Required: an expert to fix broken water pipes in residential apartments.",
            "expected": "plumber"
        }
    ]
    
    print("\nTesting Job Classification")
    print("=" * 50)
    
    # Test predictions
    for job in test_jobs:
        try:
            # Tokenize input
            inputs = tokenizer(
                job["description"],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
            
            # Get confidence scores
            scores = probabilities.tolist()
            predictions = list(zip(labels, scores))
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nJob: {job['description']}")
            print(f"Expected: {job['expected']}")
            print(f"Predicted: {labels[predicted_class]}")
            print(f"Confidence: {scores[predicted_class]:.2f}")
            print("All predictions:")
            for label, score in predictions:
                print(f"  {label}: {score:.2f}")
            
        except Exception as e:
            print(f"Error processing job: {str(e)}")

if __name__ == "__main__":
    test_bert_classifier() 