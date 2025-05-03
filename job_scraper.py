import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Tuple
import time
from transformers import pipeline, LayoutLMTokenizer, LayoutLMForTokenClassification
import pdfkit
import os
from datetime import datetime
import pytesseract
from pdf2image import convert_from_path
import tempfile
import torch
from PIL import Image
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urljoin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd
import random
from job_classifier import JobClassifier
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLJobClassifier:
    def __init__(self, model_path: str, vectorizer_path: str):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.classifier = LogisticRegression()
        self.load_model(model_path, vectorizer_path)

    def train(self, texts: List[str], labels: List[str]):
        """
        Train the ML classifier with labeled data
        
        Args:
            texts (List[str]): List of job descriptions
            labels (List[str]): List of corresponding job categories
        """
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict the job category for a given job description
        
        Args:
            text (str): Job description text
            
        Returns:
            Tuple[str, float]: Predicted job category and confidence score
        """
        X = self.vectorizer.transform([text])
        label = self.classifier.predict(X)[0]
        confidence = self.classifier.predict_proba(X).max()
        return label, confidence

    def save_model(self, classifier_path: str, vectorizer_path: str):
        """
        Save the trained model and vectorizer to disk
        
        Args:
            classifier_path (str): Path to save the classifier
            vectorizer_path (str): Path to save the vectorizer
        """
        joblib.dump(self.classifier, classifier_path)
        joblib.dump(self.vectorizer, vectorizer_path)

    def load_model(self, classifier_path: str, vectorizer_path: str):
        """
        Load a pre-trained model and vectorizer from disk
        
        Args:
            classifier_path (str): Path to the classifier
            vectorizer_path (str): Path to the vectorizer
        """
        self.classifier = joblib.load(classifier_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def categorize_job(self, text: str) -> str:
        """
        Simplified method for quick categorization
        
        Args:
            text (str): Job description text
            
        Returns:
            str: Predicted job category
        """
        try:
            # Use the classifier to predict the category
            result = self.predict(text)
            return result[0]
        except Exception as e:
            logging.error(f"Error categorizing job: {str(e)}")
            return "unknown"

    def categorize_jobs(self, jobs: List[str]) -> List[Dict[str, str]]:
        """
        Processes multiple job descriptions
        
        Args:
            jobs (List[str]): List of job descriptions
            
        Returns:
            List[Dict[str, str]]: List of dictionaries with text and category
        """
        try:
            results = []
            for job in jobs:
                result = {
                    'text': job,
                    'category': self.categorize_job(job)
                }
                results.append(result)
            return results
        except Exception as e:
            logging.error(f"Error categorizing jobs: {str(e)}")
            return []

    def predict_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Predict job categories for a batch of job descriptions
        
        Args:
            texts (List[str]): List of job descriptions
            
        Returns:
            List[Dict[str, any]]: List of dictionaries with job description and predicted category
        """
        try:
            results = []
            for text in texts:
                result = self.predict(text)
                results.append({
                    'description': text,
                    'category': result[0],
                    'confidence': result[1]
                })
            return results
        except Exception as e:
            logging.error(f"Error predicting batch: {str(e)}")
            return []

class BERTJobClassifier:
    def __init__(self, model_path: str, labels: List[str]):
        # Initialize BERT model and tokenizer
        self.model = LayoutLMForTokenClassification.from_pretrained(model_path)
        self.tokenizer = LayoutLMTokenizer.from_pretrained(model_path)
        self.labels = labels

    def load_model(self, model_path: str):
        """
        Load a pre-trained BERT model from disk
        
        Args:
            model_path (str): Path to the pre-trained BERT model
        """
        self.model = LayoutLMForTokenClassification.from_pretrained(model_path)
        self.tokenizer = LayoutLMTokenizer.from_pretrained(model_path)

    def categorize_job(self, text: str) -> str:
        """
        Simplified method for quick categorization
        
        Args:
            text (str): Job description text
            
        Returns:
            str: Predicted job category
        """
        try:
            # Use BERT to predict the category
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits.argmax(-1).squeeze().tolist()
            
            # Map predictions to job categories
            categories = ["electrician", "sweeper", "driver", "plumber", "mechanic", 
                          "software engineer", "data scientist", "teacher", "nurse", "sales"]
            predicted_category = categories[predictions[0]]
            
            return predicted_category
        except Exception as e:
            logging.error(f"Error categorizing job: {str(e)}")
            return "unknown"

    def categorize_jobs(self, jobs: List[str]) -> List[Dict[str, str]]:
        """
        Processes multiple job descriptions
        
        Args:
            jobs (List[str]): List of job descriptions
            
        Returns:
            List[Dict[str, str]]: List of dictionaries with text and category
        """
        try:
            results = []
            for job in jobs:
                result = {
                    'text': job,
                    'category': self.categorize_job(job)
                }
                results.append(result)
            return results
        except Exception as e:
            logging.error(f"Error categorizing jobs: {str(e)}")
            return []

    def predict_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Predict job categories for a batch of job descriptions
        
        Args:
            texts (List[str]): List of job descriptions
            
        Returns:
            List[Dict[str, any]]: List of dictionaries with job description and predicted category
        """
        try:
            results = []
            for text in texts:
                result = self.predict(text)
                results.append({
                    'description': text,
                    'category': result[0],
                    'confidence': result[1]
                })
            return results
        except Exception as e:
            logging.error(f"Error predicting batch: {str(e)}")
            return []

    def predict(self, text: str) -> Dict[str, any]:
        """
        Predict the job category for a given job description
        
        Args:
            text (str): Job description text
            
        Returns:
            Dict[str, any]: Predicted job category and confidence scores
        """
        try:
            # Use BERT to predict the category
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits.argmax(-1).squeeze().tolist()
            
            # Map predictions to job categories
            categories = ["electrician", "sweeper", "driver", "plumber", "mechanic", 
                          "software engineer", "data scientist", "teacher", "nurse", "sales"]
            predicted_category = categories[predictions[0]]
            
            # Get all predictions with scores
            all_predictions = list(zip(categories, outputs.logits.softmax(-1).squeeze().tolist()))
            
            return {
                'category': predicted_category,
                'confidence': outputs.logits.softmax(-1).squeeze()[categories.index(predicted_category)],
                'metadata': {
                    'all_predictions': all_predictions
                }
            }
        except Exception as e:
            logging.error(f"Error predicting job category: {str(e)}")
            return {
                'category': 'unknown',
                'confidence': 0.0,
                'metadata': {
                    'all_predictions': []
                }
            }

class JobScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Initialize the zero-shot classifier
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.job_categories = ["electrician", "sweeper", "driver", "plumber", "mechanic", 
                             "software engineer", "data scientist", "teacher", "nurse", "sales"]
        
        # Configure PDF options
        self.pdf_options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None
        }
        
        # Configure Tesseract path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Initialize LayoutLM
        self.layoutlm_tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
        self.layoutlm_model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")
        
        # Initialize Selenium WebDriver
        self.setup_selenium()
        
        # Initialize ML classifier
        self.ml_classifier = MLJobClassifier(
            model_path="job_classifier.pkl",
            vectorizer_path="vectorizer.pkl"
        )
        
        # Initialize BERT classifier
        self.bert_classifier = BERTJobClassifier(
            model_path="your-finetuned-model",
            labels=["electrician", "plumber", "sweeper", "driver"]
        )
        
    def setup_selenium(self):
        """Setup Selenium WebDriver with appropriate options"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Run in headless mode
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            # Initialize the WebDriver
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            self.driver.set_page_load_timeout(30)
            logging.info("Selenium WebDriver initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing Selenium WebDriver: {str(e)}")
            raise

    def scrape_jobs_with_selenium(self, url: str, wait_for_element: str = None) -> List[Dict[str, str]]:
        """
        Scrape job listings using Selenium
        
        Args:
            url (str): The URL to scrape job listings from
            wait_for_element (str, optional): CSS selector to wait for before scraping
            
        Returns:
            List[Dict[str, str]]: List of job postings with their details
        """
        try:
            logging.info(f"Scraping jobs from {url} using Selenium")
            self.driver.get(url)
            
            # Wait for the page to load
            if wait_for_element:
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_element))
                    )
                except TimeoutException:
                    logging.warning(f"Timeout waiting for element: {wait_for_element}")
            
            # Find job listings
            job_elements = self.driver.find_elements(By.CSS_SELECTOR, '.job-post')
            
            jobs = []
            for job_elem in job_elements:
                try:
                    job_data = {
                        'title': job_elem.find_element(By.CSS_SELECTOR, '.job-title').text.strip(),
                        'company': job_elem.find_element(By.CSS_SELECTOR, '.company-name').text.strip(),
                        'location': job_elem.find_element(By.CSS_SELECTOR, '.location').text.strip(),
                        'description': job_elem.find_element(By.CSS_SELECTOR, '.job-description').text.strip()
                    }
                    jobs.append(job_data)
                except NoSuchElementException as e:
                    logging.warning(f"Missing element in job posting: {str(e)}")
                    continue
            
            logging.info(f"Successfully scraped {len(jobs)} jobs using Selenium")
            return jobs
            
        except Exception as e:
            logging.error(f"Error scraping jobs with Selenium: {str(e)}")
            return []
            
    def close_selenium(self):
        """Close the Selenium WebDriver"""
        try:
            if hasattr(self, 'driver'):
                self.driver.quit()
                logging.info("Selenium WebDriver closed successfully")
        except Exception as e:
            logging.error(f"Error closing Selenium WebDriver: {str(e)}")

    def get_word_boxes(self, image: Image.Image) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """
        Get word boxes from an image using Tesseract OCR
        
        Args:
            image (Image.Image): PIL Image object
            
        Returns:
            List[Tuple[str, Tuple[int, int, int, int]]]: List of (word, bounding box) tuples
        """
        try:
            # Get OCR data including bounding boxes
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Extract words and their bounding boxes
            words = []
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                if int(data['conf'][i]) > 60:  # Confidence threshold
                    (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                    word = data['text'][i].strip()
                    if word:
                        words.append((word, (x, y, x + w, y + h)))
            
            return words
            
        except Exception as e:
            logging.error(f"Error getting word boxes: {str(e)}")
            return []

    def process_with_layoutlm(self, image: Image.Image, words: List[Tuple[str, Tuple[int, int, int, int]]]) -> Dict[str, str]:
        """
        Process document with LayoutLM for structured information extraction
        
        Args:
            image (Image.Image): PIL Image object
            words (List[Tuple[str, Tuple[int, int, int, int]]]): List of (word, bounding box) tuples
            
        Returns:
            Dict[str, str]: Extracted structured information
        """
        try:
            # Prepare inputs for LayoutLM
            encoding = self.layoutlm_tokenizer(
                [word[0] for word in words],
                boxes=[word[1] for word in words],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = self.layoutlm_model(**encoding)
                predictions = outputs.logits.argmax(-1).squeeze().tolist()
            
            # Process predictions (this is a simplified example)
            # In a real application, you would map predictions to specific fields
            extracted_info = {
                'title': '',
                'company': '',
                'location': '',
                'description': ''
            }
            
            # Simple heuristic: first line is title, second is company, etc.
            lines = [word[0] for word in words]
            if len(lines) > 0:
                extracted_info['title'] = lines[0]
            if len(lines) > 1:
                extracted_info['company'] = lines[1]
            if len(lines) > 2:
                extracted_info['location'] = lines[2]
            if len(lines) > 3:
                extracted_info['description'] = ' '.join(lines[3:])
            
            return extracted_info
            
        except Exception as e:
            logging.error(f"Error processing with LayoutLM: {str(e)}")
            return {}

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """
        Extract text from a PDF file using OCR and LayoutLM
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Dict[str, str]]: List of extracted job information
        """
        try:
            logging.info(f"Extracting text from PDF: {pdf_path}")
            
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            
            extracted_jobs = []
            for image in images:
                # Get word boxes using Tesseract
                words = self.get_word_boxes(image)
                
                # Process with LayoutLM
                job_info = self.process_with_layoutlm(image, words)
                if job_info:
                    extracted_jobs.append(job_info)
            
            logging.info(f"Successfully extracted {len(extracted_jobs)} jobs")
            return extracted_jobs
            
        except Exception as e:
            logging.error(f"Error extracting text from PDF: {str(e)}")
            return []

    def process_job_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """
        Process a job PDF file, extract text, and classify jobs
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Dict[str, str]]: List of classified job postings
        """
        try:
            # Extract text and structure using OCR and LayoutLM
            jobs = self.extract_text_from_pdf(pdf_path)
            
            # Classify the jobs
            classified_jobs = self.classify_jobs(jobs)
            
            return classified_jobs
            
        except Exception as e:
            logging.error(f"Error processing PDF: {str(e)}")
            return []

    def scrape_jobs(self, url: str) -> List[Dict[str, str]]:
        """
        Scrape job listings from the given URL
        
        Args:
            url (str): The URL to scrape job listings from
            
        Returns:
            List[Dict[str, str]]: List of job postings with their details
        """
        try:
            logging.info(f"Scraping jobs from {url}")
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            soup = BeautifulSoup(response.text, 'html.parser')
            job_posts = soup.find_all('div', class_='job-post')
            
            jobs = []
            for post in job_posts:
                job_data = {
                    'title': post.find('h2', class_='job-title').get_text(strip=True) if post.find('h2', class_='job-title') else 'N/A',
                    'company': post.find('span', class_='company-name').get_text(strip=True) if post.find('span', class_='company-name') else 'N/A',
                    'location': post.find('span', class_='location').get_text(strip=True) if post.find('span', class_='location') else 'N/A',
                    'description': post.find('div', class_='job-description').get_text(strip=True) if post.find('div', class_='job-description') else 'N/A'
                }
                jobs.append(job_data)
            
            logging.info(f"Successfully scraped {len(jobs)} jobs")
            return jobs
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error scraping jobs: {str(e)}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return []

    def classify_jobs(self, jobs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Classify jobs using zero-shot classification
        
        Args:
            jobs (List[Dict[str, str]]): List of job postings
            
        Returns:
            List[Dict[str, str]]: Jobs with added classification results
        """
        classified_jobs = []
        for job in jobs:
            # Combine title and description for better classification
            job_text = f"{job['title']} {job['description']}"
            
            # Get classification results
            result = self.classifier(job_text, candidate_labels=self.job_categories)
            
            # Add classification results to job data
            job['category'] = result['labels'][0]
            job['category_confidence'] = result['scores'][0]
            classified_jobs.append(job)
            
            logging.info(f"Classified job: {job['title'][:50]}... -> {job['category']} (confidence: {job['category_confidence']:.2f})")
        
        return classified_jobs

    def generate_pdf(self, jobs: List[Dict[str, str]], output_dir: str = "output") -> str:
        """
        Generate a PDF report of the job listings
        
        Args:
            jobs (List[Dict[str, str]]): List of classified job postings
            output_dir (str): Directory to save the PDF
            
        Returns:
            str: Path to the generated PDF file
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate HTML content
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Job Listings Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .job { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
                    .title { font-size: 20px; color: #333; margin-bottom: 10px; }
                    .company { font-size: 16px; color: #666; margin-bottom: 5px; }
                    .location { font-size: 14px; color: #888; margin-bottom: 10px; }
                    .category { font-size: 14px; color: #2c7be5; margin-bottom: 10px; }
                    .description { font-size: 14px; line-height: 1.5; color: #444; }
                    .header { text-align: center; margin-bottom: 40px; }
                    .timestamp { color: #666; font-size: 12px; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Job Listings Report</h1>
                    <p class="timestamp">Generated on: {timestamp}</p>
                </div>
            """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            for job in jobs:
                html_content += f"""
                <div class="job">
                    <div class="title">{job['title']}</div>
                    <div class="company">Company: {job['company']}</div>
                    <div class="location">Location: {job['location']}</div>
                    <div class="category">Category: {job['category']} (confidence: {job['category_confidence']:.2f})</div>
                    <div class="description">{job['description']}</div>
                </div>
                """
            
            html_content += """
            </body>
            </html>
            """
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_path = os.path.join(output_dir, f"job_listings_{timestamp}.pdf")
            
            # Convert HTML to PDF
            pdfkit.from_string(html_content, pdf_path, options=self.pdf_options)
            logging.info(f"PDF report generated: {pdf_path}")
            
            return pdf_path
            
        except Exception as e:
            logging.error(f"Error generating PDF: {str(e)}")
            return ""

    def extract_links(self, url: str, base_url: str = None) -> List[Dict[str, str]]:
        """
        Extract all links from a webpage
        
        Args:
            url (str): The URL to extract links from
            base_url (str, optional): Base URL for resolving relative links
            
        Returns:
            List[Dict[str, str]]: List of links with their text and href
        """
        try:
            logging.info(f"Extracting links from {url}")
            self.driver.get(url)
            
            # Wait for the page to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get all links
            links = self.driver.find_elements(By.TAG_NAME, 'a')
            
            extracted_links = []
            for link in links:
                try:
                    href = link.get_attribute('href')
                    text = link.text.strip()
                    
                    # Skip empty links and anchors
                    if not href or href.startswith('#'):
                        continue
                        
                    # Resolve relative URLs
                    if base_url and not href.startswith(('http://', 'https://')):
                        href = urljoin(base_url, href)
                    
                    extracted_links.append({
                        'text': text,
                        'href': href,
                        'visible': link.is_displayed()
                    })
                    
                except Exception as e:
                    logging.warning(f"Error processing link: {str(e)}")
                    continue
            
            logging.info(f"Successfully extracted {len(extracted_links)} links")
            return extracted_links
            
        except Exception as e:
            logging.error(f"Error extracting links: {str(e)}")
            return []

    def find_job_links(self, url: str) -> List[Dict[str, str]]:
        """
        Find job-related links from a webpage
        
        Args:
            url (str): The URL to search for job links
            
        Returns:
            List[Dict[str, str]]: List of job-related links
        """
        try:
            # Extract all links
            all_links = self.extract_links(url, base_url=url)
            
            # Filter for job-related links
            job_keywords = ['job', 'career', 'position', 'vacancy', 'opportunity', 'hiring']
            job_links = []
            
            for link in all_links:
                # Check if link text or URL contains job-related keywords
                text = link['text'].lower()
                href = link['href'].lower()
                
                if any(keyword in text or keyword in href for keyword in job_keywords):
                    job_links.append(link)
            
            logging.info(f"Found {len(job_links)} job-related links")
            return job_links
            
        except Exception as e:
            logging.error(f"Error finding job links: {str(e)}")
            return []

    def classify_job_description(self, text: str, categories: List[str] = None) -> Dict[str, any]:
        """
        Classify a job description using zero-shot classification
        
        Args:
            text (str): The job description text to classify
            categories (List[str], optional): List of job categories to consider
            
        Returns:
            Dict[str, any]: Classification results including top label and confidence scores
        """
        try:
            # Use provided categories or default to self.job_categories
            candidate_labels = categories if categories else self.job_categories
            
            # Run classification
            result = self.classifier(text, candidate_labels=candidate_labels)
            
            # Get top prediction
            top_label = result['labels'][0]
            top_score = result['scores'][0]
            
            # Get all predictions with scores
            predictions = list(zip(result['labels'], result['scores']))
            
            return {
                'top_label': top_label,
                'top_score': top_score,
                'all_predictions': predictions
            }
            
        except Exception as e:
            logging.error(f"Error classifying job description: {str(e)}")
            return {
                'top_label': 'unknown',
                'top_score': 0.0,
                'all_predictions': []
            }

    def classify_job_listings(self, jobs: List[Dict[str, str]], categories: List[str] = None) -> List[Dict[str, any]]:
        """
        Classify multiple job listings
        
        Args:
            jobs (List[Dict[str, str]]): List of job listings to classify
            categories (List[str], optional): List of job categories to consider
            
        Returns:
            List[Dict[str, any]]: Jobs with added classification results
        """
        classified_jobs = []
        for job in jobs:
            # Combine title and description for better classification
            job_text = f"{job.get('title', '')} {job.get('description', '')}"
            
            # Classify the job
            classification = self.classify_job_description(job_text, categories)
            
            # Add classification results to job data
            job['classification'] = classification
            classified_jobs.append(job)
            
            # Log the classification
            logging.info(f"Classified job: {job.get('title', '')[:50]}... -> {classification['top_label']} (confidence: {classification['top_score']:.2f})")
        
        return classified_jobs

    def __del__(self):
        """Cleanup when the object is destroyed"""
        self.close_selenium()

    def scrape_naukri(self, location: str = "mumbai", max_pages: int = 3) -> List[Dict]:
        """Scrape job listings from Naukri.com."""
        jobs = []
        base_url = f"https://www.naukri.com/jobs-in-{location}"
        
        try:
            for page in range(1, max_pages + 1):
                url = f"{base_url}-{page}" if page > 1 else base_url
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                job_cards = soup.find_all('article', class_='jobTuple')
                
                for card in job_cards:
                    try:
                        title = card.find('a', class_='title').text.strip()
                        company = card.find('a', class_='subTitle').text.strip()
                        description = card.find('div', class_='job-description').text.strip()
                        
                        # Extract salary and experience
                        salary_text = card.find('span', class_='salary').text.strip()
                        exp_text = card.find('span', class_='experience').text.strip()
                        
                        # Parse salary (assuming format like "2.5-5 Lacs PA")
                        salary = self._parse_salary(salary_text)
                        
                        # Parse experience (assuming format like "2-5 Yrs")
                        experience = self._parse_experience(exp_text)
                        
                        jobs.append({
                            "title": title,
                            "company": company,
                            "description": description,
                            "salary": salary,
                            "experience": experience,
                            "source": "Naukri"
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error parsing job card: {str(e)}")
                        continue
                
                # Add delay between requests
                time.sleep(random.uniform(2, 4))
                
        except Exception as e:
            logger.error(f"Error scraping Naukri: {str(e)}")
        
        return jobs
    
    def scrape_indeed(self, location: str = "mumbai", max_pages: int = 3) -> List[Dict]:
        """Scrape job listings from Indeed.com."""
        jobs = []
        base_url = f"https://www.indeed.co.in/jobs?l={location}"
        
        try:
            for page in range(max_pages):
                url = f"{base_url}&start={page * 10}"
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                job_cards = soup.find_all('div', class_='job_seen_beacon')
                
                for card in job_cards:
                    try:
                        title = card.find('h2', class_='jobTitle').text.strip()
                        company = card.find('span', class_='companyName').text.strip()
                        description = card.find('div', class_='job-snippet').text.strip()
                        
                        # Extract salary and experience
                        salary_text = card.find('div', class_='salary-snippet').text.strip() if card.find('div', class_='salary-snippet') else "Not specified"
                        exp_text = card.find('div', class_='experience-snippet').text.strip() if card.find('div', class_='experience-snippet') else "Not specified"
                        
                        salary = self._parse_salary(salary_text)
                        experience = self._parse_experience(exp_text)
                        
                        jobs.append({
                            "title": title,
                            "company": company,
                            "description": description,
                            "salary": salary,
                            "experience": experience,
                            "source": "Indeed"
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error parsing job card: {str(e)}")
                        continue
                
                time.sleep(random.uniform(2, 4))
                
        except Exception as e:
            logger.error(f"Error scraping Indeed: {str(e)}")
        
        return jobs
    
    def _parse_salary(self, salary_text: str) -> float:
        """Parse salary text to numeric value."""
        try:
            # Remove currency symbols and convert to numeric
            salary_text = salary_text.lower()
            if 'lacs' in salary_text or 'lakh' in salary_text:
                # Convert lacs to thousands
                return float(''.join(filter(str.isdigit, salary_text))) * 100
            elif 'k' in salary_text:
                return float(''.join(filter(str.isdigit, salary_text)))
            else:
                return 0.0
        except:
            return 0.0
    
    def _parse_experience(self, exp_text: str) -> float:
        """Parse experience text to numeric value."""
        try:
            # Extract first number from experience text
            return float(''.join(filter(str.isdigit, exp_text.split()[0])))
        except:
            return 0.0

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

def main():
    try:
        # Initialize scraper
        scraper = JobScraper()
        
        # Scrape jobs from multiple sources
        logger.info("Scraping jobs from Naukri...")
        naukri_jobs = scraper.scrape_naukri(location="mumbai", max_pages=2)
        
        logger.info("Scraping jobs from Indeed...")
        indeed_jobs = scraper.scrape_indeed(location="mumbai", max_pages=2)
        
        # Combine all jobs
        all_jobs = naukri_jobs + indeed_jobs
        
        if not all_jobs:
            logger.error("No jobs found!")
            return
        
        # Initialize classifier
        classifier = JobClassifier("./your-finetuned-model")
        
        # Process jobs
        logger.info("Processing jobs with classifier...")
        df = classifier.process_jobs(all_jobs)
        
        # Save to Excel
        output_path = "scraped_jobs.xlsx"
        df.to_excel(output_path, index=False)
        logger.info(f"✅ Excel exported: {output_path}")
        
        # Print summary
        logger.info(f"Total jobs scraped: {len(all_jobs)}")
        logger.info(f"Jobs by source: {df['source'].value_counts().to_dict()}")
        logger.info(f"Jobs by category: {df['category'].value_counts().to_dict()}")
        logger.info(f"Jobs by level: {df['job_level'].value_counts().to_dict()}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 