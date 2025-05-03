import spacy
import logging
from typing import List, Dict, Any
import re
from transformers import pipeline
import torch

logging.basicConfig(level=logging.INFO)

class TextProcessor:
    def __init__(self):
        """Initialize text processor with spaCy model and BART summarizer"""
        try:
            # Initialize spaCy
            self.nlp = spacy.load("en_core_web_sm")
            logging.info("SpaCy model loaded successfully")
            
            # Initialize BART summarizer
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
            logging.info("BART summarizer loaded successfully")
            
        except OSError as e:
            if "spacy" in str(e):
                logging.warning("Downloading spaCy model...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            else:
                logging.error(f"Error loading models: {str(e)}")
                raise
    
    def summarize_text(self, text: str, max_length: int = 100, min_length: int = 30) -> Dict[str, Any]:
        """Summarize text using BART model"""
        try:
            # Split text into chunks if it's too long (BART has a max input length)
            max_chunk_length = 1024
            chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
            
            summaries = []
            for chunk in chunks:
                summary = self.summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )[0]['summary_text']
                summaries.append(summary)
            
            # Combine summaries if there were multiple chunks
            final_summary = ' '.join(summaries)
            
            # Get key points using spaCy
            doc = self.nlp(text)
            key_points = []
            
            # Extract sentences with important entities
            for sent in doc.sents:
                if any(ent.label_ in ['ORG', 'GPE', 'MONEY', 'DATE'] for ent in sent.ents):
                    key_points.append(sent.text.strip())
            
            return {
                "summary": final_summary,
                "key_points": key_points[:5],  # Limit to top 5 key points
                "original_length": len(text),
                "summary_length": len(final_summary)
            }
        except Exception as e:
            logging.error(f"Error summarizing text: {str(e)}")
            return {
                "summary": "",
                "key_points": [],
                "original_length": len(text),
                "summary_length": 0
            }
    
    def extract_locations(self, text: str) -> List[str]:
        """Extract location entities from text"""
        try:
            doc = self.nlp(text)
            return [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        except Exception as e:
            logging.error(f"Error extracting locations: {str(e)}")
            return []
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract various entities from text"""
        try:
            doc = self.nlp(text)
            entities = {
                "locations": [],
                "organizations": [],
                "dates": [],
                "money": [],
                "skills": []
            }
            
            for ent in doc.ents:
                if ent.label_ == "GPE":
                    entities["locations"].append(ent.text)
                elif ent.label_ == "ORG":
                    entities["organizations"].append(ent.text)
                elif ent.label_ == "DATE":
                    entities["dates"].append(ent.text)
                elif ent.label_ == "MONEY":
                    entities["money"].append(ent.text)
            
            # Extract skills using custom patterns
            skill_patterns = [
                r'\b(?:experienced|skilled|proficient|expert|knowledge of|familiar with)\s+([^.,]+)',
                r'\b(?:required|needed|must have|should have)\s+([^.,]+)',
                r'\b(?:ability to|capable of|can)\s+([^.,]+)'
            ]
            
            for pattern in skill_patterns:
                matches = re.finditer(pattern, text.lower())
                for match in matches:
                    skill = match.group(1).strip()
                    if skill not in entities["skills"]:
                        entities["skills"].append(skill)
            
            return entities
        except Exception as e:
            logging.error(f"Error extracting entities: {str(e)}")
            return {k: [] for k in entities.keys()}
    
    def extract_salary_range(self, text: str) -> Dict[str, Any]:
        """Extract salary information from text"""
        try:
            # Common salary patterns
            patterns = [
                r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:-|to)\s*\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:-|to)\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:per|/)\s*(?:year|month|hour|week)',
                r'salary:\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:-|to)\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text.lower())
                if match:
                    min_salary = float(match.group(1).replace(',', ''))
                    max_salary = float(match.group(2).replace(',', ''))
                    return {
                        "min_salary": min_salary,
                        "max_salary": max_salary,
                        "currency": "USD",
                        "period": "year" if "year" in text.lower() else "month"
                    }
            
            return {
                "min_salary": None,
                "max_salary": None,
                "currency": None,
                "period": None
            }
        except Exception as e:
            logging.error(f"Error extracting salary range: {str(e)}")
            return {
                "min_salary": None,
                "max_salary": None,
                "currency": None,
                "period": None
            }
    
    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information from text"""
        try:
            # Email pattern
            email_pattern = r'[\w\.-]+@[\w\.-]+'
            email_match = re.search(email_pattern, text)
            
            # Phone pattern
            phone_pattern = r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
            phone_match = re.search(phone_pattern, text)
            
            return {
                "email": email_match.group(0) if email_match else None,
                "phone": phone_match.group(0) if phone_match else None
            }
        except Exception as e:
            logging.error(f"Error extracting contact info: {str(e)}")
            return {"email": None, "phone": None}
    
    def process_job_description(self, text: str) -> Dict[str, Any]:
        """Process job description and extract all relevant information"""
        try:
            # Get summary and key points
            summary_info = self.summarize_text(text)
            
            # Extract other information
            entities = self.extract_entities(text)
            salary_info = self.extract_salary_range(text)
            contact_info = self.extract_contact_info(text)
            
            return {
                "summary": summary_info["summary"],
                "key_points": summary_info["key_points"],
                "entities": entities,
                "salary": salary_info,
                "contact": contact_info,
                "raw_text": text,
                "text_stats": {
                    "original_length": summary_info["original_length"],
                    "summary_length": summary_info["summary_length"],
                    "compression_ratio": summary_info["summary_length"] / summary_info["original_length"] if summary_info["original_length"] > 0 else 0
                }
            }
        except Exception as e:
            logging.error(f"Error processing job description: {str(e)}")
            return {
                "summary": "",
                "key_points": [],
                "entities": {},
                "salary": {},
                "contact": {},
                "raw_text": text,
                "text_stats": {
                    "original_length": len(text),
                    "summary_length": 0,
                    "compression_ratio": 0
                }
            } 