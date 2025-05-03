import numpy as np
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Any
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import re
from text_processor import TextProcessor

logging.basicConfig(level=logging.INFO)

class JobSearchAgent:
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9):
        """Initialize the job search agent with Q-learning parameters"""
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.actions = [
            'click_job_listing',
            'click_next_page',
            'extract_job_details',
            'classify_job',
            'save_job'
        ]
        
        # Initialize web driver and text processor
        self.driver = None
        self.text_processor = TextProcessor()
        self._setup_driver()
    
    def _setup_driver(self):
        """Setup Selenium WebDriver"""
        try:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            self.driver = webdriver.Chrome(options=options)
        except Exception as e:
            logging.error(f"Error setting up WebDriver: {str(e)}")
            raise
    
    def get_state(self) -> str:
        """Get current DOM structure and convert to state representation"""
        try:
            # Get page source and parse with BeautifulSoup
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # Extract relevant features for state representation
            state_features = []
            
            # Check for job listings
            job_listings = soup.find_all('div', class_=re.compile(r'job|listing|posting', re.I))
            state_features.append(f"job_listings_{len(job_listings)}")
            
            # Check for pagination
            pagination = soup.find_all('a', class_=re.compile(r'page|next|pagination', re.I))
            state_features.append(f"pagination_{len(pagination)}")
            
            # Check for job details
            job_details = soup.find_all('div', class_=re.compile(r'detail|description|content', re.I))
            state_features.append(f"job_details_{len(job_details)}")
            
            return '_'.join(state_features)
        except Exception as e:
            logging.error(f"Error getting state: {str(e)}")
            return "error_state"
    
    def choose_action(self, state: str) -> str:
        """Choose next action using epsilon-greedy policy"""
        epsilon = 0.1  # Exploration rate
        
        if np.random.random() < epsilon:
            # Explore: choose random action
            return np.random.choice(self.actions)
        else:
            # Exploit: choose best action from Q-table
            state_actions = self.q_table[state]
            if not state_actions:
                return np.random.choice(self.actions)
            return max(state_actions.items(), key=lambda x: x[1])[0]
    
    def execute_action(self, action: str) -> Tuple[bool, float, Dict[str, Any]]:
        """Execute the chosen action and return success status, reward, and extracted data"""
        try:
            if action == 'click_job_listing':
                # Find and click first job listing
                job_links = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="job"]')
                if job_links:
                    job_links[0].click()
                    return True, 0.1, {}  # Small positive reward for finding job listing
            
            elif action == 'click_next_page':
                # Find and click next page button
                next_buttons = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="page"]')
                if next_buttons:
                    next_buttons[0].click()
                    return True, 0.1, {}  # Small positive reward for navigation
            
            elif action == 'extract_job_details':
                # Extract job details from current page
                job_details = self.driver.find_elements(By.CSS_SELECTOR, 'div[class*="job-detail"]')
                if job_details:
                    # Process job description with text processor
                    job_text = job_details[0].text
                    processed_info = self.text_processor.process_job_description(job_text)
                    return True, 0.2, processed_info  # Reward for finding job details
            
            elif action == 'classify_job':
                # Classify current job (using your existing classifier)
                # This would integrate with your JobClassifier
                return True, 0.3, {}  # Reward for successful classification
            
            elif action == 'save_job':
                # Save job to database/storage
                return True, 0.4, {}  # Reward for saving job
            
            return False, -0.1, {}  # Small negative reward for failed action
            
        except Exception as e:
            logging.error(f"Error executing action {action}: {str(e)}")
            return False, -0.5, {}  # Larger negative reward for errors
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value using Q-learning update rule"""
        # Get current Q-value
        current_q = self.q_table[state][action]
        
        # Get maximum Q-value for next state
        next_state_actions = self.q_table[next_state]
        max_next_q = max(next_state_actions.values()) if next_state_actions else 0
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        # Update Q-table
        self.q_table[state][action] = new_q
    
    def train(self, num_episodes: int = 100):
        """Train the agent through multiple episodes"""
        for episode in range(num_episodes):
            logging.info(f"Starting episode {episode + 1}")
            
            # Reset environment
            self.driver.get("https://example.com/jobs")  # Replace with actual job site
            
            total_reward = 0
            steps = 0
            max_steps = 50  # Prevent infinite loops
            extracted_jobs = []
            
            while steps < max_steps:
                # Get current state
                state = self.get_state()
                
                # Choose and execute action
                action = self.choose_action(state)
                success, reward, data = self.execute_action(action)
                
                # Get next state
                next_state = self.get_state()
                
                # Update Q-value
                self.update_q_value(state, action, reward, next_state)
                
                # Store extracted job data
                if action == 'extract_job_details' and success:
                    extracted_jobs.append(data)
                
                total_reward += reward
                steps += 1
                
                # End episode if we've found and saved a job
                if action == 'save_job' and success:
                    break
            
            logging.info(f"Episode {episode + 1} completed. Total reward: {total_reward}")
            logging.info(f"Extracted {len(extracted_jobs)} jobs")
    
    def save_model(self, path: str = "job_search_agent.pkl"):
        """Save the trained Q-table"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_model(self, path: str = "job_search_agent.pkl"):
        """Load a trained Q-table"""
        import pickle
        with open(path, 'rb') as f:
            self.q_table = defaultdict(lambda: defaultdict(float), pickle.load(f))
    
    def __del__(self):
        """Cleanup WebDriver"""
        if self.driver:
            self.driver.quit() 