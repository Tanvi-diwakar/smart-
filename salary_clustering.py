import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from typing import List, Dict, Optional

def load_job_data(file_path: str) -> List[Dict]:
    """Load job data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading job data: {str(e)}")
        return []

def cluster_salaries(jobs: List[Dict], n_clusters: int = 3) -> List[Dict]:
    """Cluster job salaries using KMeans"""
    try:
        # Extract salaries and reshape for KMeans
        salaries = np.array([job['salary'] for job in jobs]).reshape(-1, 1)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(salaries)
        salary_labels = kmeans.predict(salaries)
        
        # Get salary ranges for each cluster
        cluster_ranges = []
        for i in range(n_clusters):
            cluster_salaries = salaries[salary_labels == i]
            min_salary = cluster_salaries.min()
            max_salary = cluster_salaries.max()
            cluster_ranges.append((min_salary, max_salary))
        
        # Add salary bands to jobs
        for i, job in enumerate(jobs):
            job['salary_band'] = f"Band {salary_labels[i]}"
            job['salary_range'] = f"₹{cluster_ranges[salary_labels[i]][0]:,.0f} - ₹{cluster_ranges[salary_labels[i]][1]:,.0f}"
        
        return jobs
    except Exception as e:
        print(f"Error clustering salaries: {str(e)}")
        return jobs

def classify_experience_levels(jobs: List[Dict]) -> List[Dict]:
    """Classify jobs into experience levels using decision tree"""
    try:
        # Prepare features and labels
        X = [[job["salary"], job["experience"]] for job in jobs]
        y = ["entry" if job["experience"] < 2 else "mid" if job["experience"] < 5 else "senior" for job in jobs]
        
        # Train decision tree
        tree = DecisionTreeClassifier(max_depth=3, random_state=0)
        tree.fit(X, y)
        
        # Get predictions
        predictions = tree.predict(X)
        
        # Add experience level to jobs
        for i, job in enumerate(jobs):
            job['experience_level'] = predictions[i]
        
        # Print decision rules
        print("\nExperience Level Classification Rules")
        print("=" * 50)
        rules = export_text(tree, feature_names=["salary", "experience"])
        print(rules)
        
        return jobs
    except Exception as e:
        print(f"Error classifying experience levels: {str(e)}")
        return jobs

def save_job_data(jobs: List[Dict], file_path: str):
    """Save job data to JSON file"""
    try:
        with open(file_path, 'w') as f:
            json.dump(jobs, f, indent=2)
    except Exception as e:
        print(f"Error saving job data: {str(e)}")

def filter_jobs(jobs: List[Dict], filters: Dict) -> List[Dict]:
    """Filter jobs based on specified criteria"""
    try:
        filtered_jobs = jobs.copy()
        
        # Apply category filter
        if 'category' in filters:
            filtered_jobs = [job for job in filtered_jobs if job['category'] == filters['category']]
        
        # Apply experience level filter
        if 'experience_level' in filters:
            filtered_jobs = [job for job in filtered_jobs if job['experience_level'] == filters['experience_level']]
        
        # Apply salary band filter
        if 'salary_band' in filters:
            # Extract min and max salary from the band string
            salary_range = filters['salary_band'].replace('₹', '').replace('k', '000').split('–')
            min_salary = int(salary_range[0])
            max_salary = int(salary_range[1])
            
            filtered_jobs = [
                job for job in filtered_jobs 
                if min_salary <= job['salary'] <= max_salary
            ]
        
        return filtered_jobs
    except Exception as e:
        print(f"Error filtering jobs: {str(e)}")
        return []

def main():
    # Load job data
    jobs = load_job_data('job_data.json')
    if not jobs:
        print("No job data found")
        return
    
    # Cluster salaries
    jobs = cluster_salaries(jobs)
    
    # Classify experience levels
    jobs = classify_experience_levels(jobs)
    
    # Save updated job data
    save_job_data(jobs, 'job_data_with_bands.json')
    
    # Define filters
    filters = {
        "category": "plumber",
        "experience_level": "entry",
        "salary_band": "₹15k–₹20k"
    }
    
    # Filter jobs
    filtered_jobs = filter_jobs(jobs, filters)
    
    # Print results
    print("\nFiltered Job Results")
    print("=" * 50)
    print(f"Filters applied: {filters}")
    print(f"Number of matching jobs: {len(filtered_jobs)}")
    print("\nMatching Jobs:")
    for job in filtered_jobs:
        print(f"\nTitle: {job['title']}")
        print(f"Category: {job['category']}")
        print(f"Salary: ₹{job['salary']:,.0f}")
        print(f"Salary Band: {job['salary_band']}")
        print(f"Experience: {job['experience']} years")
        print(f"Experience Level: {job['experience_level']}")

if __name__ == "__main__":
    main() 