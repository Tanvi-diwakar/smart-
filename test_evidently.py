import pandas as pd
from evidently import Report
from evidently.metric_preset import DataDriftPreset
import requests
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
import numpy as np
import re
import matplotlib.pyplot as plt
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class JobClassifier:
    def __init__(self, model_path="./your-finetuned-model"):
        self.model_name = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.labels = ["plumber", "driver", "sweeper", "electrician", "chef", 
                      "carpenter", "painter", "guard", "cleaner", 
                      "computer operator", "technician", "security"]

    def predict_with_confidence(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()
        conf = torch.softmax(logits, dim=1)[0][pred].item()
        return self.labels[pred], round(conf, 2)

# Create sample data
reference_data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': ['a', 'b', 'c', 'd', 'e']
})

current_data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 6],
    'feature2': ['a', 'b', 'c', 'd', 'f']
})

# Create and run the report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)

# Save the report
report.save_html("test_drift_report.html")
print("Report generated successfully!")

def scrape_jobs(url, classifier):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')
    jobs = []
    for job in soup.select('.JobCard'):
        title = job.select_one('.JobCard_jobTitle__1X30I')
        company = job.select_one('.JobCard_companyName__3d0VI')
        location = job.select_one('.JobCard_location__3Ctnr')
        salary = job.select_one('.JobCard_salary__2oZp2')
        description = job.select_one('.JobCard_description__2X30I')
        job_data = {
            'title': title.text.strip() if title else "Not Mentioned",
            'company': company.text.strip() if company else "Not Mentioned",
            'location': location.text.strip() if location else "Not Mentioned",
            'salary': salary.text.strip() if salary else "Not Mentioned",
            'description': description.text.strip() if description else ""
        }
        # Classify using the new transformer model
        job_data['category'], job_data['confidence'] = classifier.predict_with_confidence(
            job_data['title'] + " " + job_data['description']
        )
        jobs.append(job_data)
    for i, job in enumerate(jobs):
        print(f"Job {i}: {job}")
    return jobs

def parse_salary(salary_str):
    if not salary_str or "Not Mentioned" in salary_str:
        return np.nan
    salary_str = salary_str.replace('₹', '').replace(',', '').strip().lower()
    match = re.match(r'(\d+)(?:\s*-\s*(\d+))?', salary_str)
    if match:
        low = int(match.group(1))
        high = int(match.group(2)) if match.group(2) else low
        return (low + high) // 2
    if 'k' in salary_str:
        return int(float(salary_str.replace('k', '')) * 1000)
    try:
        return int(salary_str)
    except ValueError:
        return np.nan

def parse_experience(exp_str):
    if not exp_str or "Fresher" in exp_str:
        return 0
    match = re.search(r'(\d+)', exp_str)
    if match:
        return int(match.group(1))
    return np.nan

# Example DataFrame
# df = pd.DataFrame({'salary': [...], 'experience': [...]})

# Apply parsing
df['salary_num'] = df['salary'].apply(parse_salary)
df['experience_num'] = df['description'].apply(parse_experience)

# Drop rows with missing values
X = df[['salary_num', 'experience_num']].dropna()

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
df.loc[X.index, 'level'] = kmeans.labels_

# Map cluster numbers to level names (adjust as needed based on your data)
level_map = {0: 'junior', 1: 'mid', 2: 'senior'}
df['level_name'] = df['level'].map(level_map)

# Now filter for plumbers, senior level, and salary > 25000
results = df[(df['category'] == 'plumber') & 
             (df['level_name'] == 'senior') & 
             (df['salary_num'] > 25000)]

print(results[['title', 'salary', 'experience_num', 'level_name']].head())

# Create the scatter plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['experience_num'], df['salary_num'], 
                     c=df['level'], cmap='viridis', alpha=0.6)

# Add labels and title
plt.xlabel("Experience (years)", fontsize=12)
plt.ylabel("Salary (₹)", fontsize=12)
plt.title("Salary vs Experience Clusters", fontsize=14, pad=20)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Cluster', fontsize=12)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Format y-axis to show salary in thousands
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{int(x/1000)}k'))

# Add cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 1], centers[:, 0], c='red', marker='x', s=200, 
           linewidths=3, label='Cluster Centers')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

# Example usage:
if __name__ == "__main__":
    # Initialize the classifier with the fine-tuned model
    classifier = JobClassifier()
    
    url = "https://www.workindia.in/jobs-in-mumbai/"
    jobs = scrape_jobs(url, classifier)
    df = pd.DataFrame(jobs)
    
    # Add numeric columns
    df['salary_num'] = df['salary'].apply(parse_salary)
    df['experience_num'] = df['description'].apply(parse_experience)

    # Drop missing values
    X = df[['salary_num', 'experience_num']].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
    df.loc[X.index, 'level'] = kmeans.labels_

    # Map clusters to names
    level_map = {0: 'junior', 1: 'mid', 2: 'senior'}
    df['level_name'] = df['level'].map(level_map)

    # Save results
    df.to_excel("classified_jobs.xlsx", index=False)
    print("✅ Excel exported: classified_jobs.xlsx")

    # Display results in Streamlit
    st.title("Job Scraper & Analyzer Demo")
    st.dataframe(df)

    print(df.columns)  # See all columns
    print(df.head())   # See first few rows for a quick check

    st.write("Columns:", df.columns.tolist())
    st.dataframe(df.head())

    print(df.isnull().sum())  # Shows count of nulls per column
    print(df[df.isnull().any(axis=1)])  # Shows rows with any nulls

    st.write("Null values per column:", df.isnull().sum())
    st.write("Rows with missing values:", df[df.isnull().any(axis=1)])

    st.write("Sample jobs:", df.head(10))

    # Add filters
    category = st.selectbox("Category", df['category'].unique())
    level = st.selectbox("Level", ['junior', 'mid', 'senior'])
    min_salary = st.slider("Minimum Salary", int(df['salary_num'].min()), int(df['salary_num'].max()), 25000)
    filtered = df[(df['category'] == category) & (df['level_name'] == level) & (df['salary_num'] > min_salary)]
    st.write("Filtered Results:", filtered)

    # Optionally: add matplotlib/seaborn plots 

# Print first 5 jobs
for job in jobs[:5]:
    print(job['title'], job['description'], job['salary'], job['location'])

# Predict category and confidence for a sample text
text = "Urgently hiring experienced driver for night shift."
category, confidence = classifier.predict_with_confidence(text)
print(f"Category: {category}, Confidence: {confidence:.2f}")

# Scatter plot: Salary vs Experience Clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['experience_num'], df['salary_num'], c=df['level'], cmap='viridis')
plt.xlabel("Experience (years)")
plt.ylabel("Salary (₹)")
plt.title("Salary vs Experience Clusters")
plt.colorbar(label='Cluster')
plt.show()

# Filter for senior plumbers with salary > 25000
results = df[(df['category'] == 'plumber') & 
             (df['level_name'] == 'senior') & 
             (df['salary_num'] > 25000)]
print(results[['title', 'salary', 'experience_num']])

Index(['title', 'company', 'location', 'salary', 'description', 'category',
       'confidence', 'salary_num', 'experience_num', 'level', 'level_name'],
      dtype='object')
           title      company location   salary               description category  confidence  salary_num  experience_num  level level_name
0        Driver  ABC Logistics   Mumbai  ₹30,000  Night shift driver needed   driver       0.92      30000            5.0      2     senior
1       Plumber   XYZ Plumbing   Mumbai  ₹28,000   Experienced plumber reqd  plumber       0.88      28000            7.0      2     senior
... 