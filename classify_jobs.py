import pandas as pd
from sklearn.cluster import KMeans

# Sample scraped data (simulate or use real scrape)
data = [
    {"title": "Urgent electrician needed", "description": "Fix wiring and install fuse boxes", "salary": 18000, "experience": 2},
    {"title": "Chef required for hotel", "description": "Prepare Indian and Chinese dishes", "salary": 28000, "experience": 5},
    {"title": "Driver needed", "description": "Drive company truck for delivery", "salary": 15000, "experience": 1},
    # Add more as needed
]

# Dummy prediction logic
labels = ["plumber", "driver", "sweeper", "electrician", "chef", "carpenter", "painter", "guard", "cleaner", "computer operator", "technician", "security"]
for i, job in enumerate(data):
    job["category"] = labels[i % len(labels)]
    job["confidence"] = 0.9  # Dummy confidence

# Convert to DataFrame
df = pd.DataFrame(data)

# Clustering for salary/experience levels
X = df[["salary", "experience"]]
kmeans = KMeans(n_clusters=3, random_state=42)
df["level_cluster"] = kmeans.fit_predict(X)

# Optional: Map cluster to readable level
cluster_map = {
    0: "entry-level",
    1: "mid-level",
    2: "senior-level"
}
df["job_level"] = df["level_cluster"].map(cluster_map)

# Save to Excel
df.to_excel("classified_jobs.xlsx", index=False)
print("✅ Excel exported: classified_jobs.xlsx") 