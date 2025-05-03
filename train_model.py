from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import numpy as np

# Define all job categories
labels = [
    "plumber", "electrician", "carpenter", "painter", "welder", "mason", "mechanic",
    "driver", "delivery boy", "loader",
    "sweeper", "housekeeping", "garbage collector", "pest control",
    "security guard", "watchman", "parking attendant",
    "tailor", "embroidery worker", "ironing staff",
    "chef", "kitchen helper", "dishwasher",
    "salesman", "store helper", "cashier",
    "barber", "ac technician", "gardener", "lift operator",
    "warehouse worker", "office boy", "computer operator"
]

# Sample training data with expanded categories
train_texts = [
    # Construction and Maintenance
    "Need someone to fix plumbing issues in a housing society.",
    "Required plumber to fix leaking pipelines in a flat.",
    "Looking for experienced plumber for commercial building maintenance.",
    "Need a professional to install new electrical circuits.",
    "Required electrician for commercial building maintenance.",
    "Looking for experienced electrician for home wiring.",
    "Hiring carpenter for furniture assembly work.",
    "Looking for experienced carpenter for woodwork.",
    "Need a painter to repaint school classrooms.",
    "Looking for experienced painter for house renovation.",
    "Required welder to fix metal gate in workshop.",
    "Need experienced welder for fabrication work.",
    "Wanted: mason for brickwork in new construction.",
    "Looking for experienced mason for wall construction.",
    "Need mechanic to repair car engine.",
    "Looking for experienced mechanic for bike service.",
    
    # Transportation
    "Required experienced driver for office cab service.",
    "Looking for delivery boy for local area.",
    "Need loader for warehouse work.",
    "Hiring truck driver for long distance delivery.",
    
    # Cleaning and Sanitation
    "Looking for cleaning staff in office building.",
    "Opening for a cleaner in a private school.",
    "Need housekeeping staff for hotel.",
    "Required garbage collector for residential area.",
    "Looking for pest control service provider.",
    
    # Security
    "Security guard required for night shift at hospital.",
    "Looking for watchman for apartment complex.",
    "Need parking attendant for shopping mall.",
    
    # Textile and Garments
    "Looking for a tailor to stitch school uniforms.",
    "Need embroidery worker for dress making.",
    "Hiring ironing staff for laundry service.",
    
    # Food Service
    "Opening for chef in a restaurant near station.",
    "Need kitchen helper for hotel kitchen.",
    "Dishwasher required in hotel kitchen.",
    
    # Retail
    "Hiring salesman for electronics store.",
    "Need store helper for supermarket.",
    "Looking for cashier for retail shop.",
    
    # Personal Services
    "Hiring a barber for men's salon in main market.",
    "We need an AC technician for installation and service.",
    "Looking for gardener for residential complex.",
    "Need lift operator for office building.",
    
    # Office and Warehouse
    "Warehouse helper required to load/unload goods.",
    "Looking for office boy for corporate office.",
    "Need computer operator for data entry work."
]

train_labels = [
    # Construction and Maintenance
    "plumber", "plumber", "plumber", "electrician", "electrician", "electrician",
    "carpenter", "carpenter", "painter", "painter", "welder", "welder",
    "mason", "mason", "mechanic", "mechanic",
    
    # Transportation
    "driver", "delivery boy", "loader", "driver",
    
    # Cleaning and Sanitation
    "sweeper", "sweeper", "housekeeping", "garbage collector", "pest control",
    
    # Security
    "security guard", "watchman", "parking attendant",
    
    # Textile and Garments
    "tailor", "embroidery worker", "ironing staff",
    
    # Food Service
    "chef", "kitchen helper", "dishwasher",
    
    # Retail
    "salesman", "store helper", "cashier",
    
    # Personal Services
    "barber", "ac technician", "gardener", "lift operator",
    
    # Office and Warehouse
    "warehouse worker", "office boy", "computer operator"
]

# Initialize and fit vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)

# Initialize and fit classifier
classifier = RandomForestClassifier(n_estimators=300, random_state=42)
classifier.fit(X_train, train_labels)

# Save model and vectorizer
dump(classifier, 'job_classifier.pkl')
dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved successfully!") 