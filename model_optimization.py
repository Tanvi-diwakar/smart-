import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_data():
    """Load and prepare training data with multiple examples per class"""
    train_texts = []
    train_labels = []
    
    # Helper function to add examples
    def add_examples(texts, label):
        for text in texts:
            train_texts.append(text)
            train_labels.append(label)
    
    # Electrician examples
    add_examples([
        "Need professional electrician for home wiring installation",
        "Looking for experienced electrician to fix circuit issues",
        "Required: electrician for commercial building maintenance",
        "Electrician needed for industrial electrical work",
        "Hiring certified electrician for electrical repairs"
    ], "electrician")
    
    # Plumber examples
    add_examples([
        "Plumber required to fix leaking pipes in apartment",
        "Need experienced plumber for bathroom renovation",
        "Looking for plumber to install new water system",
        "Plumber needed for drainage system repair",
        "Required: plumber for commercial plumbing work"
    ], "plumber")
    
    # Carpenter examples
    add_examples([
        "Carpenter needed for custom furniture making",
        "Looking for experienced carpenter for woodwork",
        "Required: carpenter for door and window installation",
        "Carpenter needed for home renovation work",
        "Hiring skilled carpenter for cabinet making"
    ], "carpenter")
    
    # Driver examples
    add_examples([
        "Driver required for office pickup and drop",
        "Looking for experienced driver with valid license",
        "Need professional driver for corporate transport",
        "Driver wanted for delivery service",
        "Hiring driver for school bus"
    ], "driver")
    
    # Painter examples
    add_examples([
        "Painter needed for house painting work",
        "Looking for experienced painter for wall texturing",
        "Required: painter for commercial building",
        "Painter needed for furniture refinishing",
        "Hiring painter for interior and exterior work"
    ], "painter")
    
    # AC Technician examples
    add_examples([
        "AC technician required for installation and repair",
        "Looking for experienced AC mechanic",
        "Need AC technician for regular maintenance",
        "AC repair technician wanted",
        "Hiring AC service technician for commercial units"
    ], "ac technician")
    
    # Security Guard examples
    add_examples([
        "Security guard needed for night shift",
        "Looking for experienced security personnel",
        "Required: security guard for residential complex",
        "Security guard wanted for office building",
        "Hiring security staff for mall"
    ], "security guard")
    
    # Cleaner/Sweeper examples
    add_examples([
        "Cleaner required for office maintenance",
        "Looking for cleaning staff for commercial space",
        "Need housekeeping staff for hotel",
        "Cleaner wanted for daily office cleaning",
        "Hiring cleaning personnel for school"
    ], "cleaner")
    
    # Computer Operator examples
    add_examples([
        "Computer operator needed for data entry",
        "Looking for experienced computer operator",
        "Required: computer operator with typing skills",
        "Computer operator wanted for office work",
        "Hiring data entry operator"
    ], "computer operator")
    
    # Chef examples
    add_examples([
        "Chef required for restaurant kitchen",
        "Looking for experienced chef with Indian cuisine knowledge",
        "Need head chef for hotel",
        "Chef wanted for catering service",
        "Hiring chef for fine dining restaurant"
    ], "chef")
    
    return train_texts, train_labels

def optimize_model():
    """Optimize the model using GridSearchCV and feature engineering"""
    try:
        print("\nStarting model optimization...")
        
        # Load data
        texts, labels = load_data()
        print(f"Loaded {len(texts)} training examples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print("Data split completed")
        
        # Initialize vectorizer with improved parameters
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        # Transform texts
        print("Vectorizing text data...")
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        print(f"Vectorization completed. Number of features: {len(vectorizer.get_feature_names_out())}")
        
        # Initialize base model with better parameters
        base_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Fit the model
        print("\nTraining model...")
        base_model.fit(X_train_vec, y_train)
        
        # Make predictions
        y_pred = base_model.predict(X_test_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Print results
        print("\nModel Results")
        print("=" * 50)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Save optimized model and vectorizer
        print("\nSaving model and vectorizer...")
        joblib.dump(base_model, 'optimized_job_classifier.pkl')
        joblib.dump(vectorizer, 'optimized_vectorizer.pkl')
        print("Model and vectorizer saved successfully")
        
        # Verify files exist
        if os.path.exists('optimized_job_classifier.pkl') and os.path.exists('optimized_vectorizer.pkl'):
            print("Verified: Model files saved correctly")
        else:
            print("Warning: Model files not found after saving")
        
        # Plot feature importance
        plot_feature_importance(base_model, vectorizer)
        
        return base_model, vectorizer, accuracy
        
    except Exception as e:
        logging.error(f"Error in model optimization: {str(e)}")
        return None, None, 0.0

def plot_feature_importance(model, vectorizer, top_n=20):
    """Plot feature importance"""
    try:
        print("\nGenerating feature importance plot...")
        # Get feature importance
        feature_importance = model.feature_importances_
        feature_names = vectorizer.get_feature_names_out()
        
        # Sort features by importance
        indices = np.argsort(feature_importance)[::-1]
        top_features = feature_names[indices][:top_n]
        top_importance = feature_importance[indices][:top_n]
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_importance, y=top_features)
        plt.title('Top 20 Most Important Features')
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature Name')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
        print("Feature importance plot saved as 'feature_importance.png'")
        
    except Exception as e:
        logging.error(f"Error plotting feature importance: {str(e)}")

if __name__ == "__main__":
    model, vectorizer, accuracy = optimize_model()
    if model is not None:
        print(f"\nOptimization completed successfully with accuracy: {accuracy:.4f}") 