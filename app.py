import streamlit as st
import pandas as pd
from job_classification import JobClassifier
from scam_detector import ScamDetector
from job_search_agent import JobSearchAgent
from data_monitor import DataMonitor
import json
from datetime import datetime
import time
from text_processor import TextProcessor

# Set page config
st.set_page_config(
    page_title="Blue-Collar Job Explorer",
    page_icon="💼",
    layout="wide"
)

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = JobClassifier()
if 'scam_detector' not in st.session_state:
    st.session_state.scam_detector = ScamDetector()
if 'search_agent' not in st.session_state:
    st.session_state.search_agent = JobSearchAgent()
if 'data_monitor' not in st.session_state:
    st.session_state.data_monitor = DataMonitor()
if 'jobs_df' not in st.session_state:
    st.session_state.jobs_df = pd.DataFrame()
if 'is_searching' not in st.session_state:
    st.session_state.is_searching = False

# Title and description
st.title("💼 Blue-Collar Job Explorer")
st.markdown("""
This app helps you explore and classify blue-collar job listings, with built-in scam detection and automated job search.
""")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Job Analysis", "Automated Search", "Data Monitoring"])

with tab1:
    # Sidebar for input
    st.sidebar.header("Input Options")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Use Sample Jobs", "Enter Job Description", "Upload JSON"]
    )

    # Handle different input methods
    if input_method == "Use Sample Jobs":
        sample_jobs = [
            "Required experienced driver for office cab service.",
            "Hiring a painter for house painting work.",
            "Need a welder to fix metal gate in workshop.",
            "Looking for a tailor to stitch school uniforms.",
            "Opening for chef in a restaurant near station."
        ]
        
        if st.sidebar.button("Load Sample Jobs"):
            # Get job classifications
            predictions = st.session_state.classifier.predict_batch(sample_jobs)
            df = pd.DataFrame(predictions)
            
            # Get scam detection results
            scam_results = st.session_state.scam_detector.predict_batch(sample_jobs)
            df['is_scam'] = [result['is_scam'] for result in scam_results]
            df['scam_probability'] = [result['scam_probability'] for result in scam_results]
            df['suspicious_features'] = [result['features'] for result in scam_results]
            
            st.session_state.jobs_df = df
            st.session_state.jobs_df['timestamp'] = datetime.now()

    elif input_method == "Enter Job Description":
        job_text = st.sidebar.text_area("Enter job description:", height=150)
        if st.sidebar.button("Analyze"):
            if job_text:
                processor = TextProcessor()
                job_info = processor.process_job_description(job_text)
                
                # Access summary
                print(job_info["summary"])
                
                # Access key points
                for point in job_info["key_points"]:
                    print(f"- {point}")
                
                # Access text statistics
                print(f"Compression ratio: {job_info['text_stats']['compression_ratio']:.2%}")
                
                # Get job classification
                prediction = st.session_state.classifier.predict(job_text)
                
                # Get scam detection
                scam_result = st.session_state.scam_detector.predict(job_text)
                
                # Combine results
                result = {
                    **prediction,
                    'is_scam': scam_result['is_scam'],
                    'scam_probability': scam_result['scam_probability'],
                    'suspicious_features': scam_result['features']
                }
                
                st.session_state.jobs_df = pd.DataFrame([result])
                st.session_state.jobs_df['timestamp'] = datetime.now()

    elif input_method == "Upload JSON":
        uploaded_file = st.sidebar.file_uploader("Upload JSON file with job descriptions", type=['json'])
        if uploaded_file is not None:
            try:
                jobs = json.load(uploaded_file)
                if isinstance(jobs, list):
                    # Get job classifications
                    predictions = st.session_state.classifier.predict_batch(jobs)
                    df = pd.DataFrame(predictions)
                    
                    # Get scam detection results
                    scam_results = st.session_state.scam_detector.predict_batch(jobs)
                    df['is_scam'] = [result['is_scam'] for result in scam_results]
                    df['scam_probability'] = [result['scam_probability'] for result in scam_results]
                    df['suspicious_features'] = [result['features'] for result in scam_results]
                    
                    st.session_state.jobs_df = df
                    st.session_state.jobs_df['timestamp'] = datetime.now()
                else:
                    st.error("Invalid JSON format. Please upload a list of job descriptions.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # Display results
    if not st.session_state.jobs_df.empty:
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            job_type = st.selectbox(
                "Filter by Job Category",
                ["All"] + sorted(st.session_state.jobs_df["category"].unique().tolist())
            )
        
        with col2:
            min_confidence = st.slider(
                "Minimum Classification Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1
            )
        
        with col3:
            max_scam_prob = st.slider(
                "Maximum Scam Probability",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1
            )
        
        # Apply filters
        filtered_df = st.session_state.jobs_df.copy()
        if job_type != "All":
            filtered_df = filtered_df[filtered_df["category"] == job_type]
        filtered_df = filtered_df[filtered_df["confidence"] >= min_confidence]
        filtered_df = filtered_df[filtered_df["scam_probability"] <= max_scam_prob]
        
        # Display results
        st.subheader(f"Found {len(filtered_df)} jobs")
        
        # Show detailed view
        for _, row in filtered_df.iterrows():
            # Create expander title with warning if scam
            title = f"{row['category'].title()} (Confidence: {row['confidence']:.2%})"
            if row['is_scam']:
                title = f"⚠️ {title} - Potential Scam!"
            
            with st.expander(title):
                st.write("**Job Description:**")
                st.write(row['text'])
                
                # Show scam detection details
                st.write("**Scam Analysis:**")
                scam_prob = row['scam_probability']
                color = "red" if scam_prob > 0.7 else "orange" if scam_prob > 0.3 else "green"
                st.markdown(f"Scam Probability: <span style='color:{color}'>{scam_prob:.1%}</span>", unsafe_allow_html=True)
                
                # Show suspicious features
                features = row['suspicious_features']
                if features:
                    st.write("**Suspicious Indicators:**")
                    for feature, value in features.items():
                        if value and feature != 'length':
                            st.write(f"- {feature.replace('_', ' ').title()}")
                
                st.write("**Timestamp:**", row['timestamp'].strftime("%Y-%m-%d %H:%M:%S"))
        
        # Download option
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="analyzed_jobs.csv",
            mime="text/csv"
        )
    else:
        st.info("👆 Use the sidebar to input job descriptions and get started!")

with tab2:
    st.header("🤖 Automated Job Search")
    st.markdown("""
    The AI agent will automatically search for jobs, classify them, and check for potential scams.
    """)
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        job_site = st.text_input("Job Site URL", "https://example.com/jobs")
        num_episodes = st.number_input("Number of Search Episodes", min_value=1, max_value=100, value=10)
    
    with col2:
        target_categories = st.multiselect(
            "Target Job Categories",
            st.session_state.classifier.labels,
            default=["driver", "electrician", "plumber"]
        )
        max_scam_probability = st.slider(
            "Maximum Acceptable Scam Probability",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1
        )
    
    # Start/Stop buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Search", disabled=st.session_state.is_searching):
            st.session_state.is_searching = True
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Train the agent
                status_text.text("Training agent...")
                st.session_state.search_agent.train(num_episodes=num_episodes)
                
                # Start searching
                status_text.text("Searching for jobs...")
                for episode in range(num_episodes):
                    # Update progress
                    progress = (episode + 1) / num_episodes
                    progress_bar.progress(progress)
                    
                    # Simulate job search (replace with actual search logic)
                    time.sleep(1)  # Simulate work
                    
                    # Update status
                    status_text.text(f"Episode {episode + 1}/{num_episodes} completed")
                
                st.success("Job search completed!")
                
            except Exception as e:
                st.error(f"Error during job search: {str(e)}")
            finally:
                st.session_state.is_searching = False
    
    with col2:
        if st.button("Stop Search", disabled=not st.session_state.is_searching):
            st.session_state.is_searching = False
            st.warning("Search stopped by user")
    
    # Display search results
    if not st.session_state.jobs_df.empty:
        st.subheader("Search Results")
        st.dataframe(st.session_state.jobs_df)

with tab3:
    st.header("📊 Data Monitoring")
    st.markdown("""
    Monitor data quality and drift in job listings over time.
    """)
    
    # Data monitoring controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Update Reference Data"):
            if not st.session_state.jobs_df.empty:
                st.session_state.data_monitor.update_reference_data(st.session_state.jobs_df)
                st.success("Reference data updated!")
            else:
                st.warning("No data available to set as reference.")
    
    with col2:
        if st.button("Generate Drift Report"):
            if not st.session_state.jobs_df.empty:
                try:
                    report_path = st.session_state.data_monitor.generate_drift_report(
                        st.session_state.jobs_df
                    )
                    st.success(f"Drift report generated! Saved to {report_path}")
                    
                    # Display report in iframe
                    with open(report_path, 'r') as f:
                        report_html = f.read()
                    st.components.v1.html(report_html, height=800, scrolling=True)
                except Exception as e:
                    st.error(f"Error generating drift report: {str(e)}")
            else:
                st.warning("No data available for drift analysis.")
    
    # Display drift summary
    if not st.session_state.jobs_df.empty:
        st.subheader("Data Drift Summary")
        drift_summary = st.session_state.data_monitor.get_drift_summary(st.session_state.jobs_df)
        
        # Dataset drift status
        drift_status = "⚠️ Detected" if drift_summary['dataset_drift'] else "✅ No Drift"
        st.markdown(f"**Dataset Drift Status:** {drift_status}")
        
        # Drift columns
        if drift_summary['drift_columns']:
            st.markdown("**Columns with Drift:**")
            for column in drift_summary['drift_columns']:
                score = drift_summary['drift_scores'].get(column, 0)
                st.markdown(f"- {column}: {score:.2%}")
        
        # Data quality metrics
        st.markdown("**Data Quality Metrics:**")
        quality = drift_summary['data_quality']
        st.markdown(f"""
        - Current Rows: {quality['current_rows']}
        - Reference Rows: {quality['reference_rows']}
        - Missing Values: {sum(quality['missing_values'].values())}
        """)
        
        # Visualize drift scores
        if drift_summary['drift_scores']:
            st.subheader("Drift Scores by Column")
            drift_df = pd.DataFrame({
                'Column': list(drift_summary['drift_scores'].keys()),
                'Drift Score': list(drift_summary['drift_scores'].values())
            })
            st.bar_chart(drift_df.set_index('Column'))
    else:
        st.info("👆 Use the Job Analysis tab to process some jobs first!") 