import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO)

class DataMonitor:
    def __init__(self, reference_data: pd.DataFrame = None):
        """Initialize the data monitor with reference data"""
        self.reference_data = reference_data
        self.reports_dir = "monitoring_reports"
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def prepare_job_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare job data for drift analysis"""
        # Convert text columns to numerical features
        processed_df = df.copy()
        
        # Add text length as a feature
        processed_df['text_length'] = processed_df['text'].str.len()
        
        # Add word count
        processed_df['word_count'] = processed_df['text'].str.split().str.len()
        
        # Add numerical features from text_stats if available
        if 'text_stats' in processed_df.columns:
            processed_df['compression_ratio'] = processed_df['text_stats'].apply(
                lambda x: x.get('compression_ratio', 0) if isinstance(x, dict) else 0
            )
        
        # Convert categorical columns to numerical
        if 'category' in processed_df.columns:
            processed_df['category_encoded'] = pd.Categorical(processed_df['category']).codes
        
        # Select numerical columns for drift analysis
        numerical_columns = [
            'text_length', 'word_count', 'compression_ratio',
            'confidence', 'scam_probability', 'category_encoded'
        ]
        
        return processed_df[numerical_columns].fillna(0)
    
    def generate_drift_report(self, current_data: pd.DataFrame, 
                            report_name: str = None) -> str:
        """Generate drift report comparing current data with reference data"""
        try:
            if self.reference_data is None:
                logging.warning("No reference data available. Using current data as reference.")
                self.reference_data = current_data
            
            # Prepare data for analysis
            reference_processed = self.prepare_job_data(self.reference_data)
            current_processed = self.prepare_job_data(current_data)
            
            # Create report
            report = Report(metrics=[
                DataDriftPreset(),
                DataQualityPreset(),
                DatasetDriftMetric(),
                ColumnDriftMetric(column_name='text_length'),
                ColumnDriftMetric(column_name='word_count'),
                ColumnDriftMetric(column_name='scam_probability')
            ])
            
            # Run report
            report.run(
                current_data=current_processed,
                reference_data=reference_processed
            )
            
            # Generate report filename
            if report_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_name = f"drift_report_{timestamp}.html"
            
            report_path = os.path.join(self.reports_dir, report_name)
            
            # Save report
            report.save_html(report_path)
            logging.info(f"Drift report saved to {report_path}")
            
            return report_path
            
        except Exception as e:
            logging.error(f"Error generating drift report: {str(e)}")
            raise
    
    def update_reference_data(self, new_reference: pd.DataFrame):
        """Update the reference dataset"""
        self.reference_data = new_reference
        logging.info("Reference data updated")
    
    def get_drift_summary(self, current_data: pd.DataFrame) -> dict:
        """Get a summary of data drift metrics"""
        try:
            # Prepare data
            reference_processed = self.prepare_job_data(self.reference_data)
            current_processed = self.prepare_job_data(current_data)
            
            # Calculate basic drift metrics
            drift_summary = {
                'dataset_drift': False,
                'drift_columns': [],
                'drift_scores': {},
                'data_quality': {}
            }
            
            # Check for dataset drift
            dataset_drift = DatasetDriftMetric()
            dataset_drift.calculate(
                current_data=current_processed,
                reference_data=reference_processed
            )
            drift_summary['dataset_drift'] = dataset_drift.get_result().dataset_drift
            
            # Check column drift
            for column in current_processed.columns:
                if column in reference_processed.columns:
                    drift_metric = ColumnDriftMetric(column_name=column)
                    drift_metric.calculate(
                        current_data=current_processed,
                        reference_data=reference_processed
                    )
                    result = drift_metric.get_result()
                    
                    if result.drift_detected:
                        drift_summary['drift_columns'].append(column)
                        drift_summary['drift_scores'][column] = result.drift_score
            
            # Add basic data quality metrics
            drift_summary['data_quality'] = {
                'current_rows': len(current_processed),
                'reference_rows': len(reference_processed),
                'missing_values': current_processed.isnull().sum().to_dict(),
                'column_types': current_processed.dtypes.astype(str).to_dict()
            }
            
            return drift_summary
            
        except Exception as e:
            logging.error(f"Error calculating drift summary: {str(e)}")
            return {
                'error': str(e),
                'dataset_drift': False,
                'drift_columns': [],
                'drift_scores': {},
                'data_quality': {}
            } 