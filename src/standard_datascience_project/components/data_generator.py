import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple
from ..entity.config_entity import DataGenerationConfig

class SyntheticDataGenerator:
    """Component for generating synthetic web authentication logs with anomalies"""
    
    def __init__(self, config: DataGenerationConfig):
        self.config = config
        np.random.seed(config.seed)
        
    def generate_data(self) -> pd.DataFrame:
        """Generate synthetic web authentication logs with anomalies"""
        
        # Generate user IDs
        user_ids = np.random.choice(
            range(1, self.config.num_users + 1), 
            self.config.num_records
        )
        
        # Generate timestamps over the last year
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=365), 
            periods=self.config.num_records, 
            freq='min'
        )
        
        # Simulating Login Successes
        login_statuses = np.random.choice(
            ['Success', 'Fail'], 
            self.config.num_records, 
            p=[self.config.success_login_rate, 1 - self.config.success_login_rate]
        )
        
        # Generate IP addresses
        common_ips = [f'192.168.1.{i}' for i in np.random.randint(10, 100, 10)]
        uncommon_ips = [f'10.10.10.{i}' for i in np.random.randint(1, 10, 3)]
        total_ips = common_ips + uncommon_ips
        probabilities = [(1 - self.config.uncommon_ip_prob) / len(common_ips)] * len(common_ips) + \
                       [self.config.uncommon_ip_prob / len(uncommon_ips)] * len(uncommon_ips)
        ip_addresses = np.random.choice(total_ips, self.config.num_records, p=probabilities)
        
        # Define possible device types
        device_types = ['Desktop', 'Mobile']
        devices = np.random.choice(device_types, self.config.num_records)
        
        # Define normal Locations
        locations = np.random.choice(
            ['Jodhpur', 'Jaipur', 'Udaipur', 'Kota', 'Bangalore', 'Andra Pradesh'], 
            self.config.num_records, 
            p=[0.24, 0.24, 0.24, 0.24, 0.02, 0.02]
        )
        
        # Generate session durations and behavioral scores
        session_durations = np.where(
            login_statuses == 'Success', 
            np.random.exponential(scale=300, size=self.config.num_records), 
            0
        )
        failed_attempts = np.where(
            login_statuses == 'Fail', 
            np.random.poisson(lam=2, size=self.config.num_records), 
            0
        )
        behavioral_scores = np.where(
            login_statuses == 'Success', 
            np.random.normal(loc=80, scale=10, size=self.config.num_records), 
            np.random.normal(loc=50, scale=10, size=self.config.num_records)
        )
        
        # Define anomalies based on specific conditions
        anomaly_conditions = (
            (failed_attempts > self.config.failed_attempt_limit) |
            (behavioral_scores < self.config.behavioral_score_low) |
            (behavioral_scores > self.config.behavioral_score_high) |
            (session_durations > 3 * np.mean(session_durations)) |
            (session_durations < 5) |
            (np.isin(ip_addresses, uncommon_ips)) |
            (np.isin(locations, ['Bangalore', 'Andra Pradesh']))
        )
        anomaly_labels = anomaly_conditions.astype(int)
        
        # Compile all features into a DataFrame
        df = pd.DataFrame({
            'User ID': user_ids,
            'Timestamp': timestamps,
            'Login Status': login_statuses,
            'IP Address': ip_addresses,
            'Device Type': devices,
            'Location': locations,
            'Session Duration': session_durations,
            'Failed Attempts': failed_attempts,
            'Behavioral Score': behavioral_scores,
            'Anomaly': anomaly_labels
        })
        
        return df
    
    def save_data(self, df: pd.DataFrame, file_path: str = None) -> str:
        """Save the generated data to CSV file"""
        if file_path is None:
            os.makedirs('data', exist_ok=True)
            file_path = os.path.join('data', 'synthetic_web_auth_logs.csv')
        
        df.to_csv(file_path, index=False)
        print(f"Data saved successfully to '{file_path}'.")
        return file_path 