#!/usr/bin/env python3
"""
TEMPO Data Preprocessing Pipeline for AI Model Training

This script prepares TEMPO NO₂ data for machine learning model training.
It handles data cleaning, feature engineering, and dataset splitting.

Author: NASA Apps Challenge
Date: 2024
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class TEMPODataPreprocessor:
    """
    Preprocesses TEMPO data for AI model training.
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.targets = None
        
    def load_tempo_data(self, file_pattern="*TEMPO*NO2*.nc"):
        """
        Load TEMPO NetCDF files and combine them.
        
        Args:
            file_pattern (str): Pattern to match TEMPO files
            
        Returns:
            pandas.DataFrame: Combined TEMPO data
        """
        print("Loading TEMPO data...")
        
        tempo_files = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.nc') and 'TEMPO' in file.upper() and 'NO2' in file.upper():
                tempo_files.append(os.path.join(self.data_dir, file))
        
        if not tempo_files:
            print(f"⚠ No TEMPO NetCDF files found in {self.data_dir}")
            print("Please download TEMPO NO₂ Level 3 data first!")
            return None
        
        print(f"Found {len(tempo_files)} TEMPO files")
        
        # Load and combine all files
        all_data = []
        for file_path in tempo_files:
            try:
                dataset = xr.open_dataset(file_path)
                
                # Extract key variables
                if 'NO2' in dataset.data_vars:
                    no2_data = dataset['NO2']
                    
                    # Get coordinates
                    if 'latitude' in dataset.coords:
                        lat = dataset['latitude']
                    elif 'lat' in dataset.coords:
                        lat = dataset['lat']
                    else:
                        continue
                        
                    if 'longitude' in dataset.coords:
                        lon = dataset['longitude']
                    elif 'lon' in dataset.coords:
                        lon = dataset['lon']
                    else:
                        continue
                    
                    # Get time
                    if 'time' in dataset.coords:
                        time = dataset['time']
                    else:
                        continue
                    
                    # Create meshgrid for coordinates
                    lon_mesh, lat_mesh = np.meshgrid(lon.values, lat.values)
                    
                    # Flatten data
                    df = pd.DataFrame({
                        'NO2': no2_data.values.flatten(),
                        'latitude': lat_mesh.flatten(),
                        'longitude': lon_mesh.flatten(),
                        'time': np.repeat(time.values, len(lat_mesh.flatten())),
                        'source_file': os.path.basename(file_path)
                    })
                    
                    # Remove NaN values
                    df = df.dropna()
                    all_data.append(df)
                    
                    print(f"✓ Loaded {len(df)} data points from {os.path.basename(file_path)}")
                
            except Exception as e:
                print(f"✗ Error loading {file_path}: {e}")
                continue
        
        if all_data:
            self.raw_data = pd.concat(all_data, ignore_index=True)
            print(f"✓ Total data points loaded: {len(self.raw_data)}")
            return self.raw_data
        else:
            print("✗ No data could be loaded")
            return None
    
    def create_temporal_features(self, df):
        """
        Create temporal features from time data.
        
        Args:
            df (pandas.DataFrame): Data with time column
            
        Returns:
            pandas.DataFrame: Data with temporal features
        """
        print("Creating temporal features...")
        
        # Convert time to datetime
        df['datetime'] = pd.to_datetime(df['time'])
        
        # Extract temporal features
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Seasonal features
        df['season'] = df['month'].map({
            12: 0, 1: 0, 2: 0,  # Winter
            3: 1, 4: 1, 5: 1,   # Spring
            6: 2, 7: 2, 8: 2,   # Summer
            9: 3, 10: 3, 11: 3   # Fall
        })
        
        print(f"✓ Created temporal features")
        return df
    
    def create_spatial_features(self, df):
        """
        Create spatial features from latitude/longitude.
        
        Args:
            df (pandas.DataFrame): Data with lat/lon columns
            
        Returns:
            pandas.DataFrame: Data with spatial features
        """
        print("Creating spatial features...")
        
        # Distance from center (assuming North America focus)
        center_lat, center_lon = 40.0, -100.0  # Center of US
        df['distance_from_center'] = np.sqrt(
            (df['latitude'] - center_lat)**2 + (df['longitude'] - center_lon)**2
        )
        
        # Regional features
        df['is_east_coast'] = (df['longitude'] > -80).astype(int)
        df['is_west_coast'] = (df['longitude'] < -120).astype(int)
        df['is_northern'] = (df['latitude'] > 40).astype(int)
        df['is_southern'] = (df['latitude'] < 30).astype(int)
        
        print(f"✓ Created spatial features")
        return df
    
    def create_weather_features(self, df):
        """
        Create weather-related features (placeholder for external data).
        
        Args:
            df (pandas.DataFrame): Input data
            
        Returns:
            pandas.DataFrame: Data with weather features
        """
        print("Creating weather features...")
        
        # Placeholder weather features (would need external weather data)
        # For now, create synthetic features based on location and time
        np.random.seed(42)  # For reproducible results
        
        df['temperature'] = 20 + 10 * np.sin(df['day_of_year'] * 2 * np.pi / 365) + \
                            np.random.normal(0, 5, len(df))
        
        df['wind_speed'] = 10 + 5 * np.random.random(len(df))
        df['humidity'] = 50 + 20 * np.random.random(len(df))
        df['pressure'] = 1013 + 20 * np.random.random(len(df))
        
        print(f"✓ Created weather features (synthetic)")
        return df
    
    def prepare_features_and_targets(self, df, target_column='NO2'):
        """
        Prepare features and targets for machine learning.
        
        Args:
            df (pandas.DataFrame): Processed data
            target_column (str): Column to use as target
            
        Returns:
            tuple: (features, targets)
        """
        print("Preparing features and targets...")
        
        # Define feature columns
        feature_columns = [
            'latitude', 'longitude', 'year', 'month', 'day', 'hour',
            'day_of_year', 'day_of_week', 'is_weekend', 'season',
            'distance_from_center', 'is_east_coast', 'is_west_coast',
            'is_northern', 'is_southern', 'temperature', 'wind_speed',
            'humidity', 'pressure'
        ]
        
        # Select available features
        available_features = [col for col in feature_columns if col in df.columns]
        
        self.features = df[available_features].copy()
        self.targets = df[target_column].copy()
        
        print(f"✓ Features: {len(available_features)} columns")
        print(f"✓ Targets: {len(self.targets)} samples")
        print(f"✓ Feature columns: {available_features}")
        
        return self.features, self.targets
    
    def split_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """
        Split data into train/validation/test sets.
        
        Args:
            test_size (float): Proportion for test set
            val_size (float): Proportion for validation set
            random_state (int): Random seed
            
        Returns:
            dict: Split datasets
        """
        print("Splitting data into train/validation/test sets...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.features, self.targets, test_size=test_size, random_state=random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        splits = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        }
        
        print(f"✓ Train set: {len(X_train)} samples")
        print(f"✓ Validation set: {len(X_val)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        
        return splits
    
    def scale_features(self, splits, scaler_type='standard'):
        """
        Scale features using specified scaler.
        
        Args:
            splits (dict): Data splits
            scaler_type (str): Type of scaler ('standard' or 'minmax')
            
        Returns:
            dict: Scaled data splits
        """
        print(f"Scaling features using {scaler_type} scaler...")
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        
        # Fit scaler on training data
        X_train_scaled = scaler.fit_transform(splits['X_train'])
        X_val_scaled = scaler.transform(splits['X_val'])
        X_test_scaled = scaler.transform(splits['X_test'])
        
        scaled_splits = splits.copy()
        scaled_splits['X_train'] = X_train_scaled
        scaled_splits['X_val'] = X_val_scaled
        scaled_splits['X_test'] = X_test_scaled
        scaled_splits['scaler'] = scaler
        
        print(f"✓ Features scaled successfully")
        return scaled_splits
    
    def analyze_data_quality(self, df):
        """
        Analyze data quality and provide statistics.
        
        Args:
            df (pandas.DataFrame): Data to analyze
            
        Returns:
            dict: Data quality statistics
        """
        print("Analyzing data quality...")
        
        quality_stats = {
            'total_samples': len(df),
            'missing_values': df.isnull().sum().sum(),
            'no2_stats': {
                'mean': df['NO2'].mean(),
                'std': df['NO2'].std(),
                'min': df['NO2'].min(),
                'max': df['NO2'].max(),
                'median': df['NO2'].median()
            },
            'spatial_coverage': {
                'lat_range': (df['latitude'].min(), df['latitude'].max()),
                'lon_range': (df['longitude'].min(), df['longitude'].max())
            },
            'temporal_coverage': {
                'date_range': (df['datetime'].min(), df['datetime'].max()),
                'unique_dates': df['datetime'].nunique()
            }
        }
        
        print(f"✓ Data quality analysis complete")
        print(f"  - Total samples: {quality_stats['total_samples']:,}")
        print(f"  - Missing values: {quality_stats['missing_values']}")
        print(f"  - NO₂ range: {quality_stats['no2_stats']['min']:.3f} to {quality_stats['no2_stats']['max']:.3f}")
        print(f"  - Date range: {quality_stats['temporal_coverage']['date_range'][0]} to {quality_stats['temporal_coverage']['date_range'][1]}")
        
        return quality_stats


def main():
    """
    Main function to demonstrate data preprocessing pipeline.
    """
    print("TEMPO Data Preprocessing Pipeline for AI Training")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = TEMPODataPreprocessor()
    
    # Load data
    raw_data = preprocessor.load_tempo_data()
    
    if raw_data is None:
        print("\n⚠ No TEMPO data found!")
        print("Please download TEMPO NO₂ Level 3 NetCDF files first.")
        print("Visit: https://tempo.si.edu/data/level3/tempo_no2_l3/")
        return
    
    # Process data
    processed_data = raw_data.copy()
    processed_data = preprocessor.create_temporal_features(processed_data)
    processed_data = preprocessor.create_spatial_features(processed_data)
    processed_data = preprocessor.create_weather_features(processed_data)
    
    # Analyze data quality
    quality_stats = preprocessor.analyze_data_quality(processed_data)
    
    # Prepare features and targets
    features, targets = preprocessor.prepare_features_and_targets(processed_data)
    
    # Split data
    splits = preprocessor.split_data()
    
    # Scale features
    scaled_splits = preprocessor.scale_features(splits)
    
    print(f"\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"✓ Data ready for AI model training")
    print(f"✓ Features: {scaled_splits['X_train'].shape[1]} dimensions")
    print(f"✓ Training samples: {scaled_splits['X_train'].shape[0]:,}")
    print(f"✓ Validation samples: {scaled_splits['X_val'].shape[0]:,}")
    print(f"✓ Test samples: {scaled_splits['X_test'].shape[0]:,}")
    
    print(f"\nNext steps:")
    print(f"1. Choose AI model type (regression, classification, time series)")
    print(f"2. Implement model architecture")
    print(f"3. Train and validate model")
    print(f"4. Evaluate performance")


if __name__ == "__main__":
    main()
