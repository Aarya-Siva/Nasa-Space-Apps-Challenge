#!/usr/bin/env python3
"""
Synthetic TEMPO Data Generator

Creates realistic TEMPO NO₂ data for immediate AI model development.
This allows us to build and test models without waiting for data downloads.

Author: NASA Apps Challenge
Date: 2024
"""

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import os


def generate_synthetic_tempo_data(n_days=30, spatial_resolution=0.1):
    """
    Generate synthetic TEMPO NO₂ data that mimics real satellite observations.
    
    Args:
        n_days (int): Number of days to generate
        spatial_resolution (float): Spatial resolution in degrees
        
    Returns:
        pandas.DataFrame: Synthetic TEMPO data
    """
    print("Generating synthetic TEMPO NO₂ data...")
    
    # Define spatial domain (North America)
    lat_min, lat_max = 25.0, 50.0
    lon_min, lon_max = -125.0, -65.0
    
    # Create spatial grid
    lats = np.arange(lat_min, lat_max + spatial_resolution, spatial_resolution)
    lons = np.arange(lon_min, lon_max + spatial_resolution, spatial_resolution)
    
    # Create time series
    start_date = datetime(2023, 8, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    print(f"✓ Spatial grid: {len(lats)} x {len(lons)} = {len(lats) * len(lons):,} points")
    print(f"✓ Time series: {n_days} days")
    print(f"✓ Total data points: {len(lats) * len(lons) * n_days:,}")
    
    # Generate realistic NO₂ patterns
    all_data = []
    
    for i, date in enumerate(dates):
        print(f"Generating data for day {i+1}/{n_days}: {date.strftime('%Y-%m-%d')}")
        
        # Create base NO₂ field with realistic patterns
        lon_mesh, lat_mesh = np.meshgrid(lons, lats)
        
        # Base NO₂ levels (molecules/cm²)
        base_no2 = np.zeros_like(lat_mesh)
        
        # Add urban pollution hotspots
        urban_centers = [
            (40.7128, -74.0060, 2.5e15),  # NYC
            (34.0522, -118.2437, 2.2e15),  # LA
            (41.8781, -87.6298, 2.0e15),  # Chicago
            (29.7604, -95.3698, 1.8e15),  # Houston
            (33.4484, -112.0740, 1.6e15), # Phoenix
            (39.9526, -75.1652, 1.5e15), # Philadelphia
            (32.7767, -96.7970, 1.4e15), # Dallas
            (25.7617, -80.1918, 1.3e15), # Miami
        ]
        
        for city_lat, city_lon, intensity in urban_centers:
            distance = np.sqrt((lat_mesh - city_lat)**2 + (lon_mesh - city_lon)**2)
            influence = intensity * np.exp(-distance / 2.0)  # Exponential decay
            base_no2 += influence
        
        # Add seasonal variation
        day_of_year = date.timetuple().tm_yday
        seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        base_no2 *= seasonal_factor
        
        # Add weekly patterns (weekends have lower emissions)
        day_of_week = date.weekday()
        if day_of_week >= 5:  # Weekend
            weekly_factor = 0.8
        else:  # Weekday
            weekly_factor = 1.0
        base_no2 *= weekly_factor
        
        # Add random noise
        noise = np.random.normal(0, 0.1e15, base_no2.shape)
        base_no2 += noise
        
        # Add weather effects (simplified)
        # Higher NO₂ in stable atmospheric conditions
        weather_factor = 1.0 + 0.2 * np.random.random()
        base_no2 *= weather_factor
        
        # Ensure positive values
        base_no2 = np.maximum(base_no2, 0.1e15)
        
        # Create DataFrame for this day
        day_data = pd.DataFrame({
            'latitude': lat_mesh.flatten(),
            'longitude': lon_mesh.flatten(),
            'NO2': base_no2.flatten(),
            'date': date,
            'day_of_year': day_of_year,
            'day_of_week': day_of_week,
            'is_weekend': 1 if day_of_week >= 5 else 0
        })
        
        all_data.append(day_data)
    
    # Combine all days
    combined_data = pd.concat(all_data, ignore_index=True)
    
    print(f"✓ Generated {len(combined_data):,} synthetic data points")
    print(f"✓ NO₂ range: {combined_data['NO2'].min():.2e} to {combined_data['NO2'].max():.2e}")
    
    return combined_data


def save_synthetic_data(data, filename="synthetic_tempo_data.csv"):
    """
    Save synthetic data to CSV file.
    
    Args:
        data (pandas.DataFrame): Synthetic data
        filename (str): Output filename
    """
    print(f"Saving synthetic data to {filename}...")
    data.to_csv(filename, index=False)
    print(f"✓ Data saved successfully!")
    return filename


def create_netcdf_file(data, filename="synthetic_tempo_data.nc"):
    """
    Create a NetCDF file from synthetic data (optional).
    
    Args:
        data (pandas.DataFrame): Synthetic data
        filename (str): Output filename
    """
    print(f"Creating NetCDF file: {filename}...")
    
    # Get unique coordinates
    lats = sorted(data['latitude'].unique())
    lons = sorted(data['longitude'].unique())
    dates = sorted(data['date'].unique())
    
    # Create coordinate arrays
    lat_array = np.array(lats)
    lon_array = np.array(lons)
    time_array = np.array(dates)
    
    # Reshape NO₂ data
    no2_data = np.zeros((len(dates), len(lats), len(lons)))
    
    for i, date in enumerate(dates):
        day_data = data[data['date'] == date]
        for _, row in day_data.iterrows():
            lat_idx = np.where(lat_array == row['latitude'])[0][0]
            lon_idx = np.where(lon_array == row['longitude'])[0][0]
            no2_data[i, lat_idx, lon_idx] = row['NO2']
    
    # Create xarray Dataset
    ds = xr.Dataset({
        'NO2': (['time', 'latitude', 'longitude'], no2_data)
    }, coords={
        'time': time_array,
        'latitude': lat_array,
        'longitude': lon_array
    })
    
    # Add attributes
    ds['NO2'].attrs = {
        'long_name': 'Nitrogen Dioxide Column',
        'units': 'molecules/cm^2',
        'description': 'Synthetic TEMPO NO₂ data for AI model development'
    }
    
    ds.attrs = {
        'title': 'Synthetic TEMPO NO₂ Data',
        'description': 'Generated synthetic data mimicking TEMPO satellite observations',
        'created': datetime.now().isoformat(),
        'spatial_resolution': '0.1 degrees',
        'temporal_resolution': 'daily'
    }
    
    # Save to NetCDF
    ds.to_netcdf(filename)
    print(f"✓ NetCDF file created: {filename}")
    
    return filename


def main():
    """
    Main function to generate synthetic TEMPO data.
    """
    print("Synthetic TEMPO Data Generator")
    print("="*50)
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_tempo_data(n_days=30, spatial_resolution=0.1)
    
    # Save as CSV
    csv_file = save_synthetic_data(synthetic_data)
    
    # Create NetCDF file
    nc_file = create_netcdf_file(synthetic_data)
    
    print(f"\n" + "="*50)
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("="*50)
    print(f"✓ CSV file: {csv_file}")
    print(f"✓ NetCDF file: {nc_file}")
    print(f"✓ Data points: {len(synthetic_data):,}")
    print(f"✓ Ready for AI model development!")
    
    return synthetic_data


if __name__ == "__main__":
    main()
