#!/usr/bin/env python3
"""
TEMPO Level 3 NO₂ Data Processor

This script loads TEMPO Level 3 NO₂ NetCDF files and converts them into
a flattened pandas DataFrame for analysis.

Author: NASA Apps Challenge
Date: 2024
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_tempo_no2_file(file_path):
    """
    Load a TEMPO Level 3 NO₂ NetCDF file using xarray.
    
    Args:
        file_path (str): Path to the NetCDF file
        
    Returns:
        xarray.Dataset: The loaded dataset
    """
    try:
        print(f"Loading TEMPO NO₂ file: {file_path}")
        dataset = xr.open_dataset(file_path)
        return dataset
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def print_dataset_structure(dataset):
    """
    Print the structure of the dataset including variables, dimensions, and coordinates.
    
    Args:
        dataset (xarray.Dataset): The dataset to analyze
    """
    print("\n" + "="*60)
    print("DATASET STRUCTURE")
    print("="*60)
    
    print(f"\nDataset dimensions:")
    for dim, size in dataset.dims.items():
        print(f"  {dim}: {size}")
    
    print(f"\nDataset coordinates:")
    for coord in dataset.coords:
        coord_info = dataset[coord]
        print(f"  {coord}: {coord_info.shape} - {coord_info.dtype}")
        if hasattr(coord_info, 'attrs') and coord_info.attrs:
            print(f"    Attributes: {coord_info.attrs}")
    
    print(f"\nDataset variables:")
    for var in dataset.data_vars:
        var_info = dataset[var]
        print(f"  {var}: {var_info.shape} - {var_info.dtype}")
        if hasattr(var_info, 'attrs') and var_info.attrs:
            print(f"    Attributes: {var_info.attrs}")
    
    print(f"\nDataset attributes:")
    if hasattr(dataset, 'attrs') and dataset.attrs:
        for attr, value in dataset.attrs.items():
            print(f"  {attr}: {value}")


def extract_key_variables(dataset):
    """
    Extract key variables from the TEMPO dataset.
    
    Args:
        dataset (xarray.Dataset): The loaded dataset
        
    Returns:
        dict: Dictionary containing the extracted variables
    """
    print("\n" + "="*60)
    print("EXTRACTING KEY VARIABLES")
    print("="*60)
    
    extracted_vars = {}
    
    # Extract NO2 data
    if 'NO2' in dataset.data_vars:
        extracted_vars['NO2'] = dataset['NO2']
        print(f"✓ Extracted NO2: {dataset['NO2'].shape}")
    else:
        print("⚠ NO2 variable not found. Available variables:")
        for var in dataset.data_vars:
            print(f"  - {var}")
    
    # Extract latitude
    if 'latitude' in dataset.coords:
        extracted_vars['latitude'] = dataset['latitude']
        print(f"✓ Extracted latitude: {dataset['latitude'].shape}")
    elif 'lat' in dataset.coords:
        extracted_vars['latitude'] = dataset['lat']
        print(f"✓ Extracted latitude (as 'lat'): {dataset['lat'].shape}")
    else:
        print("⚠ Latitude coordinate not found. Available coordinates:")
        for coord in dataset.coords:
            print(f"  - {coord}")
    
    # Extract longitude
    if 'longitude' in dataset.coords:
        extracted_vars['longitude'] = dataset['longitude']
        print(f"✓ Extracted longitude: {dataset['longitude'].shape}")
    elif 'lon' in dataset.coords:
        extracted_vars['longitude'] = dataset['lon']
        print(f"✓ Extracted longitude (as 'lon'): {dataset['lon'].shape}")
    else:
        print("⚠ Longitude coordinate not found. Available coordinates:")
        for coord in dataset.coords:
            print(f"  - {coord}")
    
    # Extract time
    if 'time' in dataset.coords:
        extracted_vars['time'] = dataset['time']
        print(f"✓ Extracted time: {dataset['time'].shape}")
    else:
        print("⚠ Time coordinate not found. Available coordinates:")
        for coord in dataset.coords:
            print(f"  - {coord}")
    
    return extracted_vars


def create_flattened_dataframe(extracted_vars):
    """
    Convert the extracted variables into a flattened pandas DataFrame.
    
    Args:
        extracted_vars (dict): Dictionary containing the extracted variables
        
    Returns:
        pandas.DataFrame: Flattened DataFrame with NO2, latitude, longitude, and time
    """
    print("\n" + "="*60)
    print("CREATING FLATTENED DATAFRAME")
    print("="*60)
    
    # Check if we have all required variables
    required_vars = ['NO2', 'latitude', 'longitude', 'time']
    missing_vars = [var for var in required_vars if var not in extracted_vars]
    
    if missing_vars:
        print(f"⚠ Missing required variables: {missing_vars}")
        print("Creating DataFrame with available variables only.")
    
    # Create coordinate meshgrids
    data_dict = {}
    
    # Handle NO2 data
    if 'NO2' in extracted_vars:
        no2_data = extracted_vars['NO2']
        print(f"NO2 data shape: {no2_data.shape}")
        
        # Flatten NO2 data
        data_dict['NO2'] = no2_data.values.flatten()
        print(f"Flattened NO2 data length: {len(data_dict['NO2'])}")
    
    # Handle coordinates
    if 'latitude' in extracted_vars and 'longitude' in extracted_vars:
        lat = extracted_vars['latitude']
        lon = extracted_vars['longitude']
        
        # Create meshgrid for coordinates
        lon_mesh, lat_mesh = np.meshgrid(lon.values, lat.values)
        
        data_dict['latitude'] = lat_mesh.flatten()
        data_dict['longitude'] = lon_mesh.flatten()
        
        print(f"Created coordinate meshgrids: {lon_mesh.shape}")
    
    # Handle time
    if 'time' in extracted_vars:
        time_data = extracted_vars['time']
        print(f"Time data shape: {time_data.shape}")
        
        # If time is 1D, repeat it for each spatial point
        if len(time_data.shape) == 1:
            # Calculate number of spatial points
            n_spatial_points = len(data_dict.get('latitude', []))
            if n_spatial_points > 0:
                # Repeat time for each spatial point
                data_dict['time'] = np.repeat(time_data.values, n_spatial_points)
            else:
                data_dict['time'] = time_data.values
        else:
            data_dict['time'] = time_data.values.flatten()
        
        print(f"Processed time data length: {len(data_dict['time'])}")
    
    # Create DataFrame
    df = pd.DataFrame(data_dict)
    
    print(f"\n✓ Created DataFrame with shape: {df.shape}")
    print(f"✓ DataFrame columns: {list(df.columns)}")
    
    # Remove rows with NaN values
    initial_rows = len(df)
    df_clean = df.dropna()
    final_rows = len(df_clean)
    
    if initial_rows != final_rows:
        print(f"✓ Removed {initial_rows - final_rows} rows with NaN values")
        print(f"✓ Final DataFrame shape: {df_clean.shape}")
    
    return df_clean


def save_dataframe(df, output_path=None):
    """
    Save the DataFrame to a CSV file.
    
    Args:
        df (pandas.DataFrame): The DataFrame to save
        output_path (str): Path to save the CSV file
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"tempo_no2_data_{timestamp}.csv"
    
    print(f"\nSaving DataFrame to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"✓ DataFrame saved successfully!")
    
    return output_path


def main():
    """
    Main function to process TEMPO NO₂ data.
    """
    print("TEMPO Level 3 NO₂ Data Processor")
    print("="*60)
    
    # Look for TEMPO NO₂ files in the data directory
    data_dir = "data"
    tempo_files = glob.glob(os.path.join(data_dir, "*TEMPO*NO2*.nc"))
    
    if not tempo_files:
        print(f"⚠ No TEMPO NO₂ files found in {data_dir}/ directory")
        print("Please ensure you have a file like 'TEMPO_NO2_L3_v03.nc' in the data/ folder")
        return
    
    print(f"Found {len(tempo_files)} TEMPO NO₂ file(s):")
    for file in tempo_files:
        print(f"  - {file}")
    
    # Process the first file
    file_path = tempo_files[0]
    
    # Step 1: Load the dataset
    dataset = load_tempo_no2_file(file_path)
    if dataset is None:
        return
    
    # Step 2: Print dataset structure
    print_dataset_structure(dataset)
    
    # Step 3: Extract key variables
    extracted_vars = extract_key_variables(dataset)
    
    # Step 4: Create flattened DataFrame
    df = create_flattened_dataframe(extracted_vars)
    
    # Step 5: Display DataFrame info
    print("\n" + "="*60)
    print("DATAFRAME SUMMARY")
    print("="*60)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nBasic statistics:")
    print(df.describe())
    
    # Step 6: Save DataFrame
    output_file = save_dataframe(df)
    
    print(f"\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"✓ Processed file: {file_path}")
    print(f"✓ Output file: {output_file}")
    print(f"✓ Total data points: {len(df)}")


if __name__ == "__main__":
    main()
