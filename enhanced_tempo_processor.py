#!/usr/bin/env python3
"""
Enhanced TEMPO Level 3 NO₂ Data Processor with CSV Collection Results Support

This script processes TEMPO Level 3 NO₂ NetCDF files and can work with
NASA Earthdata collection results CSV files to automatically locate
and process multiple TEMPO data files.

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
import requests
from urllib.parse import urlparse
import re

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def load_collection_results_csv(csv_path):
    """
    Load and analyze the NASA Earthdata collection results CSV file.
    
    Args:
        csv_path (str): Path to the collection results CSV file
        
    Returns:
        pandas.DataFrame: The loaded CSV data
    """
    try:
        print(f"Loading collection results CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"\n✓ CSV loaded successfully!")
        print(f"✓ Shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Display first few rows to understand structure
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        
        return df
    except Exception as e:
        print(f"Error loading CSV file {csv_path}: {e}")
        return None


def analyze_csv_structure(df):
    """
    Analyze the CSV structure to identify relevant columns for TEMPO data.
    
    Args:
        df (pandas.DataFrame): The loaded CSV data
        
    Returns:
        dict: Analysis results with identified columns
    """
    print("\n" + "="*60)
    print("CSV STRUCTURE ANALYSIS")
    print("="*60)
    
    analysis = {
        'file_path_columns': [],
        'url_columns': [],
        'date_columns': [],
        'product_columns': [],
        'tempo_files': []
    }
    
    # Look for file path columns
    file_keywords = ['file', 'path', 'filename', 'granule', 'download']
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in file_keywords):
            analysis['file_path_columns'].append(col)
            print(f"✓ Found file path column: {col}")
    
    # Look for URL columns
    url_keywords = ['url', 'link', 'download_url', 'href']
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in url_keywords):
            analysis['url_columns'].append(col)
            print(f"✓ Found URL column: {col}")
    
    # Look for date columns
    date_keywords = ['date', 'time', 'start', 'end', 'acquisition']
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in date_keywords):
            analysis['date_columns'].append(col)
            print(f"✓ Found date column: {col}")
    
    # Look for product/collection columns
    product_keywords = ['product', 'collection', 'dataset', 'shortname']
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in product_keywords):
            analysis['product_columns'].append(col)
            print(f"✓ Found product column: {col}")
    
    # Identify TEMPO files
    tempo_patterns = ['tempo', 'no2', 'nitrogen', 'dioxide']
    for col in df.columns:
        if any(pattern in col.lower() for pattern in tempo_patterns):
            print(f"✓ Found TEMPO-related column: {col}")
    
    return analysis


def extract_tempo_files_from_csv(df, analysis):
    """
    Extract TEMPO file information from the CSV data.
    
    Args:
        df (pandas.DataFrame): The loaded CSV data
        analysis (dict): Analysis results from analyze_csv_structure
        
    Returns:
        list: List of dictionaries containing TEMPO file information
    """
    print("\n" + "="*60)
    print("EXTRACTING TEMPO FILES")
    print("="*60)
    
    tempo_files = []
    
    # Look for TEMPO-related entries
    tempo_keywords = ['tempo', 'no2', 'nitrogen dioxide']
    
    for idx, row in df.iterrows():
        is_tempo = False
        file_info = {'index': idx, 'row_data': row.to_dict()}
        
        # Check if any column contains TEMPO-related keywords
        for col in df.columns:
            if pd.notna(row[col]):
                cell_value = str(row[col]).lower()
                if any(keyword in cell_value for keyword in tempo_keywords):
                    is_tempo = True
                    file_info['tempo_match_column'] = col
                    file_info['tempo_match_value'] = row[col]
                    break
        
        if is_tempo:
            # Extract file path or URL
            file_path = None
            file_url = None
            
            # Check file path columns
            for col in analysis['file_path_columns']:
                if pd.notna(row[col]) and str(row[col]).strip():
                    file_path = str(row[col]).strip()
                    file_info['file_path'] = file_path
                    break
            
            # Check URL columns
            for col in analysis['url_columns']:
                if pd.notna(row[col]) and str(row[col]).strip():
                    file_url = str(row[col]).strip()
                    file_info['file_url'] = file_url
                    break
            
            # Extract date information
            for col in analysis['date_columns']:
                if pd.notna(row[col]):
                    file_info[f'date_{col}'] = row[col]
            
            # Extract product information
            for col in analysis['product_columns']:
                if pd.notna(row[col]):
                    file_info[f'product_{col}'] = row[col]
            
            tempo_files.append(file_info)
            print(f"✓ Found TEMPO file #{len(tempo_files)}: {file_path or file_url}")
    
    print(f"\n✓ Total TEMPO files found: {len(tempo_files)}")
    return tempo_files


def download_file_from_url(url, local_path):
    """
    Download a file from URL to local path.
    
    Args:
        url (str): URL to download from
        local_path (str): Local path to save the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Downloading: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Downloaded successfully: {local_path}")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def prepare_tempo_files(tempo_files, data_dir="data"):
    """
    Prepare TEMPO files for processing by downloading or locating them.
    
    Args:
        tempo_files (list): List of TEMPO file information
        data_dir (str): Directory to store files
        
    Returns:
        list: List of local file paths ready for processing
    """
    print("\n" + "="*60)
    print("PREPARING TEMPO FILES")
    print("="*60)
    
    local_files = []
    
    for i, file_info in enumerate(tempo_files):
        print(f"\nProcessing file #{i+1}/{len(tempo_files)}")
        
        local_path = None
        
        # Check if we have a file path
        if 'file_path' in file_info:
            file_path = file_info['file_path']
            
            # If it's already a local file, check if it exists
            if os.path.exists(file_path):
                local_path = file_path
                print(f"✓ Local file found: {file_path}")
            else:
                print(f"⚠ Local file not found: {file_path}")
        
        # Check if we have a URL to download
        if local_path is None and 'file_url' in file_info:
            url = file_info['file_url']
            
            # Generate local filename
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename or not filename.endswith('.nc'):
                filename = f"tempo_file_{i+1}.nc"
            
            local_path = os.path.join(data_dir, filename)
            
            # Download the file
            if download_file_from_url(url, local_path):
                local_files.append(local_path)
            else:
                print(f"✗ Failed to download: {url}")
        
        # If we have a local path, add it to the list
        if local_path and os.path.exists(local_path):
            local_files.append(local_path)
    
    print(f"\n✓ Total files ready for processing: {len(local_files)}")
    return local_files


def process_multiple_tempo_files(file_paths):
    """
    Process multiple TEMPO files and combine the results.
    
    Args:
        file_paths (list): List of file paths to process
        
    Returns:
        pandas.DataFrame: Combined DataFrame from all files
    """
    print("\n" + "="*60)
    print("PROCESSING MULTIPLE TEMPO FILES")
    print("="*60)
    
    all_dataframes = []
    
    for i, file_path in enumerate(file_paths):
        print(f"\nProcessing file #{i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
        
        # Load the dataset
        try:
            dataset = xr.open_dataset(file_path)
            
            # Extract key variables
            extracted_vars = extract_key_variables(dataset)
            
            # Create DataFrame
            df = create_flattened_dataframe(extracted_vars)
            
            # Add source file information
            df['source_file'] = os.path.basename(file_path)
            df['file_index'] = i
            
            all_dataframes.append(df)
            print(f"✓ Processed {len(df)} data points")
            
        except Exception as e:
            print(f"✗ Error processing {file_path}: {e}")
            continue
    
    if all_dataframes:
        # Combine all DataFrames
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"\n✓ Combined dataset shape: {combined_df.shape}")
        print(f"✓ Total data points: {len(combined_df)}")
        return combined_df
    else:
        print("\n✗ No files were successfully processed")
        return pd.DataFrame()


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
        
    Returns:
        str: Path to the saved file
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"tempo_no2_combined_data_{timestamp}.csv"
    
    print(f"\nSaving DataFrame to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"✓ DataFrame saved successfully!")
    
    return output_path


def main():
    """
    Main function to process TEMPO NO₂ data with CSV collection results support.
    """
    print("Enhanced TEMPO Level 3 NO₂ Data Processor")
    print("="*60)
    
    # Look for collection results CSV file
    csv_files = glob.glob("*.csv")
    tempo_csv = None
    
    for csv_file in csv_files:
        if 'collection' in csv_file.lower() or 'edsc' in csv_file.lower():
            tempo_csv = csv_file
            break
    
    if tempo_csv:
        print(f"Found collection results CSV: {tempo_csv}")
        
        # Load and analyze CSV
        df = load_collection_results_csv(tempo_csv)
        if df is None:
            return
        
        # Analyze CSV structure
        analysis = analyze_csv_structure(df)
        
        # Extract TEMPO files
        tempo_files = extract_tempo_files_from_csv(df, analysis)
        
        if tempo_files:
            # Prepare files for processing
            local_files = prepare_tempo_files(tempo_files)
            
            if local_files:
                # Process all files
                combined_df = process_multiple_tempo_files(local_files)
                
                if not combined_df.empty:
                    # Save combined results
                    output_file = save_dataframe(combined_df)
                    
                    print(f"\n" + "="*60)
                    print("PROCESSING COMPLETE")
                    print("="*60)
                    print(f"✓ Processed {len(local_files)} TEMPO files")
                    print(f"✓ Combined dataset shape: {combined_df.shape}")
                    print(f"✓ Output file: {output_file}")
                    return
        
        print("\n⚠ No TEMPO files found in CSV or failed to process them")
    
    # Fallback: Look for direct TEMPO NetCDF files
    print("\nFalling back to direct TEMPO file processing...")
    data_dir = "data"
    tempo_files = glob.glob(os.path.join(data_dir, "*TEMPO*NO2*.nc"))
    
    if not tempo_files:
        print(f"⚠ No TEMPO NO₂ files found in {data_dir}/ directory")
        print("Please ensure you have:")
        print("1. A collection results CSV file in the project directory, OR")
        print("2. TEMPO NetCDF files in the data/ folder")
        return
    
    print(f"Found {len(tempo_files)} TEMPO NO₂ file(s):")
    for file in tempo_files:
        print(f"  - {file}")
    
    # Process the files
    combined_df = process_multiple_tempo_files(tempo_files)
    
    if not combined_df.empty:
        output_file = save_dataframe(combined_df)
        
        print(f"\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"✓ Processed {len(tempo_files)} TEMPO files")
        print(f"✓ Combined dataset shape: {combined_df.shape}")
        print(f"✓ Output file: {output_file}")


if __name__ == "__main__":
    main()
