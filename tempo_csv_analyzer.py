#!/usr/bin/env python3
"""
Simple TEMPO Collection Results Analyzer

This script analyzes the NASA Earthdata collection results CSV file
and extracts TEMPO data information without requiring external packages.

Author: NASA Apps Challenge
Date: 2024
"""

import os
import csv
import json
from datetime import datetime


def analyze_csv_file(csv_path):
    """
    Analyze the collection results CSV file using basic Python libraries.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        dict: Analysis results
    """
    print(f"Analyzing CSV file: {csv_path}")
    print("="*60)
    
    tempo_entries = []
    all_entries = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            # Read the CSV file
            csv_reader = csv.DictReader(file)
            
            # Get column names
            columns = csv_reader.fieldnames
            print(f"✓ CSV columns: {columns}")
            
            # Process each row
            for row_num, row in enumerate(csv_reader, start=2):
                all_entries.append(row)
                
                # Check if this is a TEMPO entry
                short_name = row.get('Short Name', '').upper()
                entry_title = row.get('Entry Title', '').upper()
                
                if 'TEMPO' in short_name or 'TEMPO' in entry_title:
                    tempo_entries.append({
                        'row_number': row_num,
                        'data_provider': row.get('Data Provider', ''),
                        'short_name': row.get('Short Name', ''),
                        'version': row.get('Version', ''),
                        'entry_title': row.get('Entry Title', ''),
                        'processing_level': row.get('Processing Level', ''),
                        'platform': row.get('Platform', ''),
                        'start_time': row.get('Start Time', ''),
                        'end_time': row.get('End Time', ''),
                        'full_row': row
                    })
    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    print(f"✓ Total entries: {len(all_entries)}")
    print(f"✓ TEMPO entries found: {len(tempo_entries)}")
    
    return {
        'total_entries': len(all_entries),
        'tempo_entries': tempo_entries,
        'columns': columns
    }


def categorize_tempo_data(tempo_entries):
    """
    Categorize TEMPO entries by product type and processing level.
    
    Args:
        tempo_entries (list): List of TEMPO entries
        
    Returns:
        dict: Categorized TEMPO data
    """
    print("\n" + "="*60)
    print("CATEGORIZING TEMPO DATA")
    print("="*60)
    
    categories = {
        'no2_products': [],
        'hcho_products': [],
        'o3_products': [],
        'cloud_products': [],
        'other_products': []
    }
    
    for entry in tempo_entries:
        short_name = entry['short_name'].upper()
        entry_title = entry['entry_title'].upper()
        
        if 'NO2' in short_name or 'NO2' in entry_title:
            categories['no2_products'].append(entry)
        elif 'HCHO' in short_name or 'HCHO' in entry_title or 'FORMALDEHYDE' in entry_title:
            categories['hcho_products'].append(entry)
        elif 'O3' in short_name or 'O3' in entry_title or 'OZONE' in entry_title:
            categories['o3_products'].append(entry)
        elif 'CLD' in short_name or 'CLOUD' in entry_title:
            categories['cloud_products'].append(entry)
        else:
            categories['other_products'].append(entry)
    
    # Print summary
    for category, entries in categories.items():
        if entries:
            print(f"\n{category.replace('_', ' ').title()}: {len(entries)} entries")
            for entry in entries:
                print(f"  - {entry['short_name']} ({entry['processing_level']})")
    
    return categories


def generate_download_instructions(categories):
    """
    Generate instructions for downloading TEMPO data files.
    
    Args:
        categories (dict): Categorized TEMPO data
        
    Returns:
        str: Instructions text
    """
    print("\n" + "="*60)
    print("DOWNLOAD INSTRUCTIONS")
    print("="*60)
    
    instructions = []
    instructions.append("TEMPO Data Download Instructions:")
    instructions.append("="*50)
    
    # Focus on NO2 products for this analysis
    no2_products = categories.get('no2_products', [])
    
    if no2_products:
        instructions.append(f"\nNO2 Products Found: {len(no2_products)}")
        instructions.append("-" * 30)
        
        for entry in no2_products:
            instructions.append(f"\nProduct: {entry['short_name']}")
            instructions.append(f"Title: {entry['entry_title']}")
            instructions.append(f"Processing Level: {entry['processing_level']}")
            instructions.append(f"Version: {entry['version']}")
            instructions.append(f"Platform: {entry['platform']}")
            instructions.append(f"Start Time: {entry['start_time']}")
            instructions.append(f"End Time: {entry['end_time']}")
            
            # Generate download URL pattern
            if entry['processing_level'] == '3':
                instructions.append(f"\nDownload URL Pattern:")
                instructions.append(f"https://tempo.si.edu/data/level3/{entry['short_name'].lower()}/")
                instructions.append(f"Look for files ending in .nc (NetCDF format)")
            
            instructions.append("-" * 30)
    
    # Add general instructions
    instructions.append(f"\nGeneral Download Instructions:")
    instructions.append("-" * 30)
    instructions.append("1. Visit: https://tempo.si.edu/")
    instructions.append("2. Navigate to the data section")
    instructions.append("3. Select your desired product (NO2, HCHO, O3, etc.)")
    instructions.append("4. Choose processing level (L2 or L3)")
    instructions.append("5. Download NetCDF (.nc) files")
    instructions.append("6. Place downloaded files in the 'data/' directory")
    
    instructions.append(f"\nFor Level 3 NO2 data specifically:")
    instructions.append("- Look for files like: TEMPO_NO2_L3_v03.nc")
    instructions.append("- These are gridded, ready-to-use datasets")
    instructions.append("- Perfect for analysis and visualization")
    
    return "\n".join(instructions)


def save_analysis_results(analysis_results, output_file=None):
    """
    Save analysis results to a JSON file.
    
    Args:
        analysis_results (dict): Analysis results
        output_file (str): Output file path
        
    Returns:
        str: Path to saved file
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"tempo_analysis_results_{timestamp}.json"
    
    print(f"\nSaving analysis results to: {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        print(f"✓ Analysis results saved successfully!")
        return output_file
    except Exception as e:
        print(f"✗ Error saving results: {e}")
        return None


def main():
    """
    Main function to analyze TEMPO collection results CSV.
    """
    print("TEMPO Collection Results Analyzer")
    print("="*60)
    
    # Look for the CSV file
    csv_path = "data/edsc_collection_results_export.csv"
    
    if not os.path.exists(csv_path):
        print(f"✗ CSV file not found: {csv_path}")
        print("Please ensure the file is in the data/ directory")
        return
    
    # Analyze the CSV file
    analysis_results = analyze_csv_file(csv_path)
    
    if analysis_results is None:
        print("✗ Failed to analyze CSV file")
        return
    
    # Categorize TEMPO data
    categories = categorize_tempo_data(analysis_results['tempo_entries'])
    
    # Generate download instructions
    instructions = generate_download_instructions(categories)
    
    # Print instructions
    print(instructions)
    
    # Save results
    analysis_results['categories'] = categories
    analysis_results['instructions'] = instructions
    
    output_file = save_analysis_results(analysis_results)
    
    # Summary
    print(f"\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"✓ Analyzed CSV file: {csv_path}")
    print(f"✓ Total entries: {analysis_results['total_entries']}")
    print(f"✓ TEMPO entries: {len(analysis_results['tempo_entries'])}")
    print(f"✓ NO2 products: {len(categories.get('no2_products', []))}")
    print(f"✓ Analysis saved to: {output_file}")
    
    print(f"\nNext steps:")
    print(f"1. Follow the download instructions above")
    print(f"2. Download TEMPO NO2 Level 3 NetCDF files")
    print(f"3. Place them in the data/ directory")
    print(f"4. Run the enhanced processor to analyze the data")


if __name__ == "__main__":
    main()
