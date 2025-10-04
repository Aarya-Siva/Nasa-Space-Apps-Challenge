# TEMPO Level 3 NO₂ Data Processor

This Python script processes TEMPO Level 3 NO₂ NetCDF files and converts them into flattened pandas DataFrames for analysis.

## Features

- Loads TEMPO Level 3 NO₂ NetCDF files using xarray
- Prints comprehensive dataset structure (variables, dimensions, coordinates)
- Extracts key variables: NO₂, latitude, longitude, and time
- Converts data into a flattened pandas DataFrame
- Handles missing variables gracefully
- Saves processed data to CSV format

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Or install the core dependencies manually:
```bash
pip install numpy pandas xarray netCDF4
```

## Usage

1. Place your TEMPO NO₂ NetCDF file in the `data/` directory
2. Run the script:
```bash
python tempo_no2_processor.py
```

The script will:
- Look for TEMPO NO₂ files in the `data/` directory
- Process the first file found
- Display dataset structure and statistics
- Save the flattened data to a CSV file

## File Structure

```
├── data/                          # Directory for TEMPO NetCDF files
│   └── TEMPO_NO2_L3_v03.nc       # Your TEMPO NO₂ file
├── tempo_no2_processor.py         # Main processing script
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```

## Output

The script generates:
- Console output showing dataset structure and processing steps
- A CSV file with columns: NO2, latitude, longitude, time
- Summary statistics of the processed data

## Example Output

```
TEMPO Level 3 NO₂ Data Processor
============================================================
Found 1 TEMPO NO₂ file(s):
  - data/TEMPO_NO2_L3_v03.nc

Loading TEMPO NO₂ file: data/TEMPO_NO2_L3_v03.nc

============================================================
DATASET STRUCTURE
============================================================

Dataset dimensions:
  latitude: 1800
  longitude: 3600
  time: 1

Dataset coordinates:
  latitude: (1800,) - float32
  longitude: (3600,) - float32
  time: (1,) - datetime64[ns]

Dataset variables:
  NO2: (1, 1800, 3600) - float32
    Attributes: {'units': 'mol/m^2', 'long_name': 'Nitrogen Dioxide Column'}

✓ Created DataFrame with shape: (6480000, 4)
✓ DataFrame columns: ['NO2', 'latitude', 'longitude', 'time']
```

## Requirements

- Python 3.7+
- numpy
- pandas
- xarray
- netCDF4

## Notes

- The script handles various coordinate naming conventions (lat/latitude, lon/longitude)
- Missing variables are handled gracefully with informative messages
- NaN values are automatically removed from the final DataFrame
- The script processes the first TEMPO file found in the data directory
