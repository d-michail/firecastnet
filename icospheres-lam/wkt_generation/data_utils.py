"""
Data Processing Utilities

This module contains utilities for loading and processing geographic data
from various sources including CSV files and other data formats.
"""

import pandas as pd


def get_country(country_name, filename="icospheres-lam/csv/wkt_countries_continents.csv"):
    """
    Retrieve country information from a CSV file.
    
    This function searches for a country by name, ISO code, or subunit name
    and returns the corresponding row from the CSV file.
    
    Args:
        country_name (str): The name, ISO code, or subunit name to search for
        filename (str): Path to the CSV file containing country data.
                       Defaults to "icospheres-lam/csv/wkt_countries_continents.csv"
    
    Returns:
        pandas.Series: The first matching row from the CSV file
        
    Raises:
        IndexError: If no matching country is found
        FileNotFoundError: If the CSV file doesn't exist
    """
    df = pd.read_csv(filename, encoding='latin1', keep_default_na=False, sep=',', quotechar='"')
    
    # Search by NAME, SU_A3 (ISO code), or SUBUNIT
    matching_rows = df[
        (df['NAME'] == country_name) | 
        (df['SU_A3'] == country_name) | 
        (df['SUBUNIT'] == country_name)
    ]
    
    if matching_rows.empty:
        raise ValueError(f"No country found with name/code: {country_name}")
    
    return matching_rows.iloc[0]