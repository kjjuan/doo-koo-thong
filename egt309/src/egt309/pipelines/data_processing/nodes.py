"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 1.0.0
"""

import pandas as pd
import numpy as np

def load_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Passes the raw DataFrame loaded by the Kedro catalog."""
    return df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Fill null values in loan columns with 'unknown'
    df = df.fillna({
        'Housing Loan': 'unknown',
        'Personal Loan': 'unknown'
    })

    # Normalise 'Contact Method' column values
    df['Contact Method'] = df['Contact Method'].replace('Cell', 'cellular')
    df['Contact Method'] = df['Contact Method'].replace('Telephone', 'telephone')

    # Remove unrealistic '150 years' age entries
    df = df.drop(df[df['Age'] == '150 years'].index)

    # Corrects negative entries in 'Campaign Calls'
    df['Campaign Calls'] = df['Campaign Calls'].astype(str)
    df['Campaign Calls'] = df['Campaign Calls'].str.replace('-', '', regex=False)   
    df['Campaign Calls'] = df['Campaign Calls'].astype(int)

    return df