import pandas as pd
import json
from AutoClean import AutoClean

def load_data(file_path):
    """
    Load data from various file formats (CSV, JSON, Excel).
    
    Args:
        file_path (str): Path to the data file.
        
    Returns:
        pd.DataFrame: Loaded data as a Pandas DataFrame.
    """
    if file_path.endswith('.csv'):
        dataset = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        dataset = pd.json_normalize(data)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        dataset = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    return dataset


file_path='olympics2024.csv'
dataset=load_data(file_path)
cleaned_data=AutoClean(dataset, mode='auto', duplicates=True, missing_num=True, missing_categ=True, 
                       encode_categ=False, extract_datetime=True, outliers=False, outlier_param=1.5, 
                       logfile=True, verbose=True)
print(cleaned_data)