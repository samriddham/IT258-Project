import os
import pandas as pd

def modify_csv_columns(directory, columns_to_delete, new_column_names=None):
    """
    This function goes through a directory, finds CSV files, and modifies/deletes columns.

    Parameters:
    - directory: The path to the directory containing the CSV files.
    - columns_to_delete: List of column names to delete from the CSV files.
    - new_column_names: Dictionary of column names to rename, where keys are old names and values are new names.

    The function will:
    - Delete the specified columns.
    - Rename columns as specified.
    - Save the modified CSV file with the same name.
    """
    
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Delete specified columns
            df.drop(columns=columns_to_delete, errors='ignore', inplace=True)
            
            # Rename columns if needed
            if new_column_names:
                df.rename(columns=new_column_names, inplace=True)
            
            # Save the modified CSV file
            df.to_csv(file_path, index=False)
            print(f"Modified: {filename}")

# Example usage:
directory = 'calculated_stock_data'
columns_to_delete = [
    'normalized_close', 
    'return_ratio', 
    'percentage_change_wrt_low', 
    'high_relative_to_close', 
    'low_relative_to_close'
]

# Add your column name changes here
new_column_names = {
    'normalised close_unnamed: 8_level_1': 'NORMALISED_CLOSE',
    'return ratio_unnamed: 9_level_1': 'RETURN RATIO',
    'percentage change open_unnamed: 10_level_1': 'PERCENTAGE_CHANGE_OPEN',
    'percentage change high_unnamed: 11_level_1': 'PERCENTAGE_CHANGE_CLOSE',
    'percentage change low_unnamed: 12_level_1': 'PERCENTAGE_CHANGE_LOW',
    'sector_unnamed: 13_level_1': 'Sector_Name',
    'sector encoded_unnamed: 14_level_1': 'Sector_Encoded',
    
}

modify_csv_columns(directory, columns_to_delete, new_column_names)

