import os
import pandas as pd
missing_data=[]
def load_nifty500_stock_data(directory, columns_to_extract):
    """
    This function goes through a directory, reads each CSV file, and extracts the specified columns into a dictionary.
    
    Parameters:
    - directory: The path to the directory containing the CSV files.
    - columns_to_extract: List of column names to extract from the CSV files.
    
    Returns:
    - NIFTY500_stock: A dictionary where the key is the stock name (extracted from the file name)
      and the value is the DataFrame containing the extracted columns.
    """
    NIFTY500_stock = {}
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Check if all the columns to extract exist in the file
                missing_columns = [col for col in columns_to_extract if col not in df.columns]
                if missing_columns:
                    missing_data.append(filename)
                    print(f"Warning: Missing columns in {filename}: {', '.join(missing_columns)}")
                
                # Extract only the specified columns
                extracted_df = df[columns_to_extract]
                
                # Extract the stock name from the filename (e.g., '3MINDIA_data.csv' -> '3MINDIA')
                stock_name = filename.split('_')[0]
                
                # Store the extracted data in the dictionary
                NIFTY500_stock[stock_name] = extracted_df
                
                print(f"Loaded data for {stock_name} from {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return NIFTY500_stock

# Example usage:
directory = 'calculated_stock_data'
columns_to_extract = [
    'date', 'close', 'high', 'low', 'open', 'volume', 'ma_20_unnamed: 6_level_1', 
    'ma_50_unnamed: 7_level_1', 'NORMALISED_CLOSE', 'RETURN RATIO', 'PERCENTAGE_CHANGE_OPEN', 
    'PERCENTAGE_CHANGE_CLOSE', 'PERCENTAGE_CHANGE_LOW', 'Sector_Name', 'Sector_Encoded', 
    'ma_5', 'ma_10', 'ma_15', 'ma_20', 'ma_25', 'ma_30'
]

# Load the data into the dictionary
NIFTY500_stock = load_nifty500_stock_data(directory, columns_to_extract)

# Example of accessing data for a specific stock (e.g., '3MINDIA')
if 'ONGC' in NIFTY500_stock:
    print(NIFTY500_stock['ONGC'].head())  # Show first 5 rows of 3MINDIA's stock data
else:
    print("Stock '3MINDIA' not found in the data.")

# Print the keys (stock names)
print("Stocks loaded:", NIFTY500_stock.keys())
print(len(NIFTY500_stock.keys()))
print(missing_data,"\n",len(missing_data))

