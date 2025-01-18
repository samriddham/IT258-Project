import os
import pandas as pd
from collections import defaultdict

# Input and output directories
input_dir = "fixed_stock_data"
output_dir = "calculated_stock_data"
os.makedirs(output_dir, exist_ok=True)

# List of moving average periods to calculate
ma_periods = [5, 10, 15, 20, 25, 30]

# Nested dictionary to store the data in the desired format
NIFTY500_stock = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

# Function to extract and normalize column names dynamically
def normalize_columns(stock_data):
    # Lowercase all column names
    stock_data.columns = stock_data.columns.str.lower()

    # Log the column names for debugging
    print(f"Original Columns: {stock_data.columns.tolist()}")

    # Identify the stock/company name dynamically from one of the columns
    company_name = None
    for col in stock_data.columns:
        if "close_" in col:
            company_name = col.split("_")[1]  # Extract the company name
            break

    if not company_name:
        raise ValueError("Could not determine the company name from columns.")

    # Rename columns based on the company name
    column_mapping = {
        f"date_ticker": "date",
        f"close_{company_name}": "close",
        f"high_{company_name}": "high",
        f"low_{company_name}": "low",
        f"open_{company_name}": "open",
        f"volume_{company_name}": "volume",
    }

    print(f"Column Mapping: {column_mapping}")

    stock_data.rename(columns=column_mapping, inplace=True)
    return stock_data

# Function to preprocess stock data and calculate additional fields
def process_stock_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                print(f"Processing file: {filename}")

                # Load the CSV file
                stock_data = pd.read_csv(file_path)

                # Normalize and rename columns dynamically
                stock_data = normalize_columns(stock_data)

                # Required columns (lowercase)
                required_columns = ["date", "close", "high", "low"]

                # Check if all required columns exist
                if all(col in stock_data.columns for col in required_columns):
                    # Drop rows with NaN values in key columns
                    stock_data = stock_data.dropna(subset=required_columns)

                    # Calculate normalized close price
                    stock_data["normalized_close"] = stock_data["close"] / stock_data["close"].max()

                    # Calculate return ratio
                    stock_data["return_ratio"] = stock_data["close"].pct_change()

                    # Calculate percentage change wrt low
                    stock_data["percentage_change_wrt_low"] = (
                        (stock_data["close"] - stock_data["low"]) / stock_data["low"] * 100
                    )

                    # Calculate relative to close (e.g., high relative to close)
                    stock_data["high_relative_to_close"] = stock_data["high"] / stock_data["close"]
                    stock_data["low_relative_to_close"] = stock_data["low"] / stock_data["close"]

                    # Calculate moving averages
                    for period in ma_periods:
                        column_name = f"ma_{period}"
                        stock_data[column_name] = stock_data["close"].rolling(window=period).mean()

                    # Save the updated file back to the directory
                    stock_data.to_csv(output_path, index=False)

                    # Extract the stock name (assuming the file name is the stock name without extension)
                    stock_name = os.path.splitext(filename)[0]

                    # Update the NIFTY500_stock dictionary
                    NIFTY500_stock[stock_name]["stock_price"] = {
                        "normalized_close": stock_data["normalized_close"].tolist(),
                        "return_ratio": stock_data["return_ratio"].tolist(),
                        "percentage_change_wrt_low": stock_data["percentage_change_wrt_low"].tolist(),
                        "high_relative_to_close": stock_data["high_relative_to_close"].tolist(),
                        "low_relative_to_close": stock_data["low_relative_to_close"].tolist(),
                        "moving_averages": {period: stock_data[f"ma_{period}"].tolist() for period in ma_periods},
                    }

                    print(f"Successfully processed: {filename}")
                else:
                    print(f"Skipping {filename}: Missing required columns. Found columns: {stock_data.columns.tolist()}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Process all files in the input directory
process_stock_files(input_dir)

print("Processing complete. Processed files saved in the 'calculated_stock_data' directory.")

