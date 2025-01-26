import os
import pandas as pd

# Directory containing the stock data files
stock_data_dir = "stock_data"
output_dir = "fixed_stock_data"  # Directory to save the corrected files
os.makedirs(output_dir, exist_ok=True)

def fix_headers(file_path, output_path):
    # Read the file with the first two rows as headers
    try:
        data = pd.read_csv(file_path, header=[0, 1])

        # Flatten the multi-level columns
        data.columns = ['_'.join(col).strip() for col in data.columns.values]

        # Correct the misalignment (swap "Price" and "Date" levels)
        corrected_columns = []
        for col in data.columns:
            if "Price_" in col:
                corrected_columns.append(col.replace("Price_", "Date_"))
            elif "Date_" in col:
                corrected_columns.append(col.replace("Date_", "Price_"))
            else:
                corrected_columns.append(col)

        # Assign corrected columns back to the DataFrame
        data.columns = corrected_columns

        # Save the corrected DataFrame to the output directory
        data.to_csv(output_path, index=False)
        print(f"Fixed and saved: {output_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Process all CSV files in the directory
for filename in os.listdir(stock_data_dir):
    if filename.endswith(".csv"):
        input_file_path = os.path.join(stock_data_dir, filename)
        output_file_path = os.path.join(output_dir, filename)
        fix_headers(input_file_path, output_file_path)

print(f"Header fixing complete. Fixed files saved in {output_dir}.")

