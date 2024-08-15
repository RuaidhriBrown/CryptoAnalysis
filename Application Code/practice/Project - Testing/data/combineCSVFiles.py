import os
import pandas as pd

def combine_csv_files(input_directory, output_file):
    all_files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]
    combined_df = pd.DataFrame()

    for file in all_files:
        file_path = os.path.join(input_directory, file)
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df.to_csv(output_file, index=False)
    print(f"All CSV files have been combined into {output_file}")

# Specify the directory containing the CSV files and the output file name
input_directory = "data/er20_collection/"
output_file = "combined_er20.csv"

combine_csv_files(input_directory, output_file)
