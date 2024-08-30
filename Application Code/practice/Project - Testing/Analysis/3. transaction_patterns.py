import pandas as pd
import matplotlib.pyplot as plt

import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to parse timestamps and convert to datetime
def parse_timestamps(df, timestamp_column='timestamp'):
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    return df

# Function to analyze transaction patterns over time
def transaction_patterns_over_time(df, timestamp_column='timestamp', value_column='value'):
    # Ensure timestamp is in datetime format
    df = parse_timestamps(df, timestamp_column)
    
    # Resample data by day and calculate total and average transaction values
    daily_transactions = df.resample('D', on=timestamp_column).agg({
        value_column: ['count', 'sum', 'mean']
    }).reset_index()
    
    daily_transactions.columns = ['Date', 'Transaction Count', 'Total Value', 'Average Value']
    
    return daily_transactions

# Results directory creation with date and time
results_dir = 'results'
date_time_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
path = os.path.join(results_dir, date_time_dir)
if not os.path.exists(path):
    os.makedirs(path)

# Load the data
file_path = 'transaction_dataset_even.csv'  # Adjust to your file path
df = load_data(file_path)

# Analyze transaction patterns over time
transaction_patterns_df = transaction_patterns_over_time(df, timestamp_column='Time Diff between first and last (Mins)', value_column='total Ether sent')

# Save the transaction patterns analysis to CSV
transaction_patterns_df.to_csv(os.path.join(path, 'transaction_patterns_over_time.csv'), index=False)

# Optionally, plot the results
plt.figure(figsize=(14, 7))
plt.plot(transaction_patterns_df['Date'], transaction_patterns_df['Transaction Count'], label='Transaction Count')
plt.plot(transaction_patterns_df['Date'], transaction_patterns_df['Average Value'], label='Average Value', secondary_y=True)
plt.title('Transaction Patterns Over Time')
plt.xlabel('Date')
plt.ylabel('Transaction Count')
plt.legend(loc='upper left')
plt.show()

print(f"Transaction patterns over time analysis saved in the directory: {path}")
