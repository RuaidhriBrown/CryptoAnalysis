import pandas as pd

import os
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function for value analysis
def value_analysis(df):
    # Assuming 'min val sent' and 'min value received' are in Ether and we set a threshold for 'small values'
    low_value_threshold = df[['min val sent', 'min value received']].quantile(0.25)
    print(f"Value Analysis Thresholds: {low_value_threshold}")
    high_value_transactions = df[(df['min val sent'] < low_value_threshold['min val sent']) | (df['min value received'] < low_value_threshold['min value received'])]
    return high_value_transactions

# Function for transaction frequency analysis
def transaction_frequency_analysis(df):
    # Assuming 'Avg min between sent tnx' and 'Avg min between received tnx' are in minutes and we set a threshold for 'fast transactions'
    low_frequency_threshold = df[['Avg min between sent tnx', 'Avg min between received tnx']].quantile(0.25)
    print(f"Frequency Analysis Thresholds: {low_frequency_threshold}")
    frequent_transactions = df[(df['Avg min between sent tnx'] < low_frequency_threshold['Avg min between sent tnx']) | (df['Avg min between received tnx'] < low_frequency_threshold['Avg min between received tnx'])]
    return frequent_transactions

# Function for account analysis
def account_analysis(df):
    # Accounts that move most of the received funds
    df['total ether received'] = df['total ether received'].replace(0, 1)  # Prevent division by zero
    df['move_ratio'] = df['total Ether sent'] / df['total ether received']
    high_move_accounts = df[(df['move_ratio'] > 0.9) & (df['total ether received'] > df['total ether received'].quantile(0.75))]
    return high_move_accounts

# Create a results directory and a subdirectory with the current date and time
results_dir = 'results'
date_time_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
path = os.path.join(results_dir, date_time_dir)

# Check if the directory exists, if not, create it
if not os.path.exists(path):
    os.makedirs(path)

# Load the data
file_path = 'datasets/ethereum/transaction_dataset_even.csv'  # File path adjusted to the provided one
df = load_data(file_path)

# Perform the analyses
value_analysis_result = value_analysis(df)
transaction_frequency_analysis_result = transaction_frequency_analysis(df)
account_analysis_result = account_analysis(df)

# Print the resultsr
print("Value Analysis Result:")
print(value_analysis_result)
print("\nTransaction Frequency Analysis Result:")
print(transaction_frequency_analysis_result)
print("\nAccount Analysis Result:")
print(account_analysis_result)

# Save the results in the date and time named folder
value_analysis_result.to_csv(os.path.join(path, 'value_analysis_result.csv'), index=False)
transaction_frequency_analysis_result.to_csv(os.path.join(path, 'transaction_frequency_analysis_result.csv'), index=False)
account_analysis_result.to_csv(os.path.join(path, 'account_analysis_result.csv'), index=False)

print(f"Results saved in the directory: {path}")


# Dataset:
# (number) ,Index,Address,FLAG,Avg min between sent tnx,Avg min between received tnx,Time Diff between first and last (Mins),Sent tnx,Received Tnx,Number of Created Contracts,Unique Received From Addresses,Unique Sent To Addresses,min value received,max value received ,avg val received,min val sent,max val sent,avg val sent,min value sent to contract,max val sent to contract,avg value sent to contract,total transactions (including tnx to create contract,total Ether sent,total ether received,total ether sent contracts,total ether balance, Total ERC20 tnxs, ERC20 total Ether received, ERC20 total ether sent, ERC20 total Ether sent contract, ERC20 uni