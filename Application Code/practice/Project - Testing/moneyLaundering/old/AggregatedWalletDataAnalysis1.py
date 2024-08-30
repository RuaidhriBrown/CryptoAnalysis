import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function for value analysis with fixed threshold
def value_analysis(df):
    low_value_threshold = 1  # Example fixed threshold for low values in Ether
    high_value_transactions = df[(df['min val sent'] < low_value_threshold) | (df['min value received'] < low_value_threshold)]
    return not high_value_transactions.empty

# Function for transaction frequency analysis with fixed threshold
def transaction_frequency_analysis(df):
    low_frequency_threshold = 60  # Example fixed threshold for fast transactions (in minutes)
    frequent_transactions = df[(df['Avg min between sent tnx'] < low_frequency_threshold) | (df['Avg min between received tnx'] < low_frequency_threshold)]
    return not frequent_transactions.empty

# Function for account analysis
def account_analysis(df):
    df['total ether received'] = df['total ether received'].replace(0, 1)  # Prevent division by zero
    df['move_ratio'] = df['total Ether sent'] / df['total ether received']
    high_move_accounts = df[(df['move_ratio'] > 0.9)]  # High move ratio threshold
    return not high_move_accounts.empty

# Load the aggregated wallet data
file_path_aggregated = 'transaction_dataset_even.csv'
df_aggregated = load_data(file_path_aggregated)

# Filter data for the specified wallet
wallet_address = '0x00009277775ac7d0d59eaad8fee3d10ac6c805e8'
df_aggregated_filtered = df_aggregated[df_aggregated['Address'] == wallet_address]

# Perform the analyses
value_analysis_result = value_analysis(df_aggregated_filtered)
transaction_frequency_analysis_result = transaction_frequency_analysis(df_aggregated_filtered)
account_analysis_result = account_analysis(df_aggregated_filtered)

# Determine the likelihood of money laundering
likely_money_laundering = value_analysis_result or transaction_frequency_analysis_result or account_analysis_result

# Print the results
print("Value Analysis Result:", value_analysis_result)
print("Transaction Frequency Analysis Result:", transaction_frequency_analysis_result)
print("Account Analysis Result:", account_analysis_result)
print("Likelihood of Money Laundering:", likely_money_laundering)
