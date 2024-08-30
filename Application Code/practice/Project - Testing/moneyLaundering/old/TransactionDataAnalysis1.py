import pandas as pd

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Load the transaction data
file_path_transactions = 'combined_transactions.csv'
df_transactions = load_data(file_path_transactions)

# Filter data for the specified wallet
wallet_address = '0x00009277775ac7d0d59eaad8fee3d10ac6c805e8'
df_transactions_filtered = df_transactions[df_transactions['Address'] == wallet_address]

# Convert 'value' and 'timeStamp' columns to numeric types
df_transactions_filtered['value'] = pd.to_numeric(df_transactions_filtered['value'], errors='coerce')
df_transactions_filtered['timeStamp'] = pd.to_numeric(df_transactions_filtered['timeStamp'], errors='coerce')

# Example criteria for transaction data analysis
def transaction_data_analysis(df):
    # Example fixed thresholds for suspicious activity
    large_transaction_threshold = 10000  # in Ether
    short_time_threshold = 60  # in seconds (1 minute)
    high_volume_threshold = 10  # More than 10 transactions in a short period
    circular_transaction_threshold = 5  # More than 5 transactions to the same address
    
    total_transactions = len(df)
    
    # Flag large transactions
    large_transactions = df[df['value'] > large_transaction_threshold]
    percent_large_transactions = (len(large_transactions) / total_transactions) * 100
    
    # Flag transactions with short intervals
    df['time_diff'] = df['timeStamp'].diff().abs()
    short_interval_transactions = df[df['time_diff'] < short_time_threshold]
    percent_short_interval_transactions = (len(short_interval_transactions) / total_transactions) * 100
    
    # Flag high volume of transactions within a short period
    df['transaction_count'] = df['timeStamp'].rolling(window=short_time_threshold).count()
    high_volume_transactions = df[df['transaction_count'] > high_volume_threshold]
    percent_high_volume_transactions = (len(high_volume_transactions) / total_transactions) * 100
    
    # Flag circular transactions
    circular_transactions_counts = df['to'].value_counts()
    circular_transactions = circular_transactions_counts[circular_transactions_counts > circular_transaction_threshold]
    percent_circular_transactions = (len(circular_transactions) / total_transactions) * 100
    
    # Flag unusual patterns (e.g., large amounts received then split into smaller amounts)
    large_received_transactions = df[(df['value'] > large_transaction_threshold) & (df['to'] == wallet_address)]
    split_transactions = df[(df['value'] < large_transaction_threshold) & (df['from'].isin(large_received_transactions['to']))]
    percent_split_transactions = (len(split_transactions) / total_transactions) * 100
    
    stats = {
        "percent_large_transactions": percent_large_transactions,
        "percent_short_interval_transactions": percent_short_interval_transactions,
        "percent_high_volume_transactions": percent_high_volume_transactions,
        "percent_circular_transactions": percent_circular_transactions,
        "percent_split_transactions": percent_split_transactions,
    }
    
    return (
        not large_transactions.empty or 
        not short_interval_transactions.empty or 
        not high_volume_transactions.empty or 
        not circular_transactions.empty or 
        not split_transactions.empty
    ), large_transactions, short_interval_transactions, high_volume_transactions, circular_transactions, split_transactions, stats

# Perform the transaction data analysis
transaction_analysis_result, large_transactions, short_interval_transactions, high_volume_transactions, circular_transactions, split_transactions, stats = transaction_data_analysis(df_transactions_filtered)

# Print the result of transaction data analysis
print("Transaction Data Analysis Result:", transaction_analysis_result)

# Detailed results for better understanding
print("Large Transactions:")
print(large_transactions)

print("Short Interval Transactions:")
print(short_interval_transactions)

print("High Volume Transactions:")
print(high_volume_transactions)

print("Circular Transactions:")
print(circular_transactions)

print("Split Transactions:")
print(split_transactions)

# Print the statistics
print("\nStatistics on Suspicious Transactions:")
for key, value in stats.items():
    print(f"{key}: {value:.2f}%")
