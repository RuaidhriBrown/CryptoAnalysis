import pandas as pd

# Load the data from the provided CSV files
file_path_transactions = 'combined_transactions.csv'
file_path_erc20 = 'combined_er20.csv'

transactions_df = pd.read_csv(file_path_transactions)
erc20_df = pd.read_csv(file_path_erc20)

# Ensure the value columns are converted to numeric types
transactions_df['value'] = pd.to_numeric(transactions_df['value'], errors='coerce')
erc20_df['value'] = pd.to_numeric(erc20_df['value'], errors='coerce')

# Convert timestamps to datetime
transactions_df['timeStamp'] = pd.to_datetime(transactions_df['timeStamp'], unit='s')
erc20_df['timeStamp'] = pd.to_datetime(erc20_df['timeStamp'], unit='s')

# Define the cutoff date and time
cutoff_date = pd.to_datetime("2021-05-26 11:31:00")

# Filter data to include only transactions before the cutoff date
transactions_df = transactions_df[transactions_df['timeStamp'] < cutoff_date]
erc20_df = erc20_df[erc20_df['timeStamp'] < cutoff_date]

# Function to combine data into a single line as specified
def combine_data(address, transactions_df, erc20_df):
    # Filter transactions and summary data for the given address
    summary_df = transactions_df[transactions_df['Address'] == address]
    transaction_df = transactions_df[(transactions_df['from'] == address) | (transactions_df['to'] == address)]
    erc20_df_filtered = erc20_df[(erc20_df['from'] == address) | (erc20_df['to'] == address)]
    
    if summary_df.empty:
        raise ValueError(f"No summary data found for address {address}")
    
    # Assuming first row of summary_df is the summary data
    summary_info = summary_df.iloc[0].to_dict()
    
    # Calculate average time between sent transactions
    sent_timestamps = transaction_df[transaction_df['from'] == summary_info['Address']]['timeStamp']
    avg_time_between_sent = (sent_timestamps.diff().mean().total_seconds() / 60) if len(sent_timestamps) > 1 else 0

    # Calculate average time between received transactions
    received_timestamps = transaction_df[transaction_df['to'] == summary_info['Address']]['timeStamp']
    avg_time_between_received = (received_timestamps.diff().mean().total_seconds() / 60) if len(received_timestamps) > 1 else 0

    # Calculate time difference between first and last transactions
    time_diff_first_last = (transaction_df['timeStamp'].max() - transaction_df['timeStamp'].min()).total_seconds() / 60

    # Count unique received from and sent to addresses
    unique_received_from = transaction_df[transaction_df['to'] == summary_info['Address']]['from'].nunique()
    unique_sent_to = transaction_df[transaction_df['from'] == summary_info['Address']]['to'].nunique()

    # Calculate min, max, and average values for sent and received transactions
    sent_values = transaction_df[transaction_df['from'] == summary_info['Address']]['value']
    received_values = transaction_df[transaction_df['to'] == summary_info['Address']]['value']

    min_val_received = received_values.min() / 1e18  # Assuming values are in Wei
    max_val_received = received_values.max() / 1e18  # Assuming values are in Wei
    avg_val_received = received_values.mean() / 1e18  # Assuming values are in Wei

    min_val_sent = sent_values.min() / 1e18  # Assuming values are in Wei
    max_val_sent = sent_values.max() / 1e18  # Assuming values are in Wei
    avg_val_sent = sent_values.mean() / 1e18  # Assuming values are in Wei

    # Calculate total Ether sent, received, and balance
    total_ether_sent = sent_values.sum() / 1e18  # Assuming values are in Wei
    total_ether_received = received_values.sum() / 1e18  # Assuming values are in Wei
    total_ether_balance = total_ether_received - total_ether_sent

    # Calculate ERC20 metrics
    erc20_sent_values = erc20_df_filtered[erc20_df_filtered['from'] == summary_info['Address']]['value']
    erc20_received_values = erc20_df_filtered[erc20_df_filtered['to'] == summary_info['Address']]['value']

    total_erc20_received = erc20_received_values.sum()
    total_erc20_sent = erc20_sent_values.sum()
    unique_sent_token_addresses = erc20_df_filtered[erc20_df_filtered['from'] == summary_info['Address']]['contractAddress'].nunique()
    unique_received_token_addresses = erc20_df_filtered[erc20_df_filtered['to'] == summary_info['Address']]['contractAddress'].nunique()

    min_erc20_val_received = erc20_received_values.min()
    max_erc20_val_received = erc20_received_values.max()
    avg_erc20_val_received = erc20_received_values.mean()

    min_erc20_val_sent = erc20_sent_values.min()
    max_erc20_val_sent = erc20_sent_values.max()
    avg_erc20_val_sent = erc20_sent_values.mean()

    # Calculate average time between ERC20 transactions
    erc20_sent_timestamps = erc20_df_filtered[erc20_df_filtered['from'] == summary_info['Address']]['timeStamp']
    avg_erc20_time_between_sent = (erc20_sent_timestamps.diff().mean().total_seconds() / 60) if len(erc20_sent_timestamps) > 1 else 0

    erc20_received_timestamps = erc20_df_filtered[erc20_df_filtered['to'] == summary_info['Address']]['timeStamp']
    avg_erc20_time_between_received = (erc20_received_timestamps.diff().mean().total_seconds() / 60) if len(erc20_received_timestamps) > 1 else 0

    # Determine the most sent and received token types
    most_sent_token_type = erc20_df_filtered[erc20_df_filtered['from'] == summary_info['Address']]['tokenSymbol'].mode().values[0] if not erc20_df_filtered[erc20_df_filtered['from'] == summary_info['Address']]['tokenSymbol'].mode().empty else ''
    most_received_token_type = erc20_df_filtered[erc20_df_filtered['to'] == summary_info['Address']]['tokenSymbol'].mode().values[0] if not erc20_df_filtered[erc20_df_filtered['to'] == summary_info['Address']]['tokenSymbol'].mode().empty else ''

    # Combine the extracted information into a single dictionary
    combined_info = {
        'Index': 0,
        'Address': summary_info['Address'],
        'FLAG': summary_info['FLAG'],
        'Avg min between sent tnx': avg_time_between_sent,
        'Avg min between received tnx': avg_time_between_received,
        'Time Diff between first and last (Mins)': time_diff_first_last,
        'Sent tnx': len(sent_timestamps),
        'Received Tnx': len(received_timestamps),
        'Number of Created Contracts': 0,  # Assuming no contracts created in the given data
        'Unique Received From Addresses': unique_received_from,
        'Unique Sent To Addresses': unique_sent_to,
        'min value received': min_val_received,
        'max value received': max_val_received,
        'avg val received': avg_val_received,
        'min val sent': min_val_sent,
        'max val sent': max_val_sent,
        'avg val sent': avg_val_sent,
        'min value sent to contract': 0,  # Assuming no contract data provided
        'max val sent to contract': 0,  # Assuming no contract data provided
        'avg value sent to contract': 0,  # Assuming no contract data provided
        'total transactions (including tnx to create contract': len(transaction_df),
        'total Ether sent': total_ether_sent,
        'total ether received': total_ether_received,
        'total ether sent contracts': 0,  # Assuming no contract data provided
        'total ether balance': total_ether_balance,
        'Total ERC20 tnxs': len(erc20_df_filtered),
        'ERC20 total Ether received': total_erc20_received,
        'ERC20 total ether sent': total_erc20_sent,
        'ERC20 total Ether sent contract': 0,  # Assuming no contract data provided
        'ERC20 uniq sent addr': unique_sent_token_addresses,
        'ERC20 uniq rec addr': unique_received_token_addresses,
        'ERC20 uniq sent addr.1': 0,  # Assuming no ERC20 data provided
        'ERC20 uniq rec contract addr': 0,  # Assuming no ERC20 data provided
        'ERC20 avg time between sent tnx': avg_erc20_time_between_sent,
        'ERC20 avg time between rec tnx': avg_erc20_time_between_received,
        'ERC20 avg time between rec 2 tnx': 0,  # Assuming no ERC20 data provided
        'ERC20 avg time between contract tnx': 0,  # Assuming no ERC20 data provided
        'ERC20 min val rec': min_erc20_val_received,
        'ERC20 max val rec': max_erc20_val_received,
        'ERC20 avg val rec': avg_erc20_val_received,
        'ERC20 min val sent': min_erc20_val_sent,
        'ERC20 max val sent': max_erc20_val_sent,
        'ERC20 avg val sent': avg_erc20_val_sent,
        'ERC20 min val sent contract': 0,  # Assuming no contract data provided
        'ERC20 max val sent contract': 0,  # Assuming no contract data provided
        'ERC20 avg val sent contract': 0,  # Assuming no contract data provided
        'ERC20 uniq sent token name': most_sent_token_type,
        'ERC20 uniq rec token name': most_received_token_type,
        'ERC20 most sent token type': most_sent_token_type,
        'ERC20_most_rec_token_type': most_received_token_type
    }

    # Convert the combined information into a single line DataFrame
    combined_line = pd.DataFrame([combined_info])
    
    return combined_line

# Example usage
address_to_lookup = '0x00009277775ac7d0d59eaad8fee3d10ac6c805e8'
combined_line = combine_data(address_to_lookup, transactions_df, erc20_df)

# Iterate through the combined line and display each value
for index, row in combined_line.iterrows():
    for col, value in row.items():
        print(f'{col}: {value}')
