import pandas as pd
from tqdm import tqdm

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

# Expected values for comparison
expected_values = {
    'total Ether sent': 865.6910932,
    'total ether received': 586.4666748,
    'total ether balance': -279.2244185,
    'Avg min between sent tnx': 844.26,
    'Avg min between received tnx': 1093.71,
    'Time Diff between first and last (Mins)': 704785.63,
    'Sent tnx': 721,
    'Received Tnx': 89,
    'Unique Received From Addresses': 40,
    'Unique Sent To Addresses': 118,
    'min value received': 0,
    'max value received': 45.806785,
    'avg val received': 6.589513,
    'min val sent': 0,
    'max val sent': 31.22,
    'avg val sent': 1.200681,
    'total transactions (including tnx to create contract)': 810,
    'Total ERC20 tnxs': 265,
    'ERC20 total Ether received': 35588543.78,
    'ERC20 total ether sent': 35603169.52,
    'ERC20 uniq sent addr': 30,
    'ERC20 uniq rec addr': 54,
    'ERC20 avg time between sent tnx': 0,
    'ERC20 avg time between rec tnx': 0,
    'ERC20 min val rec': 0,
    'ERC20 max val rec': 15000000,
    'ERC20 avg val rec': 265586.1476,
    'ERC20 min val sent': 0,
    'ERC20 max val sent': 16830998.35,
    'ERC20 avg val sent': 271779.92,
    'ERC20 uniq sent token name': 'Cofoundit',
    'ERC20 uniq rec token name': 'Numeraire',
    'ERC20 most sent token type': 'Cofoundit',
    'ERC20_most_rec_token_type': 'Numeraire'
}

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

    # Calculate ERC20 metrics (assuming they are in Wei and need conversion)
    erc20_sent_values = erc20_df_filtered[erc20_df_filtered['from'] == summary_info['Address']]['value'] / 1e18
    erc20_received_values = erc20_df_filtered[erc20_df_filtered['to'] == summary_info['Address']]['value'] / 1e18

    total_erc20_received = erc20_received_values.sum()
    total_erc20_sent = erc20_sent_values.sum()
    unique_sent_token_addresses = erc20_df_filtered[erc20_df_filtered['from'] == summary_info['Address']]['contractAddress'].nunique()
    unique_received_token_addresses = erc20_df_filtered[erc20_df['to'] == summary_info['Address']]['contractAddress'].nunique()

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
        'total transactions (including tnx to create contract)': len(transaction_df),
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
        'ERC20 avg val sent contract': 0,  # Assuming no contract data provided,
        'ERC20 uniq sent token name': most_sent_token_type,
        'ERC20 uniq rec token name': most_received_token_type,
        'ERC20 most sent token type': most_sent_token_type,
        'ERC20_most_rec_token_type': most_received_token_type
    }

    # Convert the combined information into a single line DataFrame
    combined_line = pd.DataFrame([combined_info])
    
    return combined_line

# Function to iteratively remove entries until totals are close
def remove_entries_until_totals_close(transactions_df, erc20_df, address, expected_values):
    # Initial totals
    total_ether_sent = transactions_df[transactions_df['from'] == address]['value'].sum() / 1e18
    total_ether_received = transactions_df[transactions_df['to'] == address]['value'].sum() / 1e18
    print(f"Initial total Ether sent: {total_ether_sent}, Initial total Ether received: {total_ether_received}")

    # Iteratively remove latest entries until totals are close to expected values
    while total_ether_sent > expected_values['total Ether sent'] or total_ether_received > expected_values['total ether received']:
        latest_transaction_timestamp = transactions_df['timeStamp'].max()
        latest_erc20_timestamp = erc20_df['timeStamp'].max()
        
        if total_ether_sent > expected_values['total Ether sent'] and latest_transaction_timestamp >= latest_erc20_timestamp:
            latest_transaction = transactions_df[transactions_df['timeStamp'] == latest_transaction_timestamp]
            latest_transaction_value = latest_transaction[latest_transaction['from'] == address]['value'].sum()
            print(f"Removing latest transaction entry at {latest_transaction_timestamp} with value {latest_transaction_value / 1e18} Ether")
            transactions_df = transactions_df[transactions_df['timeStamp'] != latest_transaction_timestamp]
            total_ether_sent = transactions_df[transactions_df['from'] == address]['value'].sum() / 1e18
        elif total_ether_received > expected_values['total ether received'] and latest_erc20_timestamp > latest_transaction_timestamp:
            latest_erc20 = erc20_df[erc20_df['timeStamp'] == latest_erc20_timestamp]
            latest_erc20_value = latest_erc20[latest_erc20['to'] == address]['value'].sum()
            print(f"Removing latest ERC20 entry at {latest_erc20_timestamp} with value {latest_erc20_value / 1e18} Ether")
            erc20_df = erc20_df[erc20_df['timeStamp'] != latest_erc20_timestamp]
            total_ether_received = transactions_df[transactions_df['to'] == address]['value'].sum() / 1e18

        print(f"Total Ether sent: {total_ether_sent}, Total Ether received: {total_ether_received}")

    return transactions_df, erc20_df

# Function to find the cutoff dates to match the unique address counts
def find_cutoff_dates(transactions_df, erc20_df, address):
    transactions_cutoff_date = None
    erc20_cutoff_date = None

    # Filter transactions and ERC20 data to include only relevant entries
    transactions_df_filtered = transactions_df[
        (transactions_df['from'] == address) | (transactions_df['to'] == address)
    ]
    erc20_df_filtered = erc20_df[
        (erc20_df['from'] == address) | (erc20_df['to'] == address)
    ]

    # Find transactions cutoff date
    for i in tqdm(range(len(transactions_df_filtered))):
        temp_df = transactions_df_filtered.iloc[:i+1]
        if (temp_df['from'].nunique() >= 118 and temp_df['to'].nunique() >= 40):
            transactions_cutoff_date = temp_df['timeStamp'].max()
            break
    
    # Find ERC20 cutoff date
    for i in tqdm(range(len(erc20_df_filtered))):
        temp_df = erc20_df_filtered.iloc[:i+1]
        if (temp_df['from'].nunique() >= 30 and temp_df['to'].nunique() >= 54):
            erc20_cutoff_date = temp_df['timeStamp'].max()
            break

    return transactions_cutoff_date, erc20_cutoff_date

# Loop to iterate through dates and find the best match
address_to_lookup = '0x00009277775ac7d0d59eaad8fee3d10ac6c805e8'

# Remove entries until totals are close to expected values
transactions_df, erc20_df = remove_entries_until_totals_close(transactions_df, erc20_df, address_to_lookup, expected_values)

# Find the initial cutoff dates where unique address counts match the required thresholds
transactions_cutoff_date, erc20_cutoff_date = find_cutoff_dates(transactions_df, erc20_df, address_to_lookup)

# Check if the cutoff dates are found
if transactions_cutoff_date is None or erc20_cutoff_date is None:
    print("Cutoff dates not found. Please check the data.")
else:
    while True:
        # Filter the data to these initial cutoff dates
        current_transactions_df = transactions_df[
            (transactions_df['timeStamp'] <= transactions_cutoff_date) &
            ((transactions_df['from'] == address_to_lookup) | (transactions_df['to'] == address_to_lookup))
        ].copy()
        current_erc20_df = erc20_df[
            (erc20_df['timeStamp'] <= erc20_cutoff_date) &
            ((erc20_df['from'] == address_to_lookup) | (erc20_df['to'] == address_to_lookup))
        ].copy()

        # Display the filtered data for comparison
        combined_line = combine_data(address_to_lookup, current_transactions_df, current_erc20_df)
        calculated_values = combined_line.iloc[0].to_dict()
        
        # Check if calculated values match the expected values
        match = all(
            abs(calculated_values[key] - expected_values[key]) < 1e-6
            for key in expected_values if key in calculated_values
        )
        
        if match:
            print("Matching values found.")
            for index, row in combined_line.iterrows():
                for col, value in row.items():
                    print(f'{col}: {value}')
            break
        else:
            print("No match yet. Removing latest entries and trying again.")
            latest_transaction_timestamp = current_transactions_df['timeStamp'].max()
            latest_erc20_timestamp = current_erc20_df['timeStamp'].max()
            if latest_transaction_timestamp >= latest_erc20_timestamp and not current_transactions_df.empty:
                latest_transaction_value = current_transactions_df[
                    (current_transactions_df['timeStamp'] == latest_transaction_timestamp) & 
                    (current_transactions_df['from'] == address_to_lookup)
                ]['value'].sum()
                print(f"Removing latest transaction entry at {latest_transaction_timestamp} with value {latest_transaction_value / 1e18} Ether")
                transactions_df = transactions_df[transactions_df['timeStamp'] != latest_transaction_timestamp]
            elif latest_erc20_timestamp > latest_transaction_timestamp and not current_erc20_df.empty:
                latest_erc20_value = current_erc20_df[
                    (current_erc20_df['timeStamp'] == latest_erc20_timestamp) & 
                    (current_erc20_df['to'] == address_to_lookup)
                ]['value'].sum()
                print(f"Removing latest ERC20 entry at {latest_erc20_timestamp} with value {latest_erc20_value / 1e18} Ether")
                erc20_df = erc20_df[erc20_df['timeStamp'] != latest_erc20_timestamp]
            transactions_cutoff_date, erc20_cutoff_date = find_cutoff_dates(transactions_df, erc20_df, address_to_lookup)
            if transactions_cutoff_date is None or erc20_cutoff_date is None:
                print("Cutoff dates not found after removing entries. Stopping.")
                break
