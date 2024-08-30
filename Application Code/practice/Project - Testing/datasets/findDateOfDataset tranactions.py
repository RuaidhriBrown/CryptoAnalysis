import pandas as pd
import numpy as np
from tqdm import tqdm

def analyze_transactions(df, address_to_lookup):
    # Convert the timestamp to a datetime object for easier calculations
    df['timeStamp'] = pd.to_datetime(df['timeStamp'], unit='s')

    # Filter by address
    address = address_to_lookup

    def calculate_metrics(dataframe):
        total_ether_sent = dataframe[dataframe['from'] == address]['value'].sum()
        total_ether_received = dataframe[dataframe['to'] == address]['value'].sum()
        total_ether_balance = total_ether_received - total_ether_sent

        sent_transactions = dataframe[dataframe['from'] == address].sort_values('timeStamp')
        time_diffs_sent = sent_transactions['timeStamp'].diff().dropna().dt.total_seconds() / 60
        avg_time_between_sent = time_diffs_sent.mean()

        received_transactions = dataframe[dataframe['to'] == address].sort_values('timeStamp')
        time_diffs_received = received_transactions['timeStamp'].diff().dropna().dt.total_seconds() / 60
        avg_time_between_received = time_diffs_received.mean()

        total_time_diff = (dataframe['timeStamp'].max() - dataframe['timeStamp'].min()).total_seconds() / 60

        sent_tnx_count = len(sent_transactions)
        received_tnx_count = len(received_transactions)

        unique_sent_to_addresses = dataframe[dataframe['from'] == address]['to'].nunique()
        unique_received_from_addresses = dataframe[dataframe['to'] == address]['from'].nunique()

        min_val_sent = sent_transactions['value'].min()
        max_val_sent = sent_transactions['value'].max()
        avg_val_sent = sent_transactions['value'].mean()

        min_val_received = received_transactions['value'].min()
        max_val_received = received_transactions['value'].max()
        avg_val_received = received_transactions['value'].mean()

        total_transactions = len(dataframe)

        erc20_transactions = dataframe[dataframe['input'].str.contains('0xa9059cbb', na=False)]
        erc20_sent = erc20_transactions[erc20_transactions['from'] == address]
        erc20_received = erc20_transactions[erc20_transactions['to'] == address]
        erc20_total_sent = erc20_sent['value'].sum()
        erc20_total_received = erc20_received['value'].sum()
        erc20_unique_sent_addr = erc20_sent['to'].nunique()
        erc20_unique_rec_addr = erc20_received['from'].nunique()
        erc20_avg_time_between_sent = erc20_sent['timeStamp'].diff().dropna().dt.total_seconds().mean() / 60
        erc20_avg_time_between_rec = erc20_received['timeStamp'].diff().dropna().dt.total_seconds().mean() / 60
        erc20_min_val_sent = erc20_sent['value'].min()
        erc20_max_val_sent = erc20_sent['value'].max()
        erc20_avg_val_sent = erc20_sent['value'].mean()
        erc20_min_val_rec = erc20_received['value'].min()
        erc20_max_val_rec = erc20_received['value'].max()
        erc20_avg_val_rec = erc20_received['value'].mean()

        return {
            'total Ether sent': total_ether_sent,
            'total ether received': total_ether_received,
            'total ether balance': total_ether_balance,
            'Avg min between sent tnx': avg_time_between_sent,
            'Avg min between received tnx': avg_time_between_received,
            'Time Diff between first and last (Mins)': total_time_diff,
            'Sent tnx': sent_tnx_count,
            'Received Tnx': received_tnx_count,
            'Unique Received From Addresses': unique_received_from_addresses,
            'Unique Sent To Addresses': unique_sent_to_addresses,
            'min value received': min_val_received,
            'max value received': max_val_received,
            'avg val received': avg_val_received,
            'min val sent': min_val_sent,
            'max val sent': max_val_sent,
            'avg val sent': avg_val_sent,
            'total transactions (including tnx to create contract)': total_transactions,
            'Total ERC20 tnxs': len(erc20_transactions),
            'ERC20 total Ether received': erc20_total_received,
            'ERC20 total ether sent': erc20_total_sent,
            'ERC20 uniq sent addr': erc20_unique_sent_addr,
            'ERC20 uniq rec addr': erc20_unique_rec_addr,
            'ERC20 avg time between sent tnx': erc20_avg_time_between_sent,
            'ERC20 avg time between rec tnx': erc20_avg_time_between_rec,
            'ERC20 min val rec': erc20_min_val_rec,
            'ERC20 max val rec': erc20_max_val_rec,
            'ERC20 avg val rec': erc20_avg_val_rec,
            'ERC20 min val sent': erc20_min_val_sent,
            'ERC20 max val sent': erc20_max_val_sent,
            'ERC20 avg val sent': erc20_avg_val_sent,
        }

    # Initial metrics with all data
    initial_results = calculate_metrics(df)
    
    # Define the target values in the correct scale
    target_total_ether_sent = 8.656910932e+20
    target_total_ether_received = 5.864666748e+20

    # Metrics after removing transactions to approach target values
    df_filtered = df.copy()

    with tqdm(total=len(df), dynamic_ncols=True) as pbar:
        while True:
            current_metrics = calculate_metrics(df_filtered)
            abs_diff_sent = abs(current_metrics['total Ether sent'] - target_total_ether_sent)
            abs_diff_received = abs(current_metrics['total ether received'] - target_total_ether_received)
            if abs_diff_sent < 1e+13 and abs_diff_received < 1e+13:  # Further adjusted threshold
                break
            if (current_metrics['total Ether sent'] < target_total_ether_sent) or (current_metrics['total ether received'] < target_total_ether_received):
                break
            last_transaction_date = df_filtered[(df_filtered['from'] == address) | (df_filtered['to'] == address)]['timeStamp'].max()
            last_transaction = df_filtered[((df_filtered['from'] == address) | (df_filtered['to'] == address)) & (df_filtered['timeStamp'] == last_transaction_date)]
            df_filtered = df_filtered.drop(last_transaction.index)
            removed_sent = last_transaction[last_transaction['from'] == address]['value'].sum()
            removed_received = last_transaction[last_transaction['to'] == address]['value'].sum()
            current_metrics['total Ether sent'] -= removed_sent
            current_metrics['total ether received'] -= removed_received
            abs_diff_sent = abs(current_metrics['total Ether sent'] - target_total_ether_sent)
            abs_diff_received = abs(current_metrics['total ether received'] - target_total_ether_received)
            pbar.set_postfix({
                'total Ether sent': current_metrics['total Ether sent'],
                'total ether received': current_metrics['total ether received']
            })
            pbar.update(1)
            
    new_results = current_metrics
    print()
    print(f"Time Diff between first and last (Mins): {current_metrics['Time Diff between first and last (Mins)']}")
    print(f"Sent tnx: {current_metrics['Sent tnx']}")
    return initial_results, new_results

# Load the CSV file
file_path = 'transactions.csv'  # Update this to your actual file path
df_new = pd.read_csv(file_path)

# Specify the address to look up
address_to_lookup = '0x00009277775ac7d0d59eaad8fee3d10ac6c805e8'

# Perform the analysis
initial_results, new_results = analyze_transactions(df_new, address_to_lookup)

# Print the results
print("Initial Transaction Analysis Results:", initial_results)
print("New Transaction Analysis Results (after removing transactions):", new_results)
