import pandas as pd
from tqdm import tqdm

# Load the datasets
transactions_df = pd.read_csv('datasets/ethereum/combined_transactions.csv')
erc20_df = pd.read_csv('datasets/ethereum/combined_er20.csv')
addressList_df = pd.read_csv('datasets/ethereum/transaction_dataset_even.csv')

# Get the unique Addresses and corresponding flags
wallet_addresses = addressList_df['Address'].unique()
address_flags = addressList_df.set_index('Address')['FLAG'].to_dict()

# Convert timeStamp to datetime for both datasets
transactions_df['timeStamp'] = pd.to_datetime(transactions_df['timeStamp'])
erc20_df['timeStamp'] = pd.to_datetime(erc20_df['timeStamp'])

# Filter transactions and ERC20 datasets to only include relevant addresses
transactions_df = transactions_df[
    transactions_df['from'].isin(wallet_addresses) | transactions_df['to'].isin(wallet_addresses)
]
erc20_df = erc20_df[
    erc20_df['from'].isin(wallet_addresses) | erc20_df['to'].isin(wallet_addresses)
]

# Initialize a list to store the combined data for each wallet
combined_data_list = []

# Loop over each Address with a progress bar
for wallet_address in tqdm(wallet_addresses, desc="Processing Wallets"):
    combined_data = {
        'Address': wallet_address,
        'FLAG': address_flags.get(wallet_address, 0),
        'Avg min between sent tnx': 0,
        'Avg min between received tnx': 0,
        'Time Diff between first and last (Mins)': 0,
        'Sent tnx': 0,
        'Received Tnx': 0,
        'Unique Received From Addresses': 0,
        'Unique Sent To Addresses': 0,
        'min value received': 0,
        'max value received': 0,
        'avg val received': 0,
        'min val sent': 0,
        'max val sent': 0,
        'avg val sent': 0,
        'total Ether sent': 0,
        'total ether received': 0,
        'total ether balance': 0,
        'Total ERC20 tnxs': 0,
        'ERC20 total Ether received': 0,
        'ERC20 total ether sent': 0,
        'ERC20 uniq sent addr': 0,
        'ERC20 uniq rec addr': 0,
        'ERC20 avg time between sent tnx': 0,
        'ERC20 avg time between rec tnx': 0,
        'ERC20 min val rec': 0,
        'ERC20 max val rec': 0,
        'ERC20 avg val rec': 0,
        'ERC20 min val sent': 0,
        'ERC20 max val sent': 0,
        'ERC20 avg val sent': 0,
        'ERC20 uniq sent token name': '',
        'ERC20 uniq rec token name': '',
        'ERC20 most sent token type': '',
        'ERC20_most_rec_token_type': ''
    }
    # Filter transactions for this Address
    wallet_transactions = transactions_df[
        (transactions_df['from'] == wallet_address) | (transactions_df['to'] == wallet_address)
    ]
    wallet_erc20 = erc20_df[
        (erc20_df['from'] == wallet_address) | (erc20_df['to'] == wallet_address)
    ]

    # Process regular transactions (Ether)
    if not wallet_transactions.empty:
        # Sent transactions
        sent_transactions = wallet_transactions[wallet_transactions['from'] == wallet_address]
        if not sent_transactions.empty:
            sent_timestamps = sent_transactions['timeStamp']
            combined_data['Avg min between sent tnx'] = (sent_timestamps.diff().mean().total_seconds() / 60) if len(sent_timestamps) > 1 else 0
            combined_data['Sent tnx'] = len(sent_timestamps)
            combined_data['total Ether sent'] = sent_transactions['value'].sum() / 1e18
            combined_data['min val sent'] = sent_transactions['value'].min() / 1e18
            combined_data['max val sent'] = sent_transactions['value'].max() / 1e18
            combined_data['avg val sent'] = sent_transactions['value'].mean() / 1e18
        else:
            combined_data['Avg min between sent tnx'] = 0
            combined_data['Sent tnx'] = 0
            combined_data['total Ether sent'] = 0
            combined_data['min val sent'] = 0
            combined_data['max val sent'] = 0
            combined_data['avg val sent'] = 0

        # Received transactions
        received_transactions = wallet_transactions[wallet_transactions['to'] == wallet_address]
        if not received_transactions.empty:
            received_timestamps = received_transactions['timeStamp']
            combined_data['Avg min between received tnx'] = (received_timestamps.diff().mean().total_seconds() / 60) if len(received_timestamps) > 1 else 0
            combined_data['Received Tnx'] = len(received_timestamps)
            combined_data['total ether received'] = received_transactions['value'].sum() / 1e18
            combined_data['min value received'] = received_transactions['value'].min() / 1e18
            combined_data['max value received'] = received_transactions['value'].max() / 1e18
            combined_data['avg val received'] = received_transactions['value'].mean() / 1e18
            combined_data['Unique Received From Addresses'] = received_transactions['from'].nunique()
        else:
            combined_data['Avg min between received tnx'] = 0
            combined_data['Received Tnx'] = 0
            combined_data['total ether received'] = 0
            combined_data['min value received'] = 0
            combined_data['max value received'] = 0
            combined_data['avg val received'] = 0
            combined_data['Unique Received From Addresses'] = 0

        # Time difference between first and last transactions
        combined_data['Time Diff between first and last (Mins)'] = (
            (wallet_transactions['timeStamp'].max() - wallet_transactions['timeStamp'].min()).total_seconds() / 60
        )
    else:
        combined_data['Time Diff between first and last (Mins)'] = 0
        combined_data['total ether received'] = 0  # Ensure this key is always present
        combined_data['total Ether sent'] = 0  # Ensure this key is always present

    # Process ERC20 transactions
    if not wallet_erc20.empty:
        erc20_sent_timestamps = wallet_erc20[wallet_erc20['from'] == wallet_address]['timeStamp']
        erc20_received_timestamps = wallet_erc20[wallet_erc20['to'] == wallet_address]['timeStamp']

        combined_data['ERC20 avg time between sent tnx'] = (erc20_sent_timestamps.diff().mean().total_seconds() / 60) if len(erc20_sent_timestamps) > 1 else 0
        combined_data['ERC20 avg time between rec tnx'] = (erc20_received_timestamps.diff().mean().total_seconds() / 60) if len(erc20_received_timestamps) > 1 else 0
        combined_data['ERC20 total Ether received'] = wallet_erc20[wallet_erc20['to'] == wallet_address]['value'].sum() / 1e18
        combined_data['ERC20 total ether sent'] = wallet_erc20[wallet_erc20['from'] == wallet_address]['value'].sum() / 1e18
        combined_data['ERC20 uniq sent addr'] = wallet_erc20[wallet_erc20['from'] == wallet_address]['contractAddress'].nunique()
        combined_data['ERC20 uniq rec addr'] = wallet_erc20[wallet_erc20['to'] == wallet_address]['contractAddress'].nunique()
        combined_data['ERC20 min val rec'] = wallet_erc20[wallet_erc20['to'] == wallet_address]['value'].min() / 1e18
        combined_data['ERC20 max val rec'] = wallet_erc20[wallet_erc20['to'] == wallet_address]['value'].max() / 1e18
        combined_data['ERC20 avg val rec'] = wallet_erc20[wallet_erc20['to'] == wallet_address]['value'].mean() / 1e18
        combined_data['ERC20 min val sent'] = wallet_erc20[wallet_erc20['from'] == wallet_address]['value'].min() / 1e18
        combined_data['ERC20 max val sent'] = wallet_erc20[wallet_erc20['from'] == wallet_address]['value'].max() / 1e18
        combined_data['ERC20 avg val sent'] = wallet_erc20[wallet_erc20['from'] == wallet_address]['value'].mean() / 1e18
        combined_data['ERC20 most sent token type'] = wallet_erc20[wallet_erc20['from'] == wallet_address]['tokenName'].mode().values[0] if not wallet_erc20[wallet_erc20['from'] == wallet_address]['tokenName'].mode().empty else ''
        combined_data['ERC20_most_rec_token_type'] = wallet_erc20[wallet_erc20['to'] == wallet_address]['tokenName'].mode().values[0] if not wallet_erc20[wallet_erc20['to'] == wallet_address]['tokenName'].mode().empty else ''
    else:
        combined_data['ERC20 avg time between sent tnx'] = 0
        combined_data['ERC20 avg time between rec tnx'] = 0
        combined_data['ERC20 total Ether received'] = 0
        combined_data['ERC20 total ether sent'] = 0
        combined_data['ERC20 uniq sent addr'] = 0
        combined_data['ERC20 uniq rec addr'] = 0
        combined_data['ERC20 min val rec'] = 0
        combined_data['ERC20 max val rec'] = 0
        combined_data['ERC20 avg val rec'] = 0
        combined_data['ERC20 min val sent'] = 0
        combined_data['ERC20 max val sent'] = 0
        combined_data['ERC20 avg val sent'] = 0
        combined_data['ERC20 most sent token type'] = ''
        combined_data['ERC20_most_rec_token_type'] = ''

    # Additional calculations
    combined_data['total ether balance'] = combined_data['total ether received'] - combined_data['total Ether sent']
    combined_data['Sent_Received_Ratio'] = combined_data['Sent tnx'] / (combined_data['Received Tnx'] + 1)
    combined_data['Min_Max_Received_Ratio'] = combined_data['min value received'] / (combined_data['max value received'] + 1)

    # Append combined data to list
    combined_data_list.append(combined_data)

# Convert the list to a DataFrame
combined_data_df = pd.DataFrame(combined_data_list)

# Save the combined data to a CSV file for future use
combined_data_df.to_csv('datasets/ethereum/combined_transaction_data.csv', index=False)
