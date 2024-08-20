import pandas as pd
import requests
import time
from tqdm import tqdm

# Load the dataset
file_path = 'datasets/ethereum/moneyLaundering/Money_Laundering_Wallets_List.csv'
wallets_df = pd.read_csv(file_path)

# Your Etherscan API key
api_key = 'RNT7YF8V9S21MYZCWFF2ICW235EAME6XT8'

# Function to get the balance of an Ethereum wallet using Etherscan API
def get_eth_balance(address):
    url = f'https://api.etherscan.io/api'
    params = {
        'module': 'account',
        'action': 'balance',
        'address': address,
        'tag': 'latest',
        'apikey': api_key
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data['status'] == '1':
        balance = int(data['result']) / 10**18  # Convert from wei to ether
        return balance
    else:
        print(f"Error fetching balance for {address}: {data['message']}")
        return None

# Limiting the number of API calls to 5 per second
def rate_limited_get_eth_balance(address):
    balance = get_eth_balance(address)
    time.sleep(0.2)  # 5 calls per second => 1 call every 0.2 seconds
    return balance

# Initialize tqdm progress bar
tqdm.pandas(desc="Fetching Ethereum Balances")

# Retrieve balances only if missing, with a progress bar
wallets_df['Balance'] = wallets_df.progress_apply(
    lambda row: rate_limited_get_eth_balance(row['Address']) if pd.isna(row['Balance']) else row['Balance'],
    axis=1
)

# Save the updated DataFrame with balances to a new CSV file
output_file_path = 'datasets/ethereum/moneyLaundering/Money_Laundering_Wallets_List_with_Balances.csv'
wallets_df.to_csv(output_file_path, index=False)

print(f"Balances retrieved and saved to {output_file_path}")
