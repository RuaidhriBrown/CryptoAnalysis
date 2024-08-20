import requests
import pandas as pd
import time
from tqdm import tqdm
from datetime import datetime
import random

# Replace with your own Etherscan API Key
API_KEY = 'RNT7YF8V9S21MYZCWFF2ICW235EAME6XT8'
BASE_URL = 'https://api.etherscan.io/api'
MAX_API_CALLS_PER_SECOND = 5  # Max 5 API calls per second
RETRY_COUNT = 3  # Number of retries for failed requests
START_TIME = '2019-11-27 04:06:40'  # Start time for filtering transactions
MAX_WALLETS_PER_LAYER = 20  # Maximum number of wallets to check per layer
MINIMUM_ETHER_SENT = 1 * 10**18  # Minimum amount (in Wei) for transactions to be considered

# Initialize global dataframes to accumulate all data
all_wallet_summaries = []
all_normal_transactions = []
all_erc20_transactions = []
visited_wallets = set()  # Set to keep track of visited wallets

def get_account_balance(address):
    url = f'{BASE_URL}?module=account&action=balance&address={address}&tag=latest&apikey={API_KEY}'
    response = requests.get(url).json()
    balance = int(response['result']) / 10**18
    return balance

def make_request(url):
    """Helper function to make API requests with retry logic."""
    for attempt in range(RETRY_COUNT):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: Received status code {response.status_code} for URL: {url}")
        except requests.exceptions.RequestException as e:
            print(f"RequestException: {e}")
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
        time.sleep(2 ** attempt)  # Exponential backoff
    return None

def get_normal_transactions(address):
    url = f'{BASE_URL}?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={API_KEY}'
    response = make_request(url)
    time.sleep(1 / MAX_API_CALLS_PER_SECOND)  # Rate limiting
    if response and response.get('status') == '1':
        df = pd.DataFrame(response['result'])
        # Explicitly cast to numeric before converting to datetime
        df['timeStamp'] = pd.to_numeric(df['timeStamp'], errors='coerce')
        df = df[(df['from'].str.lower() == address.lower()) &
                (df['value'].astype(float) >= MINIMUM_ETHER_SENT) &  # Filter by minimum Ether sent
                (pd.to_datetime(df['timeStamp'], unit='s') >= pd.to_datetime(START_TIME))]
        return df
    else:
        return pd.DataFrame()

def get_erc20_transactions(address):
    url = f'{BASE_URL}?module=account&action=tokentx&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={API_KEY}'
    response = make_request(url)
    time.sleep(1 / MAX_API_CALLS_PER_SECOND)  # Rate limiting
    if response and response.get('status') == '1':
        df = pd.DataFrame(response['result'])
        # Explicitly cast to numeric before converting to datetime
        df['timeStamp'] = pd.to_numeric(df['timeStamp'], errors='coerce')
        df = df[(df['from'].str.lower() == address.lower()) &
                (pd.to_datetime(df['timeStamp'], unit='s') >= pd.to_datetime(START_TIME))]
        return df
    else:
        return pd.DataFrame()

def detect_money_laundering(fifo_ratio, small_volume_ratio, zero_out_ratio, network_density, balance, thresholds):
    suspicious_points = 0

    if fifo_ratio > thresholds['fifo_ratio']:
        suspicious_points += 1

    if small_volume_ratio > thresholds['small_volume_ratio']:
        suspicious_points += 1

    if zero_out_ratio > thresholds['zero_out_ratio']:
        suspicious_points += 1

    if network_density > thresholds['network_density']:
        suspicious_points += 1

    if balance < thresholds['balance_min']:
        suspicious_points += 1

    # Define the minimum number of points required to flag the wallet as suspicious
    min_points_required = 1  # Adjust this as needed

    return suspicious_points >= min_points_required

def analyze_wallet(address):
    if address in visited_wallets:
        return False, 0, 0, 0, 0  # Skip if wallet was already analyzed

    visited_wallets.add(address)  # Mark the wallet as visited

    balance = get_account_balance(address)
    normal_tx = get_normal_transactions(address)
    erc20_tx = get_erc20_transactions(address)

    fifo_ratio = 0
    small_volume_ratio = 0
    zero_out_ratio = 0
    network_density = 0

    if not normal_tx.empty:
        # Convert timeStamp to numeric before converting to datetime
        normal_tx['timeStamp'] = pd.to_numeric(normal_tx['timeStamp'], errors='coerce')
        normal_tx['time_diff'] = pd.to_datetime(normal_tx['timeStamp'], unit='s').diff().dt.total_seconds()
        fifo_ratio = normal_tx[normal_tx['time_diff'] < 3600].shape[0] / normal_tx.shape[0]

        # Calculate small-volume transaction ratio
        small_volume_ratio = normal_tx[normal_tx['value'].astype(float) < 1 * 10**18].shape[0] / normal_tx.shape[0]

        # Calculate zero-out ratio (accounts that transfer out nearly all received funds)
        total_received = normal_tx[normal_tx['to'].str.lower() == address.lower()]['value'].astype(float).sum() / 10**18
        total_sent = normal_tx[normal_tx['from'].str.lower() == address.lower()]['value'].astype(float).sum() / 10**18
        zero_out_ratio = total_sent / total_received if total_received != 0 else 0

        # Placeholder for network density calculation (requires network graph analysis)
        network_density = 1  # This would be calculated using network analysis techniques

    thresholds = {
        'fifo_ratio': 0.7,
        'small_volume_ratio': 0.6,  # Threshold for small-volume transactions
        'zero_out_ratio': 0.9,  # Threshold for zero-out middle accounts
        'network_density': 1,  # Placeholder threshold for network density
        'balance_min': 0.1  # Minimum balance threshold
    }

    suspicious = detect_money_laundering(
        fifo_ratio=fifo_ratio,
        small_volume_ratio=small_volume_ratio,
        zero_out_ratio=zero_out_ratio,
        network_density=network_density,
        balance=balance,
        thresholds=thresholds
    )

    # Add 'Flag' column to transactions
    normal_tx['Flag'] = 'Yes' if suspicious else 'No'
    erc20_tx['Flag'] = 'Yes' if suspicious else 'No'

    # Save analysis summary to global list
    summary = {
        'Address': address,
        'Balance (ETH)': balance,
        'Fast-In Fast-Out Ratio': fifo_ratio,
        'Small-Volume Ratio': small_volume_ratio,
        'Zero-Out Ratio': zero_out_ratio,
        'Network Density': network_density,
        'Flagged': 'Yes' if suspicious else 'No'
    }
    all_wallet_summaries.append(summary)

    # Append transactions to global dataframes
    normal_tx['Address'] = address
    erc20_tx['Address'] = address
    all_normal_transactions.append(normal_tx)
    all_erc20_transactions.append(erc20_tx)

    return suspicious, normal_tx['Flag'].value_counts().get('Yes', 0), erc20_tx['Flag'].value_counts().get('Yes', 0), len(normal_tx), len(erc20_tx)

def analyze_interacted_wallets(address):
    normal_tx = get_normal_transactions(address)
    erc20_tx = get_erc20_transactions(address)

    # Check if 'to' column exists before using it
    if 'to' in normal_tx.columns:
        normal_to_addresses = set(normal_tx['to'].str.lower())
    else:
        normal_to_addresses = set()

    if 'to' in erc20_tx.columns:
        erc20_to_addresses = set(erc20_tx['to'].str.lower())
    else:
        erc20_to_addresses = set()

    interacted_addresses = normal_to_addresses | erc20_to_addresses
    interacted_addresses.discard(address.lower())  # Remove the original address
    interacted_addresses = list(interacted_addresses)  # Convert to list for further processing

    # Limit the number of wallets checked to MAX_WALLETS_PER_LAYER
    if len(interacted_addresses) > MAX_WALLETS_PER_LAYER:
        interacted_addresses = random.sample(interacted_addresses, MAX_WALLETS_PER_LAYER)

    total_wallets = len(interacted_addresses)
    flagged_wallets = 0
    total_normal_tx = 0
    total_flagged_normal_tx = 0
    total_erc20_tx = 0
    total_flagged_erc20_tx = 0
    flagged_addresses = []

    with tqdm(total=total_wallets, desc="Processing wallets") as pbar:
        for interacted_address in interacted_addresses:
            suspicious, flagged_normal_tx, flagged_erc20_tx, normal_tx_count, erc20_tx_count = analyze_wallet(interacted_address)
            if suspicious:
                flagged_wallets += 1
                flagged_addresses.append(interacted_address)
            total_flagged_normal_tx += flagged_normal_tx
            total_normal_tx += normal_tx_count
            total_flagged_erc20_tx += flagged_erc20_tx
            total_erc20_tx += erc20_tx_count

            # Avoid division by zero in progress bar updates
            pbar.set_postfix({
                'Flagged Wallets': f'{flagged_wallets}/{total_wallets} ({flagged_wallets/total_wallets*100:.2f}%)',
                'Flagged Normal TX': f'{total_flagged_normal_tx}/{total_normal_tx} ({(total_flagged_normal_tx/total_normal_tx*100):.2f}%)' if total_normal_tx > 0 else 'N/A',
                'Flagged ERC20 TX': f'{total_flagged_erc20_tx}/{total_erc20_tx} ({(total_flagged_erc20_tx/total_erc20_tx*100):.2f}%)' if total_erc20_tx > 0 else 'N/A'
            })
            pbar.update(1)

    # Process next layer for flagged addresses
    for flagged_address in flagged_addresses:
        print(f"Processing second layer for flagged wallet: {flagged_address}")
        analyze_interacted_wallets(flagged_address)

    # Save all data to CSV after processing all wallets
    all_wallet_summaries_df = pd.DataFrame(all_wallet_summaries)
    all_wallet_summaries_df.to_csv('all_wallet_summaries.csv', index=False)

    all_normal_transactions_df = pd.concat(all_normal_transactions, ignore_index=True)
    all_normal_transactions_df.to_csv('all_normal_transactions.csv', index=False)

    all_erc20_transactions_df = pd.concat(all_erc20_transactions, ignore_index=True)
    all_erc20_transactions_df.to_csv('all_erc20_transactions.csv', index=False)

# Example usage with the provided address
upbit_hack_address = '0xa09871AEadF4994Ca12f5c0b6056BBd1d343c029'
analyze_interacted_wallets(upbit_hack_address)
