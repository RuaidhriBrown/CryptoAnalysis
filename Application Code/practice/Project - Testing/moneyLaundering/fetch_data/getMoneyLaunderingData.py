import requests
import pandas as pd

# Replace with your own Etherscan API Key
API_KEY = 'RNT7YF8V9S21MYZCWFF2ICW235EAME6XT8'
BASE_URL = 'https://api.etherscan.io/api'

def get_account_balance(address):
    url = f'{BASE_URL}?module=account&action=balance&address={address}&tag=latest&apikey={API_KEY}'
    response = requests.get(url).json()
    balance = int(response['result']) / 10**18
    return balance

def get_normal_transactions(address):
    url = f'{BASE_URL}?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={API_KEY}'
    response = requests.get(url).json()
    if response['status'] == '1':
        return pd.DataFrame(response['result'])
    else:
        return pd.DataFrame()

def get_erc20_transactions(address):
    url = f'{BASE_URL}?module=account&action=tokentx&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={API_KEY}'
    response = requests.get(url).json()
    if response['status'] == '1':
        return pd.DataFrame(response['result'])
    else:
        return pd.DataFrame()

def detect_money_laundering(fifo_ratio, sent_received_ratio, token_diversity_index, balance, thresholds):
    """
    Detects if the account is likely involved in money laundering based on provided thresholds.
    """
    suspicious = False

    # Check Fast-In Fast-Out Ratio
    if fifo_ratio > thresholds['fifo_ratio']:
        suspicious = True
        print("Suspicious due to high Fast-In Fast-Out Ratio.")

    # Check Sent/Received Ratio
    if thresholds['sent_received_min'] <= sent_received_ratio <= thresholds['sent_received_max']:
        suspicious = True
        print("Suspicious due to Sent/Received Ratio being close to 1.")

    # Check Token Diversity Index
    if token_diversity_index > thresholds['token_diversity']:
        suspicious = True
        print("Suspicious due to high Token Diversity Index.")

    # Additional logic can be added, such as checking balance anomalies
    if balance < thresholds['balance_min']:
        suspicious = True
        print("Suspicious due to low balance after multiple transactions.")

    return suspicious

def analyze_wallet(address):
    balance = get_account_balance(address)
    normal_tx = get_normal_transactions(address)
    erc20_tx = get_erc20_transactions(address)

    print(f'Address: {address}')
    print(f'Balance: {balance} ETH')

    # Initialize analysis variables
    fifo_ratio = 0
    sent_received_ratio = 0
    token_diversity_index = 0

    # Fast-In Fast-Out Ratio
    if not normal_tx.empty:
        normal_tx['time_diff'] = pd.to_datetime(normal_tx['timeStamp'], unit='s').diff().dt.total_seconds()
        fifo_ratio = normal_tx[normal_tx['time_diff'] < 3600].shape[0] / normal_tx.shape[0]
        print(f'Fast-In Fast-Out Ratio: {fifo_ratio}')

    # Sent/Received Ratio
    if not normal_tx.empty:
        total_sent = normal_tx[normal_tx['from'].str.lower() == address.lower()]['value'].astype(float).sum() / 10**18
        total_received = normal_tx[normal_tx['to'].str.lower() == address.lower()]['value'].astype(float).sum() / 10**18
        sent_received_ratio = total_sent / total_received if total_received != 0 else float('inf')
        print(f'Sent/Received Ratio: {sent_received_ratio}')

    # ERC20 Token Diversity
    if not erc20_tx.empty:
        token_diversity_index = erc20_tx['tokenSymbol'].nunique()
        print(f'Token Diversity Index: {token_diversity_index}')

    # Save the data to CSV files (optional)
    normal_tx.to_csv(f'{address}_normal_tx.csv', index=False)
    internal_tx.to_csv(f'{address}_internal_tx.csv', index=False)
    erc20_tx.to_csv(f'{address}_erc20_tx.csv', index=False)

    # Define thresholds for money laundering detection
    thresholds = {
        'fifo_ratio': 0.7,  # Threshold for Fast-In Fast-Out Ratio
        'sent_received_min': 0.95,  # Minimum threshold for Sent/Received Ratio close to 1
        'sent_received_max': 1.05,  # Maximum threshold for Sent/Received Ratio close to 1
        'token_diversity': 50,  # Threshold for Token Diversity Index
        'balance_min': 0.05  # Threshold for minimum balance
    }

    # Detect potential money laundering
    suspicious = detect_money_laundering(
        fifo_ratio=fifo_ratio,
        sent_received_ratio=sent_received_ratio,
        token_diversity_index=token_diversity_index,
        balance=balance,
        thresholds=thresholds
    )

    if suspicious:
        print("The account is likely involved in money laundering.")
    else:
        print("The account does not appear to be involved in money laundering.")

# Example usage with the provided address
upbit_hack_address = '0xa09871AEadF4994Ca12f5c0b6056BBd1d343c029'
analyze_wallet(upbit_hack_address)