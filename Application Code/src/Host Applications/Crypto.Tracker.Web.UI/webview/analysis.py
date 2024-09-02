from django.db.models import Q
from .moneyLaundering_model_loader import load_moneyLaundering_wallet_model, load_moneyLaundering_transaction_model, load_moneyLaundering_erc20_model
from .phishings_models_loader import load_phishing_wallet_model, load_phishing_transaction_model, load_phishing_erc20_model
from .models import CompletedAnalysis, WalletAnalysis, EthereumTransaction, ERC20Transaction, Wallet
import pandas as pd
import numpy as np
from .transaction_utils import get_balance
from django.core.exceptions import ObjectDoesNotExist


def combine_data(wallet_address, transactions, erc20_transactions):
    # Convert the QuerySet to DataFrame for easier manipulation
    transactions_df = pd.DataFrame(list(transactions.values()))
    erc20_df = pd.DataFrame(list(erc20_transactions.values()))

    # Initialize values to handle cases where there are no transactions
    avg_time_between_sent = 0
    avg_time_between_received = 0
    time_diff_first_last = 0
    unique_received_from = 0
    unique_sent_to = 0
    min_val_received = 0
    max_val_received = 0
    avg_val_received = 0
    min_val_sent = 0
    max_val_sent = 0
    avg_val_sent = 0
    total_ether_sent = 0
    total_ether_received = 0
    total_ether_balance = 0
    total_erc20_received = 0
    total_erc20_sent = 0
    unique_sent_token_addresses = 0
    unique_received_token_addresses = 0
    min_erc20_val_received = 0
    max_erc20_val_received = 0
    avg_erc20_val_received = 0
    min_erc20_val_sent = 0
    max_erc20_val_sent = 0
    avg_erc20_val_sent = 0
    avg_erc20_time_between_sent = 0
    avg_erc20_time_between_received = 0
    most_sent_token_type = ''
    most_received_token_type = ''
    sent_received_ratio = 0
    min_max_received_ratio = 0

    if not transactions_df.empty:
        # Calculate average time between sent transactions
        sent_timestamps = transactions_df[transactions_df['from_address'] == wallet_address]['timestamp']
        avg_time_between_sent = (sent_timestamps.diff().mean().total_seconds() / 60) if len(sent_timestamps) > 1 else 0

        # Calculate average time between received transactions
        received_timestamps = transactions_df[transactions_df['to_address'] == wallet_address]['timestamp']
        avg_time_between_received = (received_timestamps.diff().mean().total_seconds() / 60) if len(received_timestamps) > 1 else 0

        # Calculate time difference between first and last transactions
        time_diff_first_last = (transactions_df['timestamp'].max() - transactions_df['timestamp'].min()).total_seconds() / 60

        # Count unique received from and sent to addresses
        unique_received_from = transactions_df[transactions_df['to_address'] == wallet_address]['from_address'].nunique()
        unique_sent_to = transactions_df[transactions_df['from_address'] == wallet_address]['to_address'].nunique()

        # Calculate min, max, and average values for sent and received transactions
        sent_values = transactions_df[transactions_df['from_address'] == wallet_address]['value']
        received_values = transactions_df[transactions_df['to_address'] == wallet_address]['value']

        min_val_received = received_values.min() / 1e18 if not received_values.empty else 0
        max_val_received = received_values.max() / 1e18 if not received_values.empty else 0
        avg_val_received = received_values.mean() / 1e18 if not received_values.empty else 0

        min_val_sent = sent_values.min() / 1e18 if not sent_values.empty else 0
        max_val_sent = sent_values.max() / 1e18 if not sent_values.empty else 0
        avg_val_sent = sent_values.mean() / 1e18 if not sent_values.empty else 0

        # Calculate total Ether sent, received, and balance
        total_ether_sent = sent_values.sum() / 1e18 if not sent_values.empty else 0
        total_ether_received = received_values.sum() / 1e18 if not received_values.empty else 0
        total_ether_balance = total_ether_received - total_ether_sent

        # Feature Engineering: Adding interaction terms
        sent_received_ratio = len(sent_timestamps) / (len(received_timestamps) + 1)
        min_max_received_ratio = min_val_received / (max_val_received + 1)

    if not erc20_df.empty:
        # Calculate ERC20 metrics (assuming they are in Wei and need conversion)
        erc20_sent_values = erc20_df[erc20_df['from_address'] == wallet_address]['value'] / 1e18
        erc20_received_values = erc20_df[erc20_df['to_address'] == wallet_address]['value'] / 1e18

        total_erc20_received = erc20_received_values.sum() if not erc20_received_values.empty else 0
        total_erc20_sent = erc20_sent_values.sum() if not erc20_sent_values.empty else 0
        unique_sent_token_addresses = erc20_df[erc20_df['from_address'] == wallet_address]['contract_address'].nunique()
        unique_received_token_addresses = erc20_df[erc20_df['to_address'] == wallet_address]['contract_address'].nunique()

        min_erc20_val_received = erc20_received_values.min() if not erc20_received_values.empty else 0
        max_erc20_val_received = erc20_received_values.max() if not erc20_received_values.empty else 0
        avg_erc20_val_received = erc20_received_values.mean() if not erc20_received_values.empty else 0

        min_erc20_val_sent = erc20_sent_values.min() if not erc20_sent_values.empty else 0
        max_erc20_val_sent = erc20_sent_values.max() if not erc20_sent_values.empty else 0
        avg_erc20_val_sent = erc20_sent_values.mean() if not erc20_sent_values.empty else 0

        # Calculate average time between ERC20 transactions
        erc20_sent_timestamps = erc20_df[erc20_df['from_address'] == wallet_address]['timestamp']
        avg_erc20_time_between_sent = (erc20_sent_timestamps.diff().mean().total_seconds() / 60) if len(erc20_sent_timestamps) > 1 else 0

        erc20_received_timestamps = erc20_df[erc20_df['to_address'] == wallet_address]['timestamp']
        avg_erc20_time_between_received = (erc20_received_timestamps.diff().mean().total_seconds() / 60) if len(erc20_received_timestamps) > 1 else 0

        # Determine the most sent and received token types
        most_sent_token_type = erc20_df[erc20_df['from_address'] == wallet_address]['token_name'].mode().values[0] if not erc20_df[erc20_df['from_address'] == wallet_address]['token_name'].mode().empty else ''
        most_received_token_type = erc20_df[erc20_df['to_address'] == wallet_address]['token_name'].mode().values[0] if not erc20_df[erc20_df['to_address'] == wallet_address]['token_name'].mode().empty else ''

    # Combine the extracted information into a single dictionary
    combined_info = {
        'Avg min between sent tnx': avg_time_between_sent,
        'Avg min between received tnx': avg_time_between_received,
        'Time Diff between first and last (Mins)': time_diff_first_last,
        'Sent tnx': len(sent_timestamps),
        'Received Tnx': len(received_timestamps),
        'Unique Received From Addresses': unique_received_from,
        'Unique Sent To Addresses': unique_sent_to,
        'min value received': min_val_received,
        'max value received': max_val_received,
        'avg val received': avg_val_received,
        'min val sent': min_val_sent,
        'max val sent': max_val_sent,
        'avg val sent': avg_val_sent,
        'total Ether sent': total_ether_sent,
        'total ether received': total_ether_received,
        'total ether balance': total_ether_balance,
        'Total ERC20 tnxs': len(erc20_df),
        'ERC20 total Ether received': total_erc20_received,
        'ERC20 total ether sent': total_erc20_sent,
        'ERC20 uniq sent addr': unique_sent_token_addresses,
        'ERC20 uniq rec addr': unique_received_token_addresses,
        'ERC20 avg time between sent tnx': avg_erc20_time_between_sent,
        'ERC20 avg time between rec tnx': avg_erc20_time_between_received,
        'ERC20 min val rec': min_erc20_val_received,
        'ERC20 max val rec': max_erc20_val_received,
        'ERC20 avg val rec': avg_erc20_val_received,
        'ERC20 min val sent': min_erc20_val_sent,
        'ERC20 max val sent': max_erc20_val_sent,
        'ERC20 avg val sent': avg_erc20_val_sent,
        'ERC20 uniq sent token name': most_sent_token_type,
        'ERC20 uniq rec token name': most_received_token_type,
        'ERC20 most sent token type': most_sent_token_type,
        'ERC20_most_rec_token_type': most_received_token_type,
        'Sent_Received_Ratio': sent_received_ratio,
        'Min_Max_Received_Ratio': min_max_received_ratio
    }

    return combined_info


def run_wallet_phishing_model_analysis(wallet_analysis):
    wallet_model, wallet_features = load_moneyLaundering_wallet_model()

    wallet = wallet_analysis.wallet

    # Retrieve Ethereum transactions for the wallet
    transactions = EthereumTransaction.objects.filter(from_address=wallet.address) | EthereumTransaction.objects.filter(to_address=wallet.address)
    erc20_transactions = ERC20Transaction.objects.filter(from_address=wallet.address) | ERC20Transaction.objects.filter(to_address=wallet.address)

    # Combine data to create the features for prediction
    features = combine_data(wallet.address, transactions, erc20_transactions)

    # Convert features to DataFrame and align with the trained model's features
    features_df = pd.DataFrame([features])
    features_df = features_df.reindex(columns=wallet_features, fill_value=0)

    # Perform prediction
    wallet_prediction = wallet_model.predict(features_df)[0]

    # Check if a CompletedAnalysis with the same wallet_analysis and name exists
    completed_analysis, created = CompletedAnalysis.objects.update_or_create(
        wallet_analysis=wallet_analysis,
        name='Wallet Phishing Detection',
        defaults={
            'note': f'Phishing detected: {wallet_prediction == 1}',
            'concluded_happened': (wallet_prediction == 1)
        }
    )

    # If this was an update, you may want to create a new note or update the existing one
    if not created:
        # Optionally, update the note if the analysis was updated
        completed_analysis.save()

    return wallet_prediction == 1


def run_transaction_phishing_model_analysis(wallet_analysis):
    transaction_model, transaction_features = load_phishing_transaction_model()

    wallet = wallet_analysis.wallet

    # Retrieve Ethereum transactions for the wallet
    transactions = EthereumTransaction.objects.filter(Q(from_address=wallet.address) | Q(to_address=wallet.address))

    # Extract necessary features from the transactions
    transactions_df = pd.DataFrame(list(transactions.values('value', 'gas', 'gas_price', 'nonce', 'cumulative_gas_used', 'gas_used')))

    # Feature Engineering: Add any necessary interaction terms
    transactions_df['Gas_Used_Ratio'] = transactions_df['gas_used'] / (transactions_df['gas'] + 1)
    transactions_df['Value_Gas_Ratio'] = transactions_df['value'] / (transactions_df['gas'] + 1)

    # Align the DataFrame with the model's expected features
    transactions_df = transactions_df.reindex(columns=transaction_features, fill_value=0)

    # Perform predictions for each transaction
    transaction_predictions = transaction_model.predict(transactions_df)

    # Get indices of transactions detected as phishing
    phishing_detected_indices = transactions_df.index[transaction_predictions == 1].tolist()

    # Calculate the total number of transactions analyzed
    total_transactions_analyzed = len(transactions_df)

    # Optionally save the analysis result as before
    CompletedAnalysis.objects.update_or_create(
        wallet_analysis=wallet_analysis,
        name='Transaction Phishing Detection',
        defaults={
            'note': f'Phishing detected in {len(phishing_detected_indices)} out of {total_transactions_analyzed} transactions',
            'concluded_happened': (len(phishing_detected_indices) > 0)
        }
    )

    return phishing_detected_indices



def run_erc20_phishing_model_analysis(wallet_analysis):
    erc20_model, erc20_features = load_phishing_erc20_model()

    wallet = wallet_analysis.wallet

    # Retrieve ERC20 transactions for the wallet
    erc20_transactions = ERC20Transaction.objects.filter(Q(from_address=wallet.address) | Q(to_address=wallet.address))

    # Extract necessary features from the transactions
    erc20_df = pd.DataFrame(list(erc20_transactions.values(
        'value', 
        'gas', 
        'gas_price',
        'gas_used',
        'cumulative_gas_used',
        'nonce', 
        'transaction_index',
        'confirmations'
    )))

    # Feature Engineering: Add any necessary interaction terms
    erc20_df['Gas_Used_Ratio'] = erc20_df['gas_used'] / (erc20_df['gas'] + 1)
    erc20_df['Value_Gas_Ratio'] = erc20_df['value'] / (erc20_df['gas'] + 1)
    erc20_df['Transaction_Value_Efficiency'] = erc20_df['value'] / (erc20_df['cumulative_gas_used'] + 1)

    # Align the DataFrame with the model's expected features
    erc20_df = erc20_df.reindex(columns=erc20_features, fill_value=0)

    # Perform predictions for each ERC20 transaction
    erc20_predictions = erc20_model.predict(erc20_df)

    # Get indices of ERC20 transactions detected as phishing
    phishing_detected_indices = erc20_df.index[erc20_predictions == 1].tolist()

    # Calculate the total number of ERC20 transactions analyzed
    total_transactions_analyzed = len(erc20_df)

    # Optionally save the analysis result as before
    CompletedAnalysis.objects.update_or_create(
        wallet_analysis=wallet_analysis,
        name='ERC20 Phishing Detection',
        defaults={
            'note': f'Phishing detected in {len(phishing_detected_indices)} out of {total_transactions_analyzed} ERC20 transactions',
            'concluded_happened': (len(phishing_detected_indices) > 0)
        }
    )

    return phishing_detected_indices


def get_wallet_and_transactions(address):
    """Retrieve wallet and associated transactions."""
    # Ensure the address length is valid for Ethereum addresses
    if len(address) != 42:
        raise ValueError(f"Invalid address length: {len(address)}. Expected 42 characters.")

    try:
        wallet = Wallet.objects.get(address=address)
    except Wallet.DoesNotExist:
        balance_response = get_balance(address)
        
        if 'balance' in balance_response:
            balance = balance_response['balance']
        else:
            balance = 0  # Default to 0 if there's an error fetching the balance

        wallet = Wallet.objects.create(
            address=address,
            balance=balance
        )
    
    normal_tx = EthereumTransaction.objects.filter(from_address=address) | EthereumTransaction.objects.filter(to_address=address)
    erc20_tx = ERC20Transaction.objects.filter(from_address=address) | ERC20Transaction.objects.filter(to_address=address)
    
    normal_tx_df = pd.DataFrame(list(normal_tx.values()))
    erc20_tx_df = pd.DataFrame(list(erc20_tx.values()))
    
    return wallet, normal_tx_df, erc20_tx_df


#   Money Laundering
def calculate_fifo_ratio(normal_tx_df):
    """Calculate Fast-In Fast-Out Ratio."""
    if not normal_tx_df.empty:
        normal_tx_df['time_diff'] = pd.to_datetime(normal_tx_df['timestamp']).diff().dt.total_seconds()
        fifo_ratio = normal_tx_df[normal_tx_df['time_diff'] < 3600].shape[0] / normal_tx_df.shape[0]
        return fifo_ratio
    return 0

def calculate_sent_received_ratio(normal_tx_df, address):
    """Calculate Sent/Received Ratio."""
    if not normal_tx_df.empty:
        total_sent = normal_tx_df[normal_tx_df['from_address'].str.lower() == address.lower()]['value'].astype(float).sum() / 10**18
        total_received = normal_tx_df[normal_tx_df['to_address'].str.lower() == address.lower()]['value'].astype(float).sum() / 10**18
        sent_received_ratio = total_sent / total_received if total_received != 0 else float('inf')
        return sent_received_ratio
    return 0

def calculate_token_diversity_index(erc20_tx_df):
    """Calculate ERC20 Token Diversity Index."""
    if not erc20_tx_df.empty:
        token_diversity_index = erc20_tx_df['contract_address'].nunique()
        return token_diversity_index
    return 0

def detect_money_laundering(fifo_ratio, sent_received_ratio, token_diversity_index, balance, thresholds):
    """Detect potential money laundering."""
    suspicious = False
    reasons = []

    if fifo_ratio > thresholds['fifo_ratio']:
        suspicious = True
        reasons.append("High Fast-In Fast-Out Ratio")

    if thresholds['sent_received_min'] <= sent_received_ratio <= thresholds['sent_received_max']:
        suspicious = True
        reasons.append("Sent/Received Ratio close to 1")

    if token_diversity_index > thresholds['token_diversity']:
        suspicious = True
        reasons.append("High Token Diversity Index")

    if balance is not None and balance < thresholds['balance_min']:
        suspicious = True
        reasons.append("Low balance after multiple transactions")

    return suspicious, reasons

def save_individual_analysis(wallet_analysis, analysis_name, value, threshold, suspicious, reasons):
    """Save or update an individual analysis result."""
    note = f"""
    Analysis of Wallet {wallet_analysis.wallet.address}:
    - Value: {value}
    - Threshold: {threshold}
    - Suspicious Activity Detected: {'Yes' if suspicious else 'No'}
    - Reasons: {', '.join(reasons) if reasons else 'None'}
    """

    # Use update_or_create to avoid duplicate entries
    CompletedAnalysis.objects.update_or_create(
        wallet_analysis=wallet_analysis,
        name=analysis_name,
        defaults={
            'note': note,
            'concluded_happened': suspicious
        }
    )


def analyze_wallet_for_Money_Laundering(wallet_analysis):
    wallet = wallet_analysis.wallet
    """Main function to analyze a wallet for potential money laundering."""
    walletAddress, normal_tx_df, erc20_tx_df = get_wallet_and_transactions(wallet.address)

    print(f'Address: {wallet.address}')
    print(f'Balance: {wallet.balance} ETH')

    thresholds = {
        'fifo_ratio': 0.7,
        'sent_received_min': 0.95,
        'sent_received_max': 1.05,
        'token_diversity': 50,
        'balance_min': 0.1
    }

    # Calculate and save FIFO Ratio
    fifo_ratio = calculate_fifo_ratio(normal_tx_df)
    suspicious_fifo = fifo_ratio > thresholds['fifo_ratio']
    reason_fifo = "High Fast-In Fast-Out Ratio" if suspicious_fifo else ""
    save_individual_analysis(wallet_analysis, "Fast-In Fast-Out Ratio", fifo_ratio, f"> {thresholds['fifo_ratio']}", suspicious_fifo, reason_fifo)

    # Calculate and save Sent/Received Ratio
    sent_received_ratio = calculate_sent_received_ratio(normal_tx_df, wallet.address)
    suspicious_sent_received = thresholds['sent_received_min'] <= sent_received_ratio <= thresholds['sent_received_max']
    reason_sent_received = "Sent/Received Ratio close to 1" if suspicious_sent_received else ""
    save_individual_analysis(wallet_analysis, "Sent/Received Ratio", sent_received_ratio, f"between {thresholds['sent_received_min']} and {thresholds['sent_received_max']}", suspicious_sent_received, reason_sent_received)

    # Calculate and save Token Diversity Index
    token_diversity_index = calculate_token_diversity_index(erc20_tx_df)
    suspicious_token_diversity = token_diversity_index > thresholds['token_diversity']
    reason_token_diversity = "High Token Diversity Index" if suspicious_token_diversity else ""
    save_individual_analysis(wallet_analysis, "Token Diversity Index", token_diversity_index, f"> {thresholds['token_diversity']}", suspicious_token_diversity, reason_token_diversity)

    # Additional logic to aggregate the overall suspicion level can be added here if needed
    overall_suspicious = suspicious_fifo or suspicious_sent_received or suspicious_token_diversity

    if overall_suspicious:
        print("The account is likely involved in money laundering.")
    else:
        print("The account does not appear to be involved in money laundering.")


def run_wallet_moneyLaundering_model_analysis(wallet_analysis):
    wallet_model, wallet_features = load_moneyLaundering_wallet_model()

    wallet = wallet_analysis.wallet

    # Retrieve Ethereum transactions for the wallet
    transactions = EthereumTransaction.objects.filter(from_address=wallet.address) | EthereumTransaction.objects.filter(to_address=wallet.address)
    erc20_transactions = ERC20Transaction.objects.filter(from_address=wallet.address) | ERC20Transaction.objects.filter(to_address=wallet.address)

    # Combine data to create the features for prediction
    features = combine_data(wallet.address, transactions, erc20_transactions)

    # Convert features to DataFrame and align with the trained model's features
    features_df = pd.DataFrame([features])
    features_df = features_df.reindex(columns=wallet_features, fill_value=0)

    # Perform prediction
    wallet_prediction = wallet_model.predict(features_df)[0]

    # Check if a CompletedAnalysis with the same wallet_analysis and name exists
    completed_analysis, created = CompletedAnalysis.objects.update_or_create(
        wallet_analysis=wallet_analysis,
        name='Wallet Money Laundering Detection',
        defaults={
            'note': f'Money Laundering detected: {wallet_prediction == 1}',
            'concluded_happened': (wallet_prediction == 1)
        }
    )

    # If this was an update, you may want to create a new note or update the existing one
    if not created:
        # Optionally, update the note if the analysis was updated
        completed_analysis.save()

    return wallet_prediction == 1


def run_transaction_moneyLaundering_model_analysis(wallet_analysis):
    transaction_model, transaction_features = load_moneyLaundering_transaction_model()

    wallet = wallet_analysis.wallet

    # Retrieve Ethereum transactions for the wallet
    transactions = EthereumTransaction.objects.filter(Q(from_address=wallet.address) | Q(to_address=wallet.address))

    # Extract necessary features from the transactions
    transactions_df = pd.DataFrame(list(transactions.values('value', 'gas', 'gas_price', 'nonce', 'cumulative_gas_used', 'gas_used')))

    # Feature Engineering: Add any necessary interaction terms
    transactions_df['Gas_Used_Ratio'] = transactions_df['gas_used'] / (transactions_df['gas'] + 1)
    transactions_df['Value_Gas_Ratio'] = transactions_df['value'] / (transactions_df['gas'] + 1)

    # Align the DataFrame with the model's expected features
    transactions_df = transactions_df.reindex(columns=transaction_features, fill_value=0)

    # Perform predictions for each transaction
    transaction_predictions = transaction_model.predict(transactions_df)

    # Get indices of transactions detected as money laundering
    money_laundering_detected_indices = transactions_df.index[transaction_predictions == 1].tolist()

    # Calculate the total number of transactions analyzed
    total_transactions_analyzed = len(transactions_df)

    # Optionally save the analysis result as before
    CompletedAnalysis.objects.update_or_create(
        wallet_analysis=wallet_analysis,
        name='Transaction Money Laundering Detection',
        defaults={
            'note': f'Money laundering detected in {len(money_laundering_detected_indices)} out of {total_transactions_analyzed} transactions',
            'concluded_happened': (len(money_laundering_detected_indices) > 0)
        }
    )

    return money_laundering_detected_indices


def run_erc20_moneyLaundering_model_analysis(wallet_analysis):
    erc20_model, erc20_features = load_moneyLaundering_erc20_model()

    wallet = wallet_analysis.wallet

    # Retrieve ERC20 transactions for the wallet
    erc20_transactions = ERC20Transaction.objects.filter(Q(from_address=wallet.address) | Q(to_address=wallet.address))

    # Extract necessary features from the transactions
    erc20_df = pd.DataFrame(list(erc20_transactions.values(
        'value', 
        'gas', 
        'gas_price',
        'gas_used',
        'cumulative_gas_used',
        'nonce', 
        'transaction_index',
        'confirmations'
    )))

    # Feature Engineering: Add any necessary interaction terms
    erc20_df['Gas_Used_Ratio'] = erc20_df['gas_used'] / (erc20_df['gas'] + 1)
    erc20_df['Value_Gas_Ratio'] = erc20_df['value'] / (erc20_df['gas'] + 1)
    erc20_df['Transaction_Value_Efficiency'] = erc20_df['value'] / (erc20_df['cumulative_gas_used'] + 1)

    # Align the DataFrame with the model's expected features
    erc20_df = erc20_df.reindex(columns=erc20_features, fill_value=0)

    # Perform predictions for each ERC20 transaction
    erc20_predictions = erc20_model.predict(erc20_df)

    # Get indices of ERC20 transactions detected as money laundering
    money_laundering_detected_indices = erc20_df.index[erc20_predictions == 1].tolist()

    # Calculate the total number of transactions analyzed
    total_transactions_analyzed = len(erc20_df)

    # Optionally save the analysis result as before
    CompletedAnalysis.objects.update_or_create(
        wallet_analysis=wallet_analysis,
        name='ERC20 Money Laundering Detection',
        defaults={
            'note': f'Money laundering detected in {len(money_laundering_detected_indices)} out of {total_transactions_analyzed} ERC20 transactions',
            'concluded_happened': (len(money_laundering_detected_indices) > 0)
        }
    )

    return money_laundering_detected_indices