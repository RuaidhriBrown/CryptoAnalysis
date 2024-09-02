
# utils.py
from django.db.models import Count, Q

import hashlib
from .models import Wallet

import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for rendering plots

import pandas as pd
import matplotlib.pyplot as plt
import os
from django.conf import settings
from datetime import datetime

from .models import CompletedAnalysis, EthereumTransaction, ERC20Transaction, Wallet, WalletAnalysis, UserProfile, WalletNote

def combine_data(address, transactions, erc20_transactions):
    transaction_df = transactions.filter(Q(from_address=address) | Q(to_address=address))
    erc20_df_filtered = erc20_transactions.filter(Q(from_address=address) | Q(to_address=address))
    
    if not transaction_df.exists() and not erc20_df_filtered.exists():
        raise ValueError(f"No summary data found for address {address}")
    
    summary_info = {'Address': address}

    sent_timestamps = list(transaction_df.filter(from_address=address).values_list('timestamp', flat=True))
    sent_timestamps.sort()  # Ensure timestamps are sorted

    if len(sent_timestamps) > 1:
        sent_diffs = [(sent_timestamps[i] - sent_timestamps[i - 1]).total_seconds() for i in range(1, len(sent_timestamps))]
        avg_time_between_sent = sum(sent_diffs) / len(sent_diffs) / 60
    else:
        avg_time_between_sent = 0

    received_timestamps = list(transaction_df.filter(to_address=address).values_list('timestamp', flat=True))
    received_timestamps.sort()  # Ensure timestamps are sorted

    if len(received_timestamps) > 1:
        received_diffs = [(received_timestamps[i] - received_timestamps[i - 1]).total_seconds() for i in range(1, len(received_timestamps))]
        avg_time_between_received = sum(received_diffs) / len(received_diffs) / 60
    else:
        avg_time_between_received = 0

    if transaction_df.exists():
        first_transaction = transaction_df.earliest('timestamp').timestamp
        last_transaction = transaction_df.latest('timestamp').timestamp
        time_diff_first_last = (last_transaction - first_transaction).total_seconds() / 60
    else:
        time_diff_first_last = 0

    unique_received_from = transaction_df.filter(to_address=address).values('from_address').distinct().count()
    unique_sent_to = transaction_df.filter(from_address=address).values('to_address').distinct().count()

    sent_values = [value for value in transaction_df.filter(from_address=address).values_list('value', flat=True) if value is not None]
    received_values = [value for value in transaction_df.filter(to_address=address).values_list('value', flat=True) if value is not None]

    min_val_received = min(received_values) / 1e18 if received_values else 0
    max_val_received = max(received_values) / 1e18 if received_values else 0
    avg_val_received = sum(received_values) / len(received_values) / 1e18 if received_values else 0

    min_val_sent = min(sent_values) / 1e18 if sent_values else 0
    max_val_sent = max(sent_values) / 1e18 if sent_values else 0
    avg_val_sent = sum(sent_values) / len(sent_values) / 1e18 if sent_values else 0

    total_ether_sent = sum(sent_values) / 1e18 if sent_values else 0
    total_ether_received = sum(received_values) / 1e18 if received_values else 0
    total_ether_balance = total_ether_received - total_ether_sent

    erc20_sent_values = [value for value in erc20_df_filtered.filter(from_address=address).values_list('value', flat=True) if value is not None]
    erc20_received_values = [value for value in erc20_df_filtered.filter(to_address=address).values_list('value', flat=True) if value is not None]

    total_erc20_received = sum(erc20_received_values) / 1e18 if erc20_received_values else 0
    total_erc20_sent = sum(erc20_sent_values) / 1e18 if erc20_sent_values else 0
    unique_sent_token_addresses = erc20_df_filtered.filter(from_address=address).values('contract_address').distinct().count()
    unique_received_token_addresses = erc20_df_filtered.filter(to_address=address).values('contract_address').distinct().count()

    min_erc20_val_received = min(erc20_received_values) / 1e18 if erc20_received_values else 0
    max_erc20_val_received = max(erc20_received_values) / 1e18 if erc20_received_values else 0
    avg_erc20_val_received = sum(erc20_received_values) / len(erc20_received_values) / 1e18 if erc20_received_values else 0

    min_erc20_val_sent = min(erc20_sent_values) / 1e18 if erc20_sent_values else 0
    max_erc20_val_sent = max(erc20_sent_values) / 1e18 if erc20_sent_values else 0
    avg_erc20_val_sent = sum(erc20_sent_values) / len(erc20_sent_values) / 1e18 if erc20_sent_values else 0

    erc20_sent_timestamps = list(erc20_df_filtered.filter(from_address=address).values_list('timestamp', flat=True))
    erc20_sent_timestamps.sort()  # Ensure timestamps are sorted

    if len(erc20_sent_timestamps) > 1:
        erc20_sent_diffs = [(erc20_sent_timestamps[i] - erc20_sent_timestamps[i - 1]).total_seconds() for i in range(1, len(erc20_sent_timestamps))]
        avg_erc20_time_between_sent = sum(erc20_sent_diffs) / len(erc20_sent_diffs) / 60
    else:
        avg_erc20_time_between_sent = 0

    erc20_received_timestamps = list(erc20_df_filtered.filter(to_address=address).values_list('timestamp', flat=True))
    erc20_received_timestamps.sort()  # Ensure timestamps are sorted

    if len(erc20_received_timestamps) > 1:
        erc20_received_diffs = [(erc20_received_timestamps[i] - erc20_received_timestamps[i - 1]).total_seconds() for i in range(1, len(erc20_received_timestamps))]
        avg_erc20_time_between_received = sum(erc20_received_diffs) / len(erc20_received_diffs) / 60
    else:
        avg_erc20_time_between_received = 0

    most_sent_token_type = erc20_df_filtered.filter(from_address=address).values('token_name').annotate(count=Count('token_name')).order_by('-count').first()
    most_received_token_type = erc20_df_filtered.filter(to_address=address).values('token_name').annotate(count=Count('token_name')).order_by('-count').first()

    most_sent_token_type = most_sent_token_type['token_name'] if most_sent_token_type else ''
    most_received_token_type = most_received_token_type['token_name'] if most_received_token_type else ''

    combined_info = {
        'Address': address,
        'avg_min_between_sent_tnx': avg_time_between_sent,
        'avg_min_between_received_tnx': avg_time_between_received,
        'time_diff_between_first_and_last': time_diff_first_last,
        'sent_tnx': len(sent_timestamps),
        'received_tnx': len(received_timestamps),
        'unique_received_from_addresses': unique_received_from,
        'unique_sent_to_addresses': unique_sent_to,
        'min_val_received': min_val_received,
        'max_val_received': max_val_received,
        'avg_val_received': avg_val_received,
        'min_val_sent': min_val_sent,
        'max_val_sent': max_val_sent,
        'avg_val_sent': avg_val_sent,
        'total_ether_sent': total_ether_sent,
        'total_ether_received': total_ether_received,
        'total_ether_balance': total_ether_balance,
        'total_erc20_received': total_erc20_received,
        'total_erc20_sent': total_erc20_sent,
        'unique_sent_token_addresses': unique_sent_token_addresses,
        'unique_received_token_addresses': unique_received_token_addresses,
        'avg_erc20_time_between_sent': avg_erc20_time_between_sent,
        'avg_erc20_time_between_received': avg_erc20_time_between_received,
        'most_sent_token_type': most_sent_token_type,
        'most_received_token_type': most_received_token_type
    }
    return combined_info


def generate_wallet_stats_graphs(transactions, wallet_address):
    # Convert transactions to DataFrame
    data = pd.DataFrame(list(transactions.values()))
    
    # Convert timestamp to a datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Filter the dataset for transactions involving the Address
    filtered_data = data[(data['from_address'] == wallet_address) | (data['to_address'] == wallet_address)]
    
    # Group by hour, day of the week, and month for the filtered data
    filtered_data['hour'] = filtered_data['timestamp'].dt.hour
    filtered_data['day_of_week'] = filtered_data['timestamp'].dt.day_name()
    filtered_data['month'] = filtered_data['timestamp'].dt.month_name()
    
    # Create a complete range of hours (0 to 23) and days of the week
    hours = pd.DataFrame({'hour': range(24)})
    days_of_week = pd.DataFrame({'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']})
    
    # Count transactions per hour
    filtered_hourly_counts = filtered_data.groupby('hour').size().reset_index(name='count')
    filtered_hourly_counts = hours.merge(filtered_hourly_counts, on='hour', how='left').fillna(0)
    
    # Count transactions per day of the week
    filtered_daily_counts = filtered_data.groupby('day_of_week').size().reset_index(name='count')
    filtered_daily_counts = days_of_week.merge(filtered_daily_counts, on='day_of_week', how='left').fillna(0)
    
    # Count transactions per month
    filtered_monthly_counts = filtered_data.groupby('month').size().reset_index(name='count')
    
    # Analyze peak hours
    peak_hour = filtered_hourly_counts.loc[filtered_hourly_counts['count'].idxmax()]['hour']
    
    # Estimate time zone based on peak hour
    working_hours_start = 9
    working_hours_end = 17
    estimated_timezone_offset = (peak_hour - working_hours_start) % 24
    if estimated_timezone_offset > 12:
        estimated_timezone_offset -= 24
    estimated_timezone = f"UTC{'+' if estimated_timezone_offset >= 0 else ''}{estimated_timezone_offset}"
    
    # Define a directory to save the plots
    plots_dir = os.path.join(settings.MEDIA_ROOT, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot the number of transactions per hour of the day
    hourly_plot_path = os.path.join(plots_dir, f'{wallet_address}_hourly.png')
    plt.figure(figsize=(12, 6))
    plt.bar(filtered_hourly_counts['hour'], filtered_hourly_counts['count'], color='teal')
    plt.title('Number of Transactions per Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Transactions')
    plt.xticks(range(24))
    plt.savefig(hourly_plot_path)
    plt.close()
    
    # Plot the number of transactions per day of the week
    daily_plot_path = os.path.join(plots_dir, f'{wallet_address}_day_of_week.png')
    plt.figure(figsize=(12, 6))
    plt.bar(filtered_daily_counts['day_of_week'], filtered_daily_counts['count'], color='orange')
    plt.title('Number of Transactions per Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Transactions')
    plt.xticks(rotation=45)
    plt.savefig(daily_plot_path)
    plt.close()
    
    # Plot the number of transactions per month
    monthly_plot_path = os.path.join(plots_dir, f'{wallet_address}_monthly.png')
    plt.figure(figsize=(12, 6))
    plt.bar(filtered_monthly_counts['month'], filtered_monthly_counts['count'], color='purple')
    plt.title('Number of Transactions per Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Transactions')
    plt.xticks(rotation=45)
    plt.savefig(monthly_plot_path)
    plt.close()
    
    return {
        'hourly_plot': os.path.join(settings.MEDIA_URL, 'plots', f'{wallet_address}_hourly.png'),
        'daily_plot': os.path.join(settings.MEDIA_URL, 'plots', f'{wallet_address}_day_of_week.png'),
        'monthly_plot': os.path.join(settings.MEDIA_URL, 'plots', f'{wallet_address}_monthly.png'),
        'estimated_timezone': estimated_timezone,
        'peak_hour': peak_hour
    }

def convert_to_int(value, default=None):
    """Convert a value to an integer, handling float strings and out-of-range values."""
    try:
        int_value = int(float(value))
        if int_value < -9223372036854775808 or int_value > 9223372036854775807:
            return default  # Return default if the value is out of the range for BIGINT
        return int_value
    except (ValueError, TypeError, OverflowError):
        return default  # Return default if conversion fails
    

def generate_wallet_analysis_hash(wallet_analysis):
    """
    Generate a SHA-256 hash of the entire wallet analysis including all transactions,
    wallet details, notes, and completed analysis results.
    """
    wallet = wallet_analysis.wallet
    
    # Start with wallet details
    hash_input = f"{wallet.id}{wallet.address}{wallet.balance}{wallet.total_sent_transactions}{wallet.total_received_transactions}" \
                 f"{wallet.unique_sent_addresses}{wallet.unique_received_addresses}{wallet.total_ether_sent}{wallet.total_ether_received}" \
                 f"{wallet.total_erc20_sent}{wallet.total_erc20_received}{wallet.last_updated}"

    # Include all Ethereum transactions
    eth_transactions = EthereumTransaction.objects.filter(Q(from_address=wallet.address) | Q(to_address=wallet.address))
    for transaction in eth_transactions:
        hash_input += f"{transaction.block_number}{transaction.timestamp}{transaction.hash}{transaction.nonce}{transaction.block_hash}" \
                      f"{transaction.transaction_index}{transaction.from_address}{transaction.to_address}{transaction.value}" \
                      f"{transaction.gas}{transaction.gas_price}{transaction.is_error}{transaction.txreceipt_status}{transaction.input}" \
                      f"{transaction.contract_address}{transaction.cumulative_gas_used}{transaction.gas_used}{transaction.confirmations}" \
                      f"{transaction.method_id}{transaction.function_name}{transaction.address}{transaction.last_updated}"

    # Include all ERC20 transactions
    erc20_transactions = ERC20Transaction.objects.filter(Q(from_address=wallet.address) | Q(to_address=wallet.address))
    for transaction in erc20_transactions:
        hash_input += f"{transaction.composite_id}{transaction.block_number}{transaction.timestamp}{transaction.hash}{transaction.nonce}" \
                      f"{transaction.block_hash}{transaction.from_address}{transaction.contract_address}{transaction.to_address}" \
                      f"{transaction.value}{transaction.token_name}{transaction.token_decimal}{transaction.transaction_index}" \
                      f"{transaction.gas}{transaction.gas_price}{transaction.gas_used}{transaction.cumulative_gas_used}{transaction.input}" \
                      f"{transaction.confirmations}{transaction.address}{transaction.last_updated}"

    # Include all Wallet Notes
    wallet_notes = WalletNote.objects.filter(analysis__wallet=wallet)
    for note in wallet_notes:
        hash_input += f"{note.analysis.id}{note.user.username if note.user else ''}{note.content}{note.created_at}{note.updated_at}{note.note_type}"

    # Include all Completed Analyses
    completed_analyses = CompletedAnalysis.objects.filter(wallet_analysis__wallet=wallet)
    for analysis in completed_analyses:
        hash_input += f"{analysis.wallet_analysis.id}{analysis.name}{analysis.note}{analysis.completed_at}{analysis.concluded_happened}"

    # Generate SHA-256 hash
    hash_value = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()

    # Update the analysis_hash field of the WalletAnalysis instance
    wallet_analysis.analysis_hash = hash_value
    wallet_analysis.save(update_fields=['analysis_hash'])

    return hash_value