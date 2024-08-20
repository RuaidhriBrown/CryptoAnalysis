#target_address = '0x0059b14e35dab1b4eee1e2926c7a5660da66f747'
#target_address = '0x00009277775ac7d0d59eaad8fee3d10ac6c805e8'
#target_address = "0x001eb1e90d25e8c1372c38f2b2a36b49b6634235"

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Load the unique wallets dataset
wallets_file_path = 'datasets/ethereum/transaction_dataset_even.csv'
wallets_data = pd.read_csv(wallets_file_path)

# Extract the unique list of Addresses
wallet_addresses = wallets_data['Address'].unique()

# Load the transaction dataset
transaction_file_path = 'datasets/ethereum/combined_transactions.csv'
transaction_data = pd.read_csv(transaction_file_path)

# Function to analyze each Address for smurfing patterns
def analyze_wallet(wallet_address, data):
    #wallet_address = "0x001eb1e90d25e8c1372c38f2b2a36b49b6634235"
    
    # Filter transactions related to the Address
    clean_data = data[(data['from'] == wallet_address) | (data['to'] == wallet_address)]

    # Filter the data to only include transactions where 'FLAG' is equal to '1'
    data = clean_data[clean_data['FLAG'] == 1]

    # Convert timestamp to datetime
    data['timeStamp'] = pd.to_datetime(data['timeStamp'], unit='s')

    # Sort data by time
    data.sort_values('timeStamp', inplace=True)

    # Convert 'value' column to numeric, coercing errors to NaN
    data['value'] = pd.to_numeric(data['value'], errors='coerce')

    # Drop any rows where 'value' is NaN
    data.dropna(subset=['value'], inplace=True)
    non_zero_values = data[data['value'] > 0]['value']

    # Debugging: Check data after filtering and cleaning
    print(f"\nAnalyzing wallet: {wallet_address}")
    print(f"Total transactions after filtering: {len(data)}")
    if data.empty:
        print("No data available after filtering. Please check the conditions or data.")
        return

    # Check if there are enough non-zero values to calculate a percentile
    if len(non_zero_values) == 0:
        print("No non-zero transaction values available. Skipping this wallet.")
        return

    # Automatically calculate a small transaction threshold using the 50th percentile
    small_transaction_threshold = np.percentile(non_zero_values, 50)
    print(f"Calculated small transaction threshold (50th percentile): {small_transaction_threshold:.2e} ETH")

    # Separate incoming and outgoing transactions
    incoming_data = data[data['to'] == wallet_address]
    outgoing_data = data[data['from'] == wallet_address]

    # Feature engineering for incoming transactions
    incoming_data['day'] = incoming_data['timeStamp'].dt.date
    incoming_features = incoming_data.groupby(['day']).agg(
        incoming_count=('value', 'count'),
        incoming_total=('value', 'sum'),
        incoming_unique_senders=('from', 'nunique'),
        mean_incoming_amount=('value', 'mean'),
        median_incoming_amount=('value', 'median'),
        first_time=('timeStamp', 'min'),
        last_time=('timeStamp', 'max')
    ).reset_index()

    incoming_features['time_diff'] = (incoming_features['last_time'] - incoming_features['first_time']).dt.total_seconds()

    # Filter for small incoming transactions
    small_incoming_transactions = incoming_features[incoming_features['mean_incoming_amount'] < small_transaction_threshold]

    # Feature engineering for outgoing transactions
    outgoing_data['day'] = outgoing_data['timeStamp'].dt.date
    outgoing_features = outgoing_data.groupby(['day']).agg(
        outgoing_count=('value', 'count'),
        outgoing_total=('value', 'sum'),
        max_outgoing_amount=('value', 'max')
    ).reset_index()

    # Combine features to identify smurfing patterns
    smurfing_features = pd.merge(small_incoming_transactions, outgoing_features, on='day', how='inner')

    # Check small transactions count
    print(f"Number of potential smurfing days: {len(smurfing_features)}")
    if smurfing_features.empty:
        print("No potential smurfing patterns found.")
        return

    # Select features for clustering
    features = smurfing_features[['incoming_count', 'incoming_total', 'outgoing_total', 'max_outgoing_amount']]

    # Normalize the data
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=2)  # Adjust parameters as needed
    clusters = dbscan.fit_predict(normalized_features)

    # Add cluster labels to the dataframe
    smurfing_features['cluster'] = clusters

    # Visualize all clusters in one plot
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

    suspicious_clusters = smurfing_features[smurfing_features['cluster'] != -1]

    if not suspicious_clusters.empty:
        for cluster_id in suspicious_clusters['cluster'].unique():
            cluster_data = suspicious_clusters[suspicious_clusters['cluster'] == cluster_id]
            plt.scatter(cluster_data['day'], cluster_data['outgoing_total'], 
                        c=colors[cluster_id % len(colors)], label=f'Cluster {cluster_id}', s=50, alpha=0.6)

        plt.xlabel('Date')
        plt.ylabel('Total Outgoing Transaction Amount (ETH)')
        plt.title(f'Clusters of Smurfing Patterns for {wallet_address}')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()
    else:
        print("No suspicious clusters found.")

# Iterate over each Address and perform analysis
for wallet in wallet_addresses:
    analyze_wallet(wallet, transaction_data)

