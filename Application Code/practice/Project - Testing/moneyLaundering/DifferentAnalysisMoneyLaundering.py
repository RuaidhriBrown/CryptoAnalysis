import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Function to load data from CSV files
def load_data_from_csv(address):
    normal_tx = pd.read_csv(f'datasets/ethereum/moneyLaundering/{address}_normal_tx.csv')
    internal_tx = pd.read_csv(f'datasets/ethereum/moneyLaundering/{address}_internal_tx.csv')
    erc20_tx = pd.read_csv(f'datasets/ethereum/moneyLaundering/{address}_erc20_tx.csv')
    return normal_tx, internal_tx, erc20_tx

# Load the data
upbit_hack_address = '0xa09871AEadF4994Ca12f5c0b6056BBd1d343c029'
normal_tx, internal_tx, erc20_tx = load_data_from_csv(upbit_hack_address)

# Analysis functions (same as before)
def analyze_fifo_ratio(df):
    """Calculate the Fast-In Fast-Out (FIFO) ratio."""
    df['time_diff'] = pd.to_datetime(df['timeStamp'], unit='s').diff().dt.total_seconds()
    fifo_ratio = df[df['time_diff'] < 3600].shape[0] / df.shape[0]  # Example threshold: 1 hour
    print(f'Fast-In Fast-Out (FIFO) Ratio: {fifo_ratio:.2f}')
    return fifo_ratio

def analyze_sent_received_ratio(df, address):
    """Calculate the Sent/Received Ratio."""
    total_sent = df[df['from'].str.lower() == address.lower()]['value'].astype(float).sum() / 10**18
    total_received = df[df['to'].str.lower() == address.lower()]['value'].astype(float).sum() / 10**18
    sent_received_ratio = total_sent / total_received if total_received != 0 else np.inf
    print(f'Sent/Received Ratio: {sent_received_ratio:.2f}')
    return sent_received_ratio

def analyze_token_diversity(erc20_df):
    """Calculate the Token Diversity Index."""
    token_diversity_index = erc20_df['tokenSymbol'].nunique()
    print(f'Token Diversity Index: {token_diversity_index}')
    return token_diversity_index

def transaction_volume_analysis(df):
    """Analyze transaction volume and frequency."""
    df['value_eth'] = df['value'].astype(float) / 10**18
    df['time'] = pd.to_datetime(df['timeStamp'], unit='s')
    df.set_index('time', inplace=True)

    # Resample by day to see daily transaction volumes
    daily_volume = df['value_eth'].resample('D').sum()
    daily_count = df['value_eth'].resample('D').count()

    print('Daily Transaction Volume and Count:')
    print(daily_volume)
    print(daily_count)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    daily_volume.plot(title='Daily Transaction Volume (ETH)')
    plt.subplot(2, 1, 2)
    daily_count.plot(title='Daily Transaction Count')
    plt.tight_layout()
    plt.show()

    return daily_volume, daily_count

def clustering_analysis(df):
    """Perform clustering analysis to detect dense networks of transactions."""
    df['time'] = pd.to_datetime(df['timeStamp'], unit='s')
    df['value_eth'] = df['value'].astype(float) / 10**18
    features = df[['time', 'value_eth']]

    # Convert time to seconds since epoch for clustering
    features['time'] = features['time'].apply(lambda x: x.timestamp())
    
    # DBSCAN clustering
    clustering_model = DBSCAN(eps=3600, min_samples=5)  # Example parameters: 1 hour eps, min 5 samples
    df['cluster'] = clustering_model.fit_predict(features)
    
    print(f'Clusters found: {df["cluster"].nunique()}')
    print(df['cluster'].value_counts())

    # Plotting clusters
    plt.figure(figsize=(10, 6))
    for cluster in df['cluster'].unique():
        clustered_data = df[df['cluster'] == cluster]
        plt.scatter(clustered_data['time'], clustered_data['value_eth'], label=f'Cluster {cluster}')
    plt.title('Transaction Clusters')
    plt.xlabel('Time (Epoch)')
    plt.ylabel('Transaction Value (ETH)')
    plt.legend()
    plt.show()

    return df['cluster']

# Perform the analysis
fifo_ratio = analyze_fifo_ratio(normal_tx)
sent_received_ratio = analyze_sent_received_ratio(normal_tx, upbit_hack_address)
token_diversity_index = analyze_token_diversity(erc20_tx)
daily_volume, daily_count = transaction_volume_analysis(normal_tx)
clusters = clustering_analysis(normal_tx)
