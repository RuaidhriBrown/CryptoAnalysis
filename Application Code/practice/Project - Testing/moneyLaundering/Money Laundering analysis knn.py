import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Load the dataset
file_path = 'combined_transactions.csv'
clean_data = pd.read_csv(file_path)

# Filter the data to only include transactions where 'flag' is equal to '1'
data = clean_data[clean_data['FLAG'] == 1]

# Get the first 100 unique wallets
wallets = pd.concat([data['from'], data['to']]).unique()[:1000]

# Threshold for alerting based on the number of clusters
cluster_threshold = 3

# Create a new directory for the generated files
output_dir = 'cluster_analysis_results'
os.makedirs(output_dir, exist_ok=True)

# Iterate through the first 100 wallets
for target_address in wallets:
    # Filter transactions involving the specific address
    filtered_data = data[(data['from'] == target_address) | (data['to'] == target_address)]
    
    # Skip if there are no transactions for the wallet or only one transaction
    if filtered_data.empty or len(filtered_data) < 2:
        continue
    
    # Convert 'timeStamp' to datetime
    filtered_data['timeStamp'] = pd.to_datetime(filtered_data['timeStamp'], unit='s')
    
    # Convert 'value' to numeric, coercing errors to handle non-numeric values
    filtered_data['value'] = pd.to_numeric(filtered_data['value'], errors='coerce')
    
    # Extract the numeric value of datetime for clustering
    filtered_data['time_numeric'] = filtered_data['timeStamp'].astype('int64') // 10**9
    
    # Scale the time_numeric for KNN
    scaler = MinMaxScaler()
    filtered_data['time_scaled'] = scaler.fit_transform(filtered_data[['time_numeric']])
    
    # Determine the optimal number of clusters using the Elbow Method
    inertia = []
    K_range = range(1, min(11, len(filtered_data)))  # Ensure K_range is valid
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(filtered_data[['time_scaled']])
        inertia.append(kmeans.inertia_)
    
    if len(inertia) > 1:
        # Plot the Elbow Method graph
        plt.figure(figsize=(8, 5))
        plt.plot(K_range, inertia, 'bx-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title(f'Elbow Method For Optimal k - Wallet: {target_address}')
        plt.savefig(os.path.join(output_dir, f'Elbow_Method_{target_address}.png'))
        plt.close()
        
        # Determine the optimal number of clusters (elbow point)
        optimal_k = 4# K_range[inertia.index(min(inertia[1:], key=lambda x: x - inertia[inertia.index(x) - 1]))]  # Automatically determine elbow point
    else:
        optimal_k = 1
    
    # Perform KNN clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    filtered_data['cluster'] = kmeans.fit_predict(filtered_data[['time_scaled']])
    
    # Alert if the number of clusters is higher than the threshold
    if optimal_k > cluster_threshold:
        print(f"Alert: Wallet {target_address} has {optimal_k} clusters, which is higher than the threshold of {cluster_threshold}.")
        
        # Plot the clustered transaction distribution
        plt.figure(figsize=(12, 6))
        for cluster in range(optimal_k):
            cluster_data = filtered_data[filtered_data['cluster'] == cluster]
            plt.plot(cluster_data['timeStamp'], cluster_data['value'], 'o', label=f'Cluster {cluster}', markersize=2)
        plt.xlabel('Time')
        plt.ylabel('Transaction Value')
        plt.title(f'Clustered Transaction Distribution - Wallet: {target_address}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'Clustered_Distribution_{target_address}.png'))
        plt.close()
    
    # Transaction Frequency Analysis within clusters
    transaction_frequency_clustered = filtered_data.groupby(['cluster', 'from'])['hash'].count().reset_index()
    transaction_frequency_clustered.columns = ['cluster', 'wallet', 'transaction_count']
    
    # Transaction Amount Analysis within clusters
    transaction_amount_clustered = filtered_data.groupby(['cluster', 'from'])['value'].agg(['mean', 'median', 'sum']).reset_index()
    transaction_amount_clustered.columns = ['cluster', 'wallet', 'mean_amount', 'median_amount', 'total_amount']
    
    # In/Out Ratio Analysis within clusters
    incoming_transactions_clustered = filtered_data.groupby(['cluster', 'to'])['value'].sum().reset_index()
    incoming_transactions_clustered.columns = ['cluster', 'wallet', 'incoming_value']
    
    outgoing_transactions_clustered = filtered_data.groupby(['cluster', 'from'])['value'].sum().reset_index()
    outgoing_transactions_clustered.columns = ['cluster', 'wallet', 'outgoing_value']
    
    # Merge incoming and outgoing transactions within each cluster
    in_out_ratio_clustered = pd.merge(incoming_transactions_clustered, outgoing_transactions_clustered, on=['cluster', 'wallet'], how='outer').fillna(0)
    in_out_ratio_clustered['in_out_ratio'] = in_out_ratio_clustered['incoming_value'] / (in_out_ratio_clustered['outgoing_value'] + 1)
    
    # Round-Trip Transaction Analysis within clusters
    round_trip_transactions_clustered = filtered_data[filtered_data.duplicated(['from', 'to', 'cluster'], keep=False)]
    
    # Save the results to CSV files for each wallet
    # transaction_frequency_clustered.to_csv(os.path.join(output_dir, f'transaction_frequency_clustered_{target_address}.csv'), index=False)
    # transaction_amount_clustered.to_csv(os.path.join(output_dir, f'transaction_amount_clustered_{target_address}.csv'), index=False)
    # in_out_ratio_clustered.to_csv(os.path.join(output_dir, f'in_out_ratio_clustered_{target_address}.csv'), index=False)
    # round_trip_transactions_clustered.to_csv(os.path.join(output_dir, f'round_trip_transactions_clustered_{target_address}.csv'), index=False)

print("Analysis complete. Results saved to CSV files.")
