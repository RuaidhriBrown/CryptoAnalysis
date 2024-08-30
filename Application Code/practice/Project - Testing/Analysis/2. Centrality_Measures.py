# 2. Centrality Measures
# Centrality measures can help identify the most important or influential nodes within the network based on their position. Besides degree centrality 
# (already considered by identifying hubs), there are other measures like betweenness centrality, closeness centrality, and eigenvector 
# centrality that could provide additional insights into the roles of specific addresses in the network.

import pandas as pd
import networkx as nx

import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to create a graph from the dataset
def create_graph(df):
    G = nx.DiGraph()
    for idx, row in df.iterrows():
        G.add_edge(row['Address'], row['Unique Sent To Addresses'])
    return G

# Function to calculate centrality measures
def calculate_centrality_measures(G):
    # Convert to undirected graph for some centrality measures
    G_undirected = G.to_undirected()
    
    # Calculating various centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G_undirected)
    closeness_centrality = nx.closeness_centrality(G_undirected)
    eigenvector_centrality = nx.eigenvector_centrality_numpy(G_undirected)
    
    # Combine all centrality measures into a single DataFrame
    centrality_df = pd.DataFrame({
        'Address': degree_centrality.keys(),
        'Degree Centrality': degree_centrality.values(),
        'Betweenness Centrality': betweenness_centrality.values(),
        'Closeness Centrality': closeness_centrality.values(),
        'Eigenvector Centrality': eigenvector_centrality.values()
    })
    
    return centrality_df

# Results directory creation with date and time
results_dir = 'results'
date_time_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
path = os.path.join(results_dir, date_time_dir)
if not os.path.exists(path):
    os.makedirs(path)

# Load the data
file_path = 'transaction_dataset.csv'  # Adjust to your file path
df = load_data(file_path)

# Create the graph
G = create_graph(df)

# Calculate centrality measures
centrality_df = calculate_centrality_measures(G)

# Save the centrality measures to CSV
centrality_df.to_csv(os.path.join(path, 'centrality_measures_result.csv'), index=False)

print(f"Centrality measures results saved in the directory: {path}")
