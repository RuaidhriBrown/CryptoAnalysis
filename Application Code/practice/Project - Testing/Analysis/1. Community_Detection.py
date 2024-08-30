import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # python-louvain library

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

# Function for community detection
def detect_communities(G):
    # Convert to undirected graph for community detection
    G_undirected = G.to_undirected()
    # Detect communities
    partition = community_louvain.best_partition(G_undirected)
    return partition

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

# Perform community detection
partition = detect_communities(G)

# Saving the community detection result
# Convert partition dictionary to DataFrame for easier saving and analysis
community_df = pd.DataFrame(list(partition.items()), columns=['Address', 'Community'])
community_df.to_csv(os.path.join(path, 'community_detection_result.csv'), index=False)

print(f"Community detection results saved in the directory: {path}")
