import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to filter transactions related to the primary address
def filter_transactions(df, primary_address):
    sent_transactions = df[df['from'] == primary_address]
    received_transactions = df[df['to'] == primary_address]
    return pd.concat([sent_transactions, received_transactions])

# Function to create a graph from the dataset
def create_graph(df):
    G = nx.DiGraph()
    for idx, row in df.iterrows():
        G.add_edge(row['from'], row['to'])
    return G

# Function to calculate centrality measures
def calculate_centrality_measures(G, primary_address):
    # Extract the subgraph around the primary address
    neighbors = set(G.neighbors(primary_address)).union(set(G.predecessors(primary_address)))
    subgraph_nodes = neighbors.union({primary_address})
    G_subgraph = G.subgraph(subgraph_nodes)
    
    # Convert to undirected graph for some centrality measures
    G_undirected = G_subgraph.to_undirected()
    
    # Calculating various centrality measures
    degree_centrality = nx.degree_centrality(G_subgraph)
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
    
    return centrality_df, G_subgraph

# Function to visualize the graph
def visualize_graph(G, centrality_df, primary_address):
    pos = nx.spring_layout(G)  # Layout for visualization
    plt.figure(figsize=(12, 8))
    
    # Scale node sizes by degree centrality for better visibility
    node_sizes = [v * 1000 for v in nx.degree_centrality(G).values()]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.7)
    
    # Highlight the primary address
    nx.draw_networkx_nodes(G, pos, nodelist=[primary_address], node_size=2000, node_color='red')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5, edge_color='grey')
    
    # Draw labels only for a limited number of nodes for clarity
    limited_labels = {node: node[:6] for node in G.nodes() if node in set(centrality_df['Address'])}
    nx.draw_networkx_labels(G, pos, labels=limited_labels, font_size=8)
    
    plt.title('Network Graph Centered Around Primary Address')
    plt.show()

# Results directory creation with date and time
results_dir = 'results'
date_time_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
path = os.path.join(results_dir, date_time_dir)
if not os.path.exists(path):
    os.makedirs(path)

# Load the data
file_path = 'combined_transactions.csv'  # Adjust to your file path
df = load_data(file_path)

# Define the primary address
primary_address = '0x00009277775ac7d0d59eaad8fee3d10ac6c805e8'

# Filter transactions related to the primary address
filtered_df = filter_transactions(df, primary_address)

# Create the graph
G = create_graph(filtered_df)

# Calculate centrality measures
centrality_df, G_subgraph = calculate_centrality_measures(G, primary_address)

# Save the centrality measures to CSV
centrality_df.to_csv(os.path.join(path, 'centrality_measures_result.csv'), index=False)

print(f"Centrality measures results saved in the directory: {path}")

# Visualize the graph
visualize_graph(G_subgraph, centrality_df, primary_address)
