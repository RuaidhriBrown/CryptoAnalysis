import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community.community_louvain as community_louvain  # Correct import statement
import os
from datetime import datetime
import warnings
from tqdm import tqdm  # Progress bar

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Function to load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Clean column names
    df.columns = df.columns.str.strip().str.lower()
    return df

# Function to create a graph from the dataset
def create_graph(df):
    G = nx.DiGraph()
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Creating Graph"):
        G.add_edge(row['from'], row['to'])
    return G

# Function for community detection
def detect_communities(G):
    # Convert to undirected graph for community detection
    G_undirected = G.to_undirected()
    # Detect communities
    partition = community_louvain.best_partition(G_undirected)
    return partition

# Function to visualize the graph and communities
def visualize_communities(G, partition, save_path=None):
    pos = nx.spring_layout(G)  # Layout for visualizing the graph
    cmap = plt.cm.get_cmap('viridis', max(partition.values()) + 1)
    
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=20,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    
    plt.title("Community Detection in Transaction Network")
    
    if save_path:
        plt.savefig(save_path)
    else:
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

# Print column names for debugging
print("Column names:", df.columns)

# Create the graph
G = create_graph(df)

# Perform community detection
partition = detect_communities(G)

# Saving the community detection result
# Convert partition dictionary to DataFrame for easier saving and analysis
community_df = pd.DataFrame(list(partition.items()), columns=['Address', 'Community'])
community_df.to_csv(os.path.join(path, 'community_detection_result.csv'), index=False)

print(f"Community detection results saved in the directory: {path}")

# Visualize the communities and save as image
visualization_path = os.path.join(path, 'community_detection_visualization.png')
visualize_communities(G, partition, save_path=visualization_path)
print(f"Community detection visualization saved as: {visualization_path}")
