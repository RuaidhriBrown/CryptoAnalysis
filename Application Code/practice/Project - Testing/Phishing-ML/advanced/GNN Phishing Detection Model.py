import os
import pandas as pd
import networkx as nx
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#### Step 2: Load and Preprocess the Dataset ####

# Load the Ethereum transaction dataset
eth_transactions = pd.read_csv('combined_transactions.csv')

# Combine the datasets (if necessary)
transactions = pd.concat([eth_transactions], ignore_index=True)

# Drop duplicates, if any
transactions.drop_duplicates(inplace=True)

# Map addresses to integers for node identification
address_map = {address: i for i, address in enumerate(pd.concat([transactions['from'], transactions['to']]).unique())}

# Map 'from' and 'to' addresses to integer nodes
transactions['from_node'] = transactions['from'].map(address_map)
transactions['to_node'] = transactions['to'].map(address_map)

# Ensure 'value' is numeric
transactions['value'] = pd.to_numeric(transactions['value'], errors='coerce')

# Handle any NaN values in the 'value' column
transactions.dropna(subset=['value'], inplace=True)

# Now, separate the features and labels
X = transactions[['from_node', 'to_node', 'value']].values  # Exclude 'FLAG'
y = transactions['FLAG'].values  # This is your target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#### Step 3: Construct or Load the Transaction Network Graph ####

# Check if the graph file exists
if os.path.exists("transaction_graph.graphml"):
    # Load the graph if it exists
    G = nx.read_graphml("transaction_graph.graphml")
    print("Graph loaded from file.")
else:
    # If the graph doesn't exist, create it and save it
    G = nx.DiGraph()

    # Add edges to the graph (from_node -> to_node)
    for _, row in tqdm(transactions.iterrows(), total=transactions.shape[0], desc="Constructing Transaction Network"):
        from_node = row['from_node']
        to_node = row['to_node']
        G.add_node(from_node)  # Ensure the 'from_node' is in the graph
        G.add_node(to_node)    # Ensure the 'to_node' is in the graph
        G.add_edge(from_node, to_node, value=row['value'])

    # Save the graph for future use
    nx.write_graphml(G, "transaction_graph.graphml")
    print("Graph created and saved to file.")

#### Step 4: Subgraph Sampling and Data Preparation ####

def bfs_sample_and_prepare_data(graph, start_node, depth=2, label=None):
    # Perform BFS sampling
    if start_node not in graph:
        raise ValueError(f"The node {start_node} is not in the digraph.")
    sampled_nodes = set(nx.bfs_tree(graph, start_node, depth_limit=depth).nodes())

    # Create the subgraph
    subgraph = graph.subgraph(sampled_nodes).copy()

    # Create the node features tensor (dummy features for simplicity)
    node_mapping = {node: i for i, node in enumerate(subgraph.nodes())}
    num_nodes = subgraph.number_of_nodes()
    node_features = torch.randn((num_nodes, 8))  # Random features for demonstration

    # Create edge_index from the subgraph
    edge_index = torch.tensor(
        [[node_mapping[edge[0]], node_mapping[edge[1]]] for edge in subgraph.edges()],
        dtype=torch.long
    ).t().contiguous()

    if edge_index.numel() == 0:
        return None  # Return None if edge_index is empty

    # Assign the label to the Data object (target)
    if label is not None:
        y = torch.tensor([label] * num_nodes, dtype=torch.long)  # One label per node
    else:
        y = None

    return Data(x=node_features, edge_index=edge_index, y=y)

# Example usage with labels
phishing_address = '0xfd8999b60a72c51ea892db66f5ef0c58f2ecd6d3'  # Replace with actual address
phishing_node = address_map.get(phishing_address, None)

# Assume that phishing nodes have a label of 1 and non-phishing nodes have a label of 0
label = 1  # or 0 depending on the type of node

if phishing_node is not None and phishing_node in G:
    data = bfs_sample_and_prepare_data(G, phishing_node, depth=2, label=label)
    if data is None:
        raise ValueError(f"Subgraph for node {phishing_node} is too small or isolated.")
else:
    raise ValueError(f"Phishing node {phishing_address} with mapped ID {phishing_node} does not exist in the graph.")

#### Step 5: Define the Graph Neural Network (GNN) ####

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Check if edge_index is empty
        if edge_index.numel() == 0:
            raise ValueError("edge_index is empty. The subgraph might be too small or isolated.")

        # Validate edge_index before convolution
        if torch.max(edge_index) >= x.size(0) or torch.min(edge_index) < 0:
            raise ValueError(f"edge_index contains out-of-bounds indices: max {torch.max(edge_index)}, min {torch.min(edge_index)}, x.size(0): {x.size(0)}")

        print(f"x shape: {x.shape}, edge_index shape: {edge_index.shape}")  # Debugging line

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize the model, loss, and optimizer
model = GCN(in_channels=8, hidden_channels=32, out_channels=2)  # Ensure in_channels matches feature size
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#### Step 6: Train the Model ####

# Assuming 'data' is your PyTorch Geometric data object that includes node features and labels

# Convert to DataLoader with PyTorch Geometric
train_loader = DataLoader([data], batch_size=1, shuffle=True)  # Single subgraph for now

# Training loop with progress bar
for epoch in tqdm(range(100), desc="Training Epochs"):  # Number of epochs
    model.train()
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False):
        optimizer.zero_grad()
        batch = batch.to('cpu')  # Move batch to the appropriate device
        out = model(batch)

        # Debugging: Print shapes
        print(f"out shape: {out.shape}, batch.y shape: {batch.y.shape}")

        # Check if batch sizes match
        if out.shape[0] != batch.y.shape[0]:
            raise ValueError(f"Mismatch between output batch size ({out.shape[0]}) and target batch size ({batch.y.shape[0]}).")

        # Check if batch.y is not None before computing the loss
        if batch.y is not None:
            loss = criterion(out, batch.y)  # batch.y is the true label
            loss.backward()
            optimizer.step()
        else:
            print("Warning: batch.y is None, skipping loss computation for this batch.")

    if 'loss' in locals():
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    else:
        print(f'Epoch {epoch+1}, no valid batches for loss computation.')

#### Step 7: Evaluate the Model ####

def evaluate_model_on_graph(model, graph_data):
    model.eval()
    with torch.no_grad():
        out = model(graph_data)
        preds = out.argmax(dim=1)
        return preds

# Evaluate on the same data used for training (since it's a single graph)
predictions = evaluate_model_on_graph(model, data)

#### Step 8: Automate Detection ####

# Function to flag phishing accounts
def flag_phishing_nodes(graph, model):
    flagged_nodes = []
    for node in tqdm(graph.nodes(), desc="Flagging Phishing Nodes"):
        try:
            subgraph_data = bfs_sample_and_prepare_data(graph, node, depth=2)
            if subgraph_data is None:
                continue  # Skip if the subgraph is too small or isolated
            model.eval()
            with torch.no_grad():
                out = model(subgraph_data)
                if (out.argmax(dim=1) == 1).any():  # If any node in the subgraph is predicted as phishing
                    flagged_nodes.append(node)
        except ValueError as e:
            print(f"Skipping node {node}: {e}")
    return flagged_nodes

# Run phishing detection on all nodes
flagged_phishing_nodes = flag_phishing_nodes(G, model)
print(f"Flagged phishing nodes: {flagged_phishing_nodes}")

# Confusion matrix for flagged phishing nodes
y_true = [1 if node in flagged_phishing_nodes else 0 for node in G.nodes]
y_pred = [1 if node in flagged_phishing_nodes else 0 for node in G.nodes]

conf_matrix = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
