import os
import pandas as pd
import networkx as nx
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv, GINConv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#### Step 2: Load and Preprocess the Dataset ####

# Load the Ethereum transaction dataset
eth_transactions = pd.read_csv('datasets/ethereum/combined_transactions.csv')

# Combine the datasets (if necessary)
transactions = pd.concat([eth_transactions], ignore_index=True)

# Drop duplicates, if any
transactions.drop_duplicates(inplace=True)

# Ensure 'value' is numeric
transactions['value'] = pd.to_numeric(transactions['value'], errors='coerce')

# Handle any NaN values in the 'value' column
transactions.dropna(subset=['value'], inplace=True)

# Map addresses to integers for node identification
address_map = {address: i for i, address in enumerate(pd.concat([transactions['from'], transactions['to']]).unique())}

# Map 'from' and 'to' addresses to integer nodes
transactions['from_node'] = transactions['from'].map(address_map)
transactions['to_node'] = transactions['to'].map(address_map)

# Feature Engineering: Add centrality measures and additional features
G_full = nx.DiGraph()
for _, row in transactions.iterrows():
    G_full.add_edge(row['from_node'], row['to_node'], value=row['value'])

centrality = nx.degree_centrality(G_full)
transactions['centrality_from'] = transactions['from_node'].map(centrality)
transactions['centrality_to'] = transactions['to_node'].map(centrality)

# Additional feature: Number of transactions per node
transactions['num_tx_from'] = transactions.groupby('from_node')['value'].transform('count')
transactions['num_tx_to'] = transactions.groupby('to_node')['value'].transform('count')

# Normalize features
scaler = StandardScaler()
transactions[['value', 'gas', 'gasPrice', 'centrality_from', 'centrality_to', 'num_tx_from', 'num_tx_to']] = scaler.fit_transform(
    transactions[['value', 'gas', 'gasPrice', 'centrality_from', 'centrality_to', 'num_tx_from', 'num_tx_to']])

# Now, separate the features and labels
X = transactions[['from_node', 'to_node', 'value', 'gas', 'gasPrice', 'centrality_from', 'centrality_to', 'num_tx_from', 'num_tx_to']].values
y = transactions['FLAG'].values  # Ensure this is correctly labeled as the target

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Custom split to avoid data leakage and ensure phishing node is in the training set
def custom_train_test_split(X, y, test_size=0.2):
    unique_nodes = set(X[:, 0].astype(int)).union(set(X[:, 1].astype(int)))  # Ensure nodes are integers
    test_nodes = set(pd.Series(list(unique_nodes)).sample(frac=test_size, random_state=42))
    train_nodes = unique_nodes - test_nodes

    train_mask = pd.Series([((int(from_node) in train_nodes) and (int(to_node) in train_nodes)) 
                            for from_node, to_node in zip(X[:, 0], X[:, 1])])
    test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    return X_train, X_test, y_train, y_test, train_nodes, test_nodes

# Cross-validation setup
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_folds = kf.get_n_splits()  # Get the number of folds (in this case 5)
fold_accuracies = []
fold_conf_matrices = []

# Lists to accumulate true and predicted labels across all folds
all_true_labels = []
all_predicted_labels = []

#### Step 3: Define and Train Different GNN Models ####

# Define a function to create the graph data
def create_graph_data(X_train, y_train):
    G = nx.DiGraph()
    for i in range(X_train.shape[0]):
        from_node = int(X_train[i, 0])
        to_node = int(X_train[i, 1])
        value = X_train[i, 2]
        G.add_node(from_node)
        G.add_node(to_node)
        G.add_edge(from_node, to_node, value=value)

    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    num_nodes = G.number_of_nodes()
    node_features = torch.randn((num_nodes, 9))

    node_to_label = {int(X_train[i, 0]): y_train[i] for i in range(len(X_train))}
    y_labels = [node_to_label.get(node, 0) for node in node_mapping.keys()]
    y_labels = torch.tensor(y_labels, dtype=torch.long)

    edge_index = torch.tensor(
        [[node_mapping[edge[0]], node_mapping[edge[1]]] for edge in G.edges()],
        dtype=torch.long
    ).t().contiguous()

    data = Data(x=node_features, edge_index=edge_index, y=y_labels)
    return data

# Define different GNN models with advanced configurations
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_channels, 2)  # Final classification layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return x

class GraphSAGENet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5):
        super(GraphSAGENet, self).__init__()
        self.conv1 = GraphSAGE(in_channels, hidden_channels, num_layers=num_layers)
        self.conv2 = GraphSAGE(hidden_channels, out_channels, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_channels, 2)  # Final classification layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return x

class GATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, num_layers=2, dropout=0.5):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_channels, 2)  # Final classification layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return x

class GINNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GINNet, self).__init__()
        nn1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels))
        self.conv2 = GINConv(nn2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_channels, 2)  # Final classification layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return x

# Hyperparameter search space
param_grid = {
    'hidden_channels': [16, 32, 64],
    'lr': [0.01, 0.005, 0.001],
    'dropout': [0.3, 0.5, 0.7]
}

# Function to evaluate model on a graph dataset
def evaluate_model_on_graph(model, graph_data):
    model.eval()
    with torch.no_grad():
        out = model(graph_data)
        preds = out.argmax(dim=1)
        return preds

# Store the best results
best_models = []

# Calculate the total number of iterations for the overall progress bar
total_iterations = n_folds * len(param_grid['hidden_channels']) * len(param_grid['lr']) * len(param_grid['dropout']) * len([GCN, GraphSAGENet, GATNet, GINNet]) * 50

with tqdm(total=total_iterations, desc="Total Progress") as total_pbar:
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_balanced, y_balanced)):
        print(f"Fold {fold+1}")

        X_train, X_test = X_balanced[train_idx], X_balanced[test_idx]
        y_train, y_test = y_balanced[train_idx], y_balanced[test_idx]

        X_train, X_test, y_train, y_test, train_nodes, test_nodes = custom_train_test_split(X_balanced, y_balanced, test_size=0.2)

        # Create the graph data
        data = create_graph_data(X_train, y_train)

        best_fold_accuracy = 0
        best_fold_model = None

        for model_name, ModelClass in [('GCN', GCN), ('GraphSAGENet', GraphSAGENet), ('GATNet', GATNet), ('GINNet', GINNet)]:
            for hidden_channels in param_grid['hidden_channels']:
                for lr in param_grid['lr']:
                    for dropout in param_grid['dropout']:
                        model = ModelClass(in_channels=9, hidden_channels=hidden_channels, out_channels=16, dropout=dropout)
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]))

                        # Train the model
                        train_loader = DataLoader([data], batch_size=1, shuffle=True)

                        for epoch in tqdm(range(50), desc=f"Training {model_name} (Fold {fold+1})", leave=False):
                            model.train()
                            for batch in train_loader:
                                optimizer.zero_grad()
                                batch = batch.to('cpu')
                                out = model(batch)
                                loss = criterion(out, batch.y)
                                loss.backward()
                                optimizer.step()

                            total_pbar.update(1)  # Update the overall progress bar after each epoch

                        # Evaluate the model on training data (using the entire graph)
                        model.eval()
                        with torch.no_grad():
                            out = model(data)
                            preds = out.argmax(dim=1)
                            fold_accuracy = (preds == data.y).sum().item() / len(data.y)

                        if fold_accuracy > best_fold_accuracy:
                            best_fold_accuracy = fold_accuracy
                            best_fold_model = model

        best_models.append(best_fold_model)

        # Evaluate the best model on the test data
        test_data = create_graph_data(X_test, y_test)
        test_predictions = evaluate_model_on_graph(best_fold_model, test_data)

        all_true_labels.extend(test_data.y.cpu().numpy())
        all_predicted_labels.extend(test_predictions.cpu().numpy())

        fold_accuracy = (test_predictions == test_data.y).sum().item() / len(test_data.y)
        fold_accuracies.append(fold_accuracy)
        print(f"Fold {fold+1} Best Model Accuracy: {fold_accuracy:.4f}")

# After all folds, summarize the results
mean_accuracy = sum(fold_accuracies) / len(fold_accuracies) if fold_accuracies else 0
print(f"Mean Accuracy across folds: {mean_accuracy:.4f}")

# Generate a classification report
classification_report_str = classification_report(all_true_labels, all_predicted_labels, target_names=['Legitimate', 'Phishing'])
print("\nClassification Report:\n")
print(classification_report_str)

# Aggregate the confusion matrices
total_conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(total_conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Aggregated Confusion Matrix across all folds')
plt.show()
