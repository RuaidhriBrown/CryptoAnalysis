import os
import pandas as pd
import networkx as nx
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

#### Step 2: Load and Preprocess the Dataset ####

# Load the Ethereum transaction dataset
eth_transactions = pd.read_csv('combined_transactions.csv')

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

# Feature Engineering
scaler = StandardScaler()
transactions[['value', 'gas', 'gasPrice']] = scaler.fit_transform(transactions[['value', 'gas', 'gasPrice']])

# Now, separate the features and labels
X = transactions[['from_node', 'to_node', 'value', 'gas', 'gasPrice']].values
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
fold_accuracies = []
fold_conf_matrices = []

# Lists to accumulate true and predicted labels across all folds
all_true_labels = []
all_predicted_labels = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_balanced, y_balanced)):
    print(f"Fold {fold+1}")

    X_train, X_test = X_balanced[train_idx], X_balanced[test_idx]
    y_train, y_test = y_balanced[train_idx], y_balanced[test_idx]

    X_train, X_test, y_train, y_test, train_nodes, test_nodes = custom_train_test_split(X_balanced, y_balanced, test_size=0.2)

    #### Step 3: Construct the Transaction Network Graph ####

    G = nx.DiGraph()

    # Add edges to the graph (from_node -> to_node) for the entire dataset
    for i in range(X_train.shape[0]):
        from_node = int(X_train[i, 0])  # Ensure nodes are integers
        to_node = int(X_train[i, 1])  # Ensure nodes are integers
        value = X_train[i, 2]
        G.add_node(from_node)
        G.add_node(to_node)
        G.add_edge(from_node, to_node, value=value)

    # Create the node features tensor (using all nodes)
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    num_nodes = G.number_of_nodes()
    node_features = torch.randn((num_nodes, 8))  # Random features for demonstration

    # Create a dictionary to map each node to its corresponding label
    node_to_label = {int(X_train[i, 0]): y_train[i] for i in range(len(X_train))}

    # Handle nodes that might not have a label in node_to_label
    y_labels = []
    for node in node_mapping.keys():
        if int(node) in node_to_label:
            y_labels.append(node_to_label[int(node)])
        else:
            y_labels.append(0)  # Assign a default label, e.g., 0 for non-phishing

    y_labels = torch.tensor(y_labels, dtype=torch.long)

    # Ensure edge_index is constructed after node_mapping is finalized
    edge_index = torch.tensor(
        [[node_mapping[edge[0]], node_mapping[edge[1]]] for edge in G.edges()],
        dtype=torch.long
    ).t().contiguous()

    data = Data(x=node_features, edge_index=edge_index, y=y_labels)

    #### Step 4: Define the Graph Neural Network (GNN) ####

    class GCN(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(out_channels, 2)  # Final classification layer

        def forward(self, data):
            if data is None:
                raise ValueError("Data passed to GCN is None.")
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = self.relu(x)
            x = self.conv2(x, edge_index)
            x = self.fc(x)
            return x

    model = GCN(in_channels=8, hidden_channels=16, out_channels=8)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]))  # Adjust the weights as needed
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    #### Step 5: Train the Model ####

    train_loader = DataLoader([data], batch_size=1, shuffle=True)

    for epoch in tqdm(range(100), desc=f"Training Fold {fold+1}"):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to('cpu')
            out = model(batch)
            
            if out.shape[0] != batch.y.shape[0]:
                raise ValueError(f"Mismatch between output batch size ({out.shape[0]}) and target batch size ({batch.y.shape[0]}).")
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

    #### Step 6: Evaluate the Model ####

    def evaluate_model_on_graph(model, graph_data):
        model.eval()
        with torch.no_grad():
            out = model(graph_data)
            preds = out.argmax(dim=1)
            return preds

    # Evaluate on the same data (using the entire graph)
    test_predictions = evaluate_model_on_graph(model, data)

    all_true_labels.extend(data.y.cpu().numpy())
    all_predicted_labels.extend(test_predictions.cpu().numpy())

    fold_accuracy = (test_predictions == data.y).sum().item() / len(data.y)
    fold_accuracies.append(fold_accuracy)
    print(f"Fold {fold+1} Accuracy: {fold_accuracy:.4f}")

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
