from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the Ethereum transaction data
file_path = 'transaction_dataset.csv'
transaction_data = pd.read_csv(file_path)

# Trim leading and trailing spaces from column names
transaction_data.columns = transaction_data.columns.str.strip()

# Updated feature selection based on available columns
features = [
    'Avg min between sent tnx', 'Avg min between received tnx',
    'Time Diff between first and last (Mins)', 'Sent tnx', 'Received Tnx',
    'Number of Created Contracts', 'ERC20 uniq sent token name', 'ERC20 uniq rec token name'
]

# Preprocessing: Handling missing values
transaction_data_selected = transaction_data[features].fillna(0)

# Initializing the Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)

# Training the model
model.fit(transaction_data_selected)

# Predicting anomalies
transaction_data['Anomaly'] = model.predict(transaction_data_selected)

# -1 for anomalies and 1 for normal transactions
anomalies = transaction_data[transaction_data['Anomaly'] == -1]
normal = transaction_data[transaction_data['Anomaly'] == 1]

# Number of anomalies and total transactions
anomalies_count = len(anomalies)
total_transactions = len(transaction_data)

# Displaying a sample of the anomalies detected
anomalies_sample = anomalies.head(10)

print(f"Number of anomalies detected: {anomalies_count}")
print(f"Total number of transactions: {total_transactions}")
print("Sample of anomalous transactions:")
print(anomalies_sample)

# Step 1: Distribution of flagged transactions
flagged_illicit = transaction_data[transaction_data['FLAG'] == 1]
flagged_count = len(flagged_illicit)
print(f"Number of flagged illicit transactions: {flagged_count}")

# Step 2: Overlap between flagged illicit transactions and detected anomalies
detected_illicit = anomalies[anomalies['FLAG'] == 1]
detected_illicit_count = len(detected_illicit)
print(f"Number of detected illicit transactions (overlap): {detected_illicit_count}")

# Step 3: Calculate Precision, Recall, and F1-Score
true_labels = transaction_data['FLAG']
predicted_labels = transaction_data['Anomaly'].apply(lambda x: 1 if x == -1 else 0)

precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
