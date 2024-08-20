from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

# Load the Ethereum transaction data
file_path = 'datasets/ethereum/transaction_dataset_even.csv'
transaction_data = pd.read_csv(file_path)

# Trim leading and trailing spaces from column names
transaction_data.columns = transaction_data.columns.str.strip()

# Ensure no extra spaces in column names
initial_features = [
    'Avg min between sent tnx', 'Avg min between received tnx',
    'Time Diff between first and last (Mins)', 'Sent tnx', 'Received Tnx',
    'Number of Created Contracts', 'Unique Received From Addresses', 'Unique Sent To Addresses',
    'min value received', 'max value received', 'avg val received',
    'min val sent', 'max val sent', 'avg val sent'
]

# Check if all initial features exist in the dataset
missing_features = [feature for feature in initial_features if feature not in transaction_data.columns]
if missing_features:
    raise ValueError(f"Missing features in the dataset: {missing_features}")

# Preprocessing: Handling missing values
transaction_data_selected = transaction_data[initial_features].fillna(0)
true_labels = transaction_data['FLAG']

# Feature Engineering: Adding interaction terms
transaction_data_selected['Sent_Received_Ratio'] = (
    transaction_data_selected['Sent tnx'] / (transaction_data_selected['Received Tnx'] + 1)
)
transaction_data_selected['Min_Max_Received_Ratio'] = (
    transaction_data_selected['min value received'] / (transaction_data_selected['max value received'] + 1)
)

# Split the data into training and testing sets
print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    transaction_data_selected, true_labels, test_size=0.3, random_state=42, stratify=true_labels
)

# Feature selection using RandomForest
print("Training RandomForestClassifier for feature selection...")
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_selector.fit(X_train, y_train)

# Get feature importances and select top features
feature_importances = pd.Series(rf_selector.feature_importances_, index=transaction_data_selected.columns)
top_features = feature_importances.nlargest(10).index.tolist()

print(f"Top selected features: {top_features}")

# Apply SMOTE only on the training data
print("Applying SMOTE to balance the training data...")
X_train_resampled, y_train_resampled = SMOTE(sampling_strategy='auto', random_state=42).fit_resample(
    X_train[top_features], y_train
)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
print("Performing hyperparameter tuning with GridSearchCV...")
grid_search = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

# Train Random Forest with best parameters on the resampled training data
print("Training RandomForestClassifier with best parameters on the resampled training data...")
rf_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
rf_model.fit(X_train_resampled, y_train_resampled)

# Predict on the test data
print("Predicting on the test dataset...")
y_pred = rf_model.predict(X_test[top_features])

# Evaluate the model
print("Evaluating the model...")
rf_precision = precision_score(y_test, y_pred)
rf_recall = recall_score(y_test, y_pred)
rf_f1 = f1_score(y_test, y_pred)
rf_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Random Forest Precision: {rf_precision}")
print(f"Random Forest Recall: {rf_recall}")
print(f"Random Forest F1-Score: {rf_f1}")
print(f"Random Forest AUC-ROC: {rf_auc}")

# Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Cross-Validation to ensure consistency
print("Performing cross-validation...")
cross_val_scores = cross_val_score(rf_model, transaction_data_selected[top_features], true_labels, cv=5, scoring='f1', n_jobs=-1)
print(f"Cross-Validation F1-Scores: {cross_val_scores}")
print(f"Average Cross-Validation F1-Score: {np.mean(cross_val_scores)}")

# Function to construct a transaction network graph
def construct_graph(data, top_features):
    G = nx.Graph()
    for index, row in data.iterrows():
        # Add an edge for each feature with the transaction index as a node
        for feature in top_features:
            G.add_edge(index, feature, weight=row[feature])
    return G

# Constructing the transaction network graph using a sample of the data
print("Constructing the transaction network graph...")
sample_data = transaction_data_selected[top_features].sample(n=100, random_state=42)  # Sampling for visualization
G = construct_graph(sample_data, top_features)

# Visualizing the network graph
print("Visualizing the network graph...")
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)  # Using a seed for reproducibility
nx.draw(G, pos, node_size=50, with_labels=False, edge_color='gray', alpha=0.7)
plt.title('Transaction Network Graph')
plt.show()
