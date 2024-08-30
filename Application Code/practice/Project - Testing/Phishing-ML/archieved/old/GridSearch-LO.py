from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

# Load the Ethereum transaction data
file_path = 'transaction_dataset.csv'
transaction_data = pd.read_csv(file_path)

# Trim leading and trailing spaces from column names
transaction_data.columns = transaction_data.columns.str.strip()

# Feature selection
features = [
    'Avg min between sent tnx', 'Avg min between received tnx',
    'Time Diff between first and last (Mins)', 'Sent tnx', 'Received Tnx',
    'Number of Created Contracts'
]

# Preprocessing: Handling missing values
transaction_data_selected = transaction_data[features].fillna(0)
true_labels = transaction_data['FLAG']

# Define the parameter grid for GridSearchCV
param_grid = {
    'svm__nu': [0.01, 0.05, 0.1, 0.2],
    'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1]
}

# Create a pipeline with SMOTE and One-Class SVM
pipeline = Pipeline([
    ('smote', SMOTE(sampling_strategy='auto', random_state=42)),
    ('svm', OneClassSVM(kernel='rbf'))
])

# Initialize GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, scoring='f1', cv=5, n_jobs=-1)

# Fit the model
grid_search.fit(transaction_data_selected, true_labels.apply(lambda x: 1 if x == 1 else -1))

# Best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Train the best model with SMOTE resampling
best_svm_model = OneClassSVM(kernel='rbf', nu=best_params['svm__nu'], gamma=best_params['svm__gamma'])
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(transaction_data_selected, true_labels)
best_svm_model.fit(X_resampled)
transaction_data['Anomaly_SVM_Best_SMOTE'] = best_svm_model.predict(transaction_data_selected)

# Evaluate the best model
svm_best_smote_precision = precision_score(true_labels, transaction_data['Anomaly_SVM_Best_SMOTE'].apply(lambda x: 1 if x == -1 else 0))
svm_best_smote_recall = recall_score(true_labels, transaction_data['Anomaly_SVM_Best_SMOTE'].apply(lambda x: 1 if x == -1 else 0))
svm_best_smote_f1 = f1_score(true_labels, transaction_data['Anomaly_SVM_Best_SMOTE'].apply(lambda x: 1 if x == -1 else 0))

print(f"Best SVM SMOTE Precision: {svm_best_smote_precision}, Recall: {svm_best_smote_recall}, F1-Score: {svm_best_smote_f1}")
