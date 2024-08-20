from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

# Load the Ethereum transaction data
file_path = 'transaction_dataset.csv'
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
for feature in initial_features:
    if feature not in transaction_data.columns:
        print(f"Feature '{feature}' is not in the dataset. Please check for typos.")
        exit(1)

# Preprocessing: Handling missing values
transaction_data_selected = transaction_data[initial_features].fillna(0)
true_labels = transaction_data['FLAG']

# Split the data into training and testing sets
print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(transaction_data_selected, true_labels, test_size=0.3, random_state=42)

# Feature selection using RandomForest
print("Training RandomForestClassifier for feature selection...")
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X_train, y_train)

# Get feature importances and select top features
feature_importances = pd.Series(rf_selector.feature_importances_, index=initial_features)
top_features = feature_importances.nlargest(10).index.tolist()

print(f"Top selected features: {top_features}")

# Apply SMOTE only on the training data
print("Applying SMOTE to balance the training data...")
X_train_resampled, y_train_resampled = SMOTE(sampling_strategy='auto', random_state=42).fit_resample(X_train[top_features], y_train)

# Train Random Forest on the resampled training data
print("Training RandomForestClassifier on the resampled training data...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
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

# Cross-Validation to ensure consistency
print("Performing cross-validation...")
cross_val_scores = cross_val_score(rf_model, transaction_data_selected[top_features], true_labels, cv=5, scoring='f1')
print(f"Cross-Validation F1-Scores: {cross_val_scores}")
print(f"Average Cross-Validation F1-Score: {np.mean(cross_val_scores)}")
