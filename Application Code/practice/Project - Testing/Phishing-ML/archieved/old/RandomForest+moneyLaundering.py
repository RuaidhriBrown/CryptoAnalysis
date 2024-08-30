from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from tqdm import tqdm

# Function to load data with progress bar
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to analyze transactions and create new features
def transaction_data_analysis(df, wallet_address):
    # Ensure columns are correctly named
    df.columns = df.columns.str.strip()
    
    # Convert 'value' and 'timeStamp' columns to numeric types
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['timeStamp'] = pd.to_numeric(df['timeStamp'], errors='coerce')

    # Example fixed thresholds for suspicious activity
    large_transaction_threshold = 10000  # in Ether
    short_time_threshold = 60  # in seconds (1 minute)
    high_volume_threshold = 10  # More than 10 transactions in a short period
    circular_transaction_threshold = 5  # More than 5 transactions to the same address
    
    # Filter data for the specified wallet
    df_filtered = df[df['from'] == wallet_address]
    total_transactions = len(df_filtered)
    
    # Flag large transactions
    large_transactions = df_filtered[df_filtered['value'] > large_transaction_threshold]
    percent_large_transactions = (len(large_transactions) / total_transactions) * 100 if total_transactions > 0 else 0
    
    # Flag transactions with short intervals
    df_filtered['time_diff'] = df_filtered['timeStamp'].diff().abs()
    short_interval_transactions = df_filtered[df_filtered['time_diff'] < short_time_threshold]
    percent_short_interval_transactions = (len(short_interval_transactions) / total_transactions) * 100 if total_transactions > 0 else 0
    
    # Flag high volume of transactions within a short period
    df_filtered['transaction_count'] = df_filtered['timeStamp'].rolling(window=short_time_threshold).count()
    high_volume_transactions = df_filtered[df_filtered['transaction_count'] > high_volume_threshold]
    percent_high_volume_transactions = (len(high_volume_transactions) / total_transactions) * 100 if total_transactions > 0 else 0
    
    # Flag circular transactions
    circular_transactions_counts = df_filtered['to'].value_counts()
    circular_transactions = circular_transactions_counts[circular_transactions_counts > circular_transaction_threshold]
    percent_circular_transactions = (len(circular_transactions) / total_transactions) * 100 if total_transactions > 0 else 0
    
    # Flag unusual patterns (e.g., large amounts received then split into smaller amounts)
    large_received_transactions = df[(df['value'] > large_transaction_threshold) & (df['to'] == wallet_address)]
    split_transactions = df[(df['value'] < large_transaction_threshold) & (df['from'].isin(large_received_transactions['to']))]
    percent_split_transactions = (len(split_transactions) / total_transactions) * 100 if total_transactions > 0 else 0
    
    stats = {
        "Address": wallet_address,
        "percent_large_transactions": percent_large_transactions,
        "percent_short_interval_transactions": percent_short_interval_transactions,
        "percent_high_volume_transactions": percent_high_volume_transactions,
        "percent_circular_transactions": percent_circular_transactions,
        "percent_split_transactions": percent_split_transactions,
    }
    
    return stats

# Load the aggregated wallet data
file_path_aggregated = 'transaction_dataset_even.csv'
df_aggregated = load_data(file_path_aggregated)

# Load the transaction data
file_path_transactions = 'combined_transactions.csv'
df_transactions = load_data(file_path_transactions)

# Ensure columns are correctly named
df_aggregated.columns = df_aggregated.columns.str.strip()
df_transactions.columns = df_transactions.columns.str.strip()

# List to store the transaction analysis results
transaction_analysis_results = []

# Perform the transaction data analysis for each address
print("Performing transaction data analysis for each address...")
for address in tqdm(df_aggregated['Address']):
    transaction_stats = transaction_data_analysis(df_transactions, address)
    transaction_analysis_results.append(transaction_stats)

# Convert the transaction analysis results to a DataFrame
transaction_stats_df = pd.DataFrame(transaction_analysis_results)

# Merge the transaction stats with the aggregated data
df_combined = pd.merge(df_aggregated, transaction_stats_df, on='Address', how='left')

print(df_combined.head())

# Define initial features and add the new transaction stats features
initial_features = [
    'Avg min between sent tnx', 'Avg min between received tnx',
    'Time Diff between first and last (Mins)', 'Sent tnx', 'Received Tnx',
    'Number of Created Contracts', 'Unique Received From Addresses', 'Unique Sent To Addresses',
    'min value received', 'max value received', 'avg val received',
    'min val sent', 'max val sent', 'avg val sent'
]

# Add new features from transaction analysis
new_features = [
    'percent_large_transactions', 'percent_short_interval_transactions',
    'percent_high_volume_transactions', 'percent_circular_transactions', 'percent_split_transactions'
]

all_features = initial_features + new_features

# Ensure no extra spaces in column names
df_combined.columns = df_combined.columns.str.strip()

# Check if all features exist in the dataset
for feature in all_features:
    if feature not in df_combined.columns:
        print(f"Feature '{feature}' is not in the dataset. Please check for typos.")
        exit(1)

# Preprocessing: Handling missing values
df_combined_selected = df_combined[all_features].fillna(0)
true_labels = df_combined['FLAG']

# Split the data into training and testing sets
print("Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(df_combined_selected, true_labels, test_size=0.3, random_state=42)

# Feature selection using RandomForest
print("Training RandomForestClassifier for feature selection...")
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X_train, y_train)

# Get feature importances and select top features
feature_importances = pd.Series(rf_selector.feature_importances_, index=all_features)
top_features = feature_importances.nlargest(15).index.tolist()

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

print(f"Random Forest Precision: {rf_precision}")
print(f"Random Forest Recall: {rf_recall}")
print(f"Random Forest F1-Score: {rf_f1}")

# Cross-Validation to ensure consistency
print("Performing cross-validation...")
cross_val_scores = cross_val_score(rf_model, df_combined_selected[top_features], true_labels, cv=5, scoring='f1')
print(f"Cross-Validation F1-Scores: {cross_val_scores}")
print(f"Average Cross-Validation F1-Score: {np.mean(cross_val_scores)}")
