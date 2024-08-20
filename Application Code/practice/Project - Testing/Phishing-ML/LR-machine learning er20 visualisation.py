
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import joblib

# Load the Ethereum transaction data
file_path = 'datasets/ethereum/combined_er20.csv'
transaction_data = pd.read_csv(file_path)

# Feature selection based on the dataset provided in the image
initial_features = [
    'value', 'gas', 'gasPrice', 'gasUsed', 'cumulativeGasUsed',
    'nonce', 'transactionIndex', 'confirmations'
]

# Check if all selected features exist in the dataset
missing_features = [feature for feature in initial_features if feature not in transaction_data.columns]
if missing_features:
    raise ValueError(f"Missing features in the dataset: {missing_features}")

# Select the features and handle missing values
transaction_data_selected = transaction_data[initial_features].apply(pd.to_numeric, errors='coerce').fillna(0)
true_labels = transaction_data['FLAG']

# Visualize the distribution of selected features
sns.pairplot(transaction_data_selected, diag_kind='kde')
plt.suptitle("Feature Distribution Before Engineering", y=1.02)
plt.show()

# Feature Engineering: Adding interaction terms
transaction_data_selected['Gas_Used_Ratio'] = (
    transaction_data_selected['gasUsed'] / (transaction_data_selected['gas'] + 1)
)
transaction_data_selected['Value_Gas_Ratio'] = (
    transaction_data_selected['value'] / (transaction_data_selected['gas'] + 1)
)
transaction_data_selected['Transaction_Value_Efficiency'] = (
    transaction_data_selected['value'] / (transaction_data_selected['cumulativeGasUsed'] + 1)
)

# Handling infinity and very large values
transaction_data_selected.replace([np.inf, -np.inf], np.nan, inplace=True)
transaction_data_selected.fillna(transaction_data_selected.mean(), inplace=True)
transaction_data_selected = transaction_data_selected.clip(upper=1e10)

# Optional: Apply log transformation
transaction_data_selected = np.log1p(transaction_data_selected)

# Visualize the distribution of engineered features
sns.pairplot(transaction_data_selected, diag_kind='kde')
plt.suptitle("Feature Distribution After Engineering and Log Transformation", y=1.02)
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    transaction_data_selected, true_labels, test_size=0.3, random_state=42, stratify=true_labels
)

# Undersample the test set to balance the classes
rus = RandomUnderSampler(random_state=42)
X_test_balanced, y_test_balanced = rus.fit_resample(X_test, y_test)

# Visualize the class distribution after balancing
sns.countplot(x=y_test_balanced)
plt.title("Class Distribution After Undersampling")
plt.show()

# Feature selection using RandomForest
print("Training RandomForestClassifier for feature selection...")
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_selector.fit(X_train, y_train)

# Get feature importances and select top features
feature_importances = pd.Series(rf_selector.feature_importances_, index=transaction_data_selected.columns)
top_features = feature_importances.nlargest(10).index.tolist()

# Visualize the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.nlargest(10).values, y=top_features)
plt.title("Top 10 Feature Importances from RandomForest")
plt.show()

# Save the top features used during training
joblib.dump(top_features, 'random_forest_phishing_detector_erc20_features.pkl')

# Apply SMOTE only on the training data
print("Applying SMOTE to balance the training data...")
X_train_resampled, y_train_resampled = SMOTE(sampling_strategy='auto', random_state=42).fit_resample(
    X_train[top_features], y_train
)

# Visualize the class distribution after SMOTE
sns.countplot(x=y_train_resampled)
plt.title("Class Distribution After SMOTE")
plt.show()
