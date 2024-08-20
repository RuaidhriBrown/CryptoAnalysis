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

# Load the dataset
file_path = 'datasets/ethereum/combined_transactions.csv'
transaction_data = pd.read_csv(file_path)

# Feature selection based on the dataset provided
initial_features = [
    'value', 'gas', 'gasPrice', 'nonce', 'cumulativeGasUsed', 'gasUsed'
]

# Check if all selected features exist in the dataset
missing_features = [feature for feature in initial_features if feature not in transaction_data.columns]
if missing_features:
    raise ValueError(f"Missing features in the dataset: {missing_features}")

# Select the features and handle missing values
transaction_data_selected = transaction_data[initial_features].apply(pd.to_numeric, errors='coerce').fillna(0)
true_labels = transaction_data['FLAG']

# Feature Engineering: Adding interaction terms
transaction_data_selected['Gas_Used_Ratio'] = (
    transaction_data_selected['gasUsed'] / (transaction_data_selected['gas'] + 1)
)
transaction_data_selected['Value_Gas_Ratio'] = (
    transaction_data_selected['value'] / (transaction_data_selected['gas'] + 1)
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    transaction_data_selected, true_labels, test_size=0.3, random_state=42, stratify=true_labels
)

# Undersample the test set to balance the classes
rus = RandomUnderSampler(random_state=42)
X_test_balanced, y_test_balanced = rus.fit_resample(X_test, y_test)

# Feature selection using RandomForest
print("Training RandomForestClassifier for feature selection...")
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_selector.fit(X_train, y_train)

# Get feature importances and select top features
feature_importances = pd.Series(rf_selector.feature_importances_, index=transaction_data_selected.columns)
top_features = feature_importances.nlargest(10).index.tolist()

# Save the top features used during training
joblib.dump(top_features, 'random_forest_phishing_detector_transactions_features.pkl')

# Apply SMOTE only on the training data
print("Applying SMOTE to balance the training data...")
X_train_resampled, y_train_resampled = SMOTE(sampling_strategy='auto', random_state=42).fit_resample(
    X_train[top_features], y_train
)

# Hyperparameter tuning with manual progress bar
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

print("Performing hyperparameter tuning with GridSearchCV...")
best_score = 0
best_params = None
grid = ParameterGrid(param_grid)

with tqdm(total=len(grid)) as pbar:
    for params in grid:
        rf_model = RandomForestClassifier(**params, random_state=42, n_jobs=-1, class_weight='balanced')
        rf_model.fit(X_train_resampled, y_train_resampled)
        y_pred = rf_model.predict(X_test_balanced[top_features])
        score = f1_score(y_test_balanced, y_pred)
        if score > best_score:
            best_score = score
            best_params = params
        pbar.update(1)

print(f"Best parameters found: {best_params}")

# Train Random Forest with best parameters on the resampled training data
print("Training RandomForestClassifier with best parameters on the resampled training data...")
rf_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1, class_weight='balanced')
rf_model.fit(X_train_resampled, y_train_resampled)

# Save the trained model
model_filename = 'random_forest_phishing_detector_transactions.pkl'
joblib.dump(rf_model, model_filename)
print(f"Model saved to {model_filename}")

# Predict on the balanced test data
print("Predicting on the test dataset...")
y_pred = rf_model.predict(X_test_balanced[top_features])
y_prob = rf_model.predict_proba(X_test_balanced[top_features])[:, 1]  # Probability estimates for the positive class

# Evaluate the model
print("Evaluating the model...")
rf_precision = precision_score(y_test_balanced, y_pred)
rf_recall = recall_score(y_test_balanced, y_pred)
rf_f1 = f1_score(y_test_balanced, y_pred)
rf_auc = roc_auc_score(y_test_balanced, y_prob)
conf_matrix = confusion_matrix(y_test_balanced, y_pred)

print(f"Random Forest Precision: {rf_precision}")
print(f"Random Forest Recall: {rf_recall}")
print(f"Random Forest F1-Score: {rf_f1}")
print(f"Random Forest AUC-ROC: {rf_auc}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test_balanced, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test_balanced, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, lw=2, color='purple')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Plot Feature Importance
plt.figure(figsize=(10, 6))
feature_importances.sort_values(ascending=False).plot(kind='bar', color='teal')
plt.title('Feature Importance')
plt.show()

# Cross-Validation to ensure consistency
print("Performing cross-validation...")
cross_val_scores = cross_val_score(rf_model, transaction_data_selected[top_features], true_labels, cv=5, scoring='f1', n_jobs=-1)
print(f"Cross-Validation F1-Scores: {cross_val_scores}")
print(f"Average Cross-Validation F1-Score: {np.mean(cross_val_scores)}")
