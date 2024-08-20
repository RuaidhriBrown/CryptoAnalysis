import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.base import BaseEstimator, ClassifierMixin

print("Loading data from the combined CSV file...")
# Load the data from the combined CSV file
file_combined = 'combined_er20.csv'

# Read the data into a pandas dataframe
data_combined = pd.read_csv(file_combined)
print("Data loaded successfully.")

print("Dropping columns that won't be useful for prediction...")
# Drop columns that won't be useful for prediction
columns_to_drop = ['hash', 'blockHash', 'from', 'to', 'contractAddress', 'input', 'tokenName', 'tokenSymbol']
data_combined = data_combined.drop(columns=columns_to_drop)
print("Columns dropped.")

print("Handling missing values...")
# Handle missing values (fill with -1 for now, can be optimized later)
data_combined = data_combined.fillna(-1)
print("Missing values handled.")

print("Aggregating data by address...")
# Aggregate numerical features by address
agg_numerical = data_combined.groupby('Address').agg({
    'blockNumber': 'mean', 
    'timeStamp': 'mean',
    'nonce': 'sum', 
    'transactionIndex': 'mean', 
    'value': 'sum', 
    'gas': 'sum', 
    'gasPrice': 'mean', 
    'gasUsed': 'sum', 
    'cumulativeGasUsed': 'sum', 
    'confirmations': 'mean',
    'tokenDecimal': 'mean'
})

# Convert all columns to numeric, coercing errors
agg_numerical = agg_numerical.apply(pd.to_numeric, errors='coerce')

# Replace infinite values with NaN and then fill NaN with a large number
agg_numerical.replace([np.inf, -np.inf], np.nan, inplace=True)
agg_numerical.fillna(1e10, inplace=True)

# Clip values to a reasonable range
agg_numerical = agg_numerical.clip(lower=-1e10, upper=1e10)

# Aggregate the FLAG to get the maximum value per address (if any transaction is flagged, the address is flagged)
agg_flag = data_combined.groupby('Address')['FLAG'].max()

# Combine aggregated features
aggregated_data = pd.concat([agg_numerical, agg_flag], axis=1).reset_index()
print("Data aggregated.")

# Define features and target
print("Defining features and target...")
X = aggregated_data.drop(columns=['FLAG', 'Address'])
y = aggregated_data['FLAG']
print("Features and target defined.")

# Define numerical_features
print("Identifying numerical features...")
numerical_features = X.columns.tolist()
print("Numerical features identified.")

print("Setting up the preprocessor...")
# Preprocessor for numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features)
    ])
print("Preprocessor set up.")

# Preprocess the data
print("Preprocessing the data...")
X_processed = preprocessor.fit_transform(X)
print("Data preprocessed.")

# Custom Keras classifier to integrate with scikit-learn
class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn, epochs=10, batch_size=32, verbose=0):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_ = None
        self.classes_ = [0, 1]

    def fit(self, X, y, **kwargs):
        self.model_ = self.build_fn()
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, **kwargs)
        return self

    def predict(self, X, **kwargs):
        return (self.model_.predict(X, **kwargs) > 0.5).astype("int32")

    def predict_proba(self, X, **kwargs):
        return self.model_.predict(X, **kwargs)

print("Building the neural network model...")
# Define the neural network model
def create_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_processed.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wrap the model using the custom KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=32, verbose=0)

# Perform cross-validation
print("Performing cross-validation...")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_val_score(model, X_processed, y, cv=kfold, scoring='accuracy')

print("Cross-validation results:")
print(f"Mean accuracy: {results.mean():.4f}")
print(f"Standard deviation: {results.std():.4f}")

# Split the data into training and test sets for final evaluation
print("Splitting the data into training and test sets for final evaluation...")
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split into training and test sets. Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Train the model on the full training set
print("Training the model on the full training set...")
model.fit(X_train, y_train)
print("Model trained.")

print("Making predictions on the test set...")
# Predict on the test set
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")
print("Predictions made.")

print("Evaluating the model...")
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Model evaluated.")

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
