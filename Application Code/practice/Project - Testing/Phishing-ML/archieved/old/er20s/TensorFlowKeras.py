import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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

# Split the data into training and test sets
print("Splitting the data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split into training and test sets. Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Show the distribution of the target variable in the training and test sets
print("Distribution of target variable in the training set:")
print(y_train.value_counts())
print("\nDistribution of target variable in the test set:")
print(y_test.value_counts())

# Define numerical_features
print("Identifying numerical features...")
numerical_features = X_train.columns.tolist()
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
print("Preprocessing the training data...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print("Training data preprocessed.")

print("Building the neural network model...")
# Define the neural network model
def create_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train_processed.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build the model
model = create_model()
print("Neural network model built.")

print("Training the model...")
# Train the model
model.fit(X_train_processed, y_train, epochs=20, batch_size=32, verbose=1, validation_split=0.2)
print("Model trained.")

print("Making predictions on the test set...")
# Predict on the test set
y_pred_prob = model.predict(X_test_processed)
y_pred = (y_pred_prob > 0.5).astype("int32")
print("Predictions made.")

print("Evaluating the model...")
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Model evaluated.")

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
