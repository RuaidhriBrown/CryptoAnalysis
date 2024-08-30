import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

print("Loading data from the combined CSV file...")
# Load the data from the combined CSV file
file_combined = 'combined_transactions.csv'

# Read the data into a pandas dataframe
data_combined = pd.read_csv(file_combined)
print("Data loaded successfully.")

print("Dropping columns that won't be useful for prediction...")
# Drop columns that won't be useful for prediction
columns_to_drop = ['hash', 'blockHash', 'from', 'to', 'contractAddress', 'input', 'functionName']
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
    'isError': 'sum', 
    'txreceipt_status': 'sum', 
    'cumulativeGasUsed': 'sum', 
    'gasUsed': 'sum', 
    'confirmations': 'mean'
})

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

# Define numerical_features and categorical_features
print("Identifying numerical and categorical features...")
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("Numerical and categorical features identified.")

print("Setting up the preprocessor...")
# Redefine the preprocessor with proper handling of missing values
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features)
    ])
print("Preprocessor set up.")

print("Creating the Random Forest pipeline...")
# Create the Random Forest pipeline
model_pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
print("Random Forest pipeline created.")

print("Training the model...")
# Train the model
model_pipeline_rf.fit(X_train, y_train)
print("Model trained.")

print("Making predictions on the test set...")
# Predict on the test set
y_pred = model_pipeline_rf.predict(X_test)
print("Predictions made.")

print("Evaluating the model...")
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Model evaluated.")

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
