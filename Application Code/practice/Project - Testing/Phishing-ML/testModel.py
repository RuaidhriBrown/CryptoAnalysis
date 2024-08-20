import joblib
import pandas as pd

def load_model():
    # Load the trained model
    model = joblib.load('results/random_forest_phishing_detector_aggregated.pkl')

    # Load the feature names used during training
    trained_feature_names = joblib.load('results/random_forest_phishing_detector_aggregated_features.pkl')
    
    return model, trained_feature_names

def preprocess_features(features, trained_feature_names):
    # Apply necessary feature engineering steps
    features['Sent_Received_Ratio'] = features['Sent tnx'] / (features['Received Tnx'] + 1)
    features['Min_Max_Received_Ratio'] = features['min value received'] / (features['max value received'] + 1)
    
    # Ensure only expected features are included and in the correct order
    features = features.reindex(columns=trained_feature_names, fill_value=0)

    return features

def main():
    model, trained_feature_names = load_model()

    # Example features dictionary (ensure these match your actual data)
    features = {
        'Avg min between sent tnx': [5], 
        'Avg min between received tnx': [3], 
        'Time Diff between first and last (Mins)': [1000], 
        'Sent tnx': [10], 
        'Received Tnx': [8], 
        'Unique Received From Addresses': [5], 
        'Unique Sent To Addresses': [7], 
        'min value received': [0.1], 
        'max value received': [2.0], 
        'avg val received': [1.0], 
        'min val sent': [0.1], 
        'max val sent': [2.0], 
        'avg val sent': [1.0], 
        'total Ether sent': [10.0], 
        'total ether received': [12.0], 
        'total ether balance': [2.0], 
        'Total ERC20 tnxs': [5], 
        'ERC20 total Ether received': [1.0], 
        'ERC20 total ether sent': [1.0], 
        'ERC20 uniq sent addr': [3], 
        'ERC20 uniq rec addr': [4], 
        'ERC20 avg time between sent tnx': [2.0], 
        'ERC20 avg time between rec tnx': [3.0], 
        'ERC20 min val rec': [0.1], 
        'ERC20 max val rec': [2.0], 
        'ERC20 avg val rec': [1.0], 
        'ERC20 min val sent': [0.1], 
        'ERC20 max val sent': [2.0], 
        'ERC20 avg val sent': [1.0], 
        'ERC20 uniq sent token name': ['TOKEN1'], 
        'ERC20 uniq rec token name': ['TOKEN2'], 
        'ERC20 most sent token type': ['TOKEN1'], 
        'ERC20_most_rec_token_type': ['TOKEN2'],
        # Add the engineered features for consistency with the training phase
        'Sent_Received_Ratio': [10 / (8 + 1)],  # Example calculation
        'Min_Max_Received_Ratio': [0.1 / (2.0 + 1)]  # Example calculation
    }

    # Convert to DataFrame
    features_df = pd.DataFrame(features)

    # Preprocess the features to ensure they match the model's expected input
    features_df = preprocess_features(features_df, trained_feature_names)

    # Predict
    prediction = model.predict(features_df)
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
