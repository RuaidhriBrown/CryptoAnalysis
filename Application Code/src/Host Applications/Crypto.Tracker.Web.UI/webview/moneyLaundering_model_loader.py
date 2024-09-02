
import os
import pickle
import joblib
from django.conf import settings

def load_moneyLaundering_wallet_model():
    # Define the directory paths
    models_dir = os.path.join(settings.BASE_DIR, 'models')
    ethereum_dir = os.path.join(models_dir, 'ethereum')
    phishing_dir = os.path.join(ethereum_dir, 'moneyLaundering')

    # Define the paths for the wallet model and features
    wallet_model_path = os.path.join(phishing_dir, 'random_forest_moneyLaundering_detector_aggregated.pkl')
    wallet_features_path = os.path.join(phishing_dir, 'random_forest_moneyLaundering_detector_aggregated_features.pkl')

    # Load the model
    wallet_model = joblib.load(wallet_model_path)
    
    # Ensure the model is a RandomForestClassifier or similar model
    if not hasattr(wallet_model, 'predict'):
        raise ValueError("The loaded wallet model does not have a predict method.")
    
    # Load the feature names
    wallet_features = joblib.load(wallet_features_path)

    return wallet_model, wallet_features


def load_moneyLaundering_transaction_model():
    # Define the directory paths
    models_dir = os.path.join(settings.BASE_DIR, 'models')
    ethereum_dir = os.path.join(models_dir, 'ethereum')
    phishing_dir = os.path.join(ethereum_dir, 'moneyLaundering')

    # Define the paths for the transaction model and features
    transaction_model_path = os.path.join(phishing_dir, 'random_forest_moneyLaundering_detector_transactions.pkl')
    transaction_features_path = os.path.join(phishing_dir, 'random_forest_moneyLaundering_detector_transactions_features.pkl')

    # Load the transaction model
    transaction_model = joblib.load(transaction_model_path)

    if not hasattr(transaction_model, 'predict'):
        raise ValueError("The loaded wallet model does not have a predict method.")

    # Load the transaction features
    transaction_features = joblib.load(transaction_features_path)

    return transaction_model, transaction_features


def load_moneyLaundering_erc20_model():
    # Define the directory paths
    models_dir = os.path.join(settings.BASE_DIR, 'models')
    ethereum_dir = os.path.join(models_dir, 'ethereum')
    phishing_dir = os.path.join(ethereum_dir, 'moneyLaundering')

    # Define the paths for the ERC20 model and features
    erc20_model_path = os.path.join(phishing_dir, 'random_forest_moneyLaundering_detector_er20.pkl')
    erc20_features_path = os.path.join(phishing_dir, 'random_forest_moneyLaundering_detector_er20_features.pkl')
        
    # Load the transaction model
    erc20_model = joblib.load(erc20_model_path)

    if not hasattr(erc20_model, 'predict'):
        raise ValueError("The loaded wallet model does not have a predict method.")

    # Load the transaction features
    erc20_features = joblib.load(erc20_features_path)

    return erc20_model, erc20_features
