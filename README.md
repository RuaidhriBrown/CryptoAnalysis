# Crypto Analysis Tool

Crypto Analysis Tool is a tool for analyzing the transaction histories of cryptocurrencies, specifically Ethereum, to identify illicit activities such as phishing, money laundering, and other suspicious behaviors. This project encompasses the development of machine learning models for anomaly detection and a web application to make these insights accessible to users.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Local Development](#local-development)
   - [Installing Requirements](#installing-requirements)
3. [Machine Learning Model Training](#machine-learning-model-training)
   - [Phishing Detection](#phishing-detection)
   - [Money Laundering Detection](#money-laundering-detection)
4. [Web Application](#web-application)
   - [Overview](#overview)
   - [Key Features](#key-features)
   - [Running the Web Application](#running-the-web-application)
   - [Web Application Configuration](#web-application-configuration)
5. [University of Buckingham](#university-of-buckingham)

## Project Overview

The Crypto Analysis Tool aims to provide a comprehensive, automated solution for detecting illicit activities within the Ethereum blockchain. The tool includes machine learning models trained on real transaction data to identify patterns indicative of fraudulent activities such as phishing and money laundering. The models are integrated into a web application that allows users to perform in-depth analysis of blockchain transactions.

## Local Development

To set up a local development environment, use the provided Docker Compose configuration. This setup ensures all dependencies are correctly installed and the environment mirrors the production setting.

Note: We highly recommend using Docker Compose for setting up the project, as it simplifies the process and ensures that the environment is consistent across different systems. Docker Compose automatically handles all configurations, including environment variables, database settings, and static files.

You will still need to ensure the trained models are added to the models directory

[Phishing Detection](#phishing-detection)
[Money Laundering Detection](#money-laundering-detection)

### Why Use Docker Compose?
- Simplifies Configuration: Docker Compose manages all dependencies and configurations in a single file, eliminating the need for manual setup in settings.py.
- Ensures Consistency: The Docker environment mirrors the production environment, reducing the risk of inconsistencies or errors.
- Quick Setup: Set up the entire project environment with a single command.

Note: the static files may not work in the docker-compose method

### Steps to Start Local Development Using Docker Compose:
1. **Clone the repository**:
    git clone https://github.com/RuaidhriBrown/CryptoAnalysis.git
    cd 'CryptoAnalysis\Application Code\src\Host Applications\Crypto.Tracker.Web.UI\'
2. **Run Docker Compose to build and start the services**:
    Run the following: docker-compose up --build
    This command will build the Docker images and start the containers for the web application and the database. All configurations, including environment variables and database settings, are automatically managed by Docker Compose.
3. **Access the Web Application**:
    - Once the Docker containers are up and running, you can access the web application in your browser at http://localhost:8000.

### Avoid Manual Configuration
If you choose to use Docker Compose, you don't need to manually configure the settings.py file for environment-specific settings like SECRET_KEY, DEBUG, ALLOWED_HOSTS, or DATABASES. Docker Compose handles these settings through environment variables defined in the docker-compose.yml file.

However, if you still prefer manual setup, please refer to the Web Application Configuration section below.

### Installing Requirements

Before running the project, you need to install all the required Python packages for both the machine learning models and the web application.

#### Installing Requirements for Machine Learning Model Training

1. **Navigate to the project directory**:
   ```bash
   cd "CryptoAnalysis\Application Code\practice\Project - Testing\"
   ```
2. **Install the Python dependencies**:
    - Make sure you have Python and pip installed. If you're using a virtual environment (recommended), activate it first.
    - Run the following command to install the required packages for the machine learning models:
   ```bash
   pip install -r requirements.txt
   ```
3. **Installing Requirements for the Web Application**:
    - Navigate to the web application directory: 'cd CryptoAnalysis/Application\ Code/src/Host\ Applications/Crypto.Tracker.Web.UI'
    ```bash
    pip install -r requirements.txt
    ```
    - Ensure that the requirements.txt file, which contains all necessary dependencies for running the Django web application, is located in the web application directory.
Note: If you encounter any issues during installation, check that your Python version and pip are up to date. You may also need to install additional system dependencies if required by specific Python packages.

## Machine Learning Model Training

### Overview

The machine learning component of the Crypto Analysis Tool consists of several models designed to detect illicit activities on the Ethereum blockchain, specifically targeting phishing and money laundering behaviors. The models use Random Forest and Graph Neural Networks (GNNs) to analyze transaction data and identify suspicious patterns.

### Phishing Detection

Phishing detection utilizes three datasets:
- Aggregated Ethereum Wallets
- Ethereum Transaction Data
- ERC20 Token Transaction Data

The phishing dataset is based on data from [Kaggle](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset/data). This dataset provides a list of wallets from which transactions and ERC20 transactions were collected using the Etherscan API. The resulting data was used to create an aggregated Ethereum Wallets dataset, representing known phishing accounts. Note that the transactions involve recognized phishing accounts but may not necessarily represent genuine phishing activities.


### Money Laundering Detection

Money laundering detection focuses on datasets derived from incidents like the Upbit hack, including:
- Aggregated Ethereum Wallets
- Ethereum Transaction Data
- ERC20 Token Transaction Data

These datasets were created using the script located at:
`CryptoAnalysis/Application Code/practice/Project - Testing/moneyLaundering/fetch_data/getMoneyLaunderingData1Layer.py`.

#### Developing the Upbit-Hack Dataset for Money Laundering Detection

The Upbit hack dataset was developed to train models for detecting money laundering activities on the Ethereum blockchain. The dataset construction involved several key steps:

1. **API Configuration and Setup**:  
   The Etherscan API was used to retrieve Ethereum transaction data. To comply with API rate limits, a maximum of 5 API calls per second was enforced, and retries were handled for failed requests.

2. **Data Collection**:  
   Transaction data was collected using the `get_normal_transactions` and `get_erc20_transactions` functions. These functions fetch normal Ether transactions and ERC20 token transactions associated with a given Ethereum address, filtering the results to include only relevant transactions that meet specific criteria (e.g., minimum Ether sent and a specific start time).

3. **Address Analysis**:  
   Each Ethereum address (wallet) is analyzed for potential involvement in money laundering activities using the `analyze_wallet` function. Key metrics such as Fast-In Fast-Out (FIFO) ratio, small-volume transaction ratio, zero-out ratio, and network density are calculated to identify suspicious behavior.

4. **Detection of Money Laundering**:  
   The `detect_money_laundering` function applies predefined thresholds to determine if a wallet is suspicious. The thresholds include FIFO ratio, small-volume ratio, zero-out ratio, network density, and minimum balance. If a wallet meets or exceeds the thresholds, it is flagged as suspicious.

5. **Interacting Wallets**:  
   The `analyze_interacted_wallets` function identifies wallets that have interacted with the initial suspicious address. It recursively analyzes these wallets to uncover potential layers of money laundering activity, expanding the dataset to include a broader network of transactions.

6. **Data Compilation**:  
   Data is accumulated in global dataframes for all analyzed wallets, normal transactions, and ERC20 transactions. These dataframes are then saved to CSV files (`all_wallet_summaries.csv`, `all_normal_transactions.csv`, and `all_erc20_transactions.csv`) for further analysis and model training.

7. **Create 'Ethereum Transaction' and 'ERC20 Token Transaction' Datasets**:  
   Combine normal and ERC20 transactions from both suspicious and legitimate wallets.

8. **Create 'Aggregated Ethereum Wallets' Dataset**:  
   Aggregate the 'Ethereum Transaction Data' and 'ERC20 Token Transaction Data' to derive features such as:
   - Avg min between sent and received transactions
   - Avg min between received tnx
   - Time Diff between first and last (Mins)
   - Sent tnx
   - Received Tnx
   - Unique Received From Addresses
   - Unique Sent To Addresses
   - min value received
   - max value received
   - avg val received
   - min val sent
   - max val sent
   - avg val sent
   - total Ether sent
   - total ether received
   - total ether balance
   - Total ERC20 tnxs
   - ERC20 total Ether received
   - ERC20 total ether sent
   - ERC20 uniq sent addr
   - ERC20 uniq rec addr
   - ERC20 avg time between sent tnx
   - ERC20 avg time between rec tnx
   - ERC20 min val rec
   - ERC20 max val rec
   - ERC20 avg val rec
   - ERC20 min val sent
   - ERC20 max val sent
   - ERC20 avg val sent
   - ERC20 uniq sent token name
   - ERC20 uniq rec token name
   - ERC20 most sent token type
   - ERC20_most_rec_token_type

    #### Python Code for Dataset Development

    Below is a simplified version of the Python script used to develop the Upbit hack dataset:

    ```python
    import requests
    import pandas as pd
    import time
    from tqdm import tqdm
    from datetime import datetime
    import random

    # Replace with your own Etherscan API Key
    API_KEY = 'YOUR_API_KEY'
    BASE_URL = 'https://api.etherscan.io/api'
    MAX_API_CALLS_PER_SECOND = 5  # Max 5 API calls per second
    RETRY_COUNT = 3  # Number of retries for failed requests
    START_TIME = '2019-11-27 04:06:40'  # Start time for filtering transactions
    MAX_WALLETS_PER_LAYER = 20  # Maximum number of wallets to check per layer
    MINIMUM_ETHER_SENT = 1 * 10**18  # Minimum amount (in Wei) for transactions to be considered

    # Initialize global dataframes to accumulate all data
    all_wallet_summaries = []
    all_normal_transactions = []
    all_erc20_transactions = []
    visited_wallets = set()  # Set to keep track of visited wallets

    def get_account_balance(address):
        # Code to retrieve account balance
        pass

    def make_request(url):
        # Helper function for API requests with retry logic
        pass

    def get_normal_transactions(address):
        # Function to get normal Ether transactions
        pass

    def get_erc20_transactions(address):
        # Function to get ERC20 token transactions
        pass

    def detect_money_laundering(fifo_ratio, small_volume_ratio, zero_out_ratio, network_density, balance, thresholds):
        # Logic to detect potential money laundering
        pass

    def analyze_wallet(address):
        # Analyze a wallet for suspicious activity
        pass

    def analyze_interacted_wallets(address):
        # Analyze wallets that interacted with a given address
        pass

    # Start the analysis with the known Upbit hack address
    upbit_hack_address = '0xa09871AEadF4994Ca12f5c0b6056BBd1d343c029'
    analyze_interacted_wallets(upbit_hack_address)

2. **Feature Engineering and Selection**:
   - Custom features such as transaction ratios, gas usage metrics, and interaction terms are engineered to capture complex behaviors associated with illicit activities.
   - Specific features like 'Sent_Received_Ratio' and 'Fast-In Fast-Out (FIFO) Ratio' help differentiate between legitimate and fraudulent activities.

3. **Model Development**:
   - **Phishing Detection Models**: Trained using Random Forest algorithms on balanced datasets, achieving high precision and recall rates. GNNs are also explored for detecting phishing by leveraging transaction network topologies.
   - **Money Laundering Detection Models**: Focus on identifying rapid fund transfers and small-volume transactions indicative of laundering, utilizing similar machine learning approaches.

4. **Evaluation Metrics**:
   - Models are evaluated using metrics such as Precision, Recall, F1-Score, and AUC-ROC. Cross-validation is employed to ensure robustness and generalizability.

### Training the Models

To train the machine learning models locally:

#### Phishing model code:
First ensure the three following datasets are present in the directory 'CryptoAnalysis\Application Code\practice\Project - Testing\datasets\ethereum\phishing\':
1. 'combined_er20.csv'
2. 'combined_transactions.csv'
3. 'transaction_dataset_even.csv'

in the directory: 'CryptoAnalysis\Application Code\practice\Project - Testing\Phishing-ML\' there are three python scripts:
'LR-machine learning er20.py'
'LR-machine learning transactions.py'
'LR-machine learning wallets v3.py'

running these will each create a trained model and the related features PKL file in the 'CryptoAnalysis\Application Code\practice\Project - Testing\results\ directory.

#### Money Laundering model code:
First ensure the three following datasets are present in the directory 'CryptoAnalysis\Application Code\practice\Project - Testing\datasets\ethereum\MoneyLaundering\':
1. 'combined_er20_transaction_data.csv'
2. 'combined_normal_transactions_data.csv'
3. 'combined_transaction_data.csv'

In the directory: 'CryptoAnalysis\Application Code\practice\Project - Testing\moneyLaundering\' there are three python scripts:
'LR-machine learning er20.py'
'LR-machine learning transactions.py'
'LR-machine learning wallets.py'

running these will each create a trained model and the related features PKL file in the 'CryptoAnalysis\Application Code\practice\Project - Testing\results\ directory.

## Web Application

### Overview

The web application is developed using Django and serves as the user interface for interacting with the trained models and analyzing Ethereum blockchain data. It allows users to upload transaction data, run model analyses, and view detailed visualizations of the results.

### Key Features

- **Data Upload and Analysis**: Users can download Ethereum transaction data from etherscan and run analyses using trained machine learning models.
- **Visualizations and Reporting**: The app provides graphical representations of transaction patterns and allows for exporting analysis results.
- **User Interface web app**: Facilitates easy navigation, model execution, and result interpretation, supporting digital forensic investigations.

### Running the Web Application

Required:
6 models trained in the previous section saved into the respetive directories:
"CryptoAnalysis\Application Code\src\Host Applications\Crypto.Tracker.Web.UI\models\ethereum\MoneyLaundering" have the following models:
1. 'random_forest_moneyLaundering_detector_aggregated.pkl'
2. 'random_forest_moneyLaundering_detector_aggregated_features.pkl'
3. 'random_forest_moneyLaundering_detector_er20.pkl'
4. 'random_forest_moneyLaundering_detector_er20_features.pkl'
5. 'random_forest_moneyLaundering_detector_transactions.pkl'
6. 'random_forest_moneyLaundering_detector_transactions_features.pkl'
and
"CryptoAnalysis\Application Code\src\Host Applications\Crypto.Tracker.Web.UI\models\ethereum\phishing" have the following files:
1. 'random_forest_phishing_detector_aggregated.pkl'
2. 'random_forest_phishing_detector_aggregated_features.pkl'
3. 'random_forest_phishing_detector_er20.pkl'
4. 'random_forest_phishing_detector_er20_features.pkl'
5. 'random_forest_phishing_detector_transactions.pkl'
6. 'random_forest_phishing_detector_transactions_features.pkl'

## Web Application Configuration

To properly set up the Django web application for the Crypto Analysis Tool, users must adjust several settings in the `settings.py` file and configure their environment accordingly.

### Required Configuration Steps

1. **Secret Key**
   - **Location in `settings.py`**:
     ```python
     SECRET_KEY = 'b1be7575-abe1-4fe1-9c95-dce1e276f171'
     ```
   - **What to do**: 
     Replace the default `SECRET_KEY` with a unique, randomly generated string. This key is crucial for security purposes and should be kept secret. You can generate a new key using online tools or by running Django’s `get_random_secret_key()` function:
     ```bash
     python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())'
     ```

2. **Debug Mode**
   - **Location in `settings.py`**:
     ```python
     DEBUG = True
     ```
   - **What to do**: 
     Set `DEBUG = False` for production environments to prevent sensitive data from being exposed. In a development environment, keep it set to `True` to enable detailed error messages.

3. **Allowed Hosts**
   - **Location in `settings.py`**:
     ```python
     ALLOWED_HOSTS = []
     ```
   - **What to do**: 
     Add the domain names or IP addresses of your deployment environment to the `ALLOWED_HOSTS` list to allow Django to serve your app on those hosts. For example:
     ```python
     ALLOWED_HOSTS = ['localhost', '127.0.0.1', 'yourdomain.com']
     ```

4. **Database Configuration**
   - **Location in `settings.py`**:
     ```python
     DATABASES = {
         'default': {
             'ENGINE': 'django.db.backends.postgresql',
             'NAME': 'CryptoTracker',
             'USER': 'crypto-postgres',
             'PASSWORD': 'X1B2#WXYZ123a',
             'HOST': 'localhost',
             'PORT': '5432',
             'OPTIONS': {
                 'options': '-c search_path=dev_crypto_tracker,public'
             }
         }
     }
     ```
   - **What to do**:
     Ensure that PostgreSQL is installed and configured correctly on your system. You may need to modify the following database settings to match your environment:
     - `NAME`: Your PostgreSQL database name.
     - `USER`: Your PostgreSQL username.
     - `PASSWORD`: Your PostgreSQL password.
     - `HOST`: The address of your PostgreSQL server (use `localhost` for local development).
     - `PORT`: The port number for your PostgreSQL server (default is `5432`).

5. **Static and Media Files**
   - **Location in `settings.py`**:
     ```python
     STATIC_URL = '/static/'
     STATIC_ROOT = posixpath.join(*(BASE_DIR.split(os.path.sep) + ['static']))

     MEDIA_URL = '/media/'
     MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
     ```
   - **What to do**:
     - Ensure `STATIC_ROOT` and `MEDIA_ROOT` directories are correctly set for your deployment environment. These should exist and be writable by the web server.
     - In production, run `python manage.py collectstatic` to gather all static files into the `STATIC_ROOT` directory.

6. **Environment Variables**
   - **ETHERSCAN API Key**:
     The application requires an Etherscan API key to access blockchain data. This key should be stored as an environment variable.
   - **Location in `settings.py`**:
     ```python
     ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "your-default-api-key")
     ```
   - **What to do**:
     Set the `ETHERSCAN_API_KEY` environment variable in your shell or environment manager:
     - **For PowerShell**:
       ```powershell
       $env:ETHERSCAN_API_KEY = "<Your_API_Key>"
       ```
     - **For Linux/macOS Bash**:
       ```bash
       export ETHERSCAN_API_KEY="<Your_API_Key>"
       ```
     - **For Windows Command Prompt**:
       ```cmd
       set ETHERSCAN_API_KEY=<Your_API_Key>
       ```
     - Alternatively, add this to a `.env` file in your project’s root directory.

7. **Middleware and Installed Apps**
   - **Middleware**: The default middleware is set up for standard security and session management. Customize further as needed for your environment.
   - **Installed Apps**: Ensure all necessary apps, including Django default apps and `webview`, are listed under `INSTALLED_APPS` in `settings.py`.

8. **Timezone and Localization**
   - **Location in `settings.py`**:
     ```python
     LANGUAGE_CODE = 'en-us'
     TIME_ZONE = 'UTC'
     ```
   - **What to do**: Adjust `LANGUAGE_CODE` and `TIME_ZONE` according to your localization needs.

After completing these configurations, your Django application should be properly set up to run in your local or production environment.

The web application will be accessible at `http://localhost:8000`.

### Creating an Admin User

To manage your Django application through the admin interface, you need to create an admin user. Follow these steps using Visual Studio:

1. **Open the Django Project in Visual Studio**:
   - Ensure you have your Django project open in Visual Studio.

2. **Right-click on the Django Project** in the Solution Explorer pane.

3. **Navigate to Python > Django**:
   - Hover over **Python** to expand the menu.
   - Then hover over **Django** to see more options.

4. **Select 'Create Superuser'**:
   - Click on **Create Superuser** from the dropdown menu.

5. **Follow the Prompts in the Output Window**:
   - Visual Studio will prompt you in the Output window to enter the details for your superuser.
   - Provide a username, email, and password when prompted.

6. **Access the Django Admin Interface**:
   - Open your web browser and go to `http://localhost:8000/admin`.
   - Log in using the admin credentials you just created.



## Univiersity of Buckingham

This repository was developed as part of Ruaidhri Brown's MSc in Innovative Computing at the University of Buckingham. The project addresses critical challenges in digital forensics and cybersecurity by providing tools to detect and analyze illicit activities on the Ethereum blockchain, thereby aiding law enforcement and research in blockchain security.

