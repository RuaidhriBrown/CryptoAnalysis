import requests
import pandas as pd
import time
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import random

# Replace with your own Etherscan API Key
API_KEY = 'RNT7YF8V9S21MYZCWFF2ICW235EAME6XT8'
BASE_URL = 'https://api.etherscan.io/api'
MAX_API_CALLS_PER_SECOND = 5  # Max 5 API calls per second
RETRY_COUNT = 3  # Number of retries for failed requests
START_TIME = '2019-11-27 04:06:40'  # Start time for filtering transactions
MAX_WALLETS_PER_BRANCH = 5  # Maximum number of wallets to check per branch
MAX_DEPTH = 3  # Maximum depth of the analysis

# Helper function to make API requests with retry logic
def make_request(url):
    for attempt in range(RETRY_COUNT):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: Received status code {response.status_code} for URL: {url}")
        except requests.exceptions.RequestException as e:
            print(f"RequestException: {e}")
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
        time.sleep(2 ** attempt)  # Exponential backoff
    return None

# Function to get all outgoing transactions for a wallet
def get_outgoing_transactions(address):
    url = f'{BASE_URL}?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={API_KEY}'
    response = make_request(url)
    time.sleep(1 / MAX_API_CALLS_PER_SECOND)  # Rate limiting
    if response and response.get('status') == '1':
        df = pd.DataFrame(response['result'])
        # Explicitly cast to numeric before converting to datetime
        df['timeStamp'] = pd.to_numeric(df['timeStamp'], errors='coerce')
        df = df[(df['from'].str.lower() == address.lower()) &
                (pd.to_datetime(df['timeStamp'], unit='s') >= pd.to_datetime(START_TIME))]
        return df
    else:
        return pd.DataFrame()

# Function to recursively collect transactions for multiple layers
def collect_transactions(address, depth=0, graph=None, visited=None):
    if depth > MAX_DEPTH:
        return
    
    if visited is None:
        visited = set()
    if graph is None:
        graph = nx.DiGraph()

    if address in visited:
        return
    
    visited.add(address)

    # Get outgoing transactions
    transactions = get_outgoing_transactions(address)
    if transactions.empty:
        return

    # Limit to MAX_WALLETS_PER_BRANCH if there are too many transactions
    if len(transactions) > MAX_WALLETS_PER_BRANCH:
        transactions = transactions.sort_values(by='timeStamp').head(MAX_WALLETS_PER_BRANCH)
    
    for index, tx in transactions.iterrows():
        to_address = tx['to'].lower()
        graph.add_edge(address, to_address, value=float(tx['value']) / 10**18)

        # Recursively collect transactions for the next wallet
        collect_transactions(to_address, depth + 1, graph, visited)

    return graph

# Function to visualize the transactions
def visualize_transactions(graph):
    plt.figure(figsize=(12, 8))
    
    pos = nx.spring_layout(graph, k=0.5, iterations=20)  # Adjust 'k' for more or less spread
    pos = {k: (v[0], -v[1]) for k, v in pos.items()}  # Adjust layout to left-to-right

    edge_labels = nx.get_edge_attributes(graph, 'value')
    
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.title("Transaction Flow Graph (Left to Right)")
    plt.show()

# Example usage
upbit_hack_address = '0xa09871AEadF4994Ca12f5c0b6056BBd1d343c029'
graph = collect_transactions(upbit_hack_address)
visualize_transactions(graph)
