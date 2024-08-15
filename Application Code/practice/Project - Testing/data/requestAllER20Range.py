import requests
import csv
import time
import pandas as pd

def get_erc20_transactions(address, api_key):
    url = "https://api.etherscan.io/api"
    params = {
        "module": "account",
        "action": "tokentx",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "asc",
        "apikey": api_key
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

def process_accounts(illicit, file_name, api_key, output_file, start, end):
    df = pd.read_csv(file_name)
    
    # Filter accounts with FLAG = illicit
    df_flag = df[df['FLAG'] == illicit]
    
    # Get the subset of accounts based on start and end
    df_subset = df_flag.iloc[start:end]
    
    all_erc20_transactions = []
    processed_addresses = set()
    total_erc20_transactions = 0

    for index, row in df_subset.iterrows():
        address = row['Address']
        flag = row['FLAG']
        
        print(f"{index}: Processing address: {address} with FLAG: {flag}")
        
        # Get ERC20 token transactions
        erc20_transactions = get_erc20_transactions(address, api_key)
        
        if erc20_transactions and erc20_transactions['status'] == '1':
            for tx in erc20_transactions['result']:
                tx['Address'] = address
                tx['FLAG'] = flag
            all_erc20_transactions.extend(erc20_transactions['result'])
            processed_addresses.add(address)
            total_erc20_transactions += len(erc20_transactions['result'])
            print(f"ERC20 transactions for {address} have been collected.")
        else:
            print(f"Error fetching ERC20 transactions for {address} or no transactions found.")
        
        # Wait to respect the rate limit of 5 requests per second
        time.sleep(0.2)
    
    if all_erc20_transactions:
        keys = all_erc20_transactions[0].keys()
        with open(output_file, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_erc20_transactions)
        print(f"All ERC20 transactions have been saved to {output_file}")
    
    print(f"\nStatistics:")
    print(f"Total distinct accounts processed: {len(processed_addresses)}")
    print(f"Total ERC20 transactions fetched: {total_erc20_transactions}")
    print(f"ERC20 data saved in: {output_file}")

api_key = "RNT7YF8V9S21MYZCWFF2ICW235EAME6XT8"
file_name = "transaction_dataset_even.csv"

# Example usage: process accounts from index 0 to 100
start_index = 1000
count = 1180
end_index = start_index + count

illicit = 1

output_file = f"data/er20_collection/Flag_{illicit}_erc20_transactions_{start_index}_{end_index-1}.csv"
process_accounts(illicit, file_name, api_key, output_file, start_index, end_index)
