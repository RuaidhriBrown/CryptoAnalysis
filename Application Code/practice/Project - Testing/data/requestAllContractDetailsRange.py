import requests
import pandas as pd
import time

def get_contract_details(address, api_key):
    url = "https://api.etherscan.io/api"
    params = {
        "module": "contract",
        "action": "getsourcecode",
        "address": address,
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
    
    all_contract_details = []
    processed_addresses = set()
    total_contracts = 0

    for index, row in df_subset.iterrows():
        address = row['Address']
        flag = row['FLAG']
        
        print(f"{index + start}: Processing address: {address} with FLAG: {flag}")
        
        contract_details = get_contract_details(address, api_key)
        
        if contract_details and contract_details['status'] == '1':
            for contract in contract_details['result']:
                contract['Address'] = address
                contract['FLAG'] = flag
            all_contract_details.extend(contract_details['result'])
            processed_addresses.add(address)
            total_contracts += len(contract_details['result'])
            print(f"Contract details for {address} have been collected.")
        else:
            print(f"Error fetching contract details for {address} or no contract details found. Response: {contract_details}")
        
        # Wait to respect the rate limit of 5 requests per second
        time.sleep(0.2)
    
    if all_contract_details:
        contract_df = pd.DataFrame(all_contract_details)
        contract_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"All contract details have been saved to {output_file}")
    else:
        print("No contract details found for any address.")

    print(f"\nStatistics:")
    print(f"Total distinct accounts processed: {len(processed_addresses)}")
    print(f"Total contract details fetched: {total_contracts}")
    print(f"Data saved in: {output_file}")

api_key = "RNT7YF8V9S21MYZCWFF2ICW235EAME6XT8"
file_name = "transaction_dataset_even.csv"

# Example usage: process accounts from index 0 to 100
start_index = 0
count = 20
end_index = start_index + count

illicit = 1

output_file = f"contract_details_{start_index}_{end_index-1}.csv"
process_accounts(illicit, file_name, api_key, output_file, start_index, end_index)


