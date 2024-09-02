import requests
import re
from datetime import datetime
from django.db.models import Max, Q
from django.utils import timezone
from django.conf import settings
from tqdm import tqdm
from .models import EthereumTransaction, ERC20Transaction
from .utils import convert_to_int
import time

def get_transactions(address, startblock, limit=None):
    # Check if the address is in the correct format
    if not re.match(r'^0x[a-fA-F0-9]{40}$', address):
        print(f"Invalid address format: {address}")
        return {'error': 'Invalid address format'}

    url = "https://api.etherscan.io/api"
    api_key = settings.ETHERSCAN_API_KEY

    params = {
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": startblock,
        "endblock": 99999999,
        "sort": "asc",
        "apikey": api_key
    }

    if limit is not None:
        params["offset"] = limit  # Limit number of transactions

    print(f"Fetching transactions for address: {address}, with params: {params}")

    response = requests.get(url, params=params)

    print(f"Received response status: {response.status_code}, response body: {response.text}")

    if response.status_code == 200:
        return response.json()
    return {'error': 'Failed to fetch transactions'}

def get_erc20_transactions(address, startblock, limit=None):
    url = "https://api.etherscan.io/api"
    api_key = settings.ETHERSCAN_API_KEY

    params = {
        "module": "account",
        "action": "tokentx",
        "address": address,
        "startblock": startblock,
        "endblock": 99999999,
        "sort": "asc",
        "apikey": api_key
    }

    if limit is not None:
        params["offset"] = limit  # Limit number of transactions

    print(f"Fetching ERC20 transactions for address: {address}, with params: {params}")

    response = requests.get(url, params=params)

    print(f"Received response status: {response.status_code}, response body: {response.text}")

    if response.status_code == 200:
        return response.json()
    return {'error': 'Failed to fetch ERC20 transactions'}

def get_balance(address):
    """Fetch the current balance of the Ethereum address."""
    # Ensure the address is a string
    if not isinstance(address, str):
        print(f"Invalid address type: {type(address)}, expected a string")
        return {'error': 'Invalid address type, expected a string'}

    # Validate Ethereum address format
    if not re.match(r'^0x[a-fA-F0-9]{40}$', address):
        print(f"Invalid address format: {address}")
        return {'error': 'Invalid address format'}

    url = "https://api.etherscan.io/api"
    api_key = settings.ETHERSCAN_API_KEY

    params = {
        "module": "account",
        "action": "balance",
        "address": address,
        "tag": "latest",  # Get the most recent balance
        "apikey": api_key
    }

    print(f"Fetching balance for address: {address}, with params: {params}")

    response = requests.get(url, params=params)

    print(f"Received response status: {response.status_code}, response body: {response.text}")

    if response.status_code == 200:
        result = response.json()
        if result['status'] == '1':
            balance_in_wei = int(result['result'])
            balance_in_ether = balance_in_wei / 10**18
            print(f"Balance in Ether: {balance_in_ether}")
            return {'balance': balance_in_ether}
        else:
            print(f"Error fetching balance: {result.get('message', 'No message')}")
            return {'error': 'Failed to fetch balance', 'message': result.get('message', 'No message')}
    return {'error': 'Failed to fetch balance', 'status_code': response.status_code}

def update_transactions(address, eth_transactions, erc20_transactions):
    eth_transactions_to_create = []
    erc20_transactions_to_create = []

    for tx in eth_transactions:
        eth_transactions_to_create.append(
            EthereumTransaction(
                hash=tx['hash'],
                block_number=convert_to_int(tx['blockNumber'], 0),
                timestamp=timezone.make_aware(datetime.fromtimestamp(convert_to_int(tx['timeStamp'], 0))),
                nonce=convert_to_int(tx['nonce'], 0),
                block_hash=tx['blockHash'] or '',
                transaction_index=convert_to_int(tx['transactionIndex'], 0),
                from_address=tx['from'] or '',
                to_address=tx['to'] or '',
                value=convert_to_int(tx.get('value', 0), 0),
                gas=convert_to_int(tx['gas'], 0),
                gas_price=convert_to_int(tx['gasPrice'], 0),
                is_error=convert_to_int(tx.get('isError', 0), 0),
                txreceipt_status=convert_to_int(tx.get('txreceipt_status'), None),
                input=tx['input'] or '',
                contract_address=tx.get('contractAddress', None),
                cumulative_gas_used=convert_to_int(tx['cumulativeGasUsed'], 0),
                gas_used=convert_to_int(tx['gasUsed'], 0),
                confirmations=convert_to_int(tx['confirmations'], 0),
                method_id=tx.get('methodId', ''),
                function_name=tx.get('functionName', ''),
                address=address,
                last_updated=timezone.now(),
            )
        )

    for tx in erc20_transactions:
        erc20_transactions_to_create.append(
            ERC20Transaction(
                composite_id=f"{tx['hash']}_{tx['from']}_{tx['to']}",
                block_number=convert_to_int(tx['blockNumber'], 0),
                timestamp=timezone.make_aware(datetime.fromtimestamp(convert_to_int(tx['timeStamp'], 0))),
                hash=tx['hash'],
                nonce=convert_to_int(tx['nonce'], 0),
                block_hash=tx['blockHash'] or '',
                from_address=tx['from'] or '',
                contract_address=tx.get('contractAddress', None),
                to_address=tx['to'] or '',
                value=convert_to_int(tx.get('value', 0), 0),
                token_name=tx['tokenName'] or '',
                token_decimal=convert_to_int(tx['tokenDecimal'], 0),
                transaction_index=convert_to_int(tx['transactionIndex'], 0),
                gas=convert_to_int(tx['gas'], 0),
                gas_price=convert_to_int(tx['gasPrice'], 0),
                gas_used=convert_to_int(tx['gasUsed'], 0),
                cumulative_gas_used=convert_to_int(tx['cumulativeGasUsed'], 0),
                input=tx['input'] or '',
                confirmations=convert_to_int(tx['confirmations'], 0),
                address=address,
                last_updated=timezone.now(),
            )
        )

    EthereumTransaction.objects.bulk_create(eth_transactions_to_create, ignore_conflicts=True)
    ERC20Transaction.objects.bulk_create(erc20_transactions_to_create, ignore_conflicts=True)

def fetch_linked_wallets(address):
    unique_addresses = set(
        EthereumTransaction.objects.filter(
            Q(from_address=address) | Q(to_address=address)
        ).values_list('from_address', 'to_address').distinct()
    )
    unique_addresses = {addr for pair in unique_addresses for addr in pair}
    unique_addresses.discard(address)

    total_calls = len(unique_addresses) * 2
    expected_time = total_calls / 5

    start_time = time.time()
    eth_transactions_to_create = []
    erc20_transactions_to_create = []

    for addr in tqdm(unique_addresses, desc="Processing addresses"):
        highest_eth_block = EthereumTransaction.objects.filter(Q(from_address=addr) | Q(to_address=addr)).aggregate(Max('block_number'))['block_number__max'] or 0
        highest_erc20_block = ERC20Transaction.objects.filter(Q(from_address=addr) | Q(to_address=addr)).aggregate(Max('block_number'))['block_number__max'] or 0

        eth_transactions_response = get_transactions(addr, highest_eth_block)
        if 'error' in eth_transactions_response:
            continue
        eth_transactions = eth_transactions_response.get('result', [])

        erc20_transactions_response = get_erc20_transactions(addr, highest_erc20_block)
        if 'error' in erc20_transactions_response:
            continue
        erc20_transactions = erc20_transactions_response.get('result', [])

        for tx in eth_transactions:
            eth_transactions_to_create.append(
                EthereumTransaction(
                    hash=tx['hash'],
                    block_number=convert_to_int(tx['blockNumber'], 0),
                    timestamp=timezone.make_aware(datetime.fromtimestamp(convert_to_int(tx['timeStamp'], 0))),
                    nonce=convert_to_int(tx['nonce'], 0),
                    block_hash=tx['blockHash'] or '',
                    transaction_index=convert_to_int(tx['transactionIndex'], 0),
                    from_address=tx['from'] or '',
                    to_address=tx['to'] or '',
                    value=convert_to_int(tx.get('value', 0), 0),
                    gas=convert_to_int(tx['gas'], 0),
                    gas_price=convert_to_int(tx['gasPrice'], 0),
                    is_error=convert_to_int(tx.get('isError', 0), 0),
                    txreceipt_status=convert_to_int(tx.get('txreceipt_status'), None),
                    input=tx['input'] or '',
                    contract_address=tx.get('contractAddress', None),
                    cumulative_gas_used=convert_to_int(tx['cumulativeGasUsed'], 0),
                    gas_used=convert_to_int(tx['gasUsed'], 0),
                    confirmations=convert_to_int(tx['confirmations'], 0),
                    method_id=tx.get('methodId', ''),
                    function_name=tx.get('functionName', ''),
                    address=addr,
                    last_updated=timezone.now(),
                )
            )

        for tx in erc20_transactions:
            erc20_transactions_to_create.append(
                ERC20Transaction(
                    composite_id=f"{tx['hash']}_{tx['from']}_{tx['to']}",
                    block_number=convert_to_int(tx['blockNumber'], 0),
                    timestamp=timezone.make_aware(datetime.fromtimestamp(convert_to_int(tx['timeStamp'], 0))),
                    hash=tx['hash'],
                    nonce=convert_to_int(tx['nonce'], 0),
                    block_hash=tx['blockHash'] or '',
                    from_address=tx['from'] or '',
                    contract_address=tx.get('contractAddress', None),
                    to_address=tx['to'] or '',
                    value=convert_to_int(tx.get('value', 0), 0),
                    token_name=tx['tokenName'] or '',
                    token_decimal=convert_to_int(tx['tokenDecimal'], 0),
                    transaction_index=convert_to_int(tx['transactionIndex'], 0),
                    gas=convert_to_int(tx['gas'], 0),
                    gas_price=convert_to_int(tx['gasPrice'], 0),
                    gas_used=convert_to_int(tx['gasUsed'], 0),
                    cumulative_gas_used=convert_to_int(tx['cumulativeGasUsed'], 0),
                    input=tx['input'] or '',
                    confirmations=convert_to_int(tx['confirmations'], 0),
                    address=addr,
                    last_updated=timezone.now(),
                )
            )

        time.sleep(2 / 5)

    EthereumTransaction.objects.bulk_create(eth_transactions_to_create, ignore_conflicts=True)
    ERC20Transaction.objects.bulk_create(erc20_transactions_to_create, ignore_conflicts=True)

    elapsed_time = time.time() - start_time
    return elapsed_time, expected_time
