import logging
import pandas as pd

logger = logging.getLogger(__name__)

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required
from django.db.models import Q, Min, Max, Count, Prefetch
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from .models import CompletedAnalysis, EthereumTransaction, ERC20Transaction, Wallet, WalletAnalysis, UserProfile, WalletNote
from .utils import combine_data, generate_wallet_stats_graphs
from .transaction_utils import get_balance, get_transactions, get_erc20_transactions, update_transactions, fetch_linked_wallets
import time
import json
from datetime import datetime

from .models import Wallet, WalletAnalysis
from .analysis import run_wallet_phishing_model_analysis, run_transaction_phishing_model_analysis, run_erc20_phishing_model_analysis, analyze_wallet_for_Money_Laundering, run_wallet_moneyLaundering_model_analysis, run_transaction_moneyLaundering_model_analysis, run_erc20_moneyLaundering_model_analysis

def index(request):
    return render(request, 'index.html')

def cfdg(request):
    return render(request, 'cfdg-view.html')

@login_required
def profile(request):
    return render(request, 'profile.html')

@login_required
def organization(request):
    return render(request, 'Todo.html')

@login_required
def documentation(request):
    return render(request, 'Todo.html')

@login_required
def statistics(request):
    return render(request, 'Todo.html')

@login_required
def transaction_list(request):
    transaction_list = EthereumTransaction.objects.all().order_by('-timestamp')
    paginator = Paginator(transaction_list, 50)  # Show 50 transactions per page

    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'transaction_list.html', {'page_obj': page_obj})

@login_required
def transactions_by_address(request, address):
    transactions = EthereumTransaction.objects.filter(Q(from_address=address) | Q(to_address=address)).order_by('-timestamp')
    paginator = Paginator(transactions, 50)  # Show 50 transactions per page

    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'transactions_by_address.html', {'page_obj': page_obj, 'address': address})

@login_required
def wallet_list(request):
    search_query = request.GET.get('search', '')
    order_by = request.GET.get('order_by', 'address')
    order_direction = request.GET.get('order_direction', 'asc')

    # Retrieve all wallets and prefetch related analyses
    wallets = Wallet.objects.prefetch_related(
        Prefetch(
            'analyses',
            queryset=WalletAnalysis.objects.select_related('user_profile')
        )
    ).all()

    # Filter wallets based on search query
    if search_query:
        wallets = wallets.filter(address__icontains=search_query)

    # Apply ordering
    if order_by in ['address', 'total_sent_transactions', 'total_received_transactions',
                    'unique_sent_addresses', 'unique_received_addresses', 'total_ether_sent',
                    'total_ether_received', 'total_erc20_sent', 'total_erc20_received', 'last_updated']:
        order_by = order_by if order_direction == 'asc' else '-' + order_by
        wallets = wallets.order_by(order_by)

    paginator = Paginator(wallets, 50)  # Show 50 wallets per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'request': request,
    }
    return render(request, 'ether_wallet_list/wallet_list.html', context)


@login_required
def update_wallets(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        filter_date = data.get('filter_date')
        update_all = data.get('update_all')

        if update_all:
            wallets = Wallet.objects.all()
        elif filter_date:
            filter_date = datetime.strptime(filter_date, '%Y-%m-%d')
            wallets = Wallet.objects.filter(last_updated__lt=filter_date)
        else:
            wallets = Wallet.objects.none()

        start_time = time.time()

        for wallet in wallets:
            transactions = EthereumTransaction.objects.filter(Q(from_address=wallet.address) | Q(to_address=wallet.address))
            erc20_transactions = ERC20Transaction.objects.filter(Q(from_address=wallet.address) | Q(to_address=wallet.address))
            
            if transactions.exists() or erc20_transactions.exists():
                combined_info = combine_data(wallet.address, transactions, erc20_transactions)
                
                wallet.total_sent_transactions = combined_info['sent_tnx']
                wallet.total_received_transactions = combined_info['received_tnx']
                wallet.unique_sent_addresses = combined_info['unique_sent_to_addresses']
                wallet.unique_received_addresses = combined_info['unique_received_from_addresses']
                wallet.total_ether_sent = combined_info['total_ether_sent']
                wallet.total_ether_received = combined_info['total_ether_received']
                wallet.total_erc20_sent = combined_info['total_erc20_sent']
                wallet.total_erc20_received = combined_info['total_erc20_received']
                wallet.last_analyzed = timezone.now()
                wallet.save()
            else:
                wallet.notes = "No data kept"
                wallet.save()

        elapsed_time = time.time() - start_time
        return JsonResponse({'status': 'completed', 'elapsed_time': elapsed_time})

    return JsonResponse({'status': 'invalid request'}, status=400)

@login_required
def run_analysis(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        analysis_type = data.get('analysis_type')
        # Implement your analysis logic based on the analysis_type
        # This is just an example of responding
        return JsonResponse({'status': 'completed', 'analysis_type': analysis_type})
    return JsonResponse({'status': 'failed'}, status=400)

@login_required
def wallet_details(request, address):
    logger.info(f"Fetching details for wallet: {address}")
    
    # Fetch or create wallet
    try:
        wallet = Wallet.objects.get(address=address)
        logger.info(f"Wallet found in the database: {wallet}")
    except Wallet.DoesNotExist:
        logger.warning(f"Wallet not found. Creating a new wallet for address: {address}")
        # Fetch balance for the new wallet
        balance_response = get_balance(address)
        logger.debug(f"Balance response: {balance_response}")

        balance = balance_response['balance'] if 'balance' in balance_response else 0

        # Create a new wallet entry
        wallet = Wallet.objects.create(
            address=address,
            balance=balance
        )
        logger.info(f"New wallet created with balance: {balance}")

    # Fetch transactions
    last_eth_block_number = EthereumTransaction.objects.filter(
        Q(from_address=address) | Q(to_address=address)
    ).aggregate(Max('block_number'))['block_number__max'] or 0

    last_erc20_block_number = ERC20Transaction.objects.filter(
        Q(from_address=address) | Q(to_address=address)
    ).aggregate(Max('block_number'))['block_number__max'] or 0

    MAX_TRANSACTIONS_TO_ADD = 100

    # Fetch Ethereum transactions
    eth_transactions_response = get_transactions(address, startblock=last_eth_block_number + 1, limit=MAX_TRANSACTIONS_TO_ADD)
    logger.debug(f"Ethereum transactions response: {eth_transactions_response}")

    if eth_transactions_response.get('status') == '1':
        eth_transactions = eth_transactions_response.get('result', [])[:MAX_TRANSACTIONS_TO_ADD]
        save_ethereum_transactions(eth_transactions, address)
        logger.info(f"{len(eth_transactions)} Ethereum transactions saved.")

    # Fetch ERC20 transactions
    erc20_transactions_response = get_erc20_transactions(address, startblock=last_erc20_block_number + 1, limit=MAX_TRANSACTIONS_TO_ADD)
    logger.debug(f"ERC20 transactions response: {erc20_transactions_response}")

    if erc20_transactions_response.get('status') == '1':
        erc20_transactions = erc20_transactions_response.get('result', [])[:MAX_TRANSACTIONS_TO_ADD]
        save_erc20_transactions(erc20_transactions, address)
        logger.info(f"{len(erc20_transactions)} ERC20 transactions saved.")

    # Retrieve transactions related to the wallet
    transactions = EthereumTransaction.objects.filter(Q(from_address=address) | Q(to_address=address)).order_by('-timestamp')
    erc20_transactions = ERC20Transaction.objects.filter(Q(from_address=address) | Q(to_address=address)).order_by('-timestamp')
    
    unique_addresses = set(
        EthereumTransaction.objects.filter(
            Q(from_address=address) | Q(to_address=address)
        ).values_list('from_address', 'to_address').distinct()
    )
    unique_addresses = {addr for pair in unique_addresses for addr in pair if addr}  # Ensure non-empty addresses
    unique_addresses.discard(address)
    expected_time = (len(unique_addresses) * 2) * (2 / 5)

    # Ensure the WalletAnalysis for the current user and wallet exists
    wallet_analysis, created = WalletAnalysis.objects.get_or_create(
        wallet=wallet, 
        user_profile=request.user.userprofile,
        defaults={
            'owner': request.user.username,
            'tag': '',
            'believed_illicit_count': 0,
            'believed_illicit': False,
            'confirmed_illicit': False,
            'believed_crime': '',
            'updating_note': ''
        }
    )
    
    # Fetch wallet analyses, notes, and completed analyses
    wallet_analyses = WalletAnalysis.objects.filter(wallet=wallet)
    wallet_notes = WalletNote.objects.filter(analysis__wallet=wallet)
    completed_analyses = CompletedAnalysis.objects.filter(wallet_analysis__wallet=wallet)

    paginator = Paginator(transactions, 50)  # Show 50 transactions per page
    erc20_paginator = Paginator(erc20_transactions, 50)  # Show 50 ERC20 transactions per page
    
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    erc20_page_number = request.GET.get('erc20_page')
    erc20_page_obj = erc20_paginator.get_page(erc20_page_number)
    
    wallet_stats = combine_data(address, transactions, erc20_transactions)
    total_erc20_transactions = wallet_stats['total_erc20_received'] + wallet_stats['total_erc20_sent']
    
    last_ethereum_update = transactions.aggregate(last_updated=Max('last_updated'))['last_updated']
    last_erc20_update = erc20_transactions.aggregate(last_updated=Max('last_updated'))['last_updated']
    
    if last_ethereum_update and last_erc20_update:
        earliest_last_update = min(last_ethereum_update, last_erc20_update)
    elif last_ethereum_update:
        earliest_last_update = last_ethereum_update
    else:
        earliest_last_update = last_erc20_update or timezone.now()  # Fallback to current time if both are None
    
    graph_paths = generate_wallet_stats_graphs(transactions, address)

    return render(request, 'ether_wallet/wallet_details.html', {
        'address': address,
        'wallet_id': wallet.id,
        'wallet': wallet,
        'wallet_analysis': wallet_analysis,  # Include the wallet analysis for the current user
        'page_obj': page_obj,
        'erc20_page_obj': erc20_page_obj,
        'wallet_stats': wallet_stats,
        'total_erc20_transactions': total_erc20_transactions,
        'graph_paths': graph_paths,
        'estimated_timezone': graph_paths['estimated_timezone'],
        'peak_hour': graph_paths['peak_hour'],
        'last_ethereum_update': last_ethereum_update,
        'last_erc20_update': last_erc20_update,
        'earliest_last_update': earliest_last_update,
        'expected_time': expected_time,
        'linked_wallets': unique_addresses,  # Add linked wallets
        'wallet_analyses': wallet_analyses,
        'wallet_notes': wallet_notes,
        'completed_analyses': completed_analyses,
    })


def save_ethereum_transactions(transactions, address):
    """Saves Ethereum transactions to the database."""
    for tx in transactions:
        try:
            # Log the transaction being processed
            logger.info(f"Processing Ethereum transaction: {tx['hash']} for address {address}")

            # Handle empty strings or missing values for txreceipt_status
            txreceipt_status = tx.get('txreceipt_status', None)
            if txreceipt_status == '':
                txreceipt_status = None
            else:
                txreceipt_status = int(txreceipt_status)  # Ensure it's an integer

            EthereumTransaction.objects.update_or_create(
                hash=tx['hash'],
                defaults={
                    'block_number': int(tx['blockNumber']),
                    'timestamp': datetime.fromtimestamp(int(tx['timeStamp'])),  # Corrected datetime usage
                    'nonce': int(tx['nonce']),
                    'block_hash': tx['blockHash'],
                    'transaction_index': int(tx['transactionIndex']),
                    'from_address': tx['from'],
                    'to_address': tx['to'],
                    'value': int(tx['value']),
                    'gas': int(tx['gas']),
                    'gas_price': int(tx['gasPrice']),
                    'is_error': int(tx['isError']),
                    'txreceipt_status': txreceipt_status,
                    'input': tx['input'],
                    'contract_address': tx.get('contractAddress', None),
                    'cumulative_gas_used': int(tx['cumulativeGasUsed']),
                    'gas_used': int(tx['gasUsed']),
                    'confirmations': int(tx['confirmations']),
                    'method_id': tx.get('methodId', ''),
                    'function_name': tx.get('functionName', ''),
                    'address': address,
                    'last_updated': timezone.now()
                }
            )
        except Exception as e:
            # Log the error if one occurs
            logger.error(f"Error saving Ethereum transaction {tx['hash']} for address {address}: {e}")



def save_erc20_transactions(transactions, address):
    """Saves ERC20 transactions to the database."""
    
    def safe_cast_to_bigint(value, field_name):
        """Convert a value to an int and ensure it's within the bigint range."""
        try:
            value_int = int(value)
            if -9223372036854775808 <= value_int <= 9223372036854775807:
                return value_int
            else:
                raise ValueError(f"Value {value_int} for field '{field_name}' is out of range for bigint")
        except (ValueError, TypeError) as e:
            # Log the error if casting fails
            logger.error(f"Error converting value for '{field_name}' to bigint: {e}")
            return None

    for tx in transactions:
        try:
            # Log the transaction being processed
            logger.info(f"Processing ERC20 transaction: {tx['hash']} for address {address}")

            ERC20Transaction.objects.update_or_create(
                composite_id=f"{tx['hash']}_{tx['from']}_{tx['to']}",
                defaults={
                    'block_number': safe_cast_to_bigint(tx['blockNumber'], 'block_number'),
                    'timestamp': datetime.fromtimestamp(int(tx['timeStamp'])),  # Corrected datetime usage
                    'hash': tx['hash'],
                    'nonce': safe_cast_to_bigint(tx['nonce'], 'nonce'),
                    'block_hash': tx['blockHash'],
                    'from_address': tx['from'],
                    'contract_address': tx.get('contractAddress', None),
                    'to_address': tx['to'],
                    'value': safe_cast_to_bigint(tx['value'], 'value'),
                    'token_name': tx['tokenName'],
                    'token_decimal': safe_cast_to_bigint(tx['tokenDecimal'], 'token_decimal'),
                    'transaction_index': safe_cast_to_bigint(tx['transactionIndex'], 'transaction_index'),
                    'gas': safe_cast_to_bigint(tx['gas'], 'gas'),
                    'gas_price': safe_cast_to_bigint(tx['gasPrice'], 'gas_price'),
                    'gas_used': safe_cast_to_bigint(tx['gasUsed'], 'gas_used'),
                    'cumulative_gas_used': safe_cast_to_bigint(tx['cumulativeGasUsed'], 'cumulative_gas_used'),
                    'input': tx['input'],
                    'confirmations': safe_cast_to_bigint(tx['confirmations'], 'confirmations'),
                    'address': address,
                    'last_updated': timezone.now()
                }
            )
        except Exception as e:
            # Log the error if one occurs
            logger.error(f"Error saving ERC20 transaction {tx['hash']} for address {address}: {e}")


@login_required
def update_datasets(request, address):
    if request.method == 'POST':
        highest_eth_block = EthereumTransaction.objects.filter(Q(from_address=address) | Q(to_address=address)).aggregate(Max('block_number'))['block_number__max'] or 0
        highest_erc20_block = ERC20Transaction.objects.filter(Q(from_address=address) | Q(to_address=address)).aggregate(Max('block_number'))['block_number__max'] or 0

        eth_transactions_response = get_transactions(address, highest_eth_block)
        if 'error' in eth_transactions_response:
            return JsonResponse(eth_transactions_response, status=400)
        eth_transactions = eth_transactions_response.get('result', [])

        erc20_transactions_response = get_erc20_transactions(address, highest_erc20_block)
        if 'error' in erc20_transactions_response:
            return JsonResponse(erc20_transactions_response, status=400)
        erc20_transactions = erc20_transactions_response.get('result', [])

        update_transactions(address, eth_transactions, erc20_transactions)
        
        return redirect('wallet_details', address=address)

    return render(request, 'wallet_details.html', {'address': address})

@login_required
def download_linked_wallets(request, address):
    if request.method == 'POST': #
        elapsed_time, expected_time = fetch_linked_wallets(address)
        return JsonResponse({'status': 'completed', 'elapsed_time': elapsed_time})

    unique_addresses = set(
        EthereumTransaction.objects.filter(
            Q(from_address=address) | Q(to_address=address)
        ).values_list('from_address', 'to_address').distinct()
    )
    unique_addresses = {addr for pair in unique_addresses for addr in pair}
    unique_addresses.discard(address)

    total_calls = len(unique_addresses) * 2
    expected_time = total_calls / 5

    return render(request, 'ether_wallet/_analysis.html', {'expected_time': expected_time})


@login_required
def update_wallet_analysis(request, wallet_id):
    wallet = get_object_or_404(Wallet, id=wallet_id)
    wallet_analysis, created = WalletAnalysis.objects.get_or_create(wallet=wallet, user_profile=request.user.userprofile)

    if request.method == 'POST':
        wallet_analysis.owner = request.POST.get('owner')
        wallet_analysis.tag = request.POST.get('tag')
        wallet_analysis.believed_illicit_count = request.POST.get('believed_illicit_count')
        wallet_analysis.believed_illicit = request.POST.get('believed_illicit') == 'True'
        wallet_analysis.confirmed_illicit = request.POST.get('confirmed_illicit') == 'True'
        wallet_analysis.believed_crime = request.POST.get('believed_crime')
        wallet_analysis.updating_note = request.POST.get('updating_note')
        wallet_analysis.save()

        return redirect('wallet_details', address=wallet.address)

    return render(request, 'ether_wallet/wallet_details.html', {
        'wallet_analyses': wallet_analysis,
        'wallet': wallet,
    })

@login_required
def add_wallet_note(request, wallet_id):
    wallet = get_object_or_404(Wallet, id=wallet_id)
    
    # Attempt to get or create a WalletAnalysis for the current user and wallet
    wallet_analysis, created = WalletAnalysis.objects.get_or_create(
        wallet=wallet, 
        user_profile=request.user.userprofile,
        defaults={
            'owner': request.user.username,
            'tag': '',
            'believed_illicit_count': 0,
            'believed_illicit': False,
            'confirmed_illicit': False,
            'believed_crime': '',
            'updating_note': ''
        }
    )

    if request.method == 'POST':
        content = request.POST.get('content')
        if content:
            WalletNote.objects.create(
                analysis=wallet_analysis,
                user=request.user,
                content=content
            )
        return redirect('wallet_details', address=wallet.address)

    return redirect('wallet_details', address=wallet.address)

@login_required
def run_phishing_detection_W_wallets(request, wallet_id):
    wallet = get_object_or_404(Wallet, id=wallet_id)
    wallet_analysis, created = WalletAnalysis.objects.get_or_create(wallet=wallet, user_profile=request.user.userprofile)

    logger.debug(f'Running phishing analysis for wallet {wallet.address}')
    
    phishing_detected = run_wallet_phishing_model_analysis(wallet_analysis)
    
    logger.debug(f'Phishing analysis completed for wallet {wallet.address}. Detected: {phishing_detected}')

    return redirect('wallet_details', address=wallet.address)

@login_required
def run_phishing_detection_W_transactions(request, wallet_id):
    wallet = get_object_or_404(Wallet, id=wallet_id)
    wallet_analysis, created = WalletAnalysis.objects.get_or_create(wallet=wallet, user_profile=request.user.userprofile)

    logger.debug(f'Running transaction phishing analysis for wallet {wallet.address}')
    
    phishing_detected = run_transaction_phishing_model_analysis(wallet_analysis)
    
    logger.debug(f'Transaction phishing analysis completed for wallet {wallet.address}. Detected: {phishing_detected}')

    return redirect('wallet_details', address=wallet.address)


@login_required
def run_phishing_detection_W_ER20(request, wallet_id):
    wallet = get_object_or_404(Wallet, id=wallet_id)
    wallet_analysis, created = WalletAnalysis.objects.get_or_create(wallet=wallet, user_profile=request.user.userprofile)

    logger.debug(f'Running ERC20 phishing analysis for wallet {wallet.address}')
    
    phishing_detected = run_erc20_phishing_model_analysis(wallet_analysis)
    
    logger.debug(f'ERC20 phishing analysis completed for wallet {wallet.address}. Detected: {phishing_detected}')

    return redirect('wallet_details', address=wallet.address)


@login_required
def run_Money_Laundering_detection(request, wallet_id):
    wallet = get_object_or_404(Wallet, id=wallet_id)
    wallet_analysis, created = WalletAnalysis.objects.get_or_create(wallet=wallet, user_profile=request.user.userprofile)

    logger.debug(f'Running money laundering detection for wallet {wallet.address}')
    
    money_laundering_detected = analyze_wallet_for_Money_Laundering(wallet_analysis)
    
    logger.debug(f'Money laundering detection completed for wallet {wallet.address}. Detected: {money_laundering_detected}')

    return redirect('wallet_details', address=wallet.address)

@login_required
def run_moneyLaundering_detection_W_wallets(request, wallet_id):
    wallet = get_object_or_404(Wallet, id=wallet_id)
    wallet_analysis, created = WalletAnalysis.objects.get_or_create(wallet=wallet, user_profile=request.user.userprofile)

    logger.debug(f'Running Money laundering analysis for wallet {wallet.address}')
    
    moneyLaundering_detected = run_wallet_moneyLaundering_model_analysis(wallet_analysis)
    
    logger.debug(f'Money laundering analysis completed for wallet {wallet.address}. Detected: {moneyLaundering_detected}')

    return redirect('wallet_details', address=wallet.address)

@login_required
def run_moneyLaundering_detection_W_transactions(request, wallet_id):
    wallet = get_object_or_404(Wallet, id=wallet_id)
    wallet_analysis, created = WalletAnalysis.objects.get_or_create(wallet=wallet, user_profile=request.user.userprofile)

    logger.debug(f'Running transaction Money laundering analysis for wallet {wallet.address}')
    
    moneyLaundering_detected = run_transaction_moneyLaundering_model_analysis(wallet_analysis)
    
    logger.debug(f'Transaction Money laundering analysis completed for wallet {wallet.address}. Detected: {moneyLaundering_detected}')

    return redirect('wallet_details', address=wallet.address)


@login_required
def run_moneyLaundering_detection_W_ER20(request, wallet_id):
    wallet = get_object_or_404(Wallet, id=wallet_id)
    wallet_analysis, created = WalletAnalysis.objects.get_or_create(wallet=wallet, user_profile=request.user.userprofile)

    logger.debug(f'Running ERC20 Money laundering analysis for wallet {wallet.address}')
    
    moneyLaundering_detected = run_erc20_moneyLaundering_model_analysis(wallet_analysis)
    
    logger.debug(f'ERC20 Money laundering analysis completed for wallet {wallet.address}. Detected: {moneyLaundering_detected}')

    return redirect('wallet_details', address=wallet.address)


@login_required
def run_phishing_detection_all_wallets(request):
    if request.method == 'POST':
        wallets = Wallet.objects.all()
        for wallet in wallets:
            # Ensure a WalletAnalysis exists for each wallet
            wallet_analysis, created = WalletAnalysis.objects.get_or_create(wallet=wallet, defaults={'user_profile': request.user.userprofile})
            
            # Run phishing analysis for the wallet
            run_wallet_phishing_model_analysis(wallet_analysis)
        
        return JsonResponse({'status': 'completed'})
    return JsonResponse({'status': 'failed'})


@login_required
def transaction_analysis_results(request, wallet_id):
    wallet = get_object_or_404(Wallet, id=wallet_id)
    wallet_analysis, created = WalletAnalysis.objects.get_or_create(wallet=wallet, user_profile=request.user.userprofile)

    # Retrieve filter type from GET parameters
    filter_type = request.GET.get('filter_type', 'all')  # Default to 'all' if not specified

    # Retrieve Ethereum transactions for the wallet
    transactions = EthereumTransaction.objects.filter(Q(from_address=wallet.address) | Q(to_address=wallet.address)).order_by('-timestamp')
    erc20_transactions = ERC20Transaction.objects.filter(Q(from_address=wallet.address) | Q(to_address=wallet.address)).order_by('-timestamp')

    # Apply phishing and money laundering detection models
    transaction_phishing_indices = run_transaction_phishing_model_analysis(wallet_analysis)
    erc20_phishing_indices = run_erc20_phishing_model_analysis(wallet_analysis)
    transaction_money_laundering_indices = run_transaction_moneyLaundering_model_analysis(wallet_analysis)
    erc20_money_laundering_indices = run_erc20_moneyLaundering_model_analysis(wallet_analysis)

    # Combine results into DataFrame for easy manipulation
    transactions_df = pd.DataFrame(list(transactions.values()))
    erc20_transactions_df = pd.DataFrame(list(erc20_transactions.values()))

    # Add transaction type column
    transactions_df['transaction_type'] = 'Normal'
    erc20_transactions_df['transaction_type'] = 'ERC20'

    # Initialize detection columns to False
    transactions_df['phishing_detected'] = False
    transactions_df['money_laundering_detected'] = False
    erc20_transactions_df['phishing_detected'] = False
    erc20_transactions_df['money_laundering_detected'] = False

    # Set detected transactions to True using the indices
    transactions_df.loc[transaction_phishing_indices, 'phishing_detected'] = True
    transactions_df.loc[transaction_money_laundering_indices, 'money_laundering_detected'] = True
    erc20_transactions_df.loc[erc20_phishing_indices, 'phishing_detected'] = True
    erc20_transactions_df.loc[erc20_money_laundering_indices, 'money_laundering_detected'] = True

    # Determine the other party in the transactions
    if not transactions_df.empty:
        transactions_df['other_party'] = transactions_df.apply(
            lambda row: row['to_address'] if row['from_address'].lower() == wallet.address.lower() else row['from_address'], axis=1
        )

    if not erc20_transactions_df.empty:
        erc20_transactions_df['other_party'] = erc20_transactions_df.apply(
            lambda row: row['to_address'] if row['from_address'].lower() == wallet.address.lower() else row['from_address'], axis=1
        )

    # Filter transactions based on filter type if specified
    if filter_type == 'phishing':
        transactions_df = transactions_df[transactions_df['phishing_detected']]
        erc20_transactions_df = erc20_transactions_df[erc20_transactions_df['phishing_detected']]
    elif filter_type == 'money_laundering':
        transactions_df = transactions_df[transactions_df['money_laundering_detected']]
        erc20_transactions_df = erc20_transactions_df[erc20_transactions_df['money_laundering_detected']]

    return render(request, 'ether_wallet/wallet_transaction_analysis_results.html', {
        'wallet': wallet,
        'transactions': transactions_df.to_dict('records'),
        'erc20_transactions': erc20_transactions_df.to_dict('records'),
        'filter_type': filter_type
    })
