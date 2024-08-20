import pandas as pd

# Load the datasets
addressList_df = pd.read_csv('datasets/ethereum/MoneyLaundering/all_wallet_summaries.csv')
addressList_0_df = pd.read_csv('datasets/ethereum/transaction_dataset_even.csv')

# Remove duplicates from both datasets
addressList_df.drop_duplicates(inplace=True)
addressList_0_df.drop_duplicates(inplace=True)

# Take the first 416 wallets from both datasets
addressList_df_sample = addressList_df.head(416)
addressList_0_df_sample = addressList_0_df.head(416)

# Assign FLAG = 1 to the first dataset
addressList_df_sample['FLAG'] = 1

# Assign FLAG = 0 to the second dataset
addressList_0_df_sample['FLAG'] = 0

# Combine the two datasets
combined_transactions_df = pd.concat([addressList_df_sample, addressList_0_df_sample], ignore_index=True)

# Save the combined data to a CSV file
combined_transactions_df.to_csv('datasets/ethereum/MoneyLaundering/Money_Laundering_Wallets_List.csv', index=False)
