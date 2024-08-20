import pandas as pd

# Load the datasets
transactions_df = pd.read_csv('datasets/ethereum/MoneyLaundering/all_normal_transactions.csv')
transactions_0_df = pd.read_csv('datasets/ethereum/combined_transactions.csv')

# Remove duplicates from both datasets
transactions_df.drop_duplicates(inplace=True)
transactions_0_df.drop_duplicates(inplace=True)

# Assign FLAG = 1 to transactions_df
transactions_df['FLAG'] = 1

# Assign FLAG = 0 to transactions_0_df
transactions_0_df['FLAG'] = 0

# Combine the two datasets
combined_transactions_df = pd.concat([transactions_df, transactions_0_df], ignore_index=True)

# Save the combined data to a CSV file
combined_transactions_df.to_csv('datasets/ethereum/MoneyLaundering/combined_normal_transactions_data.csv', index=False)
