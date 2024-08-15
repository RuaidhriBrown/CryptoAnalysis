import pandas as pd

import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to analyze token flow
def token_flow_analysis(df, token_address_column='token_address', value_column='value', from_address_column='from_address', to_address_column='to_address'):
    # Filter transactions for a specific token if necessary
    # df = df[df[token_address_column] == 'specific_token_address']
    
    # Aggregate data to see the total token flow from each address
    token_outflow = df.groupby(from_address_column)[value_column].sum().reset_index(name='Total Token Outflow')
    token_inflow = df.groupby(to_address_column)[value_column].sum().reset_index(name='Total Token Inflow')
    
    # Merge the inflow and outflow data
    token_flow = pd.merge(token_outflow, token_inflow, left_on=from_address_column, right_on=to_address_column, how='outer').fillna(0)
    
    # Calculate net flow for each address
    token_flow['Net Token Flow'] = token_flow['Total Token Inflow'] - token_flow['Total Token Outflow']
    
    return token_flow

# Results directory creation with date and time
results_dir = 'results'
date_time_dir = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
path = os.path.join(results_dir, date_time_dir)
if not os.path.exists(path):
    os.makedirs(path)

# Load the data
file_path = 'token_transactions.csv'  # Adjust to your file path
df = load_data(file_path)

# Perform token flow analysis
token_flow_df = token_flow_analysis(df)

# Save the token flow analysis results to CSV
token_flow_df.to_csv(os.path.join(path, 'token_flow_analysis.csv'), index=False)

print(f"Token flow analysis results saved in the directory: {path}")
