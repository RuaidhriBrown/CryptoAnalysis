import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Load the dataset
file_path = 'path/to/your/Flag_0_erc20_transactions_0_9.csv'
data = pd.read_csv(file_path)

# Filter transactions involving the specific address
target_address = '0x00009277775ac7d0d59eaad8fee3d10ac6c805e8'
filtered_data = data[(data['from'] == target_address) | (data['to'] == target_address)]

# Convert 'timeStamp' to datetime
filtered_data['timeStamp'] = pd.to_datetime(filtered_data['timeStamp'], unit='s')

# Convert 'value' to numeric, coercing errors to handle non-numeric values
filtered_data['value'] = pd.to_numeric(filtered_data['value'], errors='coerce')

# Sort transactions by timestamp
filtered_data = filtered_data.sort_values(by='timeStamp')

# Calculate cumulative incoming and outgoing transactions
filtered_data['cumulative_in'] = filtered_data[filtered_data['to'] == target_address]['value'].cumsum().fillna(method='ffill')
filtered_data['cumulative_out'] = filtered_data[filtered_data['from'] == target_address]['value'].cumsum().fillna(method='ffill')

# Fill missing values with previous values
filtered_data['cumulative_in'] = filtered_data['cumulative_in'].fillna(0)
filtered_data['cumulative_out'] = filtered_data['cumulative_out'].fillna(0)

# Calculate the difference between incoming and outgoing
filtered_data['balance'] = filtered_data['cumulative_in'] - filtered_data['cumulative_out']

# Identify chunks: periods where balance increases significantly followed by a significant decrease
threshold = 0.9  # This is a threshold to consider significant balance changes
chunks = []

for i in range(1, len(filtered_data)):
    if filtered_data.iloc[i]['balance'] - filtered_data.iloc[i-1]['balance'] > threshold:
        start = i
    elif filtered_data.iloc[i-1]['balance'] - filtered_data.iloc[i]['balance'] > threshold:
        end = i
        chunks.append(filtered_data.iloc[start:end+1])

# Plot the chunks of transactions
plt.figure(figsize=(12, 6))
plt.plot(filtered_data['timeStamp'], filtered_data['balance'], label='Balance')
for chunk in chunks:
    plt.axvspan(chunk.iloc[0]['timeStamp'], chunk.iloc[-1]['timeStamp'], color='red', alpha=0.3)
plt.xlabel('Time')
plt.ylabel('Balance')
plt.title('Chunks of Transactions')
plt.legend()
plt.show()

# Save chunks to CSV files
for idx, chunk in enumerate(chunks):
    chunk.to_csv(f'chunk_{idx}.csv', index=False)

print("Analysis complete. Chunks saved to CSV files.")

