import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file from the provided path
file_path = 'combined_transactions.csv'

# Load the data into a pandas DataFrame, excluding the 'FLAG' column
data = pd.read_csv(file_path)
data = data.drop(columns=['FLAG'])

# Convert timestamp to a datetime format
data['timeStamp'] = pd.to_datetime(data['timeStamp'], unit='s')

# Filter the dataset for transactions involving the first wallet address
first_wallet_address = '0xd24400ae8bfebb18ca49be86258a3c749cf46853'
filtered_data = data[(data['from'] == first_wallet_address) | (data['to'] == first_wallet_address)]

# Group by hour, day, and week for the filtered data
filtered_data['hour'] = filtered_data['timeStamp'].dt.hour
filtered_data['day'] = filtered_data['timeStamp'].dt.date
filtered_data['week'] = filtered_data['timeStamp'].dt.to_period('W').apply(lambda r: r.start_time)

# Count transactions per hour
filtered_hourly_counts = filtered_data.groupby('hour').size()

# Count transactions per day
filtered_daily_counts = filtered_data.groupby('day').size()

# Count transactions per week
filtered_weekly_counts = filtered_data.groupby('week').size()

# Plot the number of transactions per hour of the day
plt.figure(figsize=(12, 6))
filtered_hourly_counts.plot(kind='bar')
plt.title('Number of Transactions per Hour of the Day (First Wallet Address)')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=0)
plt.show()

# Plot the number of transactions per day
plt.figure(figsize=(12, 6))
filtered_daily_counts.plot(kind='line')
plt.title('Number of Transactions per Day (First Wallet Address)')
plt.xlabel('Date')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)
plt.show()

# Plot the number of transactions per week
plt.figure(figsize=(12, 6))
filtered_weekly_counts.plot(kind='line')
plt.title('Number of Transactions per Week (First Wallet Address)')
plt.xlabel('Week')
plt.ylabel('Number of Transactions')
plt.xticks(rotation=45)
plt.show()
