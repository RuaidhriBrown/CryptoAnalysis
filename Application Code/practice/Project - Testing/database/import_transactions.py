import csv
import psycopg2
from datetime import datetime
from tqdm import tqdm
import sys

# Set a reasonable large field size limit
csv.field_size_limit(10 * 1024 * 1024)  # 10 MB

def convert_timestamp(unix_timestamp):
    return datetime.fromtimestamp(int(unix_timestamp))

def convert_to_int(value, default=None):
    """Convert a value to an integer, handling float strings and out-of-range values."""
    try:
        int_value = int(float(value))
        if int_value < -9223372036854775808 or int_value > 9223372036854775807:
            return default  # Return default if the value is out of the range for BIGINT
        return int_value
    except (ValueError, TypeError, OverflowError):
        return default  # Return default if conversion fails

def get_existing_hashes(connection):
    cursor = connection.cursor()
    query = "SELECT hash FROM dev_crypto_tracker.ethereum_transaction"
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    return {row[0] for row in results}

def import_csv_to_db(csv_file_path, connection):
    cursor = connection.cursor()
    existing_hashes = get_existing_hashes(connection)
    
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        total_rows = sum(1 for _ in open(csv_file_path, 'r')) - 1  # Get total number of rows excluding header
        
        success_count = 0
        skipped_count = 0
        failed_count = 0
        errors = []

        for row in tqdm(reader, total=total_rows, desc="Importing transactions"):
            if row['hash'] in existing_hashes:
                skipped_count += 1
                continue
            
            # Convert the timestamp from Unix to datetime
            timestamp = convert_timestamp(row['timeStamp'])
            
            # Prepare the data and convert fields to the correct types
            data = (
                convert_to_int(row['blockNumber']),
                timestamp,
                row['hash'],
                convert_to_int(row['nonce'], default=0),  # Set default to 0 if missing
                row['blockHash'] if row['blockHash'] else '',  # Set default to empty string if missing
                convert_to_int(row['transactionIndex'], default=0),  # Set default to 0 if missing
                row['from'] if row['from'] else '',  # Set default to empty string if missing
                row['to'] if row['to'] else '',  # Set default to empty string if missing
                convert_to_int(row['value'], default=0),  # Set default to 0 if missing
                convert_to_int(row['gas'], default=0),  # Set default to 0 if missing
                convert_to_int(row['gasPrice'], default=0),  # Set default to 0 if missing
                convert_to_int(row['isError'], default=0),  # Set default to 0 if missing
                convert_to_int(row['txreceipt_status']) if row['txreceipt_status'] else None,
                row['input'] if row['input'] else '',  # Set default to empty string if missing
                row['contractAddress'] if row['contractAddress'] else None,
                convert_to_int(row['cumulativeGasUsed'], default=0),  # Set default to 0 if missing
                convert_to_int(row['gasUsed'], default=0),  # Set default to 0 if missing
                convert_to_int(row['confirmations'], default=0),  # Set default to 0 if missing
                row['methodId'] if row['methodId'] else '',  # Set default to empty string if missing
                row['functionName'] if row['functionName'] else '',  # Set default to empty string if missing
                row['Address'] if row['Address'] else ''  # Set default to empty string if missing
            )
            
            # Prepare the SQL query to insert data
            query = """
                INSERT INTO dev_crypto_tracker.ethereum_transaction (
                    block_number, timestamp, hash, nonce, block_hash, 
                    transaction_index, from_address, to_address, value, gas, 
                    gas_price, is_error, txreceipt_status, input, contract_address, 
                    cumulative_gas_used, gas_used, confirmations, method_id, 
                    function_name, address
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """
            
            try:
                cursor.execute(query, data)
                connection.commit()
                success_count += 1
            except Exception as e:
                connection.rollback()
                failed_count += 1
                errors.append((row['hash'], str(e)))
        
        tqdm.write("\nImport completed.")
        tqdm.write(f"Total Success: {success_count}")
        tqdm.write(f"Total Skipped: {skipped_count}")
        tqdm.write(f"Total Failed: {failed_count}")
        if errors:
            tqdm.write("\nErrors:")
            for error in errors:
                tqdm.write(f"Transaction {error[0]}: {error[1]}")

    cursor.close()


if __name__ == "__main__":
    # Database connection parameters
    conn_params = {
        'dbname': 'CryptoTracker',
        'user': 'crypto-postgres',
        'password': 'X1B2#WXYZ123a',
        'host': 'localhost',
        'port': '5432'
    }
    
    # Connect to the database
    conn = psycopg2.connect(**conn_params)

    
    # Path to your CSV file
    csv_file_path = 'combined_transactions.csv'
    
    # Import CSV data to the database
    import_csv_to_db(csv_file_path, conn)
    
    # Close the database connection
    conn.close()
