import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def summarize_csv(file_path):
    # Read the CSV file
    print("Reading CSV file...")
    df = pd.read_csv(file_path)

    # Initialize tqdm progress bar
    pbar = tqdm(total=12, desc="Processing")

    # Select only numeric columns for statistical analysis
    numeric_cols = df.select_dtypes(include='number').columns

    # Summary of basic statistics
    pbar.set_description("Calculating basic statistics")
    stats_summary = df.describe(include='all').transpose()
    pbar.update(1)

    # Count of missing values
    pbar.set_description("Counting missing values")
    missing_values = df.isnull().sum().to_frame(name='Missing Values').transpose()
    pbar.update(1)

    # Range of numeric columns
    pbar.set_description("Calculating ranges of numeric columns")
    ranges = (df[numeric_cols].max() - df[numeric_cols].min()).to_frame(name='Range').transpose()
    pbar.update(1)

    # Additional statistics
    pbar.set_description("Calculating additional statistics")
    std_dev = df[numeric_cols].std().to_frame(name='Std Dev').transpose()
    min_val = df[numeric_cols].min().to_frame(name='Min').transpose()
    max_val = df[numeric_cols].max().to_frame(name='Max').transpose()
    median_val = df[numeric_cols].median().to_frame(name='Median').transpose()

    pbar.update(1)

    # Aggregated information
    pbar.set_description("Aggregating information")

    numeric_aggregations = {
        'missing_values': df[numeric_cols].isnull().sum(),
        'mean': df[numeric_cols].mean(),
        'std': std_dev.transpose().iloc[:, 0],
        'min': min_val.transpose().iloc[:, 0],
        'max': max_val.transpose().iloc[:, 0],
    }
    aggregate_info = pd.DataFrame(numeric_aggregations).transpose()
    
    # Adding quantiles to aggregate information
    pbar.set_description("Adding quantiles to aggregate information")
    quantiles = df[numeric_cols].quantile([0.25, 0.5, 0.75]).transpose()
    quantiles.columns = ['25%', '50%', '75%']
    aggregate_info = pd.concat([aggregate_info, quantiles.T])
    pbar.update(1)

    # Creating a summary dictionary
    pbar.set_description("Creating summary dictionary")
    summary = {
        "Basic Statistics": stats_summary,
        "Missing Values": missing_values,
        "Ranges": ranges,
        "Aggregate Information": aggregate_info
    }
    pbar.update(1)

    pbar.close()
    return summary

def generate_heatmap_and_correlations(file_path):
    df = pd.read_csv(file_path)
    
    # Ensure that FLAG is numeric for correlation computation
    df['FLAG'] = pd.to_numeric(df['FLAG'], errors='coerce')
    
    # Select only numeric columns for correlation
    numeric_cols = df.select_dtypes(include='number')
    
    # Calculate correlation matrix
    corr_matrix = numeric_cols.corr()
    
    # Extract correlations with FLAG
    flag_corr = corr_matrix[['FLAG']].sort_values(by='FLAG', ascending=False)
    
    # Generate heatmap
    plt.figure(figsize=(10, 15))
    sns.heatmap(flag_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation of Features with FLAG')
    plt.savefig('correlation_heatmap.png')
    plt.show()
    
    return flag_corr

def save_analysis_to_csv(file_path, summary, correlations):
    with pd.ExcelWriter('analysis_summary.xlsx') as writer:
        summary["Basic Statistics"].to_excel(writer, sheet_name="Basic Statistics", index=True)
        summary["Missing Values"].to_excel(writer, sheet_name="Missing Values", index=False)
        summary["Ranges"].to_excel(writer, sheet_name="Ranges", index=False)
        summary["Aggregate Information"].to_excel(writer, sheet_name="Aggregate Information", index=True)

        # Enhanced Basic Statistics Summary
        numeric_stats_summary = summary["Basic Statistics"].select_dtypes(include='number')
        enhanced_stats_summary = numeric_stats_summary.copy().transpose()
        enhanced_stats_summary['Std Dev'] = numeric_stats_summary.std()
        enhanced_stats_summary['Median'] = numeric_stats_summary.median()
        enhanced_stats_summary['Min'] = numeric_stats_summary.min()
        enhanced_stats_summary['Max'] = numeric_stats_summary.max()
        enhanced_stats_summary['Correlation with FLAG'] = correlations['FLAG']

        enhanced_stats_summary.to_excel(writer, sheet_name="Enhanced Basic Statistics", index=True)

        correlations.to_excel(writer, sheet_name="Correlations with FLAG", index=True)

# Example usage
# file_path = 'combined_er20.csv'
# file_path = 'combined_transactions.csv'
file_path = 'transaction_dataset_even.csv'

summary = summarize_csv(file_path)
correlations = generate_heatmap_and_correlations(file_path)
save_analysis_to_csv(file_path, summary, correlations)

# Displaying the summary in console (optional)
for key, value in summary.items():
    print(f"\n{key}:\n{value}\n")
