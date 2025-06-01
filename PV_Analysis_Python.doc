# IMPORT NECESSARY MODULES

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates    # Import for date formatting
from matplotlib.lines import Line2D    # Import Line2D for custom legend handles


# DATA PREPROCESSING

def data_preprocess(data_directory):
    """
    Processes PR and GHI data from the specified directory structure and generates a single CSV file. 
    Each CSV file contains data for multiple days.
    """
    
    print(f"\nStarting data preprocessing from: {data_directory}")

    pr_root_dir = os.path.join(data_directory, 'PR')
    ghi_root_dir = os.path.join(data_directory, 'GHI')

    # Validate existence of PR and GHI root directories
    print(f"\nChecking PR directory: {pr_root_dir}")
    if not os.path.isdir(pr_root_dir):
        print(f"\nError: PR directory not found or is not a directory at {pr_root_dir}")
        return None
    
    print(f"\nChecking GHI directory: {ghi_root_dir}")
    if not os.path.isdir(ghi_root_dir):
        print(f"\nError: GHI directory not found or is not a directory at {ghi_root_dir}")
        return None

    list_of_dfs = []    # List to store DataFrames from each multi-day CSV file
    
    # Collect PR file paths. Map PR file paths by their starting date (e.g., '2019-07-01' -> 'path/to/2019-07-01.csv')
    pr_files_map = {} 
    total_pr_files_found = 0
    
    # Traverse PR root directory 
    for year_month in sorted(os.listdir(pr_root_dir)):
        current_pr_month_path = os.path.join(pr_root_dir, year_month)
        if os.path.isdir(current_pr_month_path):    # Ensure it's a directory like '2019-07'
            for pr_filename in sorted(os.listdir(current_pr_month_path)):
                if pr_filename.endswith('.csv'):
                    date_key = pr_filename.replace('.csv', '')    # Extract 'YYYY-MM-DD' from filename
                    try:
                        pd.to_datetime(date_key)    # Validate if it's a valid date string
                        pr_files_map[date_key] = os.path.join(current_pr_month_path, pr_filename)
                        total_pr_files_found += 1
                    except ValueError:
                        print(f"\nSkipping non-date or malformed PR file: {pr_filename} in {current_pr_month_path}")
                else: 
                    print(f"\nSkipping non-CSV PR file: {pr_filename} in {current_pr_month_path}")

    # Process GHI files and merge with corresponding PR data. This loop processes CSV files containing data for multiple days.
    files_processed_count = 0     # Counts the number of successfully processed multi-day CSV file pairs
    total_ghi_files_found = 0
    
    # Traverse GHI root directory 
    for year_month in sorted(os.listdir(ghi_root_dir)):
        current_ghi_month_path = os.path.join(ghi_root_dir, year_month)
        
        if not os.path.isdir(current_ghi_month_path):
            print(f"\nSkipping non-directory item in GHI folder: {current_ghi_month_path}")
            continue # Skip if not a directory

        # Iterate through GHI files in the current month folder
        for ghi_filename in sorted(os.listdir(current_ghi_month_path)):
            if ghi_filename.endswith('.csv'):
                date_key = ghi_filename.replace('.csv', '')    # Extract 'YYYY-MM-DD' (e.g., '2019-07-01')
                try:
                    pd.to_datetime(date_key)    # Validate date string
                    total_ghi_files_found += 1
                    
                    # Check if a corresponding PR file exists for this date_key
                    if date_key in pr_files_map:
                        pr_file_path = pr_files_map[date_key]
                        ghi_file_path = os.path.join(current_ghi_month_path, ghi_filename)

                        #print(f"\nProcessing data from CSVs starting: {date_key}")    # Log which CSV pair is being processed

                        try:
                            # Read entire CSV files for the current data block into DataFrames. Pandas automatically handles the header and infers types
                            pr_df = pd.read_csv(pr_file_path)
                            ghi_df = pd.read_csv(ghi_file_path)

                            # Convert 'Date' columns to datetime objects for accurate merging
                            pr_df['Date'] = pd.to_datetime(pr_df['Date'])
                            ghi_df['Date'] = pd.to_datetime(ghi_df['Date'])

                            # Merge the PR and GHI dataframes for this data block on 'Date'
                            # 'outer' merge ensures all dates are kept even if one file is missing a day
                            # No suffixes needed as 'PR' and 'GHI' are distinct column names
                            ghi_pr_df = pd.merge(pr_df, ghi_df, on='Date', how='outer')
                            
                            # Ensure columns are in the required order: Date, GHI, PR. This also implicitly selects only these three columns
                            ghi_pr_df = ghi_pr_df[['Date', 'GHI', 'PR']]

                            list_of_dfs.append(ghi_pr_df)
                            files_processed_count += 1
                        except Exception as e:
                            print(f"\nWarning: Could not read or merge data from files starting {date_key}. Error: {e}")
                            print(f"\nPlease check the internal CSV format of: {pr_file_path}, {ghi_file_path}")
                    else: 
                        print(f"\nWarning: GHI file for {date_key} found, but no matching PR file in PR directory. Skipping. Path: {ghi_file_path}")
                except ValueError:
                    print(f"\nSkipping non-date or malformed GHI file: {ghi_filename} in {current_ghi_month_path}")
            else:
                print(f"\nSkipping non-CSV GHI file: {ghi_filename} in {current_ghi_month_path}")

    # --- Final Consolidation and Reporting ---
    if not list_of_dfs:
        print("\nNo data blocks found or processed. Please verify your data directory, subfolder structure, and CSV internal format.")
        return None

    # Concatenate all individual data block DataFrames into one final DataFrame
    # drop_duplicates ensures no duplicate dates if blocks overlapped or individual CSVs had redundant dates
    # sort_values and reset_index ensure chronological order and clean index
    df = pd.concat(list_of_dfs).drop_duplicates(subset=['Date']).sort_values(by='Date').reset_index(drop=True)

    # Save the processed data to a CSV file as 'Date,GHI,PR' as specified
    output_csv_path = "processed_data.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"\nProcessed data saved to '{output_csv_path}' with {len(df)} rows.")

    # Explicitly report the count discrepancy 
    print(f"\n--- Data Count Report for the Data ---")
    print(f"\nNumber of paired PR and GHI data files processed: {files_processed_count}")
    print(f"Total individual PR files found in directories: {total_pr_files_found}")
    print(f"Total individual GHI files found in directories: {total_ghi_files_found}")
    print(f"Total rows in final collated data (after combining all blocks): {len(df)}")
    
    if len(df) == 982:
        print("\nThis matches the expected 982 rows. ")
    else:
        print(f"\nThe final file should contain 982 rows, but {len(df)} were formed.")
    return df

# DATA VISUALISATION

def generate_performance_graph(df, start_date=None, end_date=None):
    """
    Generates the Performance Ratio (PR) evolution graph, including the 30-day moving average, dynamic budget line, and annotations.
    The generated graph is also saved as a PNG file.
    """

    if df is None or df.empty:
        print("\nNo data to generate graph. Please check data preprocessing.")
        return

    # Ensure 'Date' column is datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Apply date range filter
    filtered_df = df.copy()
    if start_date:
        filtered_df = filtered_df[filtered_df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        filtered_df = filtered_df[filtered_df['Date'] <= pd.to_datetime(end_date)]

    if filtered_df.empty:
        print(f"No data found for the specified date range: {start_date} to {end_date}.")
        return

    # Calculate 30-day moving average of PR
    filtered_df['PR_30d_MA'] = filtered_df['PR'].rolling(window=30, min_periods=1).mean()

    # Dynamic Budget Line Calculation
    budget_start_date = datetime(2019, 7, 1)
    initial_budget_pr = 73.9
    annual_reduction = 0.8

    # Logic for calculating Budget_PR
    filtered_df['Budget_PR'] = filtered_df['Date'].apply(
        lambda date: (
            initial_budget_pr - (
                max(0, (date.year - budget_start_date.year if date.month >= budget_start_date.month else date.year - budget_start_date.year - 1))
                * annual_reduction
            )
        ) if date >= budget_start_date else np.nan
    )

    # GHI Color Mapping for Scatter Plot
    conditions = [
        filtered_df['GHI'] < 2,
        (filtered_df['GHI'] >= 2) & (filtered_df['GHI'] < 4),
        (filtered_df['GHI'] >= 4) & (filtered_df['GHI'] < 6),
        filtered_df['GHI'] >= 6
    ]
    colors = ['navy', 'lightskyblue', 'orange', 'brown']
    ghi_colors = np.select(conditions, colors, default='grey')

    # Plotting
    plt.style.use('seaborn-v0_8-darkgrid') 
    fig, ax = plt.subplots(figsize=(14, 7))

    # Scatter plot for daily PR, colored by GHI
    scatter = ax.scatter(filtered_df['Date'], filtered_df['PR'], c=ghi_colors, s=15, alpha=0.6, zorder=2) 

    # 30-day moving average of PR
    ax.plot(filtered_df['Date'], filtered_df['PR_30d_MA'], color='red', linewidth=2, label='30-d moving average of PR', zorder=3)

    # Dynamic Budget Line
    ax.plot(filtered_df['Date'], filtered_df['Budget_PR'], color='darkgreen', linewidth=2, linestyle='-',
            label=f'Target Budget Yield Performance Ratio [1Y-{initial_budget_pr}%, 2Y-{initial_budget_pr-annual_reduction}%, 3Y-{initial_budget_pr-(2*annual_reduction)}%]', zorder=3)

    # Annotations and Labels
    # Dynamic Title based on date range
    if start_date and end_date:
        title_text = f"Performance Ratio Evolution\nFrom {start_date} to {end_date}"
    else:
        min_date_str = filtered_df['Date'].min().strftime('%Y-%m-%d')
        max_date_str = filtered_df['Date'].max().strftime('%Y-%m-%d')
        title_text = f"Performance Ratio Evolution\nFrom {min_date_str} to {max_date_str}"
        
    ax.set_title(title_text, fontsize=16, pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Performance Ratio [%]', fontsize=12)
    ax.set_ylim(0, 100) 

    # Horizontal grid lines, fainter
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.grid(axis='x', linestyle=':', alpha=0.3) 

    # Set Y-axis ticks to match original (every 10 units)
    ax.set_yticks(np.arange(0, 101, 10))

    # X-axis formatting to match original (e.g., Jul/19, Oct/19)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3)) 
    ax.xaxis.set_minor_locator(mdates.MonthLocator()) 
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right') 

    # Legend Placement
    # GHI legend (Daily Irradiation [kWh/m2]) - positioned at the top inside the plot area
    legend_elements_ghi = [
        Line2D([0], [0], marker='o', color='w', label='< 2', markerfacecolor='navy', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='2-4', markerfacecolor='lightskyblue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='4-6', markerfacecolor='orange', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='> 6', markerfacecolor='brown', markersize=8)
    ]
    first_legend = ax.legend(handles=legend_elements_ghi, loc='upper left', bbox_to_anchor=(0.01, 0.99),
                             ncol=4, title="Daily Irradiation [kWh/m2]", title_fontsize=10,
                             frameon=False, columnspacing=1.0, handletextpad=0.5) 
    ax.add_artist(first_legend) 

    # Main legend for lines (30-d moving average and Budget PR) - positioned lower left, inside plot area
    legend_elements_lines = [
        Line2D([0], [0], color='red', lw=2, label='30-d moving average of PR'),
        Line2D([0], [0], color='darkgreen', lw=2, label='Target Budget Yield Performance Ratio')
    ]
    ax.legend(handles=legend_elements_lines, loc='center left', bbox_to_anchor=(0.01, 0.5),
              frameon=False, fancybox=True, shadow=False) 

    # Calculate and display average PRs
    avg_pr_7d = filtered_df['PR'].tail(7).mean()
    avg_pr_30d = filtered_df['PR'].tail(30).mean()
    avg_pr_60d = filtered_df['PR'].tail(60).mean()
    avg_pr_90d = filtered_df['PR'].tail(90).mean()
    avg_pr_365d = filtered_df['PR'].tail(365).mean()
    avg_pr_lifetime = filtered_df['PR'].mean()

    # Calculate points above Target Budget PR
    above_budget_df = filtered_df[filtered_df['PR'] > filtered_df['Budget_PR']]
    total_points = len(filtered_df)
    points_above_budget = len(above_budget_df)
    percentage_above_budget = (points_above_budget / total_points) * 100 if total_points > 0 else 0

    # Text box for average PRs and points above budget - positioned at bottom right
    text_box_content = (
        f"Points above Target Budget PR = {points_above_budget}/{total_points} = {percentage_above_budget:.1f}%\n\n"
        f"Average PR last 7-d: {avg_pr_7d:.1f}%\n"
        f"Average PR last 30-d: {avg_pr_30d:.1f}%\n"
        f"Average PR last 60-d: {avg_pr_60d:.1f}%\n"
        f"Average PR last 90-d: {avg_pr_90d:.1f}%\n"
        f"Average PR last 365-d: {avg_pr_365d:.1f}%\n"
        f"Average PR Lifetime: {avg_pr_lifetime:.1f}%"
    )
    ax.text(0.98, 0.05, text_box_content, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8, ec='gray'))

    plt.tight_layout(rect=[0, 0, 0.98, 1]) 
    
    # Save the plot
    if start_date and end_date:
        output_filename = f"performance_ratio_evolution_{start_date}_to_{end_date}.png"
    else:
        output_filename = "performance_ratio_evolution_full_dataset.png"
    
    plt.savefig(output_filename, bbox_inches='tight', dpi=300) 
    print(f"Graph saved as '{output_filename}'")
    plt.show()


# MAIN FUNCTION

if __name__ == "__main__":
    data_directory_path = r"D:/Assignment/data" 

    # Step 1: Preprocess the data
    processed_dataframe = data_preprocess(data_directory_path)    # Read the actual files, generate a 'processed_data.csv' file, and return a DataFrame.

    # Step 2: Generate the PR evolution graph with the processed data
    if processed_dataframe is not None:
        print("\nGenerating graph for the full dataset...")
        generate_performance_graph(processed_dataframe)

        # Generate graph for a specific date range (example dates)
        print("\nGenerating graph for a specific date range (e.g., 2020-01-01 to 2021-12-31)...")
        generate_performance_graph(processed_dataframe, start_date="2020-01-01", end_date="2021-12-31")    # Adjust these dates based on the actual range of your data if needed.

        print("\nGraphs generated successfully (check plot window).")
    else:
        print("\nFailed to preprocess data. Cannot generate graph.")
