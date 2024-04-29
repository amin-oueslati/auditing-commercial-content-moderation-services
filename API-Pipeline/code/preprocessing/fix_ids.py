import os
import pandas as pd

base_dir_1 = '../../output/API_classification/'

# Loop through each folder in the base directory
for root, dirs, files in os.walk(base_dir_1):
    for file_name in files:
        # Check if the file is a CSV file
        if file_name.endswith('.csv'):
            # Construct the full path to the CSV file
            file_path = os.path.join(root, file_name)

            # Read the CSV file
            df = pd.read_csv(file_path)

            # Check if 'ids' column exists in the DataFrame
            if 'ids' in df.columns:
                # Rename 'ids' column to 'id'
                df.rename(columns={'ids': 'id'}, inplace=True)

                # Drop the index before saving to avoid "Unnamed: 0" column
                df.reset_index(drop=True, inplace=True)

                # Save the updated DataFrame back to the CSV file (overwrite existing file)
                df.to_csv(file_path, index=False)
