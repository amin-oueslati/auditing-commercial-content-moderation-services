import os

import pandas as pd
import scipy.stats as stats

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import matplotlib.cm as cm

##### Compile Aggreate PSA Classification #####

# Load csv containing the labels
token_real = pd.read_csv('../../input/psa/psa_real_data_majority.csv')
token_synthetic = pd.read_csv('../../input/psa/psa_synthetic_data_majority.csv')
token_real

# change psa_id to id
token_synthetic = token_synthetic.rename(columns={'psa_id': 'id'}).copy()
token_real = token_real.rename(columns={'psa_id': 'id'}).copy()

# Base directory where the API folders are located
base_dir = '../../output/API_classification'

# Initialize two empty DataFrames to collect results for real and synthetic datasets
psa_merged_real = pd.DataFrame()
psa_merged_synthetic = pd.DataFrame()

# List of APIs
apis = ['amazon', 'google', 'microsoft', 'gpt']

for api in apis:
    # Paths for the real dataset
    real_maj_path = os.path.join(base_dir, api, 'real', f'{api}_real_maj.csv')
    real_min_path = os.path.join(base_dir, api, 'real', f'{api}_real_min.csv')

    # Paths for the synthetic dataset
    synthetic_maj_path = os.path.join(base_dir, api, 'synthetic', f'{api}_synthetic_maj.csv') 
    synthetic_min_path = os.path.join(base_dir, api, 'synthetic', f'{api}_synthetic_min.csv')
    
    final_columns = ['api', 'id', 'text_maj', 'text_min', 'score_maj', 'score_min', 'score_difference', 'true_label_maj']

    # Process real data
    df_real_maj = pd.read_csv(real_maj_path)
    df_real_min = pd.read_csv(real_min_path)
    merged_real = pd.merge(df_real_maj, df_real_min, on='id', suffixes=('_maj', '_min'))
    merged_real['score_difference'] = merged_real['score_maj'] - merged_real['score_min']
    merged_real['api'] = api
    merged_real = merged_real[final_columns]
    psa_merged_real = pd.concat([psa_merged_real, merged_real], ignore_index=True)
    
    # Process synthetic data if available
    df_synthetic_maj = pd.read_csv(synthetic_maj_path)
    df_synthetic_min = pd.read_csv(synthetic_min_path)
    merged_synthetic = pd.merge(df_synthetic_maj, df_synthetic_min, on='id', suffixes=('_maj', '_min'))
    merged_synthetic['score_difference'] = merged_synthetic['score_maj'] - merged_synthetic['score_min']
    merged_synthetic['api'] = api
    merged_synthetic = merged_synthetic[final_columns]
    psa_merged_synthetic = pd.concat([psa_merged_synthetic, merged_synthetic], ignore_index=True)

psa_merged_real.columns
psa_merged_synthetic.columns

# count observations by api
psa_merged_real.groupby('api').size()
psa_merged_synthetic.groupby('api').size()

# merge with token data to get group column
psa_merged_real = pd.merge(psa_merged_real, token_real[['id', 'group']], on='id')
psa_merged_real

# print only google merged real
psa_merged_real[psa_merged_real['api'] == 'microsoft']

psa_merged_synthetic = pd.merge(psa_merged_synthetic, token_synthetic[['id', 'group']], on='id')
psa_merged_synthetic

# inspect groups
psa_merged_real['group'].value_counts()

# drop white and christian
psa_merged_real = psa_merged_real[psa_merged_real['group'] != 'white']
psa_merged_real = psa_merged_real[psa_merged_real['group'] != 'christians']

# filter cases which contain the str "is fun" in text min
exp = psa_merged_synthetic[psa_merged_synthetic['text_min'].str.contains('is fun')]
exp

# Filter the DataFrame for entries where the id is 306
filtered_data = psa_merged_synthetic[psa_merged_synthetic['text_min'].str.contains('is fun')]
grouped_data = filtered_data.groupby(['group'])
average_scores = grouped_data[['score_maj', 'score_min', 'score_difference']].mean()
print(average_scores)


# inspect lgbtq on synthetic
terms = ['gay', 'gays', 'bi', 'bisexual', 'trans', 'transexual', 'queer', 'lgbt', 'lgbtq', 'homosexual', 'lesbian']
lgbtq_filtered = psa_merged_synthetic[psa_merged_synthetic['text_min'].str.contains('|'.join(terms), case=False, regex=True)]

def check_contains(text):
    for term in terms:
        if term in text.lower():
            return term

lgbtq_filtered['contains_lgbtq'] = lgbtq_filtered['text_min'].apply(check_contains)
lgbtq_grouped = lgbtq_filtered.groupby('contains_lgbtq')
lgbtq_mean_scores = lgbtq_grouped['score_difference'].mean()
lgbtq_grouped_count = lgbtq_grouped.size()

print(lgbtq_mean_scores)
print(lgbtq_grouped_count)

# inspect lgbtq on non-synthetic
terms = ['gay', 'gays', 'bi', 'bisexual', 'trans', 'transexual', 'queer', 'lgbt', 'lgbtq', 'homosexual', 'lesbian']
lgbtq_filtered = psa_merged_real[psa_merged_real['text_min'].str.contains('|'.join(terms), case=False, regex=True)]

def check_contains(text):
    for term in terms:
        if term in text.lower():
            return term

lgbtq_filtered['contains_lgbtq'] = lgbtq_filtered['text_min'].apply(check_contains)
lgbtq_grouped = lgbtq_filtered.groupby('contains_lgbtq')
lgbtq_mean_scores = lgbtq_grouped['score_difference'].mean()
lgbtq_grouped_count = lgbtq_grouped.size()

print(lgbtq_mean_scores)
print(lgbtq_grouped_count)

##### Compute PSA Statistics #####

# Function to compute mean and 95% confidence interval

def compute_psa_statistics(data):
    results_list = []
    
    # Group by both 'group' and 'api'
    for (api, group, true_label_maj), group_data in data.groupby(['api', 'group', 'true_label_maj']):
        mean = group_data['score_difference'].mean()
        sem = group_data['score_difference'].sem()  # Standard Error of the Mean
        ci_lower, ci_upper = stats.t.interval(0.95, df=len(group_data) - 1, loc=mean, scale=sem)
        error = ci_upper - mean
        
        # Append results for this group and API to the list as a dictionary
        results_list.append({
            'api': api,
            'group': group,
            'toxic': true_label_maj,
            'mean_difference': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'error': error,
            'N': len(group_data)
        })

    # Convert the list of dictionaries to a DataFrame
    results_df = pd.DataFrame(results_list)
    return results_df

# For real
grouped_real_stats = compute_psa_statistics(psa_merged_real)
grouped_real_stats

# For synthetics
grouped_synthetic_stats = compute_psa_statistics(psa_merged_synthetic)
grouped_synthetic_stats

##### Visualise #####

# Define output directory
output_dir = "../../output/psa"

# Define a color-blind-friendly color map
colors = cm.tab20.colors

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.transforms import Affine2D

# Define output directory
output_dir = "../../output/psa"

# Define a color-blind-friendly color map
colors = cm.tab20.colors

def plot_and_save(grouped_data_real, grouped_data_synthetic, output_dir):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

    for ax_index, (grouped_data, data_type) in enumerate(zip([grouped_data_synthetic, grouped_data_real], ['Synthetic', 'Real'])):
        groups = grouped_data['group'].unique().tolist()
        toxic_values = grouped_data['toxic'].unique()

        for toxic_value in toxic_values:
            toxic_data = grouped_data[grouped_data['toxic'] == toxic_value]
            api_data = {}
            for api in toxic_data['api'].unique():
                api_data[api] = {
                    'mean_difference': toxic_data[toxic_data['api'] == api]['mean_difference'].tolist(),
                    'mean_error': toxic_data[toxic_data['api'] == api]['error'].tolist()
                }

            ax_row = 0 if data_type == 'Synthetic' else 1  # Determine the row based on data type
            ax_col = 0 if toxic_value == 0 else 1  # Determine the column based on toxic value

            ax = axs[ax_row, ax_col]

            for i, (api, color) in enumerate(zip(api_data.keys(), colors)):
                trans = Affine2D().translate(0.0, -i * 0.2 + 0.2 * (len(api_data) - 1) / 2) + ax.transData
                ax.errorbar(api_data[api]['mean_difference'], groups, xerr=api_data[api]['mean_error'], marker="o", linestyle="none", label=api, color=color, transform=trans)

            ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

            if ax_row == 0:  # Display title only in the first row
                title = 'Non-Toxic' if toxic_value == 0 else 'Toxic'
                ax.set_title(title, fontweight='bold', fontsize=24, pad = 20)
                if ax_col == 0:  # Set y-label only for the left column plots
                    ax.set_ylabel('Synthetic' if data_type == 'Synthetic' else 'Non-Synthetic', fontweight='bold', fontsize=22, labelpad=20)

            if ax_row == 0 and ax_col == 1:  # Add single legend for the second plot in the first row
                ax.legend(fontsize=18, loc='upper right') 

            ax.set_xlabel('Mean CTF Score', fontsize=22)
            ax.tick_params(axis='both', which='major', labelsize=20)  # Set tick font size

            ax.set_xlim((-0.20, 0.20))  # Assuming x-axis limits
            ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])

            if ax_col == 0:  # Display y-label only in the first column
                ax.set_ylabel('Synthetic' if data_type == 'Synthetic' else 'Non-Synthetic', fontweight='bold', fontsize=24, labelpad=20)  # Add y-label for groups

            for y in range(len(groups) - 1):
                ax.axhline(y + 0.5, color='gray', linestyle='-', linewidth=0.25)

    plt.tight_layout()
    output_file_path = os.path.join(output_dir, 'combined_plot.png')
    plt.savefig(output_file_path, bbox_inches='tight', dpi=300)
    plt.close()

# Plot and save
plot_and_save(grouped_real_stats, grouped_synthetic_stats, output_dir)



