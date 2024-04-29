import pandas as pd
import numpy as np

# Function to calculate average word count
def avg_word_count(text):
    return np.mean([len(t.split()) for t in text])

# Define datasets and their respective paths
datasets = {
    "toxigen": "../../input/data/toxigen_human_annotated",
    "jigsaw": "../../input/data/jigsaw",
    "megaspeech": "../../input/data/megaspeech"
}

# List of groups to include
groups_to_include = ['aggregate', 'asian', 'black', 'disability', 'female', 'jewish', 'latinx', 'lgbtq', 'muslim']

# Initialize list to hold DataFrames for descriptive statistics
descriptive_stats_list = []

# Process each dataset
for dataset_name, dataset_path in datasets.items():
    # Load dataset
    aggregate_data = pd.read_csv(f"{dataset_path}_aggregate.csv")
    group_data = pd.read_csv(f"{dataset_path}_group.csv")
    
    # Modify group column in megaspeech_group.csv
    if dataset_name == "megaspeech":
        group_data['group'] = group_data['group'].replace({'woman': 'female', 'jews': 'jewish'})
    
    # Calculate aggregate statistics
    total_count_aggregate = len(aggregate_data)
    toxic_count_aggregate = (aggregate_data['true_label'] == 1).sum()
    share_toxic_aggregate = toxic_count_aggregate / total_count_aggregate
    avg_length_aggregate = avg_word_count(aggregate_data['text'])
    
    # Add aggregate statistics to DataFrame
    descriptive_stats_list.append({
        'dataset': dataset_name,
        'group': 'aggregate',
        'total_count': total_count_aggregate,
        'share_toxic': share_toxic_aggregate,
        'average_word_count': avg_length_aggregate
    })
    
    # Calculate statistics for each group
    for group in group_data['group'].unique():
        if group in groups_to_include:
            group_df = group_data[group_data['group'] == group]
            total_count = len(group_df)
            toxic_count = (group_df['true_label'] == 1).sum()
            share_toxic = toxic_count / total_count
            avg_length = avg_word_count(group_df['text'])
            
            # Add group statistics to DataFrame
            descriptive_stats_list.append({
                'dataset': dataset_name,
                'group': group,
                'total_count': total_count,
                'share_toxic': share_toxic,
                'average_word_count': avg_length
            })

# Convert list of dictionaries to DataFrame
descriptive_stats = pd.DataFrame(descriptive_stats_list)

# Save the descriptive statistics to a CSV file
descriptive_stats.to_csv("../../input/data/descriptive_statistics_across_datasets.csv", index=False)




