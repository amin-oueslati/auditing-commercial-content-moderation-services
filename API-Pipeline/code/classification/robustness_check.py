import os
import pandas as pd

##### Compare Counts on Aggregate #####

base_dir = '../../output/API_classification'

apis = ['amazon', 'google', 'microsoft', 'gpt']
group_folders = ['toxigen_aggregate', 'megaspeech_aggregate', 'jigsaw_aggregate']

all_dfs = []

for api in apis:
    for group_folder in group_folders:
        folder_path = os.path.join(base_dir, api, group_folder)

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                df['service'] = api
                df['dataset'] = group_folder.replace('_aggregate', '')
                all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)
classified_df = combined_df[combined_df['predicted_label'].isnull() == False]

count_table = classified_df.groupby(['dataset', 'service']).size().reset_index(name='N')

data_base_dir = '../../input/data'

data_input_files = {
    'megaspeech': 'megaspeech_aggregate.csv',
    'toxigen': 'toxigen_human_annotated_aggregate.csv',
    'jigsaw': 'jigsaw_aggregate.csv'
}

original_datasets = []

for dataset, file_name in data_input_files.items():
    file_path = os.path.join(data_base_dir, file_name)
    df = pd.read_csv(file_path)
    df['service'] = 'original_dataset'
    df['dataset'] = dataset
    original_datasets.append(df)

original_df = pd.concat(original_datasets, ignore_index=True)
original_counts = original_df.groupby(['service', 'dataset']).size().reset_index(name='N')

final_count_table = pd.concat([count_table, original_counts], ignore_index=True)

# sort final count table by dataset and service
final_count_table = final_count_table.sort_values(by=['dataset', 'service']).reset_index(drop=True)
print(final_count_table)
print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA \n')
##### Focus on Google #####

google_base_dir = '../../output/API_classification/google'
group_folders = ['toxigen_aggregate', 'megaspeech_aggregate', 'jigsaw_aggregate']
google_dfs = []

for group_folder in group_folders:
    folder_path = os.path.join(google_base_dir, group_folder)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            df['service'] = 'google'
            df['dataset'] = group_folder.replace('_aggregate', '')
            google_dfs.append(df)

google_combined_df = pd.concat(google_dfs, ignore_index=True)
google_combined_df
original_df

# Are there any examples which occur in the google classification output but not in the original data?

google_not_in_original = google_combined_df[~google_combined_df['text'].isin(original_df['text'])]
print(google_not_in_original.value_counts('dataset'))

# Are there any examples where the true_label for the same text is different between the google classification output and the original data?

# merge on text
merged_df = pd.merge(google_combined_df, original_df, on='text', suffixes=('_google', '_original'))
# filter
different_true_labels = merged_df[merged_df['true_label_google'] != merged_df['true_label_original']]

different_true_labels = different_true_labels[['text', 'true_label_original', 'true_label_google', 'dataset_original', 'dataset_google']]
different_true_labels = different_true_labels.sort_values(by='text')
print(different_true_labels)

##### Compare Amazon

amazon_base_dir = '../../output/API_classification/amazon'
group_folders = ['toxigen_aggregate', 'megaspeech_aggregate', 'jigsaw_aggregate']
amazon_dfs = []

for group_folder in group_folders:
    folder_path = os.path.join(amazon_base_dir, group_folder)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            df['service'] = 'amazon'
            df['dataset'] = group_folder.replace('_aggregate', '')
            amazon_dfs.append(df)

amazon_combined_df = pd.concat(amazon_dfs, ignore_index=True)
amazon_combined_df
original_df

# Are there any examples which occur in the amazon classification output but not in the original data?

amazon_not_in_original = amazon_combined_df[~amazon_combined_df['text'].isin(original_df['text'])]
print(amazon_not_in_original.value_counts('dataset'))

# Are there any examples where the true_label for the same text is different between the amazon classification output and the original data?

# merge on text
merged_df = pd.merge(amazon_combined_df, original_df, on='text', suffixes=('_amazon', '_original'))
# filter
different_true_labels = merged_df[merged_df['true_label_amazon'] != merged_df['true_label_original']]

different_true_labels = different_true_labels[['text', 'true_label_original', 'true_label_amazon', 'dataset_original', 'dataset_amazon']]
different_true_labels = different_true_labels.sort_values(by='text')
print(different_true_labels)


##### Compare Counts on Group Level #####


# Define the base directory for the classification data
base_dir_1 = '../../output/API_classification'

# Define the APIs and group folders to iterate over
apis = ['amazon', 'google', 'microsoft', 'gpt']
group_folders = ['toxigen_group', 'megaspeech_group', 'jigsaw_group']

all_dfs = []

for api in apis:
    for group_folder in group_folders:
        folder_path = os.path.join(base_dir_1, api, group_folder)

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                df['service'] = api
                df['dataset'] = group_folder.replace('_group', '')
                minority_group = file_name.replace(api + '_', '').replace('.csv', '').replace(group_folder + '_', '')
                df['minority_group'] = minority_group
                all_dfs.append(df)

classification_combined_df = pd.concat(all_dfs, ignore_index=True)
classification_group_counts = classification_combined_df.groupby(['dataset', 'service']).size().reset_index(name='N')

# Define the base directory for the datasets
data_base_dir = '../../input/data'

# Define input file names for the datasets
dataset_files = {
    'megaspeech': 'megaspeech_group.csv',
    'toxigen': 'toxigen_human_annotated_group.csv',
    'jigsaw': 'jigsaw_group.csv'
}

dataset_dfs = []

for dataset, file_name in dataset_files.items():
    file_path = os.path.join(data_base_dir, file_name)
    df = pd.read_csv(file_path)
    df['dataset'] = dataset
    dataset_dfs.append(df)

dataset_combined_df = pd.concat(dataset_dfs, ignore_index=True)

final_count_df = pd.concat([classification_group_counts, dataset_combined_df.groupby(['dataset']).size().reset_index(name='N')], ignore_index=True)
final_count_df['service'] = final_count_df['service'].fillna('original_dataset')

final_count_df = final_count_df.sort_values(by=['dataset', 'service']).reset_index(drop=True)
print(final_count_df)

# Why does Microsoft have so many more phrases in MS?
microsoft_megaspeech_group_counts = classification_combined_df[(classification_combined_df['service'] == 'microsoft') & (classification_combined_df['dataset'] == 'megaspeech')].groupby('minority_group').size().reset_index(name='N')
print('microsoft megaspeech', '\n', 
      microsoft_megaspeech_group_counts)

megaspeech_group_counts = dataset_combined_df[(dataset_combined_df['dataset'] == 'megaspeech')].groupby('group').size().reset_index(name='N')
print('megaspeech data', '\n', 
      megaspeech_group_counts)


##### PSA #####


# Define the base directory for the API classification data
base_dir = '../../output/API_classification'

# Define the services (APIs) and folder types
services = ['amazon', 'google', 'microsoft', 'gpt']
types = ['synthetic', 'real']

# Initialize a list to hold all the dataframes
all_dfs = []

# Iterate over services and types to collect the data
for service in services:
    for type in types:
        # Construct the folder path
        folder_path = os.path.join(base_dir, service, type)
        
        # Iterate over the files in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                # Read the CSV file
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)
                
                # Add columns for 'service' and 'type'
                df['service'] = service
                df['data'] = type
                df['type'] = 'min' if 'min' in file_name else 'maj'
                
                # Append the dataframe to the list
                all_dfs.append(df)

# Combine all dataframes into a single dataframe
combined_df = pd.concat(all_dfs, ignore_index=True)

# Group by 'service', 'type', and 'label' (majority or minority), then count the occurrences
group_counts = combined_df.groupby(['service', 'data', 'type']).size().reset_index(name='count')

# Print the resulting DataFrame
print(group_counts)
