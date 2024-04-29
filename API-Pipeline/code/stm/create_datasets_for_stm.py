import pandas as pd
import os
import random

##### Google #####

# List to hold dataframes
dataframes = []
group_folders = ['toxigen_group', 'megaspeech_group', 'jigsaw_group']

# Define the base directory where the API folders are located
base_dir = '../../output/API_classification/google'

# Loop through each group folder
for group_folder in group_folders:
    # Construct the path to the group folder
    folder_path = os.path.join(base_dir, group_folder)
    
    # Loop through each file in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a CSV file
        if file_name.endswith('.csv'):
            # Read the CSV file
            df = pd.read_csv(os.path.join(folder_path, file_name))
            
            # Add 'dataset' column, derived from the folder name
            df['dataset'] = group_folder.replace('_group', '')
            
            # Add 'minority_group' column, derived from the file name
            minority_group = file_name.replace('google_', '').replace(group_folder, '').replace('.csv', '').replace('_', '')
            df['minority_group'] = minority_group
            
            # Append the dataframe to the list
            dataframes.append(df)

# Combine all dataframes into a single dataframe
google_stm_input = pd.concat(dataframes, ignore_index=True)
google_stm_input

# Assigning moderation outcome variable
google_stm_input['moderation_outcome'] = 'unknown'  # Default value
google_stm_input.loc[(google_stm_input['predicted_label'] == google_stm_input['true_label']), 'moderation_outcome'] = 'correctly_moderated'
google_stm_input.loc[(google_stm_input['predicted_label'] == 0) & (google_stm_input['true_label'] == 1), 'moderation_outcome'] = 'under_moderated'
google_stm_input.loc[(google_stm_input['predicted_label'] == 1) & (google_stm_input['true_label'] == 0), 'moderation_outcome'] = 'over_moderated'

# drop columns 'service' and Unnamed:0
google_stm_input = google_stm_input.drop(columns=['Unnamed: 0'])

google_stm_input

google_stm_input['moderation_outcome'].value_counts()

# Save the combined dataframe to a CSV file
google_stm_input_path = '../../../Structural-Topic-Model/input/data-raw/google_stm_input.csv'
google_stm_input.to_csv(google_stm_input_path, index=False)


##### Cross-Service Coonsistent Moderation #####


base_dir_1 = '../../output/API_classification/'

# Define the APIs and group folders to iterate over
apis = ['amazon', 'google', 'microsoft', 'gpt']
group_folders = ['toxigen_group', 'megaspeech_group', 'jigsaw_group']

# List to hold all the dataframes
all_dfs = []

# Loop through each service and group folder
for api in apis:
    for group_folder in group_folders:
        # Construct the path to the group folder within the service folder
        folder_path = os.path.join(base_dir_1, api, group_folder)

        # Loop through each file in the folder
        for file_name in os.listdir(folder_path):
            # Check if the file is a CSV file
            if file_name.endswith('.csv'):
                # Construct the full path to the CSV file
                file_path = os.path.join(folder_path, file_name)

                # Read the CSV file
                df = pd.read_csv(file_path)

                # Add columns for 'service', 'dataset', and extract 'minority_group' from the file name
                df['service'] = api
                df['dataset'] = group_folder.replace('_group', '')
                minority_group = file_name.replace(api + '_', '').replace('.csv', '').replace(group_folder + '_', '')
                df['minority_group'] = minority_group

                # Append the dataframe to the list
                all_dfs.append(df)

# Combine all dataframes into a single dataframe
combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df = combined_df.dropna(subset=['score'])
combined_df.value_counts('service')

combined_df = combined_df.drop_duplicates(subset=['service', 'id'], keep='first')
combined_df.value_counts('service')

# Filter out text groups with less than 3 occurrences
text_group_counts = combined_df['id'].value_counts()
valid_text_groups = text_group_counts[text_group_counts >= 4].index # check whether three or four
combined_df = combined_df[combined_df['id'].isin(valid_text_groups)]
combined_df.value_counts('service')

combined_df.sort_values(by='id')

# ensure all services predicted the same label
combined_df['unique_predicted_labels'] = combined_df.groupby('id')['predicted_label'].transform('nunique')
combined_df = combined_df[combined_df['unique_predicted_labels'] == 1].drop(columns='unique_predicted_labels')
combined_df.value_counts('service')

# Define a function to apply moderation outcome logic
def determine_moderation_outcome(group):
    # If all predicted labels are equal to true labels, it's correctly moderated
    if all(group['predicted_label'] == group['true_label']):
        return 'correctly_moderated'
    # If all predicted labels are 1 and true labels are 0, it's over-moderated
    elif all(group['predicted_label'] == 1) and all(group['true_label'] == 0):
        return 'overmoderated'
    # If all predicted labels are 0 and true labels are 1, it's under-moderated
    elif all(group['predicted_label'] == 0) and all(group['true_label'] == 1):
        return 'undermoderated'
    else:
        return 'unknown'  # If there's a mix of predicted and true labels, it's unknown

combined_df_moderated = pd.DataFrame(columns=['id' 'text', 'service', 'dataset', 'true_label', 'minority_group', 'moderation_outcome'])
moderation_entries = []

for _, group in combined_df.groupby('id'):
    moderation_outcome = determine_moderation_outcome(group)
    first_entry = group.iloc[0].copy()
    first_entry['moderation_outcome'] = moderation_outcome
    moderation_entries.append({
        'id': first_entry['id'],
        'text': first_entry['text'],
        'service': first_entry['service'],
        'dataset': first_entry['dataset'],
        'true_label': first_entry['true_label'],
        'minority_group': first_entry['minority_group'],
        'moderation_outcome': first_entry['moderation_outcome']
    })

combined_df_moderated = pd.DataFrame(moderation_entries)
combined_df_moderated.value_counts('moderation_outcome')

random_ids = []
for outcome in combined_df_moderated['moderation_outcome'].unique():
    outcome_ids = combined_df_moderated[combined_df_moderated['moderation_outcome'] == outcome]['id'].unique()
    random_ids.extend(random.sample(list(outcome_ids), 2))

explore = combined_df[combined_df['id'].isin(random_ids)][['id', 'service', 'true_label', 'predicted_label']]
explore = explore.sort_values(by='id')
print(explore)

# print the rows with random id from the combined_df_moderated
explore_1 = combined_df_moderated[combined_df_moderated['id'].isin(random_ids)][['id', 'moderation_outcome', 'service', 'true_label']]
print(explore_1)

# prepare for exports
cross_service_stm_input = combined_df_moderated[combined_df_moderated['moderation_outcome'] != 'unknown']
cross_service_stm_input.value_counts

# Save the combined dataframe to a CSV file
cross_service_stm_input_csv_path = '../../../Structural-Topic-Model/input/data-raw/cross_service_consistent_moderation_stm_input.csv'
cross_service_stm_input.to_csv(cross_service_stm_input_csv_path, index=False)