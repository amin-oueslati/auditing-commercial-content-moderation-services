import pandas as pd
import re
import numpy as np

# import token conversion xlsx
token_conversion = pd.read_excel('../../input/psa/token_conversion.xlsx')
token_conversion

# import MegaSpeech csv
MS = pd.read_csv('../../input/data/megaspeech_aggregate.csv')
MS

MS.value_counts('true_label')

# drop
MS.columns
columns_to_drop = ['Unnamed: 0.1', 'Unnamed: 0', 'group', 'id']
MS.drop(columns_to_drop, axis=1, inplace=True)
MS.columns

###### Compute PSA Minority Tokens ######

# Convert the tokens to lowercase for consistent matching
token_conversion['minority_token'] = token_conversion['minority_token'].str.lower()
minority_tokens = token_conversion['minority_token'].tolist()

# Function to find the exact whole word match
def match_minority_token(sentence, token):
    sentence_lower = sentence.lower()
    for token in token:
        # Use word boundaries to ensure exact matches only
        if re.search(r'\b{}\b'.format(re.escape(token)), sentence_lower):
            return token
    return None

# Apply the function to each sentence and filter the dataframe
MS['minority_token'] = MS['text'].apply(lambda x: match_minority_token(x, minority_tokens))

# Filter sentences where there was exactly one match
psa_real_data_minority = MS[MS['minority_token'].notnull()]
len(psa_real_data_minority)

psa_real_data_minority = psa_real_data_minority.drop_duplicates(subset='text')
len(psa_real_data_minority)

# Reset the index to add an ID column
psa_real_data_minority.reset_index(drop=True, inplace=True)
psa_real_data_minority.index += 1
psa_real_data_minority.insert(0, 'psa_id', psa_real_data_minority.index)

# change column names: sentence -> text, label --> true_label
psa_real_data_minority = psa_real_data_minority.rename(columns={'label': 'true_label'}).copy()
psa_real_data_minority.columns

# Merge only the 'majority_token' column to psa_real_data_minority based on 'minority_token'
psa_real_data_minority = psa_real_data_minority.merge(token_conversion[['minority_token', 'majority_token', 'group']], 
                                                      on='minority_token', how='left')


###### Compute Majority Token PSA Equivalents ######


# Now define a function to replace the minority token with the majority token in the sentence
def replace_token(sentence, minority_token, majority_token):
    # Using word boundaries in the regex for exact word match
    return re.sub(r'\b{}\b'.format(re.escape(minority_token)), majority_token, sentence)

# Use the function to replace the tokens
psa_real_data_majority = psa_real_data_minority.copy()  # First make a copy if necessary
psa_real_data_majority['text'] = psa_real_data_majority.apply(
    lambda x: replace_token(x['text'], x['minority_token'], x['majority_token']), axis=1)


###### Robustness ######


# for inspection: solely copy the text column of both to a new dataframe side by side
psa_real_data_combined = pd.concat([psa_real_data_minority['text'], psa_real_data_majority['text']], axis=1)

# rename first column to 'minority_text' and second column to 'majority_text'
psa_real_data_combined.columns = ['minority_text', 'majority_text']

# add id column at first position
psa_real_data_combined.insert(0, 'id', psa_real_data_combined.index)


###### Comparative Statistics ######


columns = ['N', 'share_toxic', 'avg_word_count']
groups = ['aggregate', 'asian', 'black', 'disability', 'female', 'jewish', 'latinx', 'lgbtq', 'muslim']
descriptive_stats = pd.DataFrame(columns=columns, index=groups)

def avg_word_count(words):
    if len(words) == 0:
        return np.nan  # Return NaN if the input array is empty
    else:
        return sum(len(word.split()) for word in words) / len(words)

total_aggregate = len(psa_real_data_minority)
count_1_aggregate = (psa_real_data_minority['true_label'] == 1).sum()
share_1_aggregate = count_1_aggregate / total_aggregate if total_aggregate > 0 else np.nan
avg_length_aggregate = avg_word_count(psa_real_data_minority['text'])
descriptive_stats.loc['aggregate'] = [total_aggregate, share_1_aggregate, avg_length_aggregate]

for group in descriptive_stats.index[1:]: # skip 'aggregate' in the loop
    group_df = psa_real_data_minority[psa_real_data_minority['group'] == group]
    total = len(group_df)
    share_1 = (group_df['true_label'] == 1).sum() / total if total > 0 else np.nan
    avg_length = avg_word_count(group_df['text'])
    descriptive_stats.loc[group] = [total, share_1, avg_length]

###### Export ######


output_file_minority = '../../input/psa/psa_real_data_minority.csv'
psa_real_data_minority.to_csv(output_file_minority, index=False)

output_file_majority = '../../input/psa/psa_real_data_majority.csv'
psa_real_data_majority.to_csv(output_file_majority, index=False)

output_file_inspection = '../../input/psa/psa_real_data_inspection.csv'
psa_real_data_combined.to_csv(output_file_inspection, index=False)

output_file_summary_stats = '../../input/psa/psa_real_data_descriptive_statistics.csv'
descriptive_stats.to_csv(output_file_summary_stats, index=True)