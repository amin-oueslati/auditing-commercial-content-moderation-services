import pandas as pd
import numpy as np

### load data

# read the CSV file using pandas
JS = pd.read_csv('../../input/data/jigsaw_data_raw.csv')

# overwrite ID column with index from J1 to JN
JS['id'] = ['JS' + str(i) for i in range(1, len(JS) + 1)]

# get length of the dataframe
print("initial N:", len(JS))

### assign binary label for hate speech

JS['true_label'] = (
    (JS['severe_toxicity'] > 0.5) |
    (JS['obscene'] > 0.5) |
    (JS['threat'] > 0.5) |
    (JS['insult'] > 0.5) |
    (JS['identity_attack'] > 0.5) |
    (JS['sexual_explicit'] > 0.5)
).astype(int)

# re-label comment_text to text
JS = JS.rename(columns={'comment_text': 'text'})

### filter for annotations

# filter for rows with toxicity_annotator_count >= 4
JS_aggregate = JS[JS['toxicity_annotator_count'] >= 4]
print("N post toxicity annotator activity filter:", len(JS_aggregate))

### map identity groups onto 7 groups
#print(JS['group'].unique())
JS.columns

JS_group = JS[(JS['identity_annotator_count'] >= 4) & (JS['toxicity_annotator_count'] >= 4)]
print("N post toxicity & identity annotator activity filter:", len(JS_group))

# find the index of 'asian' and 'white'
start_index = JS_group.columns.get_loc('asian')
end_index = JS_group.columns.get_loc('white')

# create the vector of column names from 'asian' to 'white'
column_vector = JS_group.columns[start_index:end_index+1].tolist()
print(column_vector)

# Map identity groups onto 7 groups
group_mapping = {
    'asian': ['asian'],
    'muslim': ['muslim'],
    'disability': ['physical_disability', 'intellectual_or_learning_disability', 
                   'other_disability', 'psychiatric_or_mental_illness'],
    'jewish': ['jewish'],
    'black': ['black'],
    'latinx': ['latino'],
    'lgbtq': ['bisexual', 'homosexual_gay_or_lesbian', 'other_sexual_orientation', 'transgender'],
    'woman': ['female'],
}

# initialize zero columns for new groups
JS_group.loc[:, 'lgbtq'] = 0.0
JS_group.loc[:, 'disability'] = 0.0

# iterate through each row
merge_groups = ['disability', 'lgbtq']

# calculate the sum for each new group
for group in merge_groups:
    columns = group_mapping[group]
    valid_columns = [col for col in columns if col in JS.columns]
    JS_group[group] = JS_group[valid_columns].sum(axis=1).clip(0, 1)

# rename latino to latinx
JS_group = JS_group.rename(columns={'latino': 'latinx'})

# identity group columns
identity_groups = ['asian', 'muslim', 'disability', 'jewish', 'black', 'latinx', 'lgbtq', 'female']

### filter for identity groups

# filter out rows w/ insufficient agreement among annotators
JS_group = JS_group[JS_group[identity_groups].max(axis=1) > 0.5]
print("N post identity annotator agreement filter:", len(JS_group))

# identify and remove rows with no single majority identity assigned (i.e. two identities with same max value)
max_values_per_row = JS_group[identity_groups].max(axis=1)
rows_with_multiple_max = JS_group[identity_groups].apply(lambda row: (row == max_values_per_row[row.name]).sum() > 1, axis=1)
count_with_multiple_max = rows_with_multiple_max.sum()
print("Number of rows with multiple max values:", count_with_multiple_max)

# rmeove these rows
JS_group = JS_group[~rows_with_multiple_max]
print("N post removing rows with multiple max values:", len(JS_group))

# detetmine identity with largest agreement among annotators
def determine_identity(row):
    max_val = row[identity_groups].max()
    # if there are more than one identity group with the same max value, return a random one
    return np.random.choice(row[row == max_val].index)

JS_group['group'] = JS_group[identity_groups].apply(determine_identity, axis=1)

# show value count for identity column
print(JS_group['group'].value_counts())

### save the preprocessed data

# create new aggregate df with id, text and true_label
JS_aggregate = JS_aggregate[['id', 'text', 'true_label']]

JS_aggregate['true_label'].value_counts()

# draw balanced sample of 50k
JS_aggregate_balanced = JS_aggregate.groupby('true_label').sample(n=25000, random_state=42)
JS_aggregate_balanced['true_label'].value_counts()

# export
JS_aggregate_balanced.to_csv("../../input/data/jigsaw_aggregate.csv", index=False)

# create new df with text, group, and true_label
JS_group = JS_group[['id', 'text', 'group', 'true_label']]

group_label_combinations = JS_group.groupby(['group', 'true_label']).size().reset_index(name='counts')
min_count_per_combination = group_label_combinations['counts'].min()

# balanced samples for each group and true_label combination
group_counts = JS_group.groupby(['group', 'true_label']).size()
group_min_counts = group_counts.unstack().min(axis=1)

def sample_group(group_df):
    group = group_df.name[0]
    min_count = group_min_counts[group]
    return group_df.sample(n=min_count, random_state=42, replace=True)

JS_group_balanced = JS_group.groupby(['group', 'true_label']).apply(sample_group).reset_index(drop=True)
JS_group_balanced.groupby(['group'])['true_label'].value_counts()

# get total N
print("N post balancing:", len(JS_group_balanced))

# export
JS_group_balanced.to_csv("../../input/data/jigsaw_group.csv", index=False)

### compute and save descriprive statistics

# calculate average word count
def avg_word_count(text):
    return np.mean([len(t.split()) for t in text])

# initialize the df to store descriptive statistics
columns = ['0', '1', 'avg_word_count']
groups = ['aggregate', 'asian', 'black', 'disability', 'female', 'jewish', 'latinx', 'lgbtq', 'muslim']
descriptive_stats = pd.DataFrame(columns=columns, index=groups)

count_0_aggregate = (JS_aggregate_balanced['true_label'] == 0).sum()
count_1_aggregate = (JS_aggregate_balanced['true_label'] == 1).sum()
avg_length_aggregate = avg_word_count(JS_aggregate_balanced['text'])
descriptive_stats.loc['aggregate'] = [count_0_aggregate, count_1_aggregate, avg_length_aggregate]

for group in descriptive_stats.index[1:]: # skip 'aggregate' in the loop
    group_df = JS_group_balanced[JS_group_balanced['group'] == group]
    count_0 = (group_df['true_label'] == 0).sum()
    count_1 = (group_df['true_label'] == 1).sum()
    avg_length = avg_word_count(group_df['text'])
    descriptive_stats.loc[group] = [count_0, count_1, avg_length]

descriptive_stats.to_csv("../../input/data/jigsaw_descriptive_statistics.csv")

print(descriptive_stats)

