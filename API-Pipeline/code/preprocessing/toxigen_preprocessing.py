from datasets import load_dataset
import pandas as pd
import numpy as np
import os

token = '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'

# get wd
wd = os.getcwd()

# load and prepare human annotated data

TG_human = load_dataset('skg/toxigen-data', 
                        name='annotated', 
                        token=token)

TG_human_1 = pd.DataFrame(TG_human['train'])
TG_human_2 = pd.DataFrame(TG_human['test'])

# get all columns in TG_human_1 and TG_human_2
TG_human_1.columns
TG_human_2.columns

# get groups in TG_human and TG_human_2
group_human_1 = TG_human_1['target_group'].unique()
group_human_2 = TG_human_2['target_group'].unique()

# create mapping to use same group categories in both datasets
mapping = {
    'black/african-american folks': 'black',
    'black folks / african-americans': 'black',
    'mexican folks': 'mexican',
    'women': 'women',
    'native american/indigenous folks': 'native_american',
    'folks with physical disabilities': 'physical_dis',
    'latino/hispanic folks': 'latino',
    'chinese folks': 'chinese',
    'middle eastern folks': 'middle_east',
    'asian folks': 'asian',
    'jewish folks': 'jewish',
    'muslim folks': 'muslim',
    'folks with mental disabilities': 'mental_dis',
    'lgbtq+ folks': 'lgbtq'
}

TG_human_2['target_group'] = TG_human_2['target_group'].map(mapping)

# merge both datasets
TG_human = pd.concat([TG_human_1, TG_human_2])

# add id column using prefix TGH and ranges from 1 to N
TG_human['id'] = ['TG' + str(i) for i in range(1, len(TG_human) + 1)]
TG_human.columns

# collapse groups
group_mapping = {
    'asian': 'asian',
    'chinese': 'asian',
    'muslim': 'muslim',
    'middle_east': 'muslim',
    'physical_dis': 'disability',
    'mental_dis': 'disability',
    'jewish': 'jewish',
    'black': 'black',
    'latino': 'latinx',
    'mexican': 'latinx',
    'lgbtq': 'lgbtq',
    'native_american': 'other',
    'women': 'female'
}

TG_human['target_group'] = TG_human['target_group'].map(group_mapping).fillna('other')

# drop rows with other group
TG_human = TG_human[TG_human['target_group'] != 'other']


# remove rows with unclear group link as per human annotators
# rule: at least one person expressed doubt regarding group membership (standard label)
TG_human_group = TG_human[~TG_human['predicted_group'].apply(lambda x: 'it is unclear who is referenced' in x)]

def label_data(df):
    new_df = pd.DataFrame()
    new_df['id'] = df['id'].tolist()
    new_df['group'] = df['target_group'].tolist()
    new_df['text'] = [i for i in df['text'].tolist()]
    new_df['true_label'] = np.where(df['toxicity_human'] > 2.5, 1, 0)
    return new_df

TG_human = label_data(TG_human)
TG_human_group = label_data(TG_human_group)

# balanced samples for each group and true_label combination
group_counts = TG_human_group.groupby(['group', 'true_label']).size()
group_min_counts = group_counts.unstack().min(axis=1)

def sample_group(group_df):
    group = group_df.name[0]
    min_count = group_min_counts[group]
    return group_df.sample(n=min_count, random_state=42, replace=True)

TG_human_group_balanced = TG_human_group.groupby(['group', 'true_label']).apply(sample_group).reset_index(drop=True)
TG_human_group_balanced.groupby(['group', 'true_label']).size()

# balance TG_human on true_label
minority_class_count = TG_human['true_label'].value_counts().min()
balanced_dfs = []
for label in TG_human['true_label'].unique():
    label_df = TG_human[TG_human['true_label'] == label].sample(n=minority_class_count, random_state=42)
    balanced_dfs.append(label_df)

TG_human_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=1).reset_index(drop=True)
TG_human_balanced['true_label'].value_counts()

TG_human_balanced.to_csv("../../input/data/toxigen_human_annotated_aggregate.csv", index=False)
TG_human_group_balanced.to_csv("../../input/data/toxigen_human_annotated_group.csv", index=False)