import os
import pandas as pd
import glob
import numpy as np 
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from datasets import load_dataset, load_metric, Dataset, load_from_disk
from transformers import BertTokenizer
# read in all CSV files

# apply correct column names for pipeline

# classify groups with LSTM

def load_lstm_model():
    
    model_file_path = '../../input/data/trained_model.keras'
    print(f"Loading model weights from {model_file_path}")
    model = load_model(model_file_path, compile=False)
    return model

def tokenize_function(prmpt, tokenizer):
   # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer(prmpt["text"], padding="max_length", truncation=True, max_length = 512)

def one_hot_encode_example(example):
    # This function will be applied to each example in the dataset
    one_hot_encoded = np.eye(10)[int(example['labels'])]
    example['labels'] = one_hot_encoded.tolist()  # Convert numpy array to list
    return example

def classify_groups(data, model):
    data['sentence'] = data['sentence'].fillna('')  # D: becausepip install tensorflow==2.15.0.post1 I got an error
    data['sentence'] = data['sentence'].astype(str)  # D: because I got an error

    #max_length = list(set(w for x in data.words for w in x ))
    dataset_dict = {
    "text": data['sentence'].tolist(),
    "labels": data['label'].tolist()  # Ensure this is just a list of integers, not one-hot encoded
    }
    dataset = Dataset.from_dict(dataset_dict)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenize the dataset
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x ,tokenizer), batched=True)

    # Apply one hot encoding to the labels
    dataset_onehot = tokenized_datasets.map(one_hot_encode_example)

    tf_ds = dataset_onehot.to_tf_dataset(
        columns=["input_ids"],
        label_cols=["labels"],
        batch_size=16,
        shuffle=False)
    return tf_ds

def make_predictions(model, tf_ds):
    # Make predictions
    predictions = model.predict(tf_ds)
   
   # groups = ['christians', 'black people', 'muslims and arabic/middle eastern people',
 #'white people', 'men', 'lgbtq+ people', 'jews', 'asian people', 'women',
 #'latinx people']
    groups = ['christians', 'black', 'muslim', 'white', 'men', 'lgbtq', 'jews', 'asian', 'woman', 'latinx']
    group_predictions = []
    for i in range(len(predictions)):
        group_predictions.append(groups[np.argmax(predictions[i])])
    return group_predictions


def read_data(sample_size = 50000):
    file_path = '../../input/data/mega_speech_hate_source_partitioned'
    # create empty dataframe
    df = pd.DataFrame()
    print("Reading in data")
    # iterate through all files in the directory
    for file in os.listdir(file_path):
        # check if the file exists
        full_file = os.path.join(file_path, file)
        if os.path.isfile(full_file):
            df_read = pd.read_csv(full_file)    
            df = pd.concat([df, df_read])
    print("Data read in")

    # sample dataframes
    df_aggregate = df
    df_group = df
    print("Data sampled")

    if not os.path.exists('../../input/data/megaspeech_allpreds.csv'):
        # classify groups
        print("Classifying groups")
        model = load_lstm_model()
        tf_ds = classify_groups(df, model)
        predictions = make_predictions(model, tf_ds)
    
        # add predictions to dataframes
        df_aggregate['group'] = predictions
        df_group['group'] = predictions
        df_aggregate.to_csv('../../input/data/megaspeech_allpreds.csv')
    else:
        df_aggregate = pd.read_csv('../../input/data/megaspeech_allpreds.csv')
        df_group = pd.read_csv('../../input/data/megaspeech_allpreds.csv')

    
    print("Balancing and sampling data")
    # balance dataframes
    df_aggregate = df_aggregate.rename(columns={'sentence': 'text'})
    df_aggregate = df_aggregate.rename(columns={'label': 'true_label'})

    df_group = df_group.rename(columns={'sentence': 'text'})
    df_group = df_group.rename(columns={'label': 'true_label'})

       # save dataframes
    df_aggregate['id'] = ['MS' + str(i) for i in range(1, len(df_aggregate) + 1)]
    df_group['id'] = ['MS' + str(i) for i in range(1, len(df_group) + 1)]


    df_aggregate = df_aggregate.groupby('true_label').sample(n=int(sample_size//2), random_state=42)
    print(df_aggregate['true_label'].value_counts)
    df_aggregate.to_csv('../../input/data/megaspeech_aggregate.csv')

    df_group = df_group.groupby('true_label').sample(n=int(sample_size//2), random_state=42)


    group_label_combinations = df_group.groupby(['group', 'true_label']).size().reset_index(name='counts')
    min_count_per_combination = group_label_combinations['counts'].min()

    # balanced samples for each group and true_label combination
    group_counts = df_group.groupby(['group', 'true_label']).size()
    group_min_counts = group_counts.unstack().min(axis=1)

    def sample_group(group_df, min_count_per_combination):
        group = group_df.name[0]
        min_count = group_min_counts[group]
        return group_df.sample(n=min_count, random_state=42, replace=True)
    
    df_group_balanced = df_group.groupby(['group', 'true_label']).apply(lambda group_df: sample_group(group_df, group_min_counts)).reset_index(drop=True)
    print(df_group_balanced.groupby(['group'])['true_label'].value_counts())
 
    df_group_balanced.to_csv('../../input/data/megaspeech_group.csv')
    print("Data classified and saved")
    

read_data(50000)