import pandas as pd
import numpy as np
from apiGPT import APIgpt
from apiMicrosoft import APImicrosoft
from apiAmazon import APIamazon
#from apiGoogle import APIgoogle
import evaluations as ev
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
from datasets import load_dataset
import asyncio
import matplotlib.pyplot as plt
import met_brewer
import os

colors = met_brewer.met_brew(name="Peru1", n=5)
auth_token = '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'

# Function to read data from jigsaw
def read_data_jigsaw(data_path = 'jigsaw-unintended-bias-in-toxicity-classification/train.csv'):
    print('Reading Data')
    df = pd.concat([chunk for chunk in tqdm(pd.read_csv(data_path, chunksize=1000), desc='Loading data')])
    print('Data Loaded')
    for i in tqdm(range(df.shape[0])):
        df.at[i, 'hate_score'] = bool(df.at[i, 'severe_toxicity'] > 0.5 or df.at[i, 'obscene'] > 0.5 or df.at[i, 'threat'] > 0.5 or df.at[i, 'insult'] > 0.5 or df.at[i, 'identity_attack'] > 0.5 or df.at[i, 'sexual_explicit'] > 0.5)
    print('Pre-processing data')
    return df 
# Function to load toxigen dataset
def load_toxigen():
    TG_data = load_dataset("skg/toxigen-data", name="train", token=auth_token) # 250k training examples
    TG_annotations = load_dataset("skg/toxigen-data", name="annotated", token=auth_token) # Human study
    return TG_data, TG_annotations

# Function to predict with API
def api_pred(df, sampl = 0.001):
    print('Predicting with API')
    df = df.sample(int(sampl*df.shape[0]))
    #df_ev = ev.hate_evaluation_gpt(df.comment_text, df.hate_score)
    df_ev = ev.hate_evaluation_gpt(df.prompt, df.prompt_label)
    return df_ev

# Function to evaluate dataset
def evaluate_dataset(data_path):
    df_ev= pd.read_csv(data_path)
    f1 =f1_score(df_ev['Decision'], df_ev['Label'], average='macro')
    accuracy = accuracy_score(df_ev['Decision'], df_ev['Label'])
    precision = precision_score(df_ev['Decision'], df_ev['Label'], average='macro')
    recall = recall_score(df_ev['Decision'], df_ev['Label'], average='macro')
    return [f1, accuracy, precision, recall]

# Function to predict for a group
def group_api_pred(df, group, sampl = 0.01):
    df_ev = api_pred(df, sampl)
    df_ev.to_csv(f'data_output/gpt/gpt_toxigen_{group}.csv')
    return evaluate_dataset(f'data_output/gpt/gpt_toxigen_{group}.csv')
   
# Function to concurrently predict for groups
def main():
    tg_data, tg_anott = load_toxigen()
    df = pd.DataFrame(tg_data['train'])
    groups = df.group.unique()
    for group in groups:
        df_group = df[df.group == group]
        df_ev = api_pred(df_group)
        df_ev.to_csv(f'data_output/gpt/gpt_toxigen_{group}.csv')
    #preds = np.array([group_api_pred(df[df.group == group], group, 0.1) if not os.path.isfile('data_output/gpt/gpt_toxigen_{group}.csv') ])
    preds = np.array([evaluate_dataset(f'data_output/gpt/gpt_toxigen_{group}.csv') for group in groups])
    ind = np.arange(len(groups))
    fig, ax = plt.subplots(figsize = (6,12))
    width = 0.2
    for i in range(4):
        ax.barh(ind -1.5*width + i*width, preds[:,i], width, label = ['F1 Score', 'Accuracy', 'Precision', 'Recall'][i], color = colors[i])
    ax.set_yticks(ind, groups)
    ax.set_ylabel('Groups')
    ax.set_xticks(np.arange(0,1.1,0.25))
    ax.set_xlabel('scores')
    ax.set_title('Scores for different groups in Toxigen dataset')
    ax.set_yticklabels(groups, fontsize = 10, rotation = 45)
    ax.legend()
    plt.savefig('gpt_scores_toxigen.png', dpi=300)
    
    #df_group = df[df.group == group]
    #df_ev = api_pred(df_group, 0.001)
    #    df_ev.to_csv(f'gpt_toxigen_{group}.csv')
    #    evaluate_dataset(f'gpt_toxigen_{group}.csv')
    #df_ev.to_csv('gpt_scores_toxigen.csv')


tg_data, tg_anott = load_toxigen()
tg_data

if __name__ == "__main__":
    main()


#print(f'GPT: {APIgpt(text)[0]} \n Microsoft: {APImicrosoft(text)[0]} \n Amazon: {APIamazon(text)[0]} \n Google: {APIgoogle(text)}')
#print(APIgpt('a')[0])
#def hate_std(prompt): #function calling on evaluations.py to get the standard deviation between all API's for the hate category
#    return ev.hate_evaluation(APIamazon(prompt)[0], APIgoogle(prompt), APIgpt(prompt)[0], APImicrosoft(prompt)[0])[2]
