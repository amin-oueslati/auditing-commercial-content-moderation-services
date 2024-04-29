import os

import pandas as pd
import glob
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import met_brewer
import os

colors = met_brewer.met_brew(name="Peru1", n=5)

import os

import pandas as pd
import glob
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

import matplotlib.pyplot as plt
import met_brewer
import os

### compute performance metrics for each dataset

import numpy as np
import glob
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

import os
import glob
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score

def compute_metrics(df, true_label, pred_label, pred_score):
    df = df.dropna(subset=[pred_label, pred_score])

    y_true = df[true_label].astype(int)
   
    y_pred_binary = df[pred_label].astype(int)
    y_pred_score = df[pred_score]
    
    f1 = f1_score(y_true, y_pred_binary, average='weighted')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    roc_auc = roc_auc_score(y_true, y_pred_binary)
    
    return {
        'F1 Score': f1,
        'False Positive Rate': fpr,
        'False Negative Rate': fnr,
        'ROC AUC': roc_auc,
    }

def calculate_pinned_auc(df_subgroup_examples, df_full_dataset_examples, true_label, pred_label, pred_score):
    # Combine examples from subgroup and random samples from full dataset
    pinned_dataset = pd.concat([df_subgroup_examples, df_full_dataset_examples]) 
    # remove null values
    pinned_dataset = pinned_dataset.dropna(subset=[pred_label, pred_score])
    # Calculate ROC-AUC for the pinned dataset
    pinned_auc = roc_auc_score(pinned_dataset[true_label].astype(int), pinned_dataset[pred_score])

    return pinned_auc

def process_aggregate_case(api, dataset_name, performance_metrics_path):
    file_path = f"../../output/API_classification/{api}/{dataset_name}/{api}_{dataset_name}.csv"
    df = pd.read_csv(file_path)
    metrics = compute_metrics(df, 'true_label', 'predicted_label', 'score')
    metrics['API'] = api
    metrics['Group'] = 'aggregate'
    return metrics

def process_group_case(api, dataset_name, performance_metrics_path):
    results = []
    dataset_name_agg = dataset_name.split('_')[0] + '_aggregate'
    file_pattern = f"../../output/API_classification/{api}/{dataset_name}/{api}_{dataset_name}_*.csv"
    file_path_agg = f"../../output/API_classification/{api}/{dataset_name_agg}/{api}_{dataset_name_agg}.csv"
    df_agg = pd.read_csv(file_path_agg)
    for file_path in glob.glob(file_pattern):
        group = os.path.basename(file_path).split('_')[-1].split('.')[0]
        df = pd.read_csv(file_path)
        metrics = compute_metrics(df, 'true_label', 'predicted_label', 'score')
        df_sample_group = df.sample(frac = 0.5)
        metrics['Pinned ROC AUC'] = calculate_pinned_auc(df_sample_group, df.sample(df_sample_group.shape[0]), 'true_label', 'predicted_label', 'score')
        metrics['API'] = api
        metrics['Group'] = group
        results.append(metrics)
    return results


def compute_performance_metrics(dataset_name):
    api_list = ['amazon', 'gpt', 'google', 'microsoft']
    performance_metrics_path = '../../output/performance_metrics'

    # Directory for the dataset within the performance_metrics folder
    dataset_directory = os.path.join(performance_metrics_path, dataset_name)

    # Create the directory only if it does not exist
    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)

    all_results = []

    for api in api_list:
        directory_path = f"../../output/API_classification/{api}/{dataset_name}/"
        file_paths = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

        if len(file_paths) == 1:
            metrics = process_aggregate_case(api, dataset_name, performance_metrics_path)
            all_results.append(metrics)
            column_order = ['API', 'Group', 'F1 Score', 'ROC AUC', 'False Positive Rate', 'False Negative Rate']
        elif len(file_paths) > 1:
            results = process_group_case(api, dataset_name, performance_metrics_path)
            all_results.extend(results)
            column_order = ['API', 'Group', 'F1 Score', 'ROC AUC', 'False Positive Rate', 'False Negative Rate', 'Pinned ROC AUC']
        else:
            print(f"No files found for API '{api}' in dataset '{dataset_name}'.")

    combined_metrics_df = pd.DataFrame(all_results)

    combined_metrics_df = combined_metrics_df[column_order]

    # Save the combined metrics to a CSV file in the dataset directory
    output_path = os.path.join(dataset_directory, f"{dataset_name}_performance_metrics.csv")
    combined_metrics_df.to_csv(output_path, index=False)

    print(f"Performance metrics for dataset {dataset_name} saved to {output_path}")

# compute performance metrics for each dataset
compute_performance_metrics('toxigen_group')
compute_performance_metrics('toxigen_aggregate')
compute_performance_metrics('jigsaw_group')
compute_performance_metrics('jigsaw_aggregate')
compute_performance_metrics('megaspeech_group')
compute_performance_metrics('megaspeech_aggregate')

### visualize performance metrics for each dataset

def visualize_metrics(dataset_name):
    csv_file_path = f'../../output/performance_metrics/{dataset_name}/{dataset_name}_performance_metrics.csv'
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"No file found for dataset '{dataset_name}' in 'output/performance_metrics/'")

    data = pd.read_csv(csv_file_path)

    output_dir = f'../../output/performance_metrics/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)

    metrics = ['F1 Score', 'ROC AUC', 'False Positive Rate', 'False Negative Rate', 'Pinned ROC AUC']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Group', y=metric, hue='API', data=data, palette='Set2')

        plt.title(f'{dataset_name.capitalize()} - {metric}')
        plt.xlabel('Group')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        
        # Moving the legend to the top left and removing the title
        plt.legend(loc='upper left', title=None)

        jpeg_path = os.path.join(output_dir, f'{dataset_name}_{metric.replace(" ", "_")}.jpeg')
        plt.savefig(jpeg_path, format='jpeg', bbox_inches='tight')
        plt.close()

def create_heatmap(data, apis, categories, dataset_name, output_dir, metrics):
    data.replace({'Group': categories}, inplace=True)
    output_dir = os.path.join(output_dir, 'heatmap')
    os.makedirs(output_dir, exist_ok=True)
    for metric in metrics:
        if metric not in data.columns:
            continue
        heatmap_data = data.pivot(index='API', columns='Group', values=metric)
        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(heatmap_data, annot=True, fmt=".0%", cmap='rocket_r', vmin=0, vmax=1,
                         cbar_kws={'label': ''}, annot_kws={"size": 20})
        # Increase the fontsize for xticks and yticks
        plt.xticks(rotation=45, fontsize=24)
        plt.yticks(rotation=45, fontsize=24)
        ax.set_xlabel('')
        ax.set_ylabel('')
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=22)
        cbar.ax.yaxis.label.set_size(24)
        jpeg_path = os.path.join(output_dir, f'{dataset_name}_heatmap_{metric}.png')
        plt.savefig(jpeg_path, format='png', bbox_inches='tight', dpi = 300)
        plt.close()

def visualize_metrics_heatmap(dataset_name):
    csv_file_path = f'../../output/performance_metrics/{dataset_name}/{dataset_name}_performance_metrics.csv'
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"No file found for dataset '{dataset_name}' in 'output/performance_metrics/'")

    data = pd.read_csv(csv_file_path)

    output_dir = f'../../output/performance_metrics/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)

    metrics = ['F1 Score', 'ROC AUC', 'False Positive Rate', 'False Negative Rate', 'Pinned ROC AUC']
    apis = ['Amazon', 'Google', 'OpenAI', 'Microsoft']#, 'Perspective API']
    categories = ['Asian', 'Black', 'Disability', 'Female', 'Jewish', 'Latinx', 'LGBTQ', 'Muslims']
    # Create a heatmap for each metric
    create_heatmap(data, apis, categories, dataset_name, output_dir, metrics)

def create_heatmap_all(data, apis, categories, output_dir, metrics):
    # only show the groups that are present in 'categories'
    # redefine all women to female
    data['Group'] = data['Group'].replace('woman', 'female')
    data['Group'] = data['Group'].replace('jews', 'jewish')

    categories = [category.lower() for category in categories]
    data = data[data['Group'].isin(categories)]
    # change gpt to openai
    data['API'] = data['API'].replace('gpt', 'OpenAI')
    data['API'] = data['API'].replace('amazon', 'Amazon')
    data['API'] = data['API'].replace('google', 'Google')
    data['API'] = data['API'].replace('microsoft', 'Microsoft')

    data['dataset'] = data['dataset'].replace('toxigen_group', 'ToxiGen')
    data['dataset'] = data['dataset'].replace('jigsaw_group', 'Jigsaw')
    data['dataset'] = data['dataset'].replace('megaspeech_group', 'MegaSpeech')

    # Prepare a DataFrame to hold the concatenated API and dataset names
    data['dataset_api'] = data.apply(lambda row: r"$\bf{" + f"{row['dataset']}"+ r"}$: " + f"{row['API']}", axis=1)    # Iterate over each metric
    
    for i, metric in enumerate(metrics):
       
        # Create a pivot table for the current metric
        pivot_table = data.pivot(index='dataset_api', columns='Group', values=metric)

        ## Add NaN values to create separation between blocks
        nan_row = pd.DataFrame(np.nan, index=[''], columns=pivot_table.columns)
        pivot_table = pd.concat([nan_row, pivot_table.iloc[:4], nan_row, pivot_table.iloc[4:8], nan_row, pivot_table.iloc[8:]])

        # make the group Labels upper case and change lqbtq to LQBTQ
        pivot_table.columns = [x.upper() for x in pivot_table.columns]
        pivot_table.columns = pivot_table.columns.str.replace('lqbtq', 'LGBTQ')

        # only write dataset every fourth row:
        pivot_table.index = [pivot_table.index[i+1].split(':')[0] if i % 5 == 0 else pivot_table.index[i].split(':')[1] for i in range(len(pivot_table))]
        # Plot the heatmap
        plt.figure(figsize=(16, 8))
        # only plot legend for the second metric
        ax = sns.heatmap(pivot_table, annot=True, fmt=".0%", cmap='rocket_r', vmin=0, vmax=1,
                        cbar_kws={'label': ''}, annot_kws={"size": 18})
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=18)
        cbar.ax.yaxis.label.set_size(18)
        cbar.ax.xaxis.label.set_size(18)
        plt.xticks(rotation=45, fontsize=24)
        plt.yticks(rotation=0, fontsize=24)
        ax.hlines(np.arange(0.5, len(pivot_table), 5), *ax.get_xlim(), color='black', linestyle='dotted')

        ax.set_xlabel('')
        ax.set_ylabel('')
        
        jpeg_path = os.path.join(output_dir, f'heatmap_{metric}.png')
        plt.savefig(jpeg_path, format='png', bbox_inches='tight', dpi=300)
        plt.close()

def visualize_metrics_heatmap_all_datasets(average = False):
    datasets = ['toxigen_group', 'jigsaw_group', 'megaspeech_group']   
    dataframes = []
    for dataset_name in datasets:
        csv_file_path = f'../../output/performance_metrics/{dataset_name}/{dataset_name}_performance_metrics.csv'
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"No file found for dataset '{dataset_name}' in 'output/performance_metrics/'")
        df = pd.read_csv(csv_file_path)
        df['dataset'] = dataset_name
        dataframes.append(df)
    data = pd.concat(dataframes)

    output_dir = f'../../output/performance_metrics/all_datasets'
    os.makedirs(output_dir, exist_ok=True)

    metrics = ['F1 Score', 'ROC AUC', 'False Positive Rate', 'False Negative Rate', 'Pinned ROC AUC']
    apis = ['Amazon', 'Google', 'OpenAI', 'Microsoft']#, 'Perspective API']
    categories = ['Asian', 'Black', 'Disability', 'Female', 'Jewish', 'Latinx', 'LGBTQ', 'Muslim']
    # Create a heatmap for each metric
    create_heatmap_all(data, apis, categories, output_dir, metrics)    


# Example usage:
visualize_metrics_heatmap("toxigen_group")
visualize_metrics_heatmap("toxigen_aggregate")
visualize_metrics_heatmap("jigsaw_aggregate")  
visualize_metrics_heatmap("jigsaw_group")
visualize_metrics_heatmap("megaspeech_group")
visualize_metrics_heatmap("megaspeech_aggregate")

visualize_metrics_heatmap_all_datasets(average = False)