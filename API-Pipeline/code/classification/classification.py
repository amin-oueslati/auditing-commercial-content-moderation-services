import os
#os.chdir('code/classification')
import sys 
import pandas as pd
import numpy as np

from apiGPT import hate_evaluation_gpt
from apiAmazon import hate_evaluation_amazon
from apiGoogle import hate_evaluation_google
from apiMicrosoft import hate_evaluation_microsoft
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

auth_token = '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'

# Initialize logging and set up log file
if not os.path.exists('../../output/log'):
    os.makedirs('../../output/log')

class ExcludeHttpsFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith('HTTPS')

date = pd.Timestamp.now().strftime('%Y-%m-%d')
time = pd.Timestamp.now().strftime('%H-%M')
logname = f'../../output/log/API_classification_{date}_{time}.log'
logging.basicConfig(filename=logname,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("Running Classification API Pipeline")
logger = logging.getLogger()
logger.addFilter(ExcludeHttpsFilter())

# input and output paths
input_path = '../../input/data/'
output_path = '../../output/'
### data loading

#os.path.abspath()


# load pre-processed toxigen dataset
toxigen_group = pd.read_csv(input_path + 'toxigen_human_annotated_group.csv')
toxigen_aggregate = pd.read_csv(input_path + 'toxigen_human_annotated_aggregate.csv')

# load pre-processed jigsaw dataset
jigsaw_group = pd.read_csv(input_path + 'jigsaw_group.csv')
jigsaw_aggregate = pd.read_csv(input_path + 'jigsaw_aggregate.csv')

# load pre-processed megaspeech dataset

megaspeech_group = pd.read_csv(input_path + 'megaspeech_group.csv')
megaspeech_aggregate = pd.read_csv(input_path + 'megaspeech_aggregate.csv')

### prediction

def api_pred(df, sampl=1, api = 'all'):
    
    sampled_df = df.sample(frac=sampl)

    if api == 'all':
        results = {
            'gpt': hate_evaluation_gpt(sampled_df.text, sampled_df.true_label, sampled_df.id),
            'amazon': hate_evaluation_amazon(sampled_df.text, sampled_df.true_label, sampled_df.id),
            'google': hate_evaluation_google(sampled_df.text, sampled_df.true_label, sampled_df.id),
            'microsoft': hate_evaluation_microsoft(sampled_df.text, sampled_df.true_label, sampled_df.id, max_workers=10)
        }

    elif api == 'gpt':
        results = {
            'gpt': hate_evaluation_gpt(sampled_df.text, sampled_df.true_label, sampled_df.id)
        }

    elif api == 'amazon':
        results = {
            'amazon': hate_evaluation_amazon(sampled_df.text, sampled_df.true_label, sampled_df.id)
        }

    elif api == 'google':
        results = {
            'google': hate_evaluation_google(sampled_df.text, sampled_df.true_label, sampled_df.id)
        }

    elif api == 'microsoft':
        results = {
            'microsoft': hate_evaluation_microsoft(sampled_df.text, sampled_df.true_label, sampled_df.id, max_workers=10)
        }                    

    return results

def classify_group(df, dataset_name, sampl=1, api = 'all'):
    groups = df.group.unique()

    for idx, group in enumerate(groups):
        logging.info(f"Processing group {idx+1}/{len(groups)}: {group}")
        df_group = df[df.group == group]
        
        # see if a file already exists
        if os.path.exists(f'../../output/API_classification/{api}/{dataset_name}/{api}_{dataset_name}_{group}.csv'):
            df_existing = pd.read_csv(f'../../output/API_classification/{api}/{dataset_name}/{api}_{dataset_name}_{group}.csv')
            # see if all ids were processed and have a result
            if len(df_existing) == len(df_group.sample(frac=sampl)):
                logging.info("All IDs were already processed and have a result. Skipping.")
                continue
            else:
                logging.info(f"Resuming processing for {df_existing.predicted_label.isnull().sum()} IDs")
                # find the IDs that were not processed and the ones that are in df_group but not in df_existing
                df_group = df_group[df_group.id.isin(df_existing.id) == False]
    
        results = api_pred(df_group, sampl=sampl, api=api)

        file_paths = {
            'gpt': f'../../output/API_classification/gpt/{dataset_name}/gpt_{dataset_name}_{group}.csv',
            'amazon': f'../../output/API_classification/amazon/{dataset_name}/amazon_{dataset_name}_{group}.csv',
            'google': f'../../output/API_classification/google/{dataset_name}/google_{dataset_name}_{group}.csv',
            'microsoft': f'../../output/API_classification/microsoft/{dataset_name}/microsoft_{dataset_name}_{group}.csv'
        }
        for api_name, df_ev in results.items():
            output_file = file_paths[api_name]
            if os.path.exists(output_file):
                df_existing = pd.read_csv(output_file)
                df_ev = pd.concat([df_existing, df_ev], ignore_index=True)
            else:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                df_ev.to_csv(output_file)

            # logging 
            logging.info(f"Saved {api_name} results for group {group} to {output_file}")
            logging.info(f"Results: {df_ev.predicted_label.value_counts()}")
            logging.info("")


def classify_aggregate(df, dataset_name, sampl=1, api = 'all'):
    # see if a file already exists
    if os.path.exists(f'../../output/API_classification/{api}/{dataset_name}/{api}_{dataset_name}.csv'):
        df_existing = pd.read_csv(f'../../output/API_classification/{api}/{dataset_name}/{api}_{dataset_name}.csv')
        # see if all ids were processed and have a result
        
        if len(df_existing) == len(df.sample(frac=sampl)):
            logging.info("All IDs were already processed and have a result. Skipping.")
            return
        else:
            logging.info(f"Resuming processing for {df_existing.predicted_label.isnull().sum()} IDs")
            df = df[df.id.isin(df_existing.id) == False]
    
    logging.info("Processing aggregate dataset")
   
    results = api_pred(df, sampl, api=api)  # Call the API prediction function on the entire DataFrame

    # Define the output file paths for each API
    file_paths = {
        'gpt': f'../../output/API_classification/gpt/{dataset_name}/gpt_{dataset_name}.csv',
        'amazon': f'../../output/API_classification/amazon/{dataset_name}/amazon_{dataset_name}.csv',
        'google': f'../../output/API_classification/google/{dataset_name}/google_{dataset_name}.csv',
        'microsoft': f'../../output/API_classification/microsoft/{dataset_name}/microsoft_{dataset_name}.csv'
    }

    # Create directories if they don't exist
    for output_file in file_paths.values():
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the results to CSV files
    for api_name, df_ev in results.items():
        if os.path.exists(file_paths[api_name]):
            df_existing = pd.read_csv(file_paths[api_name])
            df_ev = pd.concat([df_existing, df_ev], ignore_index=True)
        else:
            output_file = file_paths[api_name]
            df_ev.to_csv(output_file)
    # logging 
    logging.info(f"Saved {api_name} results for aggregate dataset to {output_file}")
    logging.info(f"Results: {df_ev.predicted_label.value_counts()}")
    logging.info("")
### run predictions

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        logging.info("Please provide arguments as follows: classification.py --dataset <dataset_name> --sample_frac <sample_frac> --api <api_name>")
        return
    if args[0] == '--dataset':
        dataset_name = args[1]
    else:
        logging.exception("Please provide dataset name with the argument --dataset")
        return
    if args[2] == '--sample_frac':
        sample_frac = float(args[3])
    else:
        logging.exception("Please provide sample fraction with the argument --sample_frac")
        return
    if len(args) > 4 and args[4] == '--api':
        api = args[5]
    else:
        api = 'all'

    if dataset_name == 'toxigen':
        classify_aggregate(toxigen_aggregate, 'toxigen_aggregate', sample_frac, api)
        classify_group(toxigen_group, 'toxigen_group', sample_frac, api)
    elif dataset_name == 'jigsaw':
        classify_aggregate(jigsaw_aggregate, 'jigsaw_aggregate', sample_frac, api)
        classify_group(jigsaw_group, 'jigsaw_group', sample_frac, api)
    elif dataset_name == 'megaspeech':
        classify_aggregate(megaspeech_aggregate, 'megaspeech_aggregate', sample_frac, api)
        classify_group(megaspeech_group, 'megaspeech_group', sample_frac, api)
    else:  
        logging.exception("Please provide valid dataset name. Choose from toxigen, jigsaw, megaspeech")
        return


if __name__ == '__main__':
    main()
