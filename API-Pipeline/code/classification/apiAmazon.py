import boto3
import pandas as pd
from apiGPT import APIgpt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
def APIamazon(prompt):
    """
    INPUT: string of text that will be evaluated
    OUPUT: Pandas Data Frame with the categories and values evaluated in content moderation AND overall flagging
    """
    client = boto3.client("comprehend", region_name = 'us-east-1', aws_access_key_id='AKIA4GG7FM6JNYYCHKSL', aws_secret_access_key='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') # Specification of access key
    response = client.detect_toxic_content(
        TextSegments=[
            {
                'Text': prompt
            },
        ],
        LanguageCode='en'#set language, possible other languages:'es'|'fr'|'de'|'it'|'pt'|'ar'|'hi'|'ja'|'ko'|'zh'|'zh-TW'
    )
    return pd.DataFrame(response['ResultList'][0]['Labels']), response['ResultList'][0]['Toxicity'] > 0.5

def amazon_worker(text):
    try:
        amazon_list, is_toxic = APIamazon(text)
        max_column = amazon_list['Score'].idxmax()
        idx = amazon_list.index.get_loc(max_column)
        score = amazon_list.iat[idx, 1]
        dec = int(1 if is_toxic else 0) # leverage true/false to determine classification
        return score, dec
    except Exception as e:
        logging.info(f"Error processing text: {text}. Error: {e}")
        return None, None


def hate_evaluation_amazon(text_list, labels, ids, max_workers=10):
    logging.info('Generating Amazon Scores')

    all_scores = [None] * len(text_list)
    all_labels = [None] * len(text_list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(amazon_worker, text): idx for idx, text in enumerate(text_list)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            idx = futures[future]
            score, dec = future.result()
            all_scores[idx] = score
            all_labels[idx] = dec

    df = pd.DataFrame({
        'ids': ids[:len(all_scores)],
        'text': text_list,
        'score': all_scores,
        'predicted_label': all_labels,
        'true_label': labels[:len(all_scores)]
    })    

    return df