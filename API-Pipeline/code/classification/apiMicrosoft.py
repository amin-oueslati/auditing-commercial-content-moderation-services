import http.client, urllib.request, urllib.parse, urllib.error, base64
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time 
def APImicrosoft(prompt):
    """
    INPUT: string of text that will be evaluated
    OUPUT: Pandas Data Frame with the categories and values evaluated in content moderation AND overall flagging
    """
    headers = {
        # Request headers
        'Content-Type': 'text/plain',
        'Ocp-Apim-Subscription-Key': 'f53b0b43ffc445c084ee1a2afda3b8f3', #API-Key Access
    }

    params = urllib.parse.urlencode({
        # Request parameters
        #'autocorrect': '{boolean}',
        #'PII': '{boolean}',
        #'listId': '{string}',
        'classify': 'true',
        'language': 'eng',
    })

    conn = http.client.HTTPSConnection('germanywestcentral.api.cognitive.microsoft.com')
    prompt = prompt.encode('utf-8')
    conn.request("POST", "/contentmoderator/moderate/v1.0/ProcessText/Screen?%s" % params, prompt, headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()

    data_dict = json.loads(data.decode('utf-8'))
   
    classify = data_dict["Classification"]
    
    new = (('sexually_explicit', classify['Category1']['Score']), 
           ('sexually_suggestive', classify['Category2']['Score']), 
           ('offensive', classify['Category3']['Score'])
    )
    return pd.DataFrame(columns = ['category', 'score'], data = new), classify['ReviewRecommended']

def microsoft_worker(text):
    try:
        microsoft_list, review_recommended = APImicrosoft(text) 
        max_column = microsoft_list['score'].idxmax()
        idx = microsoft_list.index.get_loc(max_column)
        score = microsoft_list.iat[idx, 1]
        dec = int(1 if review_recommended else 0)
        return score, dec
    except Exception as e:
        print(f"Error processing text: {text}. Error: {e}")
        return None, None


def hate_evaluation_microsoft(text_list, labels, ids, max_workers=5):
    print('Generating Microsoft Scores')

    all_scores = [None] * len(text_list)
    all_labels = [None] * len(text_list)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(microsoft_worker, text): idx for idx, text in enumerate(text_list)}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            idx = futures[future]
            score, dec = future.result()
            all_scores[idx] = score
            all_labels[idx] = dec

    df = pd.DataFrame({
        'id': ids[:len(all_scores)],
        'text': text_list,
        'score': all_scores,
        'predicted_label': all_labels,
        'true_label': labels[:len(all_scores)]
    })    

    return df
