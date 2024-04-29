from google.cloud import language
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

def APIgoogle(prompt, client) -> language.ModerateTextResponse:
    """
    INPUT: string of text that will be evaluated
    OUPUT: Pandas Data Frame with the categories and values evaluated in content moderation
    """

    document = language.Document(
        content=prompt,
        type_=language.Document.Type.PLAIN_TEXT
    )
    response = client.moderate_text(document=document)
    def confidence(category: language.ClassificationCategory) -> float:
        return category.confidence

    columns = ['category', 'score'] #label setting
    categories = sorted(response.moderation_categories, key=confidence, reverse=True)
    data = ((category.name, category.confidence) for category in categories) #tuple of type ((category1, score1), ..., (categoryN,scoreN))
    return pd.DataFrame(columns=columns, data=data)

# Worker Function
def google_worker(text, client):
    google_list = APIgoogle(text, client)
    # just take the first 6 categories of hate 
    #google_list = google_list['Derogatory']
    #print(google_list)
    max_column = google_list['score'].idxmax()
    idx = google_list.index.get_loc(max_column)
    score = google_list.iat[idx, 1]
   # score = google_list[google_list['category'] =='Derogatory']['score']
    
    dec = int(score > 0.5) # no classification provided by API, need to set a threshold
    return score, dec

# Evaluation Function
def hate_evaluation_google(text_list, labels, ids, max_workers=1):
    print('Generating Google Scores')

    all_scores = [None] * len(text_list)
    all_labels = [None] * len(text_list)

    client = language.LanguageServiceClient(client_options={'api_key': 'AIzaSyCjt7g7LT1_Odr9DZCodOgs7_yqYmaSzv0'}) #access key details

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {executor.submit(google_worker, text, client): idx for idx, text in enumerate(text_list)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            idx = futures[future]
            try:
                score, dec = future.result()
                all_scores[idx] = score
                all_labels[idx] = dec
            except Exception as e:
                print(f"Error processing index {idx}: {e}")

    df = pd.DataFrame({
        'ids': ids,
        'text': text_list,
        'score': all_scores,
        'predicted_label': all_labels,
        'true_label': labels
    })    

    return df