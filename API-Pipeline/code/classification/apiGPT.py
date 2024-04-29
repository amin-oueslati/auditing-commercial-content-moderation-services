from openai import OpenAI
import pandas as pd
from tqdm import tqdm


def APIgpt(prompt):

    """
    INPUT: string of text that will be evaluated
    OUPUT: Pandas Data Frame with the categories and values evaluated in content moderation AND overall flagging
    """

    client = OpenAI() #checking for Access Key 

    output = client.moderations.create(input=prompt)
    output_dict = output.model_dump()['results']#transforms GPT4 Output in dictionary with relevent entries
    output_df = []
    for i in range(len(output_dict)):
        output_df.append(pd.DataFrame(output_dict[i]))
    #flagged = output_dict.pop('flagged')

    return output_df

def hate_evaluation_gpt(text_list, labels, ids):
    """
    """
    scores = []
    predicted_label = []
    print('Generating GPT Scores')
    size = 20
    text_chunks = [text_list[pos:pos + size] for pos in range(0, len(text_list), size)]
    text_chunks = text_chunks[0:len(text_chunks)-1] # drop the last chunk because of varying size
    for text_c in tqdm(text_chunks):
        text_c = text_c.to_list()
        gpt_list = APIgpt(text_c)
        for i in range(size):
            gpt_output = gpt_list[i]
            max_column = gpt_output['category_scores'].idxmax()
            idx  = gpt_output.index.get_loc(max_column)
            score = gpt_output.iat[idx,1]
            is_hate_speech = gpt_output.iat[idx, 2]  # third column captures whether content is classified as potentially harmful
            dec = int(1 if is_hate_speech else 0) 
            scores.append(score)
            predicted_label.append(dec)
        #other_dec.append(gpt.iat[0].to_list())
        #other_scr.append(gpt.iat[1].to_list())
    df = pd.DataFrame(
    { 'ids': ids[0:len(scores)],
     'text': [text for text_list in text_chunks for text in text_list],
     'score': scores,
     'predicted_label': predicted_label,
        'true_label': labels[0:len(scores)]
    })    

    return df