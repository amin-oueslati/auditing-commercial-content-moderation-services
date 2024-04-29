import pandas as pd
import numpy as np
from apiGPT import APIgpt
from apiAmazon import APIamazon
from tqdm import tqdm
import asyncio


def hate_evaluation(amazon, google, gpt, microsoft):
    """
    INPUT: Pandas Dataframe of Categories and Scores from Content Moderation
    OUTPUT: Pandas Dataframe of scores for each API and it's deviation from the mean, overall mean, overall standard deviation
    """
    labels = ("Amazon", "Google", "GPT", "Microsoft")
    scores = [amazon.iat[1,1], google.iat[1,1], gpt.iat[2,1], microsoft.iat[2,1]] #Getting score for category relating to "hate"
    mean = np.mean(scores)
    dev = [abs(scores[i]-mean) for i in range(4)]
    return pd.DataFrame(zip(scores, dev), index = labels, columns = ["Score", "Delta_mean"]), mean, np.std(scores)


def sexual_content_evaluation(amazon, google, gpt, microsoft):
    """
    INPUT: Pandas Dataframe of Categories and Scores from Content Moderation
    OUTPUT: Pandas Dataframe of scores for each API and it's deviation from the mean, overall mean, overall standard deviation
    """
    labels = ("Amazon", "Google", "GPT", "Microsoft")
    scores = [amazon.iat[5,1], google.iat[15,1], gpt.iat[7,1], microsoft.iat[0,1]] #Getting score for category relating to "sexual-content"
    mean = np.mean(scores)
    dev = [abs(scores[i]-mean) for i in range(4)]
    return pd.DataFrame(zip(scores, dev), index = labels, columns = ["Score", "Delta_mean"]), mean, np.std(scores)

def hate_evaluation_gpt(prmpt_list, labels):
    """
    INPUT: Pandas Dataframe of Categories and Scores from Content Moderation
    OUTPUT: Pandas Dataframe of scores for each API and it's deviation from the mean, overall mean, overall standard deviation
    """
    scores = []
    decisions = []
    print('Evaluating GPT Scores')
    size = 20
    prmpt_chunks = [prmpt_list[pos:pos + size] for pos in range(0, len(prmpt_list), size)]
    prmpt_chunks = prmpt_chunks[0:len(prmpt_chunks)-1] # drop the last chunk because of varying size
    for prmpt_c in tqdm(prmpt_chunks):
        prmpt_c = prmpt_c.to_list()
        gpt_list = APIgpt(prmpt_c)
        for i in range(size):
            gpt_output = gpt_list[i]
            max_column = gpt_output['category_scores'].idxmax()
            idx  = gpt_output.index.get_loc(max_column)
            score = gpt_output.iat[idx,1]
            dec = bool(gpt_output.iat[idx,0])
            scores.append(score)
            decisions.append(dec)
        #other_dec.append(gpt.iat[0].to_list())
        #other_scr.append(gpt.iat[1].to_list())
    df = pd.DataFrame(
    {'Prompt': [prmpt for prmpt_list in prmpt_chunks for prmpt in prmpt_list],
     'Score': scores,
     'Decision': decisions,
        'Label': labels[0:len(scores)]
    })    

    return df