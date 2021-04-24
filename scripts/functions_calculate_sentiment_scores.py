                                                                                
############################################################################### 
# Import Python Libraries                                                       
############################################################################### 
import logging                                                                  
import os                                                                       
import sys                                                                      
import pandas as pd                                                             
import numpy as np                                                              
import inspect                                                                  
import time                                                                     
from tqdm import tqdm                                                           
from collections import Counter                                                 
from datetime import datetime                                                   
from functools import wraps                                                     
                                                                                
############################################################################### 
# Import Project Modules                                                        
############################################################################### 
from functions_decorators import *                                              
from functions_utility import *  

###############################################################################
# Python Package Settings
###############################################################################
logging.basicConfig(level=logging.INFO,
                    #filename='logging.info'
                    )
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


###############################################################################
# Functions 
###############################################################################

# ### Create Token Dictionary (Key = Token, Value = Score)

@my_timeit
def convert_list_mod_toks_scores(data, dict_sent, tok_type, mod_tok_names,
        write2file, dir_output, project_folder):
    """
    Function translates matched anchor word and modifying tokens to scores.
    
    Args:
        data:
        dict_sent:
        mod_tok_names:

    Returns:
        

    """
     
    # Create Dictionary Key = Token, Value = Sentiment Score
    sent_converter={x:y for x,y in zip(dict_sent['TokensClean'].values,
        dict_sent['Score'].values)}

    # Get Anchor Word Score
    data['ScoreAnchorWord'] = [sent_converter[x] for x in
            data['anchor_word'].values]
    
    # Get Anchor Word Normalized Score
    data['ScoreAnchorWordNormalized'] = data['ScoreAnchorWord'].values /\
            data['window_word_cnt'].values

    # Iterate Modifying Token Columns ('uncertain', 'degree', 'negator', 'modal')
    for mod_tok in mod_tok_names:

        # New Column List
        SCORELIST=[]
        SCORECNT=[]
        SCOREPROD=[]

        # Iterate Rows of Each Column
        for values in data[mod_tok].values:

            # If not empty string
            if values:
                # If values includes a comma
                if ',' not in values:
                    # Append To Score List For Single Token
                    SCORELIST.append(str(sent_converter[values]))
                    SCORECNT.append(1)
                    SCOREPROD.append(float(sent_converter[values]))
                
                # Multiple tokens
                else:
                    # Split Tokens
                    tokens=values.split(',')
                    # Return List of Tokens Converted to Scores
                    scorelist=[sent_converter[tok] for tok in tokens]
                    # Get Count of Modifying Tokens
                    SCORECNT.append(len(scorelist))
                    # Convert to numpy array
                    score_arr = np.array(scorelist)
                    # Append a str of comma separated scores.
                    SCORELIST.append(','.join([str(x) for x in scorelist]))
                
                    ######################################################
                    # Get Product of Scores
                    ######################################################
                    # Check if there are an even number of scores
                    if len(scorelist)%2==0:
                        # Identify Which Scores are negative
                        negative_scores = [x < 0 for x in scorelist]
                        # If All Negative then we need to multiply by
                        #-1 so the score remains negative
                        if all(negative_scores):
                            SCOREPROD.append(np.prod(score_arr) * -1)
                        else:
                            SCOREPROD.append(np.prod(score_arr))
                    # If Odd number scores
                    else:
                        SCOREPROD.append(np.prod(score_arr))

            # The value in the cell is null
            else:
                SCORELIST.append(0)
                SCORECNT.append(0)
                SCOREPROD.append(0)

        # Add Column to Original DataFrame
        data[mod_tok + 'score_list'] = SCORELIST
        data[mod_tok + 'score_cnt'] = SCORECNT
        data[mod_tok + 'score_product'] = SCOREPROD
        
        # Get Normalized Product Score
        word_cnt = data['window_word_cnt'].values
        data[mod_tok + 'score_norm'] = data[mod_tok + 'score_product'].values /\
                                               data[mod_tok + 'score_cnt'].values
    # Get Final Modifying Score By Multiplying Normalized Modifying Work Scores
    data['ModifyingTokensProduct'] = np.nan_to_num(data['modalscore_norm'].values, nan=1) *\
                                     np.nan_to_num(data['negatorscore_norm'].values, nan=1) *\
                                     np.nan_to_num(data['degreescore_norm'].values, nan=1) *\
                                     np.nan_to_num(data['uncertainscore_norm'].values, nan=1)
    # Get Final Anchor Word Score
    anchor_word_final_score = []
    for modscore, anchorscore in zip(data['ModifyingTokensProduct'].values,
                                     data['ScoreAnchorWordNormalized'].values):
        # If Both Scores Negative
        if modscore < 0 and anchorscore < 0:
            anchor_word_final_score.append((modscore * anchorscore) * -1)
        else:
            anchor_word_final_score.append((modscore * anchorscore))

    data['AnchorWordFinalScore'] = anchor_word_final_score

    # Write2file
    if write2file:
        filename=f'final_{tok_type}_anchor_word_window_sentiment_score.csv'
        write2csv(data, dir_output, project_folder, filename)

    # Return Data w/ New Columns
    return data
    


















