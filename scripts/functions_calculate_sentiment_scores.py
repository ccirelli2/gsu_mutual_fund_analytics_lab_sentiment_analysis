                                                                                
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
logging.basicConfig(level=logging.INFO)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


###############################################################################
# Functions 
###############################################################################

# ### Create Token Dictionary (Key = Token, Value = Score)

@my_timeit
def get_window_sentiment_score(data, dict_sent, tok_type, mod_tok_names,
        write2file, dir_output, project_folder):
    """
    Function translates matched anchor word and modifying tokens to scores.
    
    Args:
        data:
        dict_sent:
        mod_tok_names:

    Returns:
        

    """
    ########################################################################### 
    # Create Score Converter: Token > Score
    ########################################################################### 
    # Create Dictionary Key = Token, Value = Sentiment Score
    sent_converter={x:y for x,y in zip(dict_sent['TokensClean'].values,
        dict_sent['Score'].values)}

    # Get Anchor Word Score
    data['ScoreAnchorWord'] = [sent_converter[x] for x in
            data['anchor_word'].values]
    
    # Get Anchor Word Normalized Score
    data['ScoreAnchorWordNormalized'] = data['ScoreAnchorWord'].values /\
            data['window_word_cnt'].values

    ########################################################################### 
    # Iterate Modifying Token - Calculate Scores Single & Multiple Tokens 
    ########################################################################### 
    for mod_tok in mod_tok_names:

        # New Column List
        SCORELIST=[] 
        SCORECNT=[]
        SCOREPROD=[]

        # Iterate Rows of Each Column
        for values in data[mod_tok].values:

            # If modifying token found in cell
            if values:
                ###############################################################
                # If only one token
                ###############################################################
                if ',' not in values:
                    # Append Score, Score Count To Score Lists
                    SCORELIST.append(str(sent_converter[values]))
                    SCORECNT.append(1)
                    SCOREPROD.append(float(sent_converter[values]))
                
                ###############################################################
                # Multiple tokens
                ###############################################################
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
                    # If Odd number scores no need to multiply by -1
                    else:
                        SCOREPROD.append(np.prod(score_arr))

            # The value in the cell is null append zeros
            else:
                SCORELIST.append(0)
                SCORECNT.append(0)
                SCOREPROD.append(0)

        #######################################################################
        # Add Column to Original DataFrame
        #######################################################################
        data[mod_tok + 'score_list'] = SCORELIST
        data[mod_tok + 'score_cnt'] = SCORECNT
        data[mod_tok + 'score_product'] = SCOREPROD
        
        #######################################################################
        # Get Normalized Product Score
        #######################################################################
        word_cnt = data['window_word_cnt'].values
        data[mod_tok + 'score_norm']=data[mod_tok + 'score_product'].values /\
                                     data[mod_tok + 'score_cnt'].values

    ###########################################################################
    # Get Final Mod Score As Product of All Normalized Modifying Scores
    ###########################################################################
    data['ModifyingTokensProduct'] =\
            np.nan_to_num(data['modalscore_norm'].values, nan=1) *\
            np.nan_to_num(data['negatorscore_norm'].values, nan=1) *\
            np.nan_to_num(data['degreescore_norm'].values, nan=1) *\
            np.nan_to_num(data['uncertainscore_norm'].values, nan=1)
    

    ###########################################################################
    # Get Final Anchor Word Score
    ###########################################################################
    anchor_word_final_score = []

    for modscore, anchorscore in zip(data['ModifyingTokensProduct'].values,
                                     data['ScoreAnchorWordNormalized'].values):
        # If Both Scores Negative
        if modscore < 0 and anchorscore < 0:
            anchor_word_final_score.append((modscore * anchorscore) * -1)
        else:
            anchor_word_final_score.append((modscore * anchorscore))
    
    # Add to original dataset
    data['AnchorWordFinalScore'] = anchor_word_final_score

    # Write2file
    if write2file:
        subfolder=create_project_folder(
                dir_output=os.path.join(dir_output, project_folder),
                name='window_sentiment_scores')
        dir_output=os.path.join(dir_output, project_folder)
        filename=f'final_{tok_type}_anchor_word_window_sentiment_score.csv'
        write2csv(data, dir_output, subfolder, filename)

    # Return Data w/ New Columns
    return data
    

@my_timeit
def calculate_sentence_lvl_sentiment_score(df, write2file, dir_output,
        project_folder, token_type):
    """
    Function to calculate sentence lvl sentiment score. 
    Args:
        df:
        write2file:
        dir_output:
        project_folder:
        name:

    Returns:
        

    """
    # Calculate Score
    sent_lvl_score= df.groupby(['accession_num', 'sent_pkey'])[
            'AnchorWordFinalScore'].sum().reset_index().rename(
                    columns={'AnchorWordFinalScore':
                        f'SentenceFinal{token_type}Score'})      
    # Write to file
    if write2file:
       subfolder=create_project_folder(
                dir_output=os.path.join(dir_output, project_folder),
                name='sentence_sentiment_scores')
       dir_output=os.path.join(dir_output, project_folder)
       filename=f'sentence_sentiment_score_{token_type}.csv'
       write2csv(sent_lvl_score, dir_output, subfolder, filename)

    # return
    return sent_lvl_score




#### END
