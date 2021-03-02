############################################################################### 
# Import Python Libraries                                                       
############################################################################### 
import numpy as np                                                              
import pandas as pd                                                             
import logging                                                                  
import os                                                                       
import sys                                                                      
                                                                                
                                                                                
############################################################################### 
# Directories                                                                   
############################################################################### 
dir_repo=r'/home/cc2/Desktop/repositories/mutual_fund_analytics_lab_sentiment_analysis'
dir_scripts=os.path.join(dir_repo, 'scripts')                                      
dir_results=os.path.join(dir_repo, 'results')
[sys.path.append(x) for x in [dir_repo, dir_scripts]]                           
                                                                                
                                                                                   
############################################################################### 
# Import Project Modules                                                           
############################################################################### 
from functions_utility import *                                                    
from functions_decorators import *                                                 


############################################################################### 
# Package Conditions                                                               
############################################################################### 
logging.basicConfig(level=logging.INFO)    
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


############################################################################### 
# Data                                                               
############################################################################### 
conn, my_cursor = conn_mysql('Gsu2020!', 'mutual_fund_lab')

# Queries
query_sent_dict="""
    SELECT * FROM GSUMUTUALFUNDLAB_sentiment_dictionary_add_irreg_token_rules
    """
query_window="""
    SELECT * FROM GSUMUTUALFUNDLAB_principal_risk_legal_word_win_add_pkey
    """
# Data
sentiment_dict=load_mysql_data(conn, query_sent_dict)
data_windows=load_mysql_data(conn, query_window)

############################################################################### 
# Transformations
############################################################################### 

# Limit Sentiment to targettype
TOKEN_TYPE='Modal'

############################################################################### 
# Function Execution                                                               
############################################################################### 

def identify_mod_tokens(df_windows, sentiment_dict, token_type):
    """

    Args:
        df_windows: Dataframe; contains windows around anchor word
        sentiment_dict: DataFrame; contains modifying tokens
        token_type: Str; determines which mod token type to select

    Returns:
        Dataframe with modifying token matches joined to original
        df_windows

    """
    # Limit Sentiment Dictionary to target token type
    sentiment_dict=sentiment_dict[sentiment_dict['TokenType']==token_type]
    tokens=sentiment_dict['TokensClean'].values
    
    # Get Window Primary key
    win_pkey=df_windows['window_pkey'].values

    # Create Emtpy Dictionary w/ Keys = Tokens
    tk_match_dict={x:[] for x in tokens}
    tk_match_dict['window_pkey'] = win_pkey
    
    # Iterate Windows
    for row in df_windows.iterrows():
        series=row[1]
        # Some Windows Are Empty, So we Use Try and Except Statements
        try:
            win_left=series['window_left'].split(',')
        except AttributeError:
            wind_left=[]
        try:
            win_right=series['window_right'].split(',')
        except AttributeError:
            win_right=[]

        # Concat Windows Into One List
        win_both=win_left+win_right
        print(win_both)
        
        # Iterare Tokens
        for tk in tokens:
            if len(tk.split(' ')) == 1:
                # Check if token in window
                if tk in win_both:
                    tk_match_dict[tk].append(1)
                else:
                    tk_match_dict[tk].append(0)
            else:
                # Create Ngram of window
                tk_tokenized=tk.split(' ')
                ngrams=len(tk_tokenized)
                win_ngrams=ngrams(win_both, ngrams)
                if tuple(tk_tokenized) in win_ngrams:
                    tk_match_dict[tk].append(1)
                else:
                    tk_match_dict[tk].append(0)
    
    # Create DataFrame of Token Match Dict
    df_tk_match=pd.DataFrame.from_dict(tk_match_dict, orient='index').transpose()
    
    # Join Original Window DataFrame & Token Match DataFrame on Window Pkey
    df_final = df_windows.merge(df_tk_match, left_on='window_pkey', right_on='window_pkey')

    # Quality Check
    logging.info('---- original window dataframe dimensions => {}'.format(
        df_windows.shape))
    logging.info('---- final window dataframe dimensions => {}'.format(
        df_final.shape))
    
    # Return Merged DataFrame
    return df_final 
    

# Execution
result=identify_mod_tokens(data_windows, sentiment_dict, TOKEN_TYPE)
print(result.head())






















