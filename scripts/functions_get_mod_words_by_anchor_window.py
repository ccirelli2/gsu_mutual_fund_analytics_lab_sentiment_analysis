"""
Description:
    Functions to find modifying words within each anchor word window.
"""


############################################################################### 
# Import Python Libraries                                                       
############################################################################### 
import logging                                                                  
import os                                                                       
import sys                                                                      
import pandas as pd                                                             
import inspect                                                                  
import time                                                                     
from tqdm import tqdm                                                           
from collections import Counter                                                 
from datetime import datetime                                                   
from functools import wraps                                                     
                                                                                
from nltk import ngrams
from nltk.tokenize import word_tokenize                                         
from nltk.tokenize import sent_tokenize                                         
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters   


############################################################################### 
# Project Modules                                                       
############################################################################### 
from functions_decorators import *
from functions_utility import *


############################################################################### 
# Package Configurations                                                       
############################################################################### 
logging.basicConfig(level=logging.INFO)
pd.set_option("max_rows", None)
pd.set_option("max_columns", None)


############################################################################### 
# Functions                                                       
############################################################################### 

@my_timeit
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
                num_grams=len(tk_tokenized)
                win_ngrams=ngrams(win_both, num_grams)
                if tuple(tk_tokenized) in win_ngrams:
                    tk_match_dict[tk].append(1)
                else:
                    tk_match_dict[tk].append(0)

    # Create DataFrame of Token Match Dict
    df_tk_match=pd.DataFrame.from_dict(tk_match_dict, orient='index').transpose()

    # Join Original Window DataFrame & Token Match DataFrame on Window Pkey
    df_final = df_windows.merge(df_tk_match, left_on='window_pkey', right_on='window_pkey')
    df_final.drop('sentences', inplace=True, axis=True)
    df_final.drop('sentences_tok', inplace=True, axis=True)

    # Add Sum Column At End
    df_final['sum_all_matches'] = df_final.iloc[:, 7:].sum(axis=1)
    
    # Quality Check
    logging.info('---- original window dataframe dimensions => {}'.format(
        df_windows.shape))
    logging.info('---- final window dataframe dimensions => {}'.format(
        df_final.shape))

    # Return Merged DataFrame
    return df_final



@my_timeit
def get_mod_tokens_by_window_pkey(data, tok_start, name):
    """
    Function to convert binary token match to a single col.
    
    Input:
        data; DataFrame
            Table containing tokens
        tok_start; Int
            Index value of the start of the columns that contain
            the token match values (1|0)
        name; Str
            Name of the type of modifying token.  Example: Uncertain
            
    
    Returns:
        Dataframe
        With sentence primary key and a single column containing
        comma separated values for each matching token.
    
    """
    # Logging
    logging.info(f'dataset => {name}\ndimension => {data.shape}\n')
    # Unique list of sentence primary keys
    WINDOW_PKEY=list(set(data['window_pkey'].values))
    # Declare Result Objects
    MODTOKS={x:[] for x in WINDOW_PKEY}
    # Iterate DataFrame By Row (returns a tuple with pos 1 = series obj)
    for row in data.iterrows():
        # Get Series
        series=row[1]
        # Get Accession & Window Primary Key
        accession_num=series[0]
        window_pkey=series[6]
        # Limit Series to Columns Containing Token Matches (1,0)
        series_lim=series[tok_start:-1]
        series_lim=series_lim[series_lim.values > 0]
        mod_tokens=series_lim.index.tolist()
        # If Series Not Empty
        if mod_tokens:
            # Append Modifying Tokens To Primary Key
            [MODTOKS[window_pkey].append(tok) for tok in mod_tokens]
    # Convert List Value to String
    for key in MODTOKS:
        MODTOKS[key] = ','.join(MODTOKS[key]) 
    # Convert Dictionary to DataFrame
    df_modtoks = pd.DataFrame({})
    df_modtoks['window_pkey'] = MODTOKS.keys()
    df_modtoks[f'{name}'] = MODTOKS.values()

    # return df_modtoks
    return df_modtoks


@my_timeit
def group_mods_by_window_pkey(df_windows, modal, negator, degree,
        uncertain, tok_start, anchor_type, write2file, dir_output,
        project_folder):                                               
    """
    Function to consolidate all of the modifying token matches into single
    columns by type.  
    --------------------------------------
    df_windows: DataFrame
        The dataframe that contains all of the mathed windows for a given
        anchor word type (ex: Positive)
    modal: DataFrame
        Matched modifying tokens in window.
    tok_start: Int
        The column number where the modifying word matches start
    ---------------------------------------
    Return
        Dataframe with 5 columns:
            window_pkey, modal matches, negator matches, degree matches,
            uncertain_matches

    """

    # Collect all matching tokens into a single column                         
    col_modal=get_mod_tokens_by_window_pkey(modal, tok_start, 'modal')         
    col_negator=get_mod_tokens_by_window_pkey(negator, tok_start, 'negator')
    col_degree=get_mod_tokens_by_window_pkey(degree, tok_start, 'degree')   
    col_uncertain=get_mod_tokens_by_window_pkey(uncertain, tok_start, 'uncertain')
                                                                               
    # Create New DataFrame                                                     
    df_final=pd.DataFrame({})                                                  
    df_final['window_pkey'] = df_windows['window_pkey'].values

    # Merge Each Collected Column on Window Primary Key                     
    frames=[col_modal, col_negator, col_degree, col_uncertain]              
    # Iterate Frames & Merge on Window Primary Key
    for frame in frames:                                                       
        df_final=df_final.merge(frame, left_on='window_pkey', right_on='window_pkey')

    if write2file:
        subfolder=create_project_folder(
                dir_output=os.path.join(dir_output, project_folder),
                name='modifying_words')
        dir_output=os.path.join(dir_output, project_folder)
        filename = f'{anchor_type}_consolidated_modifying_tokens.csv'
        write2csv(df_final, dir_output, subfolder, filename)

    return df_final   





