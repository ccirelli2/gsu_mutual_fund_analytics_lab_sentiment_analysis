
################################################################################
# Import Python Libraries
################################################################################
import logging
import os
import sys
import pandas as pd
import numpy as np
import string
from datetime import datetime

                                                                                
############################################################################### 
# Set up logging parameters & Package Conditions                                
############################################################################### 
logging.basicConfig(level=logging.INFO)                                         
pd.set_option('display.max_columns', None)                                      
pd.set_option('display.max_rows', None)                                         
                                       

############################################################################### 
# Declare Variables                                                             
############################################################################### 
dir_repo = r'/home/cc2/Desktop/repositories/mutual_fund_analytics_lab_sentiment_analysis' 
dir_scripts = os.path.join(dir_repo, 'scripts')                                 
sys.path.append(dir_repo)
sys.path.append(dir_scripts)


############################################################################### 
# Import Project Modules                                                        
############################################################################### 
from functions_utility import *                                                 
from functions_decorators import *      


############################################################################### 
# Functions                                                        
############################################################################### 


def get_word_window(accession_num, sentence_pkey, sentence, anchor_word,
        window_width):                                     
    """                                                                            
    Function to obtain window of tokens around anchor word.

    Args:                                                                          
        sentence: Str;
            sentence within which the anchor word was found
        anchor_word: Str;
            anchor word
        window_width: Int;
            aumber of tokens around anchor word to capture within window.
                                                                                   
    Returns:                                                                       
        Dataframe containing the accession number, sentence primary key,
        original sentence, tokenized sentence,
        anchor word, position of each anchor word and the windows to the 
        left and right of the anchor word.
                                                                                   
    """                                                                            
    ########################################################################### 
    # Preprocess & Tokenize Sentence                                               
    ########################################################################### 
    sent_lower=sentence.lower()                                                    
    punk=string.punctuation                                                        
    sent_nopunk=''.join(list(map(lambda x: x if x not in punk else '', sent_lower)))
    sent_tokens=sent_nopunk.split(' ')                                             
    ########################################################################### 
    # Get Location of Anchor Word                                                      
    ########################################################################### 
    # Record Len of tokenized sentence                                             
    sent_len = len(sent_tokens)                                                    
    # Convert to Numpy Array                                                       
    sent_tokens_arr=np.array(sent_tokens)                                          
    # Get Index Position of Anchor Word (Single Gram Only)                         
    # np.where returns a tuple                                                     
    anchor_pos = np.where(sent_tokens_arr==anchor_word)[0].tolist()                     
    logging.debug(f'---- anchor position => {anchor_pos}')                         
    ###########################################################################
    # Get Window For Each Instance of Anchor Word in Sentence 
    ###########################################################################
    results_dict={
        'accession_num':[], 'sentence_pkey':[], 'sentences':[],
        'sentences_tok':[], 'anchor_word_pos':[], 'window_left':[],         
        'anchor_word':[], 'window_right':[]} 


    # Iterate Position of Each Anchor Word
    for pos in anchor_pos:
        # Get Left Window (return 0 if pos negative)
        pos_left = (lambda x: x if x >= 0 else 0)(pos-(window_width))
        window_left=sent_tokens[pos_left: pos]
        # Get Right Window (return len(sent_tokens) if > len(sent_tokens)
        pos_right = (lambda x: x+1 if x <= sent_len else sent_len)(pos+window_width)
        window_right=sent_tokens[pos+1 : pos_right]                                         
        # Add Results to Results Dictionary
        results_dict['accession_num'].append(accession_num)
        results_dict['sentence_pkey'].append(sentence_pkey)
        results_dict['sentences'].append(sentence)
        results_dict['sentences_tok'].append(','.join(sent_tokens))
        results_dict['anchor_word_pos'].append(pos)
        results_dict['anchor_word'].append(anchor_word)
        results_dict['window_left'].append(','.join(window_left))
        results_dict['window_right'].append(','.join(window_right))
        # Logging
        logging.debug(f'--- sentence toks => {sent_tokens}')
        logging.debug(f'--- left window => {window_left}')
        logging.debug(f'--- anchor => {anchor_word}')
        logging.debug(f'--- right window => {window_right}')
    ###########################################################################
    # Return Results 
    ###########################################################################
    df_results = pd.DataFrame(results_dict)
    return df_results                          
                  


def get_anchor_word_window_by_sent(data, anchor_word_source, window_width,
        dir_output, project_folder, write2file):
    """ 
    Function to get the window of n words around each anchor word.
    
    The input are sentences that have been identified to contain one of three
    types of anchor words (positive, negative, legal).

    This function obtains a window of n words around each anchor word.


    Args:
        data: DataFrame;
            Sentences that contain anchor words.
        anchor_word_source: String;
            The type of anchor word (positive, negative, legal).

    Returns:
    df_results: DataFrame; 
        It returns a row for each anchor word with the following
        structure: accession_num, sentence_pkey, anchor_word_type, left window,
        anchor, right_window.
        

    """
    ###########################################################################
    # Iterate Each Row of DataFrame (iterrow returns tuple w/ [1] = series)
    ###########################################################################
    frames = []

    for row in data.iterrows():
        # Series
        series=row[1]
        # Get Primary Keys
        accession_num=series['accession_num']
        sentence_pkey=series['sent_pkey']
        sentence=series['sentences']
        #######################################################################
        # Remove identifier keys (none tokens)
        #######################################################################
        token_keys=series[8:-1]
        # Double check that no identifier keys remain
        list_non_tok_cols=['Unnamed: 0', 'Unnamed: 0.1', 'acceson_num',
                'sent_pkey', 'num_toks', 'num_chars', 'unverified_match',
                'sum_matches']
        if [x for x in list_non_tok_cols if x in series.index]:
            token_keys=series[
                    (series.index != 'Unnamed: 0') &
                    (series.index != 'Unnamed: 0.1') &
                    (series.index != 'accession_num') &
                    (series.index != 'sent_pkey') &
                    (series.index != 'sentences') &
                    (series.index != 'num_toks') &
                    (series.index != 'num_chars') &
                    (series.index != 'unverified_match') &
                    (series.index != 'sum_matches')]

        # Limit Results to only tokens with at min 1 match
        token_matches=token_keys[token_keys > 0]

        #######################################################################
        # Obtain Window Around Anchor Words 
        #######################################################################
        # Iterate Keys And Return Key Value
        for anchor_word in token_matches.index:
            anchor_window=get_word_window(
                    accession_num, sentence_pkey, sentence, anchor_word,
                    window_width)
            frames.append(anchor_window)

    ###########################################################################
    # Concat Results & Write to file
    ###########################################################################
    df_concat = pd.concat(frames)

    if write2file:
        filename=f'{anchor_word_source}_anchor_words_windows_size_{window_width}.csv'
        write2csv(df_concat, dir_output, project_folder, filename)
    # Return Results
    return df_concat



















