###############################################################################
# Import Python Libraries
###############################################################################
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import pandas as pd

import os
import sys
import string
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

###############################################################################
# Import Project Modules 
###############################################################################
from functions_utility import *
from functions_decorators import *

###############################################################################
# Functions
###############################################################################


def clean_sentence(sent):
    """
    Function that removes punctuation from sentence, tokenizes sent
    and lemmatizes tokens

    Args:
        sent:

    Returns:

    String object no punctuation.
    """
    # Remove New Line Characters
    sent = sent.replace('\\n', ' ')
    
    # Remove|Replace Punctuation
    punctuation = list(string.punctuation)  
    punkt=list(string.punctuation)
    punkt.remove('-')
    punkt_nospace=[punkt[6], punkt[-5]]     # ["'", '`']
    punkt_space=[x for x in punkt if x not in punkt_nospace]    
    # Replace punctuation with no-space
    sent_nopunkt=''.join(list(map(lambda x: x if x not in punkt_nospace else '', sent)))
    # Replace punctuation with space
    sent_nopunkt=''.join(list(map(lambda x: x if x not in punkt_space else ' ', sent_nopunkt)))
    # Return Clean Sentence
    return sent_nopunkt

    
def clean_tok_sentence(sent, lemmatize):
    # Clean Sentence
    sent_nopunkt=clean_sentence(sent)
    # Tokenize
    sent_tok = word_tokenize(sent_nopunkt)
    # Lemmatize
    if lemmatize:
        lemmer = WordNetLemmatizer()
        sent_tok = [lemmer.lemmatize(x) for x in sent_tok]
    # Return Results
    return sent_tok


def get_window_by_direction(anchor_word, window_direction, window_size,
        sent_tokens):
    """
    Get window around anchor word.
    Args:
        window_direction: Str; either right or left of anchor word
        anchor_word: Str; center of window.
        window_size: Int; must be positive and reps num tokens in window.
        sent_tokens: List; tokenized sentence.

    Returns:
       list object containing  

    """
    ###########################################################################
    # Identify Index Location of Anchor Word
    ###########################################################################

    # Get Lenth of Sentence
    sent_len=len(sent_tokens)

    # Convert sent_tokens to numpy array
    sent_arr=np.array(sent_tokens)
    # Get Position of Anchor Word (Result can be more than 1 match)
    anchor_pos=np.where(sent_arr==anchor_word)[0]

    ###########################################################################
    # Iterate Each Anchor Word Found & Get Window
    ###########################################################################
    # Window list
    window_list=[]

    # If Anchor word found
    if anchor_pos.size:
        # Get Right Window
        if str(window_direction).lower() == 'right':
            # Get Range Right Window (x is = the anchor pos plus window)
            for pos_n in anchor_pos:
                pos_right = (lambda x: x+1 if x <= sent_len-1 else sent_len)(
                        pos_n + window_size)
                # Window:
                right_window=sent_tokens[pos_n + 1 : pos_right]
                # Append Window To Result Object
                window_list.append(right_window)
        # Get Left Window
        elif str(window_direction).lower() == 'left':
            for pos_n in anchor_pos:
                pos_left = (lambda x: x if x >= 0 else 0)(pos_n-window_size)
                # Get Window
                left_window=sent_tokens[pos_left : pos_n]
                # Append Window
                window_list.append(left_window)
        # Get Both
        elif str(window_direction).lower() == 'both':
            for pos_n in anchor_pos:
                pos_right = (lambda x: x+1 if x <= sent_len-1 else sent_len)(
                        pos_n + window_size)
                # Window:
                right_window=sent_tokens[pos_n + 1 : pos_right]
                # Append Window To Result Object
                window_list.append(right_window)
            for pos_n in anchor_pos:
                pos_left = (lambda x: x if x >= 0 else 0)(pos_n-window_size)
                # Get Window
                left_window=sent_tokens[pos_left : pos_n]
                # Append Window
                window_list.append(left_window)
                
    ###########################################################################
    # Return Window List        
    ###########################################################################
    logging.debug(f'---- number of anchor windows found => {len(window_list)}')
    logging.debug(f'---- windows => {window_list}')
    return window_list



def irregular_token_conditions(irregular_token, sent_tokens, window_direction,
        window_size, anchor_word, targets, token_condition=1):
    """
    Test whether the existence of target tokens around our irregular token
    results in counting or not conting the irregular token.

    # Step1: Check if anchor word in text.  If so, get window based on
        direction (left of anchor, right of anchor, both)
    # Step2: Check if target tokens are in the window.
    # Step3: Determine whether a match or the absence therefore results in
        counting or not the irregular token

    # Condition Interpretation
        - if condition exists, don't count token

    Args:
        irregular_token: Str; token for which we are testing the condition.
        sent_tokens: List; list of tokens representing the preprocessed (cleaned)
            tokenized text.
        window_type: Str; one of forward, backward, both
            Indications in which direction around the anchor to create the
            windwow.
        window_size: Int; number of tokens to consider.  Ex: if 2, and
            window_type==both, then a window of two tokens will be created
            in front of and behind of the anchor token.
        anchor: Str; establishes the position of the window. *Note: this need
            not be the irregular token as we could want to find the irregular
            token in the window of another token.
        targets: List; list of target tokens to find in the window.
        token_condition: Str; either 1=confirm or 0=reverse.
            Default is if the condition applies don't count.
            If reverse, then if condition count.

    Returns:
        An object with a 1 for count and 0 for do not count.
        This intenger is the value that will replace whatever is in the
        current table for this particular sentence and count.
    """
    ###########################################################################
    # Get Window
    ###########################################################################
    # Logging
    logging.debug('---- if condition do not count irregular token => {}'.format(
        token_condition))

    # Assert Input
    assert isinstance(sent_tokens, list), "Sent tokens should be list"
    assert isinstance(targets, list), "Sent tokens should be list"

    # Get Window
    token_window=get_window_by_direction(
            anchor_word=anchor_word,
            window_direction=window_direction,
            window_size=window_size,
            sent_tokens=sent_tokens)

    
    
    ###########################################################################
    # Find Target Token(s) in Window
    ###########################################################################
    target_token_match=False
    # If Window Found
    if token_window:
        # Find target tokens in window
        for window in token_window:
            for target in targets:
                if target in window:
                    target_token_match=True

    # Logging
    logging.debug(f'---- target tokens to match => {targets}')
    logging.debug(f'---- condition exists => {target_token_match}')

    ###########################################################################
    # Determine how condition is recognized
    ###########################################################################
    if token_condition==1:
        # If condition exists
        if target_token_match:
            # Return false as we don't want to count token
            target_token_match=0   # 0 means don't count
        else:
            target_token_match=1

    else:
        if target_token_match:
            # Return false as we don't want to count token
            target_token_match=1
        else:
            target_token_match=0

    ###########################################################################
    # Function Get Forward Window
    ###########################################################################
    logging.debug('---- irregular token should be counted => {}'.format(
        target_token_match))
    return token_window, target_token_match



@my_timeit
def execute_irregular_token_condition_function(data_matched_sent,
        sentiment_dict, token_type):

    # Subset dataframe to irregular tokens
    sentiment_dict_irreg=sentiment_dict[
            (sentiment_dict['Irregular']==1) &
            (sentiment_dict['TokenType']==token_type)]
    # Get List of Irregular Tokens
    irregular_tokens = sentiment_dict_irreg['TokensClean'].values.tolist()
    
    # Limit to those tokens in matched sent columns
    irregular_tokens = [x for x in irregular_tokens if x in
            data_matched_sent.columns.values.tolist()]
    if irregular_tokens:
        logging.info(f'---- applying condition for the following irreg tokens => {irregular_tokens}')
    else:
        logging.info(f'---- no irregular tokens found in sentences')

    # Lists to capture data on affected columns and rows 
    accession_num_list=[]
    sent_pkey_list=[]
    sent_dirty_list=[]
    sent_clean_list=[]
    irreg_tok_list=[]
    pos_tok_list=[]
    pos_index_list=[]
    condition_list=[]
    anchor_token_list=[]
    target_tokens_list=[]
    window_size_list=[]
    window_direction_list=[]
    window_list=[]
    original_match_val=[]
    new_match_val=[]

    ###########################################################################
    # Get Token Column & Meta Data
    ###########################################################################
    for irreg_tok in irregular_tokens:
        # Get Sentence & Irregular Token Columns
        accession_num_col=data_matched_sent['accession_num'].values
        sent_pkey_col=data_matched_sent['sent_pkey'].values
        sent_dirty_col=data_matched_sent['sentences'].values
        irreg_tok_col=data_matched_sent[irreg_tok].values
        
        # Get Token MetaData From Sentiment Dictionary
        metadata_irregular_tok=sentiment_dict_irreg[
                sentiment_dict_irreg['TokensClean']==irreg_tok]
        confirm_condition=int(metadata_irregular_tok['ConfirmCondition'].values[0])
        anchor_token=metadata_irregular_tok['Anchor_token'].values[0]
        target_tokens=metadata_irregular_tok['Target_tokens'].values.tolist()
        window_size=int(metadata_irregular_tok['Window_size'].values[0])
        window_direction=metadata_irregular_tok['Window_direction'].values[0]
        # Keep track row index
        pos_index=0
        
        #######################################################################
        # Iterate DataFrame w/ Matched Sentences   
        #######################################################################
        for accession_num, sent_pkey, sent_dirty, irreg_tok_val in zip(
                accession_num_col, sent_pkey_col, sent_dirty_col,
                irreg_tok_col):
            # Clean Tokenize Sentences
            sent_toks=clean_tok_sentence(sent_dirty, False)
            # Determine if Irregular token had a match
            if irreg_tok_val > 0:
                print(irreg_tok, irreg_tok_val)

                ###############################################################
                # Apply Operative Code To Test If We Should Count Match
                ###############################################################
                irreg_tok_window, irreg_tok_count=\
                        irregular_token_conditions(
                                irregular_token=irreg_tok,
                                sent_tokens=sent_toks,
                                window_direction=window_direction,
                                window_size=window_size,
                                anchor_word=anchor_token,
                                targets=target_tokens,
                                token_condition=confirm_condition)
                
                # Update DataFrame Using Row Column Indexing
                data_matched_sent.at[pos_index, irreg_tok]=irreg_tok_count
   
                # Append Values to Result Lists
                accession_num_list.append(accession_num)
                sent_pkey_list.append(sent_pkey)
                sent_dirty_list.append(sent_dirty)
                sent_clean_list.append(','.join(sent_toks))
                irreg_tok_list.append(irreg_tok)
                pos_index_list.append(pos_index)
                condition_list.append(confirm_condition)
                anchor_token_list.append(anchor_token)
                target_tokens_list.append(','.join(target_tokens))
                window_size_list.append(window_size)
                window_direction_list.append(window_direction)
                window_list.append(irreg_tok_window)
                original_match_val.append(1)  # default to 1, see line 108
                new_match_val.append(irreg_tok_count)
            
            # Increase Position Index Val 
            pos_index+=1
            
    # Build DataFrame Containing Affected Columns / Rows
    df_row_col_affected=pd.DataFrame({
        'accession_num':accession_num_list,
        'sent_pkey': sent_pkey_list,
        'sent_clean':sent_clean_list,
        'irreg_tok':irreg_tok_list,
        'pos_index':pos_index_list,
        'condition':condition_list,
        'anchor_token':anchor_token_list,
        'target_tokens':target_tokens_list,
        'window_size':window_size_list,
        'window_direction':window_direction_list,
        'window':window_list,
        'original_match_val':original_match_val,
        'new_match_val':new_match_val})

    # Return Original Matched Sentence DataFrame w/ New Assignments
    return data_matched_sent, df_row_col_affected
   


