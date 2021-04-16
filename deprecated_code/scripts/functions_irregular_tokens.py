
###############################################################################
# Import Python Libraries
###############################################################################
import numpy as np
import pandas as pd
import logging
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
# Directories
###############################################################################
dir_repo=r'/home/cc2/Desktop/repositories/mutual_fund_analytics_lab_sentiment_analysis'
dir_scripts=os.path.join(dir_repo, 'scripts')
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



























