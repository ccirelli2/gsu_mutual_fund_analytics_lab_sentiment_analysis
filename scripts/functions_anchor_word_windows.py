
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
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

###############################################################################
# Set up logging parameters & Package Conditions
###############################################################################
logging.basicConfig(level=logging.INFO)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

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
    #punk=string.punctuation
    #sent_nopunk=''.join(list(map(lambda x: x if x not in punk else '', sent_lower)))
    #sent_tokens=sent_nopunk.split(' ')
    sent_clean=clean_sentence(sent_lower)
    sent_tokens=clean_tok_sentence(sent_clean, lemmatize=False)

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
        'accession_num':[], 'sent_pkey':[], 'sentences':[],
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
        results_dict['sent_pkey'].append(sentence_pkey)
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


@my_timeit
def get_anchor_word_window_by_sent(data, anchor_word_source, window_width,
        dir_output, project_folder, write2file):
    """
    Function to get the window of n words around each anchor word.

    The input are sentences that have been identified to contain one of three
    types of anchor words (positive, negative, legal).

    Note that there can be multiple anchor words per sentence.


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
        token_keys=series[6:-1]
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
    #df_concat.drop('sentences', axis=1, inplace=True)
    if write2file:
        filename=f'{anchor_word_source}_anchor_words_windows_size_{window_width}.csv'
        write2csv(df_concat, dir_output, project_folder, filename)

    # Return Results
    return df_concat

