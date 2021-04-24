"""
Description:
    This script is the control script for the sentiment analysis program.

Requirements:
    User will need to pip install the requirements file prior to execution.

Directories:
    The user will need to define the path of the dir_repo object to comport with
    the location on their file system.
    
Logging:
    In the event that the user wants to create a log file they can just comment
    out the line that defines the logging filename under Python Package
    Settings. Othewise, logging will go to stdout.

Model Parameters:
    pkey_col_name; Str
        The name of the primary key in your dataframe that is unique to each
        pargraph. 
    :paragraph_col_name; Str
        The name of the column in the dataframe that contains
        the text for which the user would like generate a sentiment score.

Data:
    Test: The user can run the sentiment function on test data that
    can be found at dir_repo/data/test_data.csv.  The delimeter for
    this file is '|'.

Output files:
    Consolidated modifying token table csv separator "|".  Values within
    columns are separated by "," so do not use this to load the file.


Last updated: 04/14/2021
"""


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

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

from dotenv import load_dotenv; load_dotenv()

###############################################################################
# Declare Directories
###############################################################################
dir_repo = os.getenv("ROOTDIR") # env variable that points to root of this repo 
dir_scripts = os.path.join(dir_repo, 'scripts')
dir_results = os.path.join(dir_repo, 'results')
dir_reports = os.path.join(dir_repo, 'reports')
dir_data = os.path.join(dir_repo, 'data')
[sys.path.append(path) for path in [dir_scripts, dir_results, dir_reports,
        dir_data]]


###############################################################################
# Import Project Modules 
###############################################################################
from functions_decorators import *
from functions_utility import *
import functions_sentence_extraction as m_sent_extr
import functions_anchor_token_matching as m_anchor_toks
import functions_irregular_token_conditions as m_irreg_cond
import functions_anchor_word_windows as m_windows
import functions_get_mod_words_by_anchor_window as m_mod


###############################################################################
# Python Package Settings 
###############################################################################
logging.basicConfig(level=logging.INFO, 
                    #filename='logging.info'
                    )
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


###############################################################################
# Load Data 
###############################################################################

# Paragraphs
path2file=os.path.join(dir_data, 'test_data.csv')
data=load_file(path2file, name='test', delimiter="|")

# Sentiment Dictionary
path2file=os.path.join(dir_data, 'sentiment_dict.csv')
sent_dict = load_file(path2file, name='sentiment_dictionary',
        delimiter=',')

# Create Project Folder
project_folder=create_project_folder(dir_results,
        input("Please select a name for your project folder => "))



###############################################################################
# Model Parameters 
###############################################################################

# Sentence Extraction
mode='run'
tokenizer='custom'
pkey_col_name='accession_num'
para_col_name='principal_risks'

# Sentence Extraction Debug
max_num_tokens=3
max_num_chars=10
sample_pct=1.0

# Output
write2file=True


###############################################################################
# Main Function 
###############################################################################
def get_sentiment_score(data, para_col_name, pkey_col_name, mode, tokenizer,
        sample_pct, max_num_tokens, max_num_chars, sent_dict, dir_output,
        project_folder, write2file):
    """
    
    Args:
        data: DataFrame;
            Pandas dataframe that contains the paragraphs for which the user
            would like to generate a sentiment score.
        para_col_name: Str;
            Name of column containing text for which the user would like to
            generate the sentiment score.
    Returns:
        

    """
    ###########################################################################
    # Extract Sentences From Paragraph
    ###########################################################################
    sentences=m_sent_extr.sentence_segmenter(data, para_col_name, pkey_col_name,
            mode, tokenizer, sample_pct, max_num_tokens, max_num_chars,
            dir_output, project_folder, write2file=False)
   
    
    # Get Anchor Tokens
    tokens_positive = sent_dict[sent_dict['TokenType'] =='Positive'][
            'TokensClean'].values.tolist() 
    tokens_negative = sent_dict[sent_dict['TokenType'] =='Negative'][
            'TokensClean'].values.tolist()
    tokens_legal = sent_dict[sent_dict['TokenType'] =='Legal'][
            'TokensClean'].values.tolist()


    ###########################################################################
    # Get Sentences Matching Anchor Tokens
    ###########################################################################
    # Positive
    sent_pos_tokens = m_anchor_toks.get_sentences_matching_tokens(
            sentences, tokens_positive, 'positive', dir_results,
            project_folder, write2file)
    # Negative
    sent_neg_tokens = m_anchor_toks.get_sentences_matching_tokens(
            sentences, tokens_negative, 'negative', dir_results,
            project_folder, write2file)
    # Legal
    sent_legal_tokens = m_anchor_toks.get_sentences_matching_tokens(
            sentences, tokens_legal, 'legal', dir_results, project_folder,
            write2file)


    ###########################################################################
    # Apply Irregular Token Conditions 
    ###########################################################################
    sent_pos_tokens, pos_adj=\
            m_irreg_cond.execute_irregular_token_condition_function(
                    sent_pos_tokens, sent_dict, token_type='Positive')
    sent_neg_tokens, neg_adj=\
            m_irreg_cond.execute_irregular_token_condition_function(
                    sent_pos_tokens, sent_dict, token_type='Negative')
    sent_legal_tokens, legal_adj=\
            m_irreg_cond.execute_irregular_token_condition_function(
                    sent_pos_tokens, sent_dict, token_type='Legal')


    ###########################################################################
    # Get Anchor Word Windows & Create Window Primary Key
    ###########################################################################
    # Positive 
    df_windows_pos=m_windows.get_anchor_word_window_by_sent(sent_pos_tokens,
            anchor_word_source='Positive', window_width=5,
            dir_output=dir_results, project_folder=project_folder,
            write2file=write2file)
    # Negative
    df_windows_neg=m_windows.get_anchor_word_window_by_sent(sent_pos_tokens,
            anchor_word_source='Negative', window_width=5,
            dir_output=dir_results, project_folder=project_folder,
            write2file=write2file)
    # Legal 
    df_windows_legal=m_windows.get_anchor_word_window_by_sent(sent_pos_tokens,
            anchor_word_source='Legal', window_width=5,
            dir_output=dir_results, project_folder=project_folder,
            write2file=write2file)


    ###########################################################################
    # Get Modifying Words Within Anchor Word Windows  
    ###########################################################################
    # Positive
    df_pos_mod_modal=m_mod.identify_mod_tokens(df_windows_pos,
            sent_dict, token_type="Modal")
    df_pos_mod_negator=m_mod.identify_mod_tokens(df_windows_pos,
            sent_dict, token_type="Negator")
    df_pos_mod_degree=m_mod.identify_mod_tokens(df_windows_pos,
            sent_dict, token_type="Degree")
    df_pos_mod_uncert=m_mod.identify_mod_tokens(df_windows_pos,
            sent_dict, token_type="Uncertain")
    df_pos_mod_all=m_mod.group_mods_by_window_pkey(
            df_windows_pos, df_pos_mod_modal, df_pos_mod_negator,
            df_pos_mod_degree, df_pos_mod_uncert, 8, 'positive', write2file,
            dir_output, project_folder)
    # Negative 
    df_neg_mod_modal=m_mod.identify_mod_tokens(df_windows_neg,
            sent_dict, token_type="Modal")
    df_neg_mod_negator=m_mod.identify_mod_tokens(df_windows_neg,
            sent_dict, token_type="Negator")
    df_neg_mod_degree=m_mod.identify_mod_tokens(df_windows_neg,
            sent_dict, token_type="Degree")
    df_neg_mod_uncert=m_mod.identify_mod_tokens(df_windows_neg,
            sent_dict, token_type="Uncertain")
    df_neg_mod_all=m_mod.group_mods_by_window_pkey(
            df_windows_neg, df_neg_mod_modal, df_neg_mod_negator,
            df_neg_mod_degree, df_neg_mod_uncert, 8, 'negative', write2file,
            dir_output, project_folder)
    # Legal 
    df_legal_mod_modal=m_mod.identify_mod_tokens(df_windows_legal,
            sent_dict, token_type="Modal")
    df_legal_mod_negator=m_mod.identify_mod_tokens(df_windows_legal,
            sent_dict, token_type="Negator")
    df_legal_mod_degree=m_mod.identify_mod_tokens(df_windows_legal,
            sent_dict, token_type="Degree")
    df_legal_mod_uncert=m_mod.identify_mod_tokens(df_windows_legal,
            sent_dict, token_type="Uncertain")
    df_legal_mod_all=m_mod.group_mods_by_window_pkey(
            df_windows_legal, df_legal_mod_modal, df_legal_mod_negator,
            df_legal_mod_degree, df_legal_mod_uncert, 8, 'legal', write2file,
            dir_output, project_folder)


    ###########################################################################
    # Left Join Modifying Tokens & Sentiment Scores 
    ###########################################################################
    # Positive
    df_windows_pos_join_mods=df_windows_pos.merge(
            df_pos_mod_all, left_on='window_pkey', right_on='window_pkey',
            how='left').merge(
                    sent_dict[['TokensClean', 'Score']], left_on='anchor_word',
                    right_on='TokensClean', how='left').drop(
                            'TokensClean', axis=1) 
    
    # Negative
    df_windows_neg_join_mods=df_windows_neg.merge(
            df_neg_mod_all, left_on='window_pkey', right_on='window_pkey',
            how='left').merge(
                    sent_dict[['TokensClean', 'Score']], left_on='anchor_word',
                    right_on='TokensClean', how='left').drop(
                            'TokensClean', axis=1)
    # Legal
    df_windows_legal_join_mods=df_windows_legal.merge(
            df_legal_mod_all, left_on='window_pkey', right_on='window_pkey',
            how='left').merge(
                    sent_dict[['TokensClean', 'Score']], left_on='anchor_word',
                    right_on='TokensClean', how='left').drop(
                            'TokensClean', axis=1)
    # Write2file
    if write2file:
        write2csv(df_windows_pos_join_mods, dir_output, project_folder,
                'final_prescore_word_window_positive.csv')
        write2csv(df_windows_neg_join_mods, dir_output, project_folder,
                'final_prescore_word_window_negative.csv')
        write2csv(df_windows_pos_join_mods, dir_output, project_folder,
                'final_prescore_word_window_legal.csv')
    

    ###########################################################################
    # Calculate Sentiment Score 
    ###########################################################################
    

###############################################################################
# Execution 
###############################################################################

df_sentences = get_sentiment_score(data, para_col_name, pkey_col_name, mode,
    tokenizer, sample_pct, max_num_tokens, max_num_chars, sent_dict,
    dir_results, project_folder, write2file)



















