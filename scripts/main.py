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
    paragraph_col_name; Str
        The name of the column in the dataframe that contains
        the text for which the user would like generate a sentiment score.

Data:
    The user can run the sentiment function on test data that can be found at
    dir_repo/data/test_data.csv.  The delimeter for this file is |

Last updated: 04/14/2021
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

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters


###############################################################################
# Declare Directories
###############################################################################
dir_repo = r'/home/cc2/Desktop/repositories/mutual_fund_analytics_lab_sentiment_analysis'
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
            dir_output, project_folder, write2file)
   
    # Create Sentence Primary Key
    sentences=m_sent_extr.create_pkey(sentences, 'accession_num')

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
            sentences, tokens_positive, dir_results, project_folder,
            write2file)
    # Negative
    sent_neg_tokens = m_anchor_toks.get_sentences_matching_tokens(
            sentences, tokens_negative, dir_results, project_folder,
            write2file)
    # Legal
    sent_legal_tokens = m_anchor_toks.get_sentences_matching_tokens(
            sentences, tokens_legal, dir_results, project_folder,
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
    # Get Anchor Word Windows 
    ###########################################################################
    df_windows=get_anchor_word_window_by_sent(sent_pos_tokens,
            anchor_word_source='Positive', window_width=5,
            dir_results=dir_results, project_folder=project_folder,
            write2file=write2file)




    return sentences 



###############################################################################
# Execution 
###############################################################################

df_sentences = get_sentiment_score(data, para_col_name, pkey_col_name, mode,
    tokenizer, sample_pct, max_num_tokens, max_num_chars, sent_dict,
    dir_results, project_folder, write2file)



















