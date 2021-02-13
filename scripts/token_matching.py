# -*- coding: utf-8 -*-
"""
Purpose:    Create token frequency table for tokenized sentences  


Created on Wed Jan 13 18:23:56 2021
@author: chris.cirelli
"""

###############################################################################
# Import Python Libraries
###############################################################################
import logging
import os
import sys
import pandas as pd
import inspect
import string
import copy
import time
from tqdm import tqdm
from random import randint
from datetime import datetime
from collections import Counter

from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

###############################################################################
# Set up logging parameters & Package Conditions
###############################################################################
today = datetime.today().strftime("%d_%m%Y")
logging.basicConfig(level=logging.INFO)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

###############################################################################
# Declare Variables
###############################################################################
dir_repo = r'/home/cc2/Desktop/repositories/mutual_fund_analytics_lab_sentiment_analysis' 
dir_data = os.path.join(dir_repo, 'data')
dir_scripts = os.path.join(dir_repo, 'scripts')
dir_results = os.path.join(dir_repo, 'results')
dir_reports = os.path.join(dir_repo, 'reports')
[sys.path.append(x) for x in [dir_repo, dir_data, dir_scripts, dir_results,
    dir_reports]]


###############################################################################
# Import Project Modules
###############################################################################
from functions_utility import *
from functions_decorators import *
import functions_token_matching as m_tkm


###############################################################################
# Function Parameters 
###############################################################################
write2file=True
quality_control=False
sample_pct=1.0 
lemmatize=False


##############################################################################
# Import Data
###############################################################################
token_type_name = 'negative_tokens'
token_filename = 'tx_negative_con.xlsx'


project_folder = create_project_folder(dir_results, 'sent_tok_match_{}'.format(
    token_type_name))

if lemmatize:
    tokens = pd.read_excel(os.path.join(dir_reports,
        'clean_sentiment_dictionary', token_filename))[
        'lemmatized_tokens'].dropna(axis=0).values
else:
    tokens = pd.read_excel(os.path.join(dir_reports,
        'clean_sentiment_dictionary', token_filename
        ))['original_tokens'].dropna(axis=0)

# Create Set of Tokens (Remove Any Duplicates)
tokens_unique = list(set([str(x).lower() for x in tokens]))
print(tokens_unique)

###############################################################################
# Iterate Chunked Files & Match PH Tokens
###############################################################################

start = datetime.now()
def run(): 
    for i in range(10):
        logging.info(f'---- iteration => {i}')
        sentence_filename = f'sentences_tokenized_iteration_{i}.csv'
        sent_chunk = pd.read_csv(os.path.join(
            dir_results, 'tokenized_sentences', sentence_filename))
        
        # Get Sample of Dataset
        if sample_pct != 1.0:
            sent_chunk = sent_chunk.sample(frac=sample_pct)

        # Run Function
        """i+=1  : added because many of the functions have a boolean cond for i
                   where 0 equates to False, which is not the case"""
        # Increase i by 1
        i+=1               
        df_iter_result = m_tkm.get_sentences_matching_tokens_v2(
                sent_chunk, tokens_unique, dir_results, project_folder,
                write2file, quality_control, lemmatize, iteration=i)

run()

duration = (datetime.now() - start).total_seconds()
logging.info('Program finished.  Total duration in seconds => {}'.format(
    duration))



