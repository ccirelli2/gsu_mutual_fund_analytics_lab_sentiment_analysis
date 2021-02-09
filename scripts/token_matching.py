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
print(today)
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
[sys.path.append(x) for x in [dir_repo, dir_data, dir_scripts, dir_results]]


###############################################################################
# Import Project Modules
###############################################################################
from functions_utility import *
from functions_decorators import *
import functions_token_matching as m_tkm


##############################################################################
# Import Data
###############################################################################
token_type_name = 'legal_tokens'
project_folder = create_project_folder(dir_results, 'sent_match_{}'.format(
    token_type_name))

legal_tokens = pd.read_excel(os.path.join(dir_data,
    'clean_sentiment_dictionary', 'tx_legal_con.xlsx'))[
    'lemmatized_tokens'].values
legal_tokens_unique = [x for x in list(set(legal_tokens)) if isinstance(x, str)]


###############################################################################
# Function Parameters 
###############################################################################
write2file=True
quality_control=True

###############################################################################
# Iterate Chunked Files & Match PH Tokens
###############################################################################

"""
create_unique_set_lemmatized_tokens('sentiment_dictionary.xlsx',
        list_sheet_names, dir_data, dir_data, project_folder, write2file)
"""


for i in range(10):
    logging.info(f'---- iteration => {i}')
    chunk_filename = f'sentences_tokenized_iteration_{i}.csv'
    sent_chunk = pd.read_csv(
        os.path.join(dir_results, 'tokenized_sentences',
            chunk_filename))
    
    # Run Function
    df_iter_result = m_tkm.get_sentences_matching_tokens(
            sent_chunk, legal_tokens_unique, i, dir_results, project_folder,
            write2file, quality_control)

    # Write Results to File
    write2csv(df_iter_result, dir_results, project_folder,
            project_folder + f'_matching_tokens_{i}.csv')






