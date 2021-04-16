# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 18:23:56 2021
@author: chris.cirelli

Description : Function to obtain window of n tokens around each anchor word
              identified in the target sentences.
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
import string
import copy
import time
import json
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
dir_chunked_sent = os.path.join(dir_results, 'chunk_tokenized_sent')
[sys.path.append(x) for x in [dir_repo, dir_data, dir_scripts, dir_results,
    dir_reports, dir_chunked_sent]]


###############################################################################
# Import Project Modules
###############################################################################
from functions_utility import *
from functions_decorators import *
import functions_anchor_word_windows as m_win

###############################################################################
# Function Parameters 
###############################################################################
write2file=True
quality_control=False
sample_pct=1.0 
nrows=100
lemmatize=False
anchor_word_source='positive'
window_width=5


##############################################################################
# Import Data
###############################################################################
data=pd.read_csv(os.path.join(dir_results, 'sent_tok_match_positive_tokens',
                'final_results_verified_matches_with_original_cols_1.csv'),
                nrows=nrows)
logging.info('---- {} dimensions => {}'.format('data', data.shape))
project_folder = create_project_folder(dir_results, 
        f'anchor_word_window_{anchor_word_source}')


###############################################################################
# Execute Main Function 
###############################################################################

m_win.get_anchor_word_window_by_sent(data, anchor_word_source, window_width,
        dir_results, project_folder, write2file)





