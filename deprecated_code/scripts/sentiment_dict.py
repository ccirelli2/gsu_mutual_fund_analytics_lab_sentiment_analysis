# -*- coding: utf-8 -*-
"""
Purpose:    Clean up sentiment dictionary

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
filename='tx_positive_st_con.xlsx'
data=load_file2(os.path.join(dir_reports, 'sentiment_dictionary',
    filename), 'positive_tokens')
punctuation=list(string.punctuation)

###############################################################################
# Functions 
###############################################################################
def check_punct_in_token(list_tokens, list_punctuation):
    # Iterate Tokens
    for token in list_tokens:
        # Iterate punctuation
        for punck in punctuation:
            if punck in str(token):
                print(token)


data['tokens'] = list(map(
    lambda x: x.lower() if isinstance(x, str) else x, data['tokens'].values))

data.to_csv(os.path.join(dir_reports, 'sentiment_dictionary',
    'tx_positive_st_con.csv'))










