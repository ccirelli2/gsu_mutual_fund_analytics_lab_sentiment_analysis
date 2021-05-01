""" Code that builds the sentiment dictionary from individual token files.
    Note that the input file must conform to the following format requirements:
    1.) 
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
import string

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
import functions_create_sentiment_dict as m_sent_dict 

###############################################################################
# Package Settings 
###############################################################################
pd.set_option('max_columns', None)


###############################################################################
# Import Project Modules 
###############################################################################


###############################################################################
# Declare Variables 
###############################################################################
sheetnames=['negator', 'negative', 'positive', 'legal', 'uncertain', 'degree',
            'modal', 'irregular']


###############################################################################
# Import Data 
###############################################################################
data_dict=pd.ExcelFile(os.path.join(dir_data, 'dict_input_files',
                       'dict_input.xlsx'))


###############################################################################
# Create Clean Token & Punct Columns 
###############################################################################
negator_prepared=m_sent_dict.create_sent_dict(
        pd.read_excel(data_dict, 'negator'), 'negator')
negative_prepared=m_sent_dict.create_sent_dict(
        pd.read_excel(data_dict, 'negative'), 'negative')
positive_prepared=m_sent_dict.create_sent_dict(
        pd.read_excel(data_dict, 'positive'), 'positive')
legal_prepared=m_sent_dict.create_sent_dict(
        pd.read_excel(data_dict, 'legal'), 'legal')
uncertain_prepared=m_sent_dict.create_sent_dict(
        pd.read_excel(data_dict, 'uncertain'), 'uncertain')
degree_prepared=m_sent_dict.create_sent_dict(
        pd.read_excel(data_dict, 'degree'), 'degree')
modal_prepared=m_sent_dict.create_sent_dict(
        pd.read_excel(data_dict, 'modal'), 'modal')
irregular_prepared=m_sent_dict.create_sent_dict(
        pd.read_excel(data_dict,'irregular'), 'irregular')


###############################################################################
# Concatinate Sheets To A Single DataFrame 
###############################################################################

frames=[negator_prepared, negator_prepared, positive_prepared, legal_prepared,
        uncertain_prepared, degree_prepared, modal_prepared, irregular_prepared]

sent_dict_prepared=pd.concat(frames)


sent_dict_prepared.to_csv(os.path.join(dir_data, 'sent_dict_prepared.csv'))
































































































