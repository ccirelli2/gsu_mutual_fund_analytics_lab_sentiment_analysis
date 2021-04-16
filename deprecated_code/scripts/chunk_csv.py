# -*- coding: utf-8 -*-
"""
Description: Tokenize sentences of paragraphs

Created 02/07/2020
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
import time
from tqdm import tqdm
from collections import Counter
from datetime import datetime

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

###############################################################################
# Directories & Path
###############################################################################
dir_repo = r'/home/cc2/Desktop/repositories/mutual_fund_analytics_lab_sentiment_analysis'
dir_scripts = os.path.join(dir_repo, 'scripts')
dir_data = os.path.join(dir_repo, 'data')
dir_reports = os.path.join(dir_repo, 'reports')
dir_results = os.path.join(dir_repo, 'results')
dir_tok_sent = os.path.join(dir_results, 'tokenized_sentences')
[sys.path.append(x) for x in [dir_repo, dir_scripts, dir_data, dir_reports,
    dir_results]]


###############################################################################
# Set up logging parameters & Package Conditions
###############################################################################

# Logger
a_logger = logging.getLogger()
a_logger.setLevel(logging.DEBUG)
output_file_handler = logging.FileHandler(
        os.path.join(dir_reports, "output.log"))
stdout_handler = logging.StreamHandler(sys.stdout)
a_logger.addHandler(output_file_handler)
a_logger.addHandler(stdout_handler)

# Other Packages
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


###############################################################################
# Import Project Modules
###############################################################################
import functions_sentence_extraction as m_extract
import functions_get_data as m_data
from functions_utility import *
from functions_decorators import *

###############################################################################
# Import Data
###############################################################################

# Define project folder within dir_output
project_folder = create_project_folder(dir_results, 'chunk_tokenized_sent')


###############################################################################
# Function Parameters 
###############################################################################
write2file=True


###############################################################################
# Execution 
###############################################################################

# Iterate Tokenized Sentence Files

for i in range(10):
    # Load Csv Files
    tokenized_sentences = load_file(f'sentences_tokenized_iteration_{i}.csv',
            dir_tok_sent)
    # Create Sub Project Folder
    sub_project_folder = create_project_folder(
            os.path.join(dir_results, project_folder),
            f'chunk_tokenized_sent_{i}')
    # Chunk csv file
    chunk_csv_file(tokenized_sentences, 10, 'tokenized_sentence_chunk',
            os.path.join(dir_results, project_folder, sub_project_folder),
            write2file)





