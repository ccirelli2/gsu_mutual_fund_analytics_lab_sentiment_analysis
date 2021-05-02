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


############################################################################### 
# Function                                                        
############################################################################### 

def create_sent_dict(data, sheetname):                                                     
    """
    Function to prepare the individual sheets of the sentiment dictionary
    Args:
        data: Individual sheets containing information on each token type.
        sheetname: Name of sheet

    Returns:
        

    """
    print(f'Creating sheetname => {sheetname}') 
    ###########################################################################
    # Get List of Punctuation Symbols                                           
    ###########################################################################
    punctuation = list(string.punctuation)                                      
    punkt=list(string.punctuation)                                              
    # Punctuation that should be replaced by ''                                 
    punkt_nospace=[punkt[6], punkt[-5]]                                         
    # Punctutation that should be replaced by ' '                               
    punkt_space=[x for x in punkt if x not in punkt_nospace]                    
                                                                                
    ###########################################################################
    # Remove rows with null values                                           
    ###########################################################################
    data=data[data['Tokens'].isna() == False]

    ###########################################################################
    # List Clean Tokens (result)                                                
    ###########################################################################
    tokens_nopunkt=[]                                                           
                                                                                
    # Iterate Tokens                                                            
    for tok in data['Tokens'].values:                                           
        
        # If String                                            
        if isinstance(tok, str):
            # Convert to lowercase
            tok=tok_lower=tok.lower()                                                   
                                                                                
            # Replace punctuation with no space                                     
            tok_nopunkt=''.join(list(map(                                           
                lambda x: x if x not in punkt_nospace and
                    isinstance(x, str) else '', tok)))         
                                                                                    
            # Replace punctuation with space                                        
            tok_nopunkt=''.join(list(map(                                           
                lambda x: x if x not in punkt_space and
                    isinstance(x, str) else ' ', tok_nopunkt)))        
                                                                                    
            # Append Cleaned Token                                                  
            tokens_nopunkt.append(tok_nopunkt)                                      

        else:
            tokens_nopunkt.append(None)

    ###########################################################################
    # Add New column to dataset                                                 
    ###########################################################################
    data['TokensClean'] = tokens_nopunkt                                        
                                                                                
    # Rename Original Token File                                                
    data.rename(columns={'Tokens':'TokensDirty'}, inplace=True)                 
                                                                                
    ###########################################################################
    # Get Binary Flag Tokens w/ Punctuation                                     
    ###########################################################################
    def get_punkt_flag(data):
        flag=[]
        for x, y in zip(
                data['TokensDirty'].values,
                data['TokensClean'].values):
            if isinstance(x, str) and isinstance(y, str):
                if len(x) != len(y):
                    flag.append(1)
                else:
                    flag.append(0)
            else:
                flag.append(0)
        data['PunktFlag']=flag

        return data
        
    data=get_punkt_flag(data)
                                                                                
    ###########################################################################
    # Add Null Columns 
    ###########################################################################
    col_names=['DateCreated', 'Irregular', 'RuleDescription', 'Application',
            'ConfirmCondition', 'Anchor_token', 'Target_tokens', 'Window_size',
            'Window_direction']

    for name in col_names:
        if sheetname != 'irregular':
            if name == 'Irregular':
                data['Irregular'] = np.zeros((data.shape[0], 1))
            else:
                data[name] = np.nan

        elif sheetname == 'irregular':
            data['DateCreated'] = np.nan

        else:
            print('Incorrect sheet name passed to function')

    # Create Uniform Column Order
    data=data[['TokensDirty', 'Score', 'Irregular', 'TokenType',
        'TokensClean', 'PunktFlag', 'DateCreated', 'RuleDescription',
        'Application', 'ConfirmCondition', 'Anchor_token',
        'Target_tokens', 'Window_size', 'Window_direction']]
    

    # Print Final Column Structure
    print(f'Final column structure => {data.columns.tolist()}')
    print(f'Dataframe shape => {data.shape}\n')
    # Return Sentiment Dictionary                                               
    return data 



























