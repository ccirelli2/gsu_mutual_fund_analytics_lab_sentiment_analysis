#################################################################################
# Load Python Libraries
#################################################################################
import logging
from datetime import datetime
import sys
import pandas as pd
import numpy as np
import inspect
from tqdm import tqdm
import time
from collections import Counter
import string
import copy
from random import randint
from functools import wraps

from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


###############################################################################
# Set up logging parameters & Package Conditions
###############################################################################
today = datetime.today().strftime("%d_%m%Y")
logging.basicConfig(level=logging.INFO)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

###############################################################################
# Import project modules 
###############################################################################
from functions_decorators import *
from functions_utility import *


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

@my_timeit
def get_sentences_matching_tokens(                                              
        data_original, tokens, token_type, dir_output, project_folder,                      
        write2file, quality_control=False, lemmatize=False, iteration=None):    
    """                                                                         
    Identify sentences that contain the list of input tokens.                     
                                                                                
    Steps:                                                                      
    Unverified Match : find if any tokens are in sentence.                      
    Verified Match : Involves actually cleaning and tokenizing sentences        
                    and matching one to one with tokens                         
                                                                                
    Args:                                                                       
        data: DataFrame; Rows = each row is a tokenized sentences.              
                A primary key links the sentence to the original paragraphs.    
        tokens: List; contains lowercase single ngram tokens                                 
        interation : Int; because the tokenized sentences are chunked, the      
                iteration value is passed to the write2file function to match   
                each input file to output.                                      
        dir_output:                                                             
        project_folder:                                                         
        write2file: Boolean; Whether to write to file or not.                   
                                                                                
    Return:                                                                     
    -------------                                                               
    data with the addition of columns for each of the n tokens containing       
    binary values that represent a match with the sentence or not.              
    """                                                                         
    ##########################################################################  
    # Data Transformations                                                             
    ##########################################################################  
    # Create Deep Copy of dataset                                               
    data_cp = copy.deepcopy(data_original)                                      
    # Add Sentence Primary Key (accession# key is not unique at the sent lvl    
                                                                    
    ##########################################################################  
    # Result Object                                                             
    ##########################################################################  
    unverified_match = []                                                       
    sent_counter = 0                                                            
                                                                                
    # Logging                                                                   
    logging.info(f'---- dimensions original dataset => {data_original.shape}')  
    logging.info(f'---- number of tokens => {len(tokens)}')                     
    logging.info('---- getting unverified matches')                             
                                                                                
    ########################################################################### 
    # Get Unverified Match of Tokens                                            
    ########################################################################### 
                                                                                
    # Iterate Pkey & Sentences                                                  
    for sent in data_cp['sentences'].values:
        # Clean Sentence
        sent=clean_sentence(sent)
        # Token Counter                                                         
        tokens_cp = copy.deepcopy(tokens)                                       
        # Iterate Tokens                                                        
        for tk in tokens:                                                       
            tk = str(tk)                                                        
            # Remove Token From List                                            
            tokens_cp.pop(tokens_cp.index(tk))                                  
            # If list not empty                                                 
            if tokens_cp:                                                       
                if tk in sent:                                                  
                    logging.debug('---- match found')                           
                    unverified_match.append(1)                                  
                    break                                                       
            else:                                                               
                logging.debug(f'---- no match found')                           
                unverified_match.append(0)                                      
                                                                                
    # Add Unverified Match to Data                                              
    data_cp['unverified_match'] = unverified_match                              
                                                                                
    # Lim Data To Only Sentences Unverified Match                               
    data_lim = data_cp[data_cp['unverified_match'] == 1]                        
                                                                                
    # Log number of matches                                                     
    logging.info(f'---- number unverified matches => {data_lim.shape[0]}')    

    ########################################################################### 
    # Get Verified Match of Tokens                                              
    ########################################################################### 
    # Create Empty Result Object                                                
    tk_match_dict = {x:[] for x in tokens}                                                                                              
    # Iterate Sentences                                                         
    for sent in data_lim['sentences'].values:                                   
        # Clean & Tokenize Sentence                                             
        sent_clean_tok = clean_tok_sentence(sent, lemmatize)                    
        # Iterate Tokens                                                        
        for tk in tokens:                                                       
            # If Our Token is a 1 Gram                                          
            if len(tk.split(' ')) == 1:                                         
                # Check if token in tokenized sentence                          
                if tk in sent_clean_tok:                                        
                    # Append Initial Result                                     
                    tk_match_dict[tk].append(1)                                 
                else:                                                           
                    tk_match_dict[tk].append(0)                                 
                                                                                
            # Otherwise We need to Create Ngrams of Sentence                    
            else:                                                               
                # Tokenize token (Assumes tokens do not have punctuation)       
                tk_tokenized = tk.split(' ')                                    
                # Create ngram of sentence = len(ph token)                      
                sentence_ngrams = ngrams(sent_clean_tok, len(tk_tokenized))     
                # If the ngram of token in ngram of sentence                    
                if tuple(tk_tokenized) in sentence_ngrams:                      
                    tk_match_dict[tk].append(1)                                 
                else:                                                           
                    tk_match_dict[tk].append(0)    
    
    ########################################################################### 
    # Create Results DataFrame & Join to Original Dataset                       
    ########################################################################### 
    tk_matches = pd.DataFrame(tk_match_dict)                                    
    tk_matches['sent_pkey'] = data_lim['sent_pkey'].values                      
    # Get Sum of Matches Across Tokens                                          
    tk_col = tk_matches[tokens]                                                 
    tk_matches['sum_matches'] = tk_col.sum(axis=1)
    # Limit Tk Matches To Sum > 0                                               
    tk_matches_lim = tk_matches[tk_matches['sum_matches'].values > 0]           
    # Merge df_tk_matchs w/ data_lim                                            
    df_final = pd.merge(data_lim, tk_matches_lim, left_on='sent_pkey',          
            right_on='sent_pkey')                                               
    
    ########################################################################### 
    # Write Output                       
    ########################################################################### 
    if write2file:
        subfolder=create_project_folder(
                dir_output=os.path.join(dir_output, project_folder),
                name='sentences_anchor_token_matching')
        dir_output=os.path.join(dir_output, project_folder)
        filename=f'match_sentences_anchor_toks_{token_type}.csv'
        write2csv(df_final, dir_output, subfolder, filename)


    if quality_control:                                                         
        logging.info(f'---- dim unverified matches before => {data_lim.shape}') 
        logging.info(f'---- dim unverified matches after => {tk_matches.shape}')
        logging.info(f'---- dim verified matches => {tk_matches_lim.shape}')    
        logging.info(f'---- dim merged data => {df_final.shape}')               
                                                                                
    # Return final dataframe                                                    
    return df_final  









































