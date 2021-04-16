# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 18:34:49 2021

@author: chris.cirelli
"""
###############################################################################
# Import Python Libraries
###############################################################################
import logging; logging.basicConfig(level=logging.INFO)
from datetime import datetime
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
from nltk.tokenize.punkt import PunktParameters
import pandas as pd
from tqdm import tqdm
import re
import inspect
from collections import Counter
import os

###############################################################################
# Import Project Modules 
###############################################################################
from functions_utility import *
from functions_decorators import *

###############################################################################
# Function
###############################################################################


@my_timeit
def get_list_words_end_dot_provided():
    return ['dr.', 'mr.', 'bro.', 'bro', 'mrs.', 'ms.',                                
            'jr.', 'sr.', 'e.g.', 'vs.', 'u.s.',                                    
            'etc.', 'j.p.', 'inc.', 'llc.', 'co.', 'l.p.',                          
            'ltd.', 'jan.', 'feb.', 'mar.', 'apr.', 'i.e.',                         
            'jun.', 'jul.', 'aug.', 'oct.', 'dec.', 's.e.c.',                       
            'inv. co. act']  

@my_timeit
def get_tokens_end_dot(data, num, dir_output, project_folder,
                       write2file):
    """
    Function to obtain tokens ending or containing a period.

    Parameters
    ----------
    data : Dataframe
        Contains sentences of interest.
    sample_pct : Float 
        Sample percentage of rows to choose.

    Returns
    -------
    Dictionary with word:count pair.

    """
    
    sentences = data['sentences'].values.tolist()

    # Declare Results Object
    word_end_dot = []

    # Iterate Sample Setences
    for i in tqdm(range(len(sentences))):
        if num == 1:
            regex = re.compile('[a-z]+\.')
        elif num == 2:
            regex = re.compile('[a-z]\.[a-z]\.')
        elif num == 3:
            regex = re.compile('[a-z]\.[a-z]\.[a-z]\.')
        elif num == 4:
            regex = re.compile('[a-z]\.[a-z]\.[a-z]\.[a-z]\.')
        else:
            logging.error('Error: num must be <= 4')
            
        matches = re.findall(regex, sentences[i])
        [word_end_dot.append(x) for x in matches]

    # Get Count of Words ending with a dot (returns dictionary)
    cnt_wrds_end_dot = Counter(word_end_dot)
    
    # Get DataFrame Of Results
    df = pd.DataFrame(cnt_wrds_end_dot, index=['cnt']).transpose()

    # Write 2 file
    if write2file:
        write2csv(df, dir_output, project_folder,
                  f'sample_set_words_ending_{num}_dot.csv')

    # Return Results
    return cnt_wrds_end_dot


@my_timeit
def train_nltk_sentence_tokenizer(paragraphs, print_abbrevs=False):       
    """                                                                         
    Function to train NLTK PunctSentTokenizer Class on unique body of text.     
                                                                                
    Supposed to work better than out of box sent tokenizer                      
                                                                                
    Args:
    paragraphs : paragraphs on which to train tokenizer
    print_abbrevs : if you want to print the abbreviations that were
                    identified by the the trained tokenizer
    Returns :                                                                   
    -------------                                                               
    trained sentence tokenizer                                                  
    """                                                                         
    # Ensure that paragraphs are strings
    paragraphs = [str(x) for x in paragraphs]
    # Join all paragraphs into a single body of text
    raw_text = ''.join(paragraphs)                                              
                                                                                
    # Instantiate & Train Tokenizer Training Class                              

    # Manually Add abbreviations
    abbrevs = """llc., e.g., u.s., i.e., n.a., p.o., j.p., a.m., p.m., u.k.,
    dr., mr., bro., bro, mrs., ms., jr., sr., e.g., vs., etc., j.p., inc.,
    llc., co., l.p., ltd., jan., feb., mar., apr., i.e., jun., jul., aug.,
    oct., dec., s.e.c., inv. co. act, fmr llc., (i.e."""
    trainer = PunktTrainer()
    # Train on raw text
    trainer.train(raw_text, finalize=False, verbose=True)
    # Add additional bbreviations
    trainer.train(abbrevs, finalize=False, verbose=True)
    # Finalize Tokenizer
    tokenizer = PunktSentenceTokenizer(trainer.get_params())

    # Return Tokenizer                                                          
    return tokenizer                                                       



@my_timeit
def get_sentences_max_num_tokens_chars(df_sentences, max_num_tokens,
        max_num_chars, dir_output, project_folder, write2file):
    """
    Function to obtain sentences with <= maximum number of characters or tokens.
    """
    logging.info('---- Identifying sentences w/ max tokens {} chars {}'.format(
        max_num_tokens, max_num_chars))

    # <= Max Num Tokens                                                     
    sentences_min_tokens = [                                                
            x for x in df_sentences['sentences'].values                     
            if len(word_tokenize(x)) <= max_num_tokens]                     
    
    df_sent_min_toks = pd.DataFrame.from_dict(Counter(sentences_min_tokens),
            orient='index').rename(columns={0:'Count'}).sort_values(by=     
                    'Count', ascending=False)                               
                                                                            
    # <= Max Num Chars                                                      
    sentences_min_chars = [                                                 
            x for x in df_sentences['sentences'].values                     
            if len(x) <= max_num_chars]                                     
    
    df_sent_min_chars = pd.DataFrame.from_dict(Counter(sentences_min_chars),
            orient='index').rename(columns={0:'Count'}).sort_values(by=     
                    'Count', ascending=False)                               
    
    # Logging                                                               
    logging.info('---- Top sentences <= {} tokens \n{}'.format(           
        max_num_tokens, df_sent_min_toks.head()))                                           
    logging.info('---- Top sentences <= {} chars \n{}'.format(            
        max_num_chars, df_sent_min_chars.head()))

    if write2file:
        filename = 'sentences_max_number_tokens.csv'
        write2csv(df_sent_min_toks, dir_output, project_folder, filename)
        filename = 'sentences_max_number_chars.csv'
        write2csv(df_sent_min_chars, dir_output, project_folder, filename)
    
    # Results
    return df_sent_min_toks, df_sent_min_chars


@my_timeit
def get_incorrectly_tokenized_sentences(df_sentences, dir_output,
        project_folder, write2file):                                                            
    """
    Function that identifies possibly incorrectly tokenized sentences.

    Certain tokens contain periods that may be incorrectly identified by the
    tokenizer and ends of sentences.  This function identifies those tokens
    and checks to see if the tokenized sentences end with them.

    df_sentences : DataFrame; tokenized sentences
    
    Return
    --------
    DataFrame with sentences and tokens

    """
    logging.info(f'---- Testing {df_sentences.shape[0]} tokenized sentences')   
    ###########################################################################
    # Get Tokens Contain One or Two Dots                                       
    ###########################################################################
    tokens_2_dots = get_tokens_end_dot(df_sentences, 2,          
            dir_output, project_folder, write2file)                             
    tokens_3_dots = get_tokens_end_dot(df_sentences, 3,          
            dir_output, project_folder, write2file)                             
    tokens_n_dots = get_list_words_end_dot_provided()                    
    tokens_all_dots = list(tokens_2_dots.keys()) + list(tokens_3_dots.keys()) +\
            tokens_n_dots                                                      
    logging.info('---- Tokens to search for at end of sentence => {}'.format(   
        tokens_all_dots))                                                       
    logging.info('---- Searching for possibly incorrectly tokenized sentences')

    ##########################################################################
    # Identify Sentences Ending In Dot Tokens                                                 
    ##########################################################################
    pkey_list = []
    result_sentences = []                                                       
    result_token = []                                                           
                                                                                
    # Iterate Sentences                                                         
    for i in tqdm(range(df_sentences.shape[0])):
        pkey = df_sentences['accession#'].values.tolist()[i]
        sent = df_sentences['sentences'].values.tolist()[i]                               
        
        # Tokenize Sentence                                                     
        tokens = word_tokenize(sent)
        num_toks = 3
        if len(tokens) >= num_toks:
            try:                                                                    
                if ''.join(tokens[-2] + tokens[-1]) in tokens_all_dots:             
                    pkey_list.append(pkey)
                    result_sentences.append(sent)                                   
                    result_token.append(tokens[-2] + tokens[-1]) 
            except IndexError:                                                      
                pass                                                                
                                                                                
    df_results = pd.DataFrame({
        'accession#':pkey_list,
        'sentences': result_sentences,
        'tokens':result_token})
    
    df_tok_frequencey = pd.DataFrame.from_dict(Counter(result_token),
            orient='index')

    if write2file:
        filename = 'sentences_incorrectly_tokenized.csv'
        write2csv(df_results, dir_output, project_folder, filename) 
        filename = 'sentences_incorrectly_tokenized_token_frequency.csv'
        write2csv(df_tok_frequencey, dir_output, project_folder, filename) 
    
    ###########################################################################
    # Return Results
    ###########################################################################
    logging.info('---- Number of possible eroneous sentences => {}'.
        format(df_results.shape[0]))                           
    logging.info('---- Pct of sentences => {}%'.format(                         
        round((df_results.shape[0] / df_sentences.shape[0])*100,2))) 
    
    # Return Results
    return df_results


@my_timeit
def tokenizer_quality_control(df_sentences, max_num_tokens, max_num_chars,      
        dir_output, project_folder, write2file):                                
    """                                                                         
    Function to check the quality of the sentence tokenizer.                    
                                                                                
    Args:                                                                       
        df_sentences: DataFrame; Contains rows w/ sentences                     
        max_num_tokens: Int; Identify sentences <= max num tokens               
        max_num_chars: Int; Identify sentences <= max num chars                 
        dir_output: String; output directory                                    
        project_folder: String;                                                 
        write2file: Boolean                                                     
                                                                                
    Return :                                                                    
        Returns a dataframe with the sentences of interest & a report with      
        metrics on the quality of the tokenization                              
                                                                                
    """                                                                         
    logging.info(f'---- Testing {df_sentences.shape[0]} tokenized sentences')   
                                                                                
    # Get Sentences <= Min Number Tokens or Chars                               
    df_sent_min_toks, df_sent_min_chars =\
            get_sentences_max_num_tokens_chars(df_sentences,          
                    3, 10, dir_output, project_folder, write2file)              
                                                                                
    # Sentences ending in dot & end of sentence                                 
    df_sent_end_dot_toks = get_incorrectly_tokenized_sentences(       
            df_sentences, dir_output, project_folder, write2file)               
                                                                                
    # Return Results                                                            
    return df_sent_min_toks, df_sent_min_chars, df_sent_end_dot_toks 


@my_timeit
def get_paragraphs_for_incorrectly_tokenized_sentences(
        data, df_sentences, dir_output, project_folder, write2file):
    """

    Args:
        data:
        df_sentences:

    """
    
    # Get Primary Keys for Incorrectly Tokenized Sentences
    pkeys = df_sentences['accession#'].values

    # Get Paragraphs
    df_paragraphs = data.merge(df_sentences, left_on='accession#',
            right_on='accession#').rename(columns={'principal_risk':
                'paragraphs'})

    if write2file:
        filename = 'incurrectly_tokenized_paragraphs.csv'
        write2csv(df_paragraphs, dir_output, project_folder, filename)
    
    # Return Results                                                            
    return df_paragraphs


@my_timeit                                                                      
def sentence_segmenter(data, para_col_name, pkey_col_name, mode,
        tokenizer, sample_pct, max_num_tokens, max_num_chars,
        dir_output, project_folder, write2file, iteration=None):
    """                                                                         
    Function to segment sentences from a body of text.                          
                                                                                
    Parameters                                                                  
    ----------                                                                  
        data : DataFrame, contains a col of text.
        para_col_name : String; Name of column containing paragraphs/txt
        pkey_col_name : String; Name of primary key column
        mode : String; If debug mode selected, the program will run through a
            sub-program to test whether sentences were split correctly.
            Parameters max_num_tokens, max_num_chars and sample_pct are all used
            during the debug mode.
        sample_pct : Float; Utilized to set the sample of the dataset.
            Important -> when in debug mode pct should be ~0.25 or less.
        max_num_tokens : Int; threshold to identify sentences with a min
            number of tokens.                                            
        max_num_chars : Int; threshold to identify sentences with a min
            number of characters.                                            
        tokenizer : String; Name of tokenizer to use. If out-of-box then the
            standard nltk word_tokenizer is used.  Otherwise, the function will
            use a tokenizer that is fit to this dataset and acronyms / abbreviations
            mined from this dataset.
        dir_output : String, Directory to write output
        project_folder : String, Sub directory in dir_output to save results.
        write2file : Boolean, Whether to write output to file.                  
                                                                                
    Return                                                                      
    ---------                                                                   
    DataFrame containing the paragraph primary key, tokenized sentences and
    number of characters in each tokenized sentence.    
    """                                                                         
    ########################################################################### 
    # Logging
    ########################################################################### 
    logging.info(f'---- Running in mode => {mode}; tokenizer => {tokenizer}') 

    ########################################################################### 
    # Prepare Data & Train Tokenizer                                            
    ########################################################################### 
    # Drop Na Values in the Principal Risk Column (contains text)               
    data_num_rows = data.shape[0]
    data.dropna(subset=[para_col_name], inplace=True)                       
    logging.info('---- number of nan rows dropped => {}'.format(
        data_num_rows - data.shape[0]))

    # Create Sample of Data                                                     
    if sample_pct < 1.0:                                                        
        logging.info(f'---- Generating Data sample size => {sample_pct}')          
        data = data.sample(frac=sample_pct, random_state=1)                     
                                                                                
    # Get List of Paragraphs & Keys                                                     
    list_paragraphs = data[para_col_name].values.tolist()                   
    list_pkeys = data[pkey_col_name].values.tolist()                             
                                                                                
    # Remove Any Non-ASCII Characters (Essentially utf-8 chars)                 
    list_paragraphs = [                                                         
            sentence.encode("ascii", "ignore").decode() for sentence            
            in list_paragraphs]                                                 
                                                                                
    # Train NLTK Tokenizer                                                      
    if tokenizer != 'out-of-box':                                               
        trained_sent_tokenizer = train_nltk_sentence_tokenizer(          
            list_paragraphs, print_abbrevs=False)

    ########################################################################### 
    # Tokenize Sentences                                                        
    ########################################################################### 
    result_pkey = []                                                            
    result_sentences = []                                                       
    result_num_chars_sentence = []                                              
    result_num_toks_sentence = []
    logging.info(f'---- Tokenizing {len(list_paragraphs)} Paragraphs')          
                                                                                
    # Iterate Paragraphs & Tokenize Sentences                                   
    for i in tqdm(range(len(list_paragraphs))):                                 
        # Iterate Sentences In Paragraph                                        
        if tokenizer == 'out-of-box':                                           
            tokenized_sentences = sent_tokenize(list_paragraphs[i])             
        else:                                                                   
            tokenized_sentences=\
                    trained_sent_tokenizer.tokenize(list_paragraphs[i])         
        for sentence in tokenized_sentences:                                    
            # Append Results to List Objects                                    
            result_pkey.append(list_pkeys[i])                                   
            result_sentences.append(sentence)                                   
            result_num_toks_sentence.append(len(sentence.split(' ')))
            result_num_chars_sentence.append(len(sentence))                     
                                                                                
    # Construct Results DataFrame                                               
    df_sentences = pd.DataFrame({                                               
        'accession_num': result_pkey,                                              
        'sentences': result_sentences,                                          
        'num_toks' : result_num_toks_sentence,
        'num_chars': result_num_chars_sentence})                                
    
    ########################################################################### 
    # Run Test Diagnostics On Tokenized Sentences                                   
    ###########################################################################
    if mode == 'Debug' or mode == 'debug':                                      
        # Get Incorrectly Tokenized Sentences & Those With Min Length           
        df_sent_min_toks, df_sent_min_chars, df_incorrect_sentences=\
                tokenizer_quality_control(                            
                        df_sentences, max_num_tokens, max_num_chars,            
                        dir_output, project_folder, write2file)                 
        # Get Paragraphs For Incorrectly Tokenized Sentences                    
        df_paragraphs_incurrect=\
                get_paragraphs_for_incorrectly_tokenized_sentences(   
                        data, df_incorrect_sentences, dir_output,               
                        project_folder, write2file)                             
                                                                                
    ########################################################################### 
    # Write & Return Results                                                      
    ###########################################################################
    if write2file:                                                              
        if iteration is not None:                                               
            filename=f'sentences_tokenized_iteration_{iteration}.csv'           
        else:                                                                   
            filename=f'sentences_tokenized_sample_pct_{sample_pct}.csv'         
        write2csv(df_sentences, dir_output, project_folder, filename)           
                                                                                
    # Return Results                                                            
    return df_sentences 











