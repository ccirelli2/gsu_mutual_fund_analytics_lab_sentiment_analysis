############################################################################### 
# Import Python Libraries                                                       
############################################################################### 
import numpy as np                                                              
import pandas as pd                                                             
import logging                                                                  
import os                                                                       
import sys                                                                      
                                                                                
                                                                                
############################################################################### 
# Directories                                                                   
############################################################################### 
dir_repo=r'/home/cc2/Desktop/repositories/mutual_fund_analytics_lab_sentiment_analysis'
dir_scripts=os.path.join(dir_repo, 'scripts')                                      
dir_results=os.path.join(dir_repo, 'results')
dir_irreg_toks=os.path.join(dir_results, 'irregular_tokens')
[sys.path.append(x) for x in [dir_repo, dir_scripts]]                           
                                                                                
                                                                                   
############################################################################### 
# Import Project Modules                                                           
############################################################################### 
from functions_utility import *                                                    
from functions_decorators import *                                                 
import functions_irregular_tokens as m1


############################################################################### 
# Package Conditions                                                               
############################################################################### 
logging.basicConfig(level=logging.INFO)    

############################################################################### 
# Connect 2 Database                                                               
############################################################################### 
conn, my_cursor = conn_mysql(, 'mutual_fund_lab')
query_matched_sent = """SELECT *
            FROM GSUMUTUALFUNDLAB_principal_risk_token_matches_negative_h1"""
query_sent_dict = """SELECT *
            FROM GSUMUTUALFUNDLAB_sentiment_dictionary_add_irreg_token_rules"""

data_matched_sent=pd.read_sql(query_matched_sent, conn)
sentiment_dict=pd.read_sql(query_sent_dict, conn) 
token_type='Negative'

############################################################################### 
# Function Execution                                                               
############################################################################### 


def execute_irregular_token_condition_function(data_matched_sent,
        sentiment_dict, token_type):

    # Subset dataframe to irregular tokens
    sentiment_dict_irreg=sentiment_dict[
            (sentiment_dict['Irregular']==1) &
            (sentiment_dict['TokenType']==token_type)]
    # Get List of Irregular Tokens
    irregular_tokens = sentiment_dict_irreg['TokensClean'].values.tolist()
    
    # Limit to those tokens in matched sent columns
    irregular_tokens = [x for x in irregular_tokens if x in
            data_matched_sent.columns.values.tolist()]
    logging.info(f'---- irreg tokens => {irregular_tokens}')

    # Lists to capture data on affected columns and rows 
    accession_num_list=[]
    sent_pkey_list=[]
    sent_dirty_list=[]
    sent_clean_list=[]
    irreg_tok_list=[]
    pos_tok_list=[]
    pos_index_list=[]
    condition_list=[]
    anchor_token_list=[]
    target_tokens_list=[]
    window_size_list=[]
    window_direction_list=[]
    window_list=[]
    original_match_val=[]
    new_match_val=[]

    ###########################################################################
    # Get Token Column & Meta Data
    ###########################################################################
    for irreg_tok in irregular_tokens:
        # Get Sentence & Irregular Token Columns
        accession_num_col=data_matched_sent['accession_num'].values
        sent_pkey_col=data_matched_sent['sent_pkey'].values
        sent_dirty_col=data_matched_sent['sentences'].values
        irreg_tok_col=data_matched_sent[irreg_tok].values
        
        # Get Token MetaData From Sentiment Dictionary
        metadata_irregular_tok=sentiment_dict_irreg[
                sentiment_dict_irreg['TokensClean']==irreg_tok]
        confirm_condition=int(metadata_irregular_tok['ConfirmCondition'].values[0])
        anchor_token=metadata_irregular_tok['Anchor_token'].values[0]
        target_tokens=metadata_irregular_tok['Target_tokens'].values.tolist()
        window_size=int(metadata_irregular_tok['Window_size'].values[0])
        window_direction=metadata_irregular_tok['Window_direction'].values[0]
        # Keep track row index
        pos_index=0
        
        #######################################################################
        # Iterate DataFrame w/ Matched Sentences   
        #######################################################################
        for accession_num, sent_pkey, sent_dirty, irreg_tok_val in zip(
                accession_num_col, sent_pkey_col, sent_dirty_col,
                irreg_tok_col):
            # Clean Tokenize Sentences
            sent_toks=m1.clean_tok_sentence(sent_dirty, False)
            # Determine if Irregular token had a match
            if irreg_tok_val:
                ###############################################################
                # Apply Operative Code To Test If We Should Count Match
                ###############################################################
                irreg_tok_window, irreg_tok_count=\
                        m1.irregular_token_conditions(
                                irregular_token=irreg_tok,
                                sent_tokens=sent_toks,
                                window_direction=window_direction,
                                window_size=window_size,
                                anchor_word=anchor_token,
                                targets=target_tokens,
                                token_condition=confirm_condition)
                
                # Update DataFrame Using Row Column Indexing
                data_matched_sent.at[pos_index, irreg_tok]=irreg_tok_count
   
                # Append Values to Result Lists
                accession_num_list.append(accession_num)
                sent_pkey_list.append(sent_pkey)
                sent_dirty_list.append(sent_dirty)
                sent_clean_list.append(','.join(sent_toks))
                irreg_tok_list.append(irreg_tok)
                pos_index_list.append(pos_index)
                condition_list.append(confirm_condition)
                anchor_token_list.append(anchor_token)
                target_tokens_list.append(','.join(target_tokens))
                window_size_list.append(window_size)
                window_direction_list.append(window_direction)
                window_list.append(irreg_tok_window)
                original_match_val.append(1)  # default to 1, see line 108
                new_match_val.append(irreg_tok_count)
            
            # Increase Position Index Val 
            pos_index+=1
            
    # Build DataFrame Containing Affected Columns / Rows
    df_row_col_affected=pd.DataFrame({
        'accession_num':accession_num_list,
        'sent_pkey': sent_pkey_list,
        'sent_dirty':sent_dirty_list,
        'sent_clean':sent_clean_list,
        'irreg_tok':irreg_tok_list,
        'pos_index':pos_index_list,
        'condition':condition_list,
        'anchor_token':anchor_token_list,
        'target_tokens':target_tokens_list,
        'window_size':window_size_list,
        'window_direction':window_direction_list,
        'window':window_list,
        'original_match_val':original_match_val,
        'new_match_val':new_match_val})

    # Return Original Matched Sentence DataFrame w/ New Assignments
    return data_matched_sent, df_row_col_affected

data_matched_sent, df_row_col_affected=\
    execute_irregular_token_condition_function(
                data_matched_sent, sentiment_dict, token_type)




df_row_col_affected.to_csv(os.path.join(dir_irreg_toks, 'irreg_tok_matches.csv'))












