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
# Function Execution                                                               
############################################################################### 

# Test sentence
test_txt = """today is a good day to code"""
sent_toks = test_txt.split(' ')
print(sent_toks)


# Determine if to count or not count irregular token based on set of rules
count_irregular_token = m1.irregular_token_conditions(
        irregular_token='day', sent_tokens=sent_toks, window_direction='both',
        window_size=2, anchor_word='day', targets=['code'],
        token_condition='confirm')







