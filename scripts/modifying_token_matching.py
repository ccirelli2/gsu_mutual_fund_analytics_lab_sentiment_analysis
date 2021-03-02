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
[sys.path.append(x) for x in [dir_repo, dir_scripts]]                           
                                                                                
                                                                                   
############################################################################### 
# Import Project Modules                                                           
############################################################################### 
from functions_utility import *                                                    
from functions_decorators import *                                                 


############################################################################### 
# Package Conditions                                                               
############################################################################### 
logging.basicConfig(level=logging.INFO)    
pd.set_option("max_rows", None)
pd.set_option("max_cols", None)


############################################################################### 
# Connect 2 Database                                                               
############################################################################### 
conn, my_cursor = conn_mysql('Gsu2020!', 'mutual_fund_lab')




############################################################################### 
# Function Execution                                                               
############################################################################### 







