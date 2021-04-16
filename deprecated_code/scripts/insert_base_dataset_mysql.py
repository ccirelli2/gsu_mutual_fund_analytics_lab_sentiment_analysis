###############################################################################
# Import Python Libraries
###############################################################################
import logging
import os
import sys
import pandas as pd
import mysql.connector

###############################################################################
# Set Up Function Parameters 
###############################################################################
logging.basicConfig(level=logging.INFO)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

###############################################################################
# Declare Path Variables 
###############################################################################
dir_base = r'/home/cc2/Desktop/repositories/mutual_fund_analytics_lab_sentiment_analysis'
dir_data = os.path.join(dir_base, 'data')
dir_scripts = os.path.join(dir_base, 'scripts')

###############################################################################
# Import Project Modules 
###############################################################################
from functions_utility import *
from functions_decorators import *


###############################################################################
# Import Data 
###############################################################################

# Instantiate Connection to Mysql Database
conn, my_cursor = conn_mysql('Gsu2020!', 'mutual_fund_lab')


# Read Data From Csv File
data = load_file('data_text.csv', dir_data)
data.drop(['Unnamed: 0', 'Unnamed: 0.1'], inplace=True, axis=1)
cols_rename = {'accession#':'accession_num'}
data.rename(columns=cols_rename, inplace=True)
data = data[['accession_num', 'principal_strategies', 'principal_risks']]


# Set Up Insert Statements
sql = '''INSERT INTO mutual_fund_lab.paragraphs_sentiment_analysis (
            accession_num, principal_strategies, principal_risks)
            VALUES (%s, %s, %s)
      '''

count = 1
for row in data.itertuples():
    vals = [row[1], row[2], row[3]]
    my_cursor.execute(sql, vals)
    conn.commit()
    if count%10000 == 0:
        pct = round((count / data.shape[0])*100, 3)
        print(f'Count => {count}', f'Pct total => {pct}')
    # Increase Count 
    count += 1












