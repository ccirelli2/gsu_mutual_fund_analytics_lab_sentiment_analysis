
import os
import pandas as pd

dir_repo = r'/home/cc2/Desktop/repositories/mutual_fund_analytics_lab_sentiment_analysis'
dir_data = os.path.join(dir_repo, 'data')
filename = r'paragraphs_sentiment_analysis.csv'
data = pd.read_csv(os.path.join(dir_data, filename))
