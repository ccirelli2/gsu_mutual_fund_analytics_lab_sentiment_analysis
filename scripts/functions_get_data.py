"""
Descriptions : Functions associated with sourcing data from database.

"""


###############################################################################
# Import Python Libraries 
###############################################################################







###############################################################################
# Functions
###############################################################################

def get_paragraphs():                                                              
    query = """                                                                    
    SELECT accession_num, principal_strategies, principal_risks                    
    FROM mutual_fund_lab.paragraphs_sentiment_analysis                             
    """                                                                            
    return query  







