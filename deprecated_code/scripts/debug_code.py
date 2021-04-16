import logging
logging.basicConfig(level=logging.DEBUG)
import string
import numpy as np
from nltk import ngrams                                                         
from nltk.tokenize import word_tokenize                                         
from nltk.tokenize import sent_tokenize                                         
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters         
from nltk.stem import WordNetLemmatizer                                         
from nltk.stem import PorterStemmer            

"""
sent = 'today is a good day to code. today is a good day to sleep. today is a good day'
sent_tok = word_tokenize(sent)
bigram = ngrams(sent_tok, 2)
#bigram_arr = np.array([x for x in bigram])
#test = np.where(bigram_arr == ('today','is'))


token = 'today is'
tk_tokenized = token.split(' ')
sentence = 'today is a good day to code. today is a good day to code'
tk_sentence = sentence.split(' ')
sentence_ngram = ngrams(tk_sentence, len(tk_tokenized))
sentence_ngram_arr = np.array([x for x in sentence_ngram])
match = np.where(sentence_ngram_arr == tuple(tk_tokenized))[0]

if any(match):
    print(match)
    print(int(len(match) / len(tk_tokenized)))
"""

                                                                                
def clean_tok_sentence(sent):                                                   
    """                                                                         
    Function that removes punctuation from sentence, tokenizes sent             
    and lemmatizes tokens                                                       
                                                                                
    Args:                                                                       
        sent:                                                                   
                                                                                
    Returns:                                                                    
                                                                                
    String object no punctuation.                                               
    """                                                                         
    # Deal With New Line Characters                                             
    sent = sent.replace('\\n', ' ')                                             
    # Clean Up Punctuation                                                      
    punctuation = list(string.punctuation)                                      
    sent_nopunct = ''.join(list(map(                                            
        lambda x: x if x not in punctuation else ' ', sent)))                   
    # Tokenize & Lemmatize                                                      
    sent_tok = word_tokenize(sent_nopunct)                                      
    lemmer = WordNetLemmatizer()                                                
    lemm_tok = [lemmer.lemmatize(x) for x in sent_tok]                          
    return lemm_tok     



def debug_code(list_sentences, tokens):
    # Create Empty Result Object                                                
    tk_match_dict = {x:[] for x in tokens}                                      
    sent_counter = 0                                                            
                                                                                
    # Iterate Sentences                                                         
    for sent in list_sentences:                                   
        # Clean & Tokenize Sentence                                             
        sent_clean_tok = clean_tok_sentence(sent)                               
        # Iterate Tokens                                                        
        for tk in tokens:                                                       
            # If Our Token is a 1 Gram                                          
            if len(tk.split(' ')) == 1:                                         
                # Use np.where to get index pos of each match                   
                match = [x for x in sent_clean_tok if x == tk]                  
                # If Match Not Empty                                            
                if match:                                                       
                    # Append len(match) which equates to count
                    tk_match_dict[tk].append(len(match))                        
                    logging.info(f'---- 1gram match => {match}')
                                                                            
                else:                                                           
                    tk_match_dict[tk].append(0)                                    
                                                                                
            # Otherwise We need to Create Ngrams of Sentence                    
            else:                                                                  
                # Tokenize token (Assumes tokens do not have punctuation)       
                tk_tokenized = tk.split(' ')                                       
                # Create ngram of sentence = len(list tokens)                    
                sentence_ngrams = ngrams(sent_clean_tok, len(tk_tokenized))        
                # Get All Matches                                               
                match = [x for x in sentence_ngrams if x == tuple(tk_tokenized)]
                # If Match list not empty                                       
                if match[0]:                                                       
                    logging.info(f'---- ngram match => {match}')
                    # Append the length of list / num grams                     
                    tk_match_dict[tk].append(len(match))                        
                else:                                                           
                    tk_match_dict[tk].append(0) 


list_sentences = ["""if the proxy contest, or the new management, is not successful,
                  the market price of the companys securities will typically fall.
                  to the new management. or the new management"""]
tokens = ["proxy", "proxy contest", "or the new management"]


debug_code(list_sentences, tokens)







