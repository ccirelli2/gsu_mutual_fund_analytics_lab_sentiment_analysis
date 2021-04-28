# GSU Mutual Fund Analytics Lab - Sentiment Analysis


### Project Description
- 	Create a program that calculates the sengiment for mutual fund disclosures
	at the sentence and paragraph level.
- 	Users pass in a list or dataframe including the paragraphs from a
	mutual fund disclosure and the program returns the sentiment. 

# Main File (main.py)
Description:                                                                    
    This script is the control script for the sentiment analysis program.          
                                                                                
Directories:                                                                    
   See setup file.  Presently, this script uses an environmental variable          
   to define the root directly.  The user can replace this variable with a         
   string or create their own environmental variable.                           
                                                                                
Logging:                                                                        
    Presently, this script logs to stdout.  The user can change this by         
    passing the name of a file to the logging.basicConfig() method.             
                                                                                
Python libraries:                                                               
    See requirements file.                                                      
                                                                                
Data:                                                                           
    This script loads a test dataset.  Any other dataset passed to this         
    function must be passed as a pandas Data Frame and contain two columns:        
    a column with paragraphs separated by rows and a paragraph primary key         
    that should be titled 'accession_num'                                       
    Note: when loading the test dataset the separator is '|'.  

Model Parameters:                                                               
    mode: str;                                                                  
        options include 'run' or 'debug'.                                       
        If the program is run in debug mode a number of additional              
        processes will run in order to identify incorrectly tokenized           
        sentences, and tokens that include a minimum                            
        number of characters, sentences that include a minimum number of        
        tokens.                                                                 
        Note: If the user is running the program in debug mode and              
        has a large dataset (> 1000 paragraphs) it is recommended that          
        the sample percentage be set to between 0.1 to 0.25 pct.                
    max_num_tokens: Int;                                                        
        Utilized in debug mode and defines the maximum number of tokens         
        to be included as possibly eroneously                                   
        tokenized sentence.                                                     
    max_num_chars: Int;                                                         
        Utilized in the debug mode and defines the maximum number of            
        characters for a token to be identified as possibly an                  
        enroneous token.                                                        
    sample_pct: Float,                                                          
        A user defined value that will take a percentage sample of the          
        number of rows in the dataframe passed to this function.                
    tokenizer: str;                                                             
        options include 'out-of-box' and 'untrained'.                           
        If out-of-box is chosen a pre-trained sentence tokenizer will be        
        used.  If untrained, this program will train a tokenizer on the         
        text provided.                                                          
        Note: if the user has a small dataset then it is recommended            
        that the user select the out-of-box tokenizer.                          
    pkey_col_name: str;                                                         
        The name of the column that includes the primary key.                   
        The default value is 'accession_num'.                                   
    para_col_name: str;                                                         
        The name of the column that includes the paragraphs.                    
        The default value is 'principal_risks'.                                 
    mod_token_names: list;                                                      
        A list containing the names of the groups of modifying tokens.          
        This list object should not be changes.                                 
                                                                                
Output files:                                                                   
    This program will write a step-wise output to the to the output directory   
    in individual sub-folders.                                                  
                                                                                
Last updated: 04/27/2021                


---------------------------------------------------------------------------
#GSU Information

### Project Description
- 	Create a program that calculates the sengiment for mutual fund disclosures
	at the sentence and paragraph level.
- 	Users pass in a list or dataframe including the paragraphs from a
	mutual fund disclosure and the program returns the sentiment. 


### Role
- Lead developer of code found in this repository.

### Final Work Product
- Turn key sentiment generator. 

### How the program works:



### Pending Code Enhancements:
1. 	Sentiment Dictionary: Needs to be updated to incorporate the most recent
	tokens.

2. 	Headings: Address the issue that the code does not distinguish between
	paragraphs and headings.

3. 	Paragraph Primary Key:
	- It is assumed that the user will have a primary key associated with
	the paragraphs that are input into the function.
	- If one is not provided, then a function should be created to
	automatically create one. 

4. 	Token Size threshold:  Presently, the code does not incorporate a threshold
	on the number of characters a token may contain
   	in order to be included in a tokenized sentence.  Below is an example
	of a word window that includes single letter tokens that are counted
	as words. 

	Example From Test Dataset
	accession_num	0000035315-10-000049
	sentence pkey	0000035315-10-000049-53
	sentence toks	for,every,10,000,you,invested,heres,how,much,you,would,pay,in,total,
			expenses,if,you,sell,all,of,your,shares,at,the,end,of,each,
			time,period,indicated,and,if,you,hold,your,shares,class,a,
			class,t,class,b,class,c,sell,all,shares,hold,shares,sell,all,
			shares,hold,shares,sell,all,shares,hold,shares,sell,all,shares,
			hold,shares,1,year,475,475,478,478,654,154,255,155,3,years,636,
			636,645,645,777,477,480,480,5,years,811,811,826,826,1,024,824,
			829,829,10,years,1,316,1,316,1,350,1,350,1,508,1,508,1,813,1,813,
			portfolio,turnover,the,fund,pays,transaction,costs,such,as,
			commissions,when,it,buys,and,sells,securities,or,turns,over,its,portfolio
	window left	if,you,hold,your,shares
	anchor word	class
	window right	a,class,t,class,b
	
5. 	Parameter Names: They are hard coded in the main.py script.  If, for
	instance, the dictionary column names are changed or added to
   	(ex: an additional modifying token category were to be added) the
	paramter names would need to be updated.

6. 	Scoring Function:
	- Needs to be reviewed and approved.  Starting on row 99 the code
	calculates the product of the modifying word scores.
	- The code addresses when there is an even and odd number of negative
	scores, which could flip the polarity of the sentiment.
	- This particular portion of the code should be reviewed and approved
	by the individual(s) responsible for the intent of the
	  sentiment calculation.
<<<<<<< HEAD
 	- Note that between lines 148 and lines 152 we take the product of all
	of the normalized modifying words but we do not address if all are
	negative, which would flip the polarity to positive.

7.  	main.py file parameters:
	includes a parameter to declare the column name that contains the
	paragraph primary key.  that being said, 'accession_num' has been
	hardcoded in the function documents, which means that either all the
	scripts need to be changed or the user must pass a Data Frame with the
	title of the primary key column = 'accession_num'

 
=======
 	- Note that between lines 148 and lines 152 we take the product of all of the normalized modifying words but we do not address if all are negative, which would flip the polarity to positive. 

7. Documentation
	- main.py: requires documentation on mode, tokenizer and quality control parameters

8. setup.py
	- needs to include input data type, environment setup, directory setup, etc.
>>>>>>> 6fedd4bcfa5ea07520a32b755efe3cb601f5e8a1
