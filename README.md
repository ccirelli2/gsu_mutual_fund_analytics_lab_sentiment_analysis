# GSU Mutual Fund Analytics Lab - Sentiment Analysis

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
 	- Note that between lines 148 and lines 152 we take the product of all
	of the normalized modifying words but we do not address if all are
	negative, which would flip the polarity to positive.

7.  	main.py file parameters:
	includes a parameter to declare the column name that contains the
	paragraph primary key.  that being said, 'accession_num' has been
	hardcoded in the function documents, which means that either all the
	scripts need to be changed or the user must pass a Data Frame with the
	title of the primary key column = 'accession_num'

 
