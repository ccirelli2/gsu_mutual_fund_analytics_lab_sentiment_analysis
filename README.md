# GSU Mutual Fund Analytics Lab - Sentiment Analysis

### Project Description
- Create a program that calculates the sengiment for mutual fund disclosures at the sentence and paragraph level.
- Users pass in a list or dataframe including the paragraphs from a mutual fund disclosure and the program returns the sentiment. 

### Role
- Lead developer of code found in this repository.

### Final Work Product
- Turn key sentiment generator. 

### How the program works:



### Pending Code Enhancements:
1. Update sentiment dictionary to comply with the format of the sentiment dictionary found in the data folder.
2. Address the issue that the code does not distinguish between paragraphs and headings.
3. Create a function that creates a paragraph primary key if one does not exist in the dataframe that the user passes to the main function.
4. Word windows include single letter values as opposed to complete words.  Could be something
   to address in how we define a token.

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
5. Parameter Names: They are hard coded in the main.py script.  If, for instance, the dictionary column names are changed or added to
   (ex: an additional modifying token category were to be added) the paramter names would need to be updated.
   
  
