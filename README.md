# GSU Mutual Fund Analytics Lab - Sentiment Analysis

### Project Description
- Sentiment analysis for mutual fund disclosures


### Requested Documentation:
1. Project description

2. Narrative description of your role in the project (i.e., corpus assembly, topic modeling, dictionary assignment, sentiment score, public health disclosure labeling….)

3. Describe the final work product (i.e., labeled disclosures).

4. Describe the steps to generate your final work product. Be sure to highlight places where you made decisions/judgments (such as setting the optimal number of topics, or choosing which approach over several tested, etc.).  If you wrote code for this project, include screen shots of the important code components implementing the step or the judgment.

5.  Identify the relevant files for the final work product (include the name and the location along with a description of the file if not self-evident from title and/or if there are multiple files.  Be over rather than under inclusive.

6.  Identify any next steps or undone work. If someone were to pick this up what else should be done, or what didn’t you have time to do this semester



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

