Starbucks_Capstone_Project - Udacity

The Project

After reviewing the three datasets, I first started by cleaning the data including addressing missing values. Since the objective of the challenge is to create a model which better predicts the probability of offer-completion, I started by filtering and merging the three datasets. My focus was to generate insight that better explain consumers decision on whether to accept and complete offers of discount/BOGO. 
 
Using the merged data, I tabulated the offers with largest completion percentage. I found that discount offers such as offer 1 and 2 lead to better offer-completion rate compared to informational offers such as offer 9 and 10. 
 
Two classifiers, Logistic Regression and Random Forest, were used to train on the training set and calculate the predictive accuracy of the model. The RandomForestClassifier yielded a relatively better result. 

Finally, I made some assessment on the relative strength of the explanatory variables. The result shows the amount of money the customer spent was the strongest predictor on whether the customer will accept the offer and complete the process.

Libraries used

Pandas
Numpy
Matplotlib
Sklearn 

File descriptions
• Starbucks_Capstone_notebook.ipynb: Jupyter notebook containing implementation and analysis.

• data/portfolio.json: contains offer_ids and metadata about each offer (duration, type, etc)

• data/ profile.json: demographic data for each customer

• data/transcript.json: transactions, offers received, offers viewed, and offers completed records

• final_data.csv: Cleaned data in csv format

Results

LogisticsRegressionClassifier
    training accuracy score: 0.8795488721804511
    test accuracy score: 0.8793042955240339
RandomForestClassifier
    training accuracy score: 0.9998066595059076
    test accuracy score: 0.9068718359981955

An assessment of the predictive powers of the features used in the RandomForest Classifier yields the following percentages.
    total_amount (consumer spend) - 57%
    month of membership, difficulty, duration, reward also have some explanatory power, falling in the top 5
    

https://github.com/kolBerR/Starbucks-Capstone-Project---Udacity

In the github repository is included a webApp (sApp)
