# Machine Learning with Python Final Project
I chose a case study designed for a top tier strategic consulting company for my final project. \
You can find following files in this repository.
* Datasets : all the provided datasets for training and testing.
* Codes : codes for data exploration, training models, and testing.
* Result : validation result output.
\
**Case Scenario** \
Company A is a major utility company providing gas and electricity to corporate, SME and residential customers. In recent years, post-liberalization of the energy market in Europe, company A has had a growing problem with increasing customer defections above industry average. The churn issue is most acute in the SME division and thus they want it to be the first priority. \
There are 3 datasets given as follows; 1. Features of customers 2. Electric prices for each customer 3. Churn data. \
I merged those 3 and explored the them to find out which model to choose then built models that can predict the probability of churn. \
In this repository, you can find the datasets and codes of modelling. \
\
**Data Processing** \
Used all the 3 files given. (detailed data description can be found at the end) \
Some of the key processes are:
* Checked the time series and avoided using those features.
* Converted date features into numerical and calculated the duration from the start date. 
* Imputed missing data with KNN.  
* Removed outliers.  
* Feature hashing encoding for high dimensional feature.  
* Up/Down sampled the imbalanced target variable.  
 \
**Modelling**
 * Considering the distribution of data and computational expense, tested Ensemble methods (Random Forest, XGboost, Gradient Boost, Adaboost)
 * Used Grid Search Cross Validation for all the models to maximize the ROC. 
 * Draw learning curves to estimate the best parameter ranges for Grid Search. 
 \
 \
**Data Description** \
![alt text](https://github.com/chierina/ESCP-ML-Python/blob/master/data_description.png)
