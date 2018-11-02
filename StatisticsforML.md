# StatisticsforML
#### 1. Major difference between statistical modeling adn machine learning
- learn algorithm?
- model shape assumption?
- prediction accuracy & confidence?
- statistical diagostic significance: p-value?
- data split: three or two & how to use them?
- used for: research purpose or production environment

#### 2. Steps in machine learning model developmenta and deployment
-  data collection: structured, web scrapping, API, chat interaction...
-  data preparation and missing/outlier treatment
    - how to deal with missing data: 
        - average or median.
        - run model to predict(y is the missing variable, x is the other variables).
        - dummy variable (country -> no_country).
- data analysis and feature engineering
    - hidden pattern and relations between variables
    - feature selection
- data training
- data testing
- algorithm deployment

#### 3. Parametric method vs Nonparametric method
- parametric model: you have strong assumption about which model exactly you will fit to the data: linear regressionline, it will fit well if the assumption is right. 
- non-parametric model: the data tells you what the 'regression' should look like: KNN, decision tree, it usually needs more computational cost.

#### 4. Comparision between logistic regression and decision trees
- theory is different: equation vs rules explained in Englilsh sentences
- parametric model?
- assumption on response: binomial or bernoulli distribution
- shape of model: logistic curve vs not predefined

#### 5. Chi-square
-  test of independence is one of the most basic and common hypothesis tests in the statistical analysis of categorical data. given 2 categorical random variables X and Y, the chi-sqaure test of independence determines whether or not there exist a statistical dependence between tehm. chi2-contingency function is stats package uses the observed table and subsequently calculates its expected table, followed by calculating the p-value.
