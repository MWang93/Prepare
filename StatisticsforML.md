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
- parametric model: models with finite number of parameters. Because you have strong assumption about which model exactly you will fit to the data, it will fit well if the assumption is right. Examples includes: linear regression, logistic regression and linear SVMs.
- non-parametric model: models with unbouded number of parameters, allowing more flexibility. The data tells you what the 'regression' should look like, it usually needs more computational cost. Examples includes: decision trees, KNN ,topic modeling using latent dirichlet analysis.

#### 4. Comparision between logistic regression and decision trees
- theory is different: equation vs rules explained in Englilsh sentences
- parametric model?
- assumption on response: binomial or bernoulli distribution
- shape of model: logistic curve vs not predefined

#### 5. Chi-square
- test of independence is one of the most basic and common hypothesis tests in the statistical analysis of categorical data. given 2 categorical random variables X and Y, the chi-sqaure test of independence determines whether or not there exist a statistical dependence between tehm. chi2-contingency function is stats package uses the observed table and subsequently calculates its expected table, followed by calculating the p-value.

#### 6. Curse of Dimension
- the difficulty of searching through a solution space becomes much harder as you have more features.
- two solutions: feature selection & feature extraction.
- difference: feature selection keeps a subset of the original features while feature extraction creates brand new ones. 

#### [7. Feature selection vs Feature extraction](https://elitedatascience.com/dimensionality-reduction-algorithms)
- feature selection
        - varaince thresholds: remove features whose values don't change much from observation to observation, which provide little values.
        - correlation thresholds: remove features that are highly correlated with others, which provide redundant information.
        - genetic algorithms (GA): they're search algorithms, having two main uses in machine learning: optimization (finding the best weights for a neural network), supervised features selection (genes represent individual feature and the organism represents a candidate set of features)
        - honorable mension: stepwise search: supervised feature selection based on sequential search, it has 2 features: forward and backward.
- feature extraction
        - PCA: a unsupervised algorithm that creates linear combination of the original features, the new features are orthogonal, which means that they are uncorrelated. Futhermore, they are ranked in order of their "explained variance".
        - LDA: a supervised algorithm that also creates linear combination of your original features. However, unlike PCA, LDA doesn't maximize explained variance, it maximizes the separabiity between classes. 
        - Autoencoders: neural networks that are trained to reconstruct their original inputs.

#### 8. Bias-Variance tradeoff
- bias: how well the model fits the data
        - low bias: close to the truth.
        - high bias: simple, have assumption, so miss important features of the data.
- variance: how much the model changes based on changes in the input
        - low variance: not overfit.
        - high variance: flexible, capture intricate features of training data set, so easily overfit and vary a lot with the training set drawn.
- simpler model: high bias, low variance. complex model: low bias, high variance.
- example
        - knn(k=1): low bias, high variance.
        - knn(k large): high bias, low variance.
        - linear regression: high bias, low variance.

#### 9. Advantages and disadvantages of decision tree?
- advantages: decision tree are easy to interpret, nonparametric(robust to outliers), relatively few parameters to tune.
- disadvantages: prone to be overfit, however, this can be addressed by ensemble method like random forest or boosted trees.

#### 10. Advantages and disadvantages of neural networks?
- advantages: good performance for unstructured dataset such as images, audio, and video, incredible flexibility to learn pattern that no other ML algorithm can learn. 
- disadvantages: require large training data to converge, also difficult to pick the right architecture and the interanl hidden layers are incomprehensible.

#### 11. How can you choose a classifier based on training set size.

#### 12. Confusion Matrix
- sensitivity(recall): TPR = 1 - FNR
- specificity: TNR = 1 - FPR
- positive predictive value = 1 - false directory rate
- negatvie predictive value = 1- false omission rate

#### 13. What's ROC and AUC
- roc: performance plot for binary classifier of TPR vs FPR.
        - (0,0): cutoff = 1
        - (1,1): cutoff = 0 
        - (0,1): perfect classification, sensitivity = 1 and specificity = 1
- auc: area udner the roc curve

#### 14. Why AUROC is better tahn raw accuracy as an out-of-sample evalutaion metrics?
- auc is robust to class imbalance, unlike raw accuracy.

#### 15. Error I and Error II
- error I & false positive rate: predict a man is pregnant.
- error II & false negative rate: predict a pregnant woman is not pregnant.
 
#### 16. The cost of false positive is higher than false negative as well as the other way?
- large false positive cost: recruiting process, cost of hiring a bad candidate is much higher than passing a good one.
- large false negative cost: cancer prediction, predicting a cancer patient is healthy means this person cannot get proper treatement in time.
- note: minimizing the cost of false positive/negative is a machine learning problem as much as a product problem.
