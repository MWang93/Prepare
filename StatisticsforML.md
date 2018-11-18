# StatisticsforML
#### 1. Major difference between statistical modeling and machine learning
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
- the difficulty of searching through a solution space becomes much harder as you have more features, which means that everything is "far away" from each other.
- two solutions: feature selection & feature extraction.
- difference: feature selection keeps a subset of the original features while feature extraction creates brand new ones. 

#### [7. Feature selection vs Feature extraction](https://elitedatascience.com/dimensionality-reduction-algorithms)
- feature selection
    - varaince thresholds: remove features whose values don't change much from observation to observation, which provide little values.
    - correlation thresholds: remove features that are highly correlated with others, which provide redundant information.
    - genetic algorithms (GA): they're search algorithms, having two main uses in machine learning: optimization (finding the best weights for a neural network), supervised features selection (genes represent individual feature and the organism represents a candidate set of features)
    - honorable mension: stepwise search: supervised feature selection based on sequential search, it has 2 features: forward and backward.
- feature extraction
    - [PCA](https://www.coursera.org/lecture/machine-learning/principal-component-analysis-algorithm-ZYIPa): a unsupervised algorithm that creates linear combination of the original features, the new features are orthogonal, which means that they are uncorrelated. Futhermore, they are ranked in order of their ["explained variance"](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c).
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
![Image](https://github.com/MWang93/Prepare/blob/master/tradeoff.png)
- example
    - knn(k=1): low bias, high variance.
    - knn(k large): high bias, low variance.
    - linear regression: high bias, low variance.
- [tradeoff](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)
    - build a good model, we need to find a good balance between bias and variance such that it minimizes the total error.
    - total error = biased^2 + variance + irreducible error
   

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
- error I & false positive rate: predict a man is pregnant, falsely infer the existence of something that is not there.
- error II & false negative rate: predict a pregnant woman is not pregnant, falsely infer the absence of something that is.
- statistical hypothesis testing, a type I error is the incorrect rejection of a true null hypotheis, while type II error is incorrectly retaining a false null hypothesis. 
 
#### 16. The cost of false positive is higher than false negative as well as the other way?
- large false positive cost: recruiting process, cost of hiring a bad candidate is much higher than passing a good one.
- large false negative cost: cancer prediction, predicting a cancer patient is healthy means this person cannot get proper treatement in time.
- note: minimizing the cost of false positive/negative is a machine learning problem as much as a product problem.

#### 17. Variables?
- nominal, dichotomous, ordinal
- interval, ratio

#### 18. Peason's correlation & Spearman's correlation
- pearson correlation coefficient measures the linear relationship between two datasets. Strictly speaking, Pearson’s correlation requires that each dataset be normally distributed. Like other correlation coefficients, this one varies between -1 and +1 with 0 implying no correlation.
- spearman's correlation is not restricted to linear relationships.

#### 20. Evaluation Metrics for Classification Model
- confustion matrix: 
    - sensitivity, specificity and detection rate. 
    - precision, recall and f1 score 
- [log loss](https://datawookie.netlify.com/blog/2015/12/making-sense-of-logarithmic-loss/): accuracy of a classifier by penalising false classifications, heavily penalises classifiers that are confident about an incorrect classification. For example, if for a particular observation, the classifier assigns a very small probability to the correct class then the corresponding contribution to the Log Loss will be very large indeed. 
- kappa: similar to Accuracy score, but it takes into account the accuracy that would have happened anyway through random predictions.
- ks statistic
- roc & auc
- concordance and discordance
- somers-d statistic
- gini coefficient: an indicator of how well the model outperforms random predictions. It can be computed from the area under the ROC curve using the following formula: Gini Coefficient = (2 * AUROC) - 1

#### 21. CBOW and Skip gram model
In CBOW the vectors from the context words are averaged before predicting the center word. In skip-gram there is no averaging of embedding vectors. It seems like the model can learn better representations for the rare words when their vectors are not averaged with the other context words in the process of making the predictions.

#### 22. [Type of Missing values](https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4)
- missing completely at random: when we say data are missing completely at random, we mean that the missingness is nothing to do with the person being studied.
- missing at random: we mean that the missingness is to do with the person but can be predicted from other information about the person. 
- missing not at random: the missingness is specifically related to what is missing.

In the first two cases, it is safe to remove the data with missing values depending upon their occurrences, while in the third case removing observations with missing values can produce a bias in the model.

#### 23. [PCA, t-SNE difference](https://www.biostars.org/p/295174/)
The main difference between t-SNE (or other manifold learning methods) and PCA is that t-SNE tries to deconvolute relationships between neighbors in high-dimensional data.

#### 24. Multi-Collinear features

#### 25. PCA and SVD
PCA (principal component analysis) is a method of extracting important variables (in form of components) from a large set of variables available in a data set. The idea is to calculate and rank the importance of features/dimensions. 
In order to do that, we use SVD (Singular value decomposition). SVD is used on covariance matrix to calculate and rank the importance of the features.
When the data has a zero mean vector PCA will have same projections as SVD, otherwise you have to centre the data first before taking SVD.

#### 26. For example, when you're given an unfeasible amount of predictors in a predictive modeling task, what're some ways to make the prediction more feasible?
- principal component analysis

#### 27. Now you have a feasible amount of predicctors, but you're fairly sure that you don't need all of them. How would you perform feature selection on the dataset?
- ridge / lasso / elastic net regression
- univariate Feature Selection where a statistical test is applied to each feature individually. You retain only the best features according to the test outcome scores
- "recursive feature elimination":
    - first, train a model with all the feature and evaluate its performance on held out data.
    - then drop let say the 10% weakest features (e.g. the feature with least absolute coefficients in a linear model) and retrain on the remaining features.
    - iterate until you observe a sharp drop in the predictive accuracy of the model.

#### 28. [Gradient Descent](https://www.kdnuggets.com/2017/04/simple-understand-gradient-descent-algorithm.html)
Gradient decent is an optimization algorithm used to minimize some fucntion by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. In ML, we use gradient descent to update the parameters of our model. Parameters refer to coefficients in linear regression and weights in neural networks.
- [learning rate = size of the steps](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html): how big the steps are that GD takes into the direction of the local minimum
    - [high learning rate](https://developers.google.com/machine-learning/crash-course/reducing-loss/learning-rate): cover more ground each step, and maybe not reach the local minimum and risk overshooting the lowest point.
    - low learning rate: more precise, but time-consuming, takes long time to get to the bottom.
- cost function: a loss function tells us "how good" our model is at making predictions for a given set of parameters.
- [terminologies](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)
    - epochs: one epoch is when an entire data set is passed forward and backward through the neural nentwork only once, as the number of epochs increase, more number of times the weight are changed, the curve goes from underfitting->optimal->overfitting.
    - batch size: number of training examples in a single batch.
    - iteration: number of batches needed to complete one epoch.
    - for 1 epoch: batch size * iterations(number of batches) = total data set, which means we can divide the data set of 2000 examples into batches of 500 then it will take 4 interations to complete 1 epoch.

#### 29. [Types of GD](https://www.hackerearth.com/blog/machine-learning/3-types-gradient-descent-algorithms-small-large-data-sets/)
- SGD: calculates the error and updates the model for **each example** in the training dataset. If we have N samples, in each epoch we will have N updates of the weights. One advantage is that the frequent updates allow us to have a pretty detailed rate of improvement.
- Batch Gradient Descent: calculates the error for **each example** in the training dataset, but only updates the model after all training examples have been evaluated and are accumulated. If we have N samples, in each epoch we will have 1 updates of the weights. It produces a stable error gradient and a stable convergence. Batch Gradient Descent has the disadvantage that the stable error gradient can sometimes result in a state of convergence that isn’t the best the model can achieve.
- Mini Batch Gradient Descent: splits the training dataset into small batches that are used to calculate model error and update model coefficients. If we divide the training set in X mini-batches, at the end of each epoch we will have X updates of the weights of the network, one for each mini-batch, for each mini-batch m, the algorithm calculates the error of each sample s, and it accumulates the gradients of the samples in m. It creates a balance between the robustness of stochastic gradient descent and the efficiency of batch gradient descent.
    
#### 30. How can you determine which features are the most important in your model?
- run the features from a Grandient Boosting Mahien or RF to generate plots of relative importance and information gain for each feature in the ensembles, these models are somewhat robust to collinearity as well so we could get the relative importance of the features.
- feature selection: it's simple and maintains interpretability of variables, but you gain no information from those variables you've dropped.
    - filter method:
        - for classification: Chi-Square, F-Test, Mutual Info
        - for regression: F-Test, Mutual Info
    - subset selection 
        - best subset selection
        - stepwise selection(forward, backward)
    - shinkage(regularization)
        - ridge regression
        - lasso
- feature extraction: we create ten “new” independent variables, where each “new” independent variable is a combination of each of the ten “old” independent variables. However, we create these new independent variables in a specific way and order these new variables by how well they predict our dependent variable.

#### 31. Unbalanced data for classification ML
[Idea](https://www.quora.com/In-classification-how-do-you-handle-an-unbalanced-training-set):
- [Resample differently](https://elitedatascience.com/imbalanced-classes)
    - up-sample minority class
    - down-sample majority class
- Try different metrics
- Penalized Models
    - a popular algorithm is Penalized-SVM
- Use tree-based algorithms
    - decision tree: always perform well on imbalanced datasets because their hierarhical structure allows them to learn signals from both classes
    - tree ensembles
- Anomaly Detection

Specific Ways:
- RF/ SMOTE boosting
- XGBoost/ hype-parameter optimisation
    - scale_pos_weight
- SVM /cost sensitive training

- CV
- Metrics

#### 32. [Activation Functions](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)
- it calculated the sum of weighted inputs, add bias, determines the output like yes or no.
- type: linear and non-linear
- non-linear:
    - sigmoid or logistic activation fucntion
    - tanh or hyperbolic tangent Activation Function
    - ReLU (Rectified Linear Unit) Activation Function
    - leaky ReLU
- [cheet sheet](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

#### 33. Different metrics for calculating KNN
- KNN is based on a simple underlying idea: classify a value of a new point based on the points close to it.
    - for prediction: use the average outcome of the neighbors.
    - for classification, use the majority of the votes of the k closet neighbors.
- requires 3 things:
    - the set of stored records
    - distance metric to compute distance 
    - the value of k: small k -> overfitting, large k -> high bias
- problem: dimensionality and scale
    - when the dimensionality is large, it's hard to find neighbors that are close enough, neighbors are far away
    - if a single numberic attribute has a large spread, it can dominate the distance metic
    

#### 34. Types of Ensemble Algorithm
- bagging: builds different classifier (high variance) by training on repeated samples (with replacement) from the data.
- boosting: combines simple base classifier (weak learners: high bias, low variance) by up-weighting data points which are classified incorrectly.
- random forest: averages many trees which are constructed with some amount of randomness.

#### 35. Logistic Regression 
- goal: estimating the log odds of an event.
- equation: ln(p/(1-p)) = a + b1x1 + b2x2 + ... 
- how: instead of using y as the dependent variable, we use a function of it, which is called logit, once the logit has been predicted, it can be mapped back to probability.

#### 36. SVM
- output SVM to probability: https://stackoverflow.com/questions/49507066/predict-probabilities-using-svm
- optimization?

# Some Resouces: 
https://www.analyticsvidhya.com/blog/2017/04/40-questions-test-data-scientist-machine-learning-solution-skillpower-machine-learning-datafest-2017/
https://ml-cheatsheet.readthedocs.io/en/latest/
