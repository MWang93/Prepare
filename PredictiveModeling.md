# Predictive Modeling
#### 1. Given a Dataset and build a model
-  Classification or regression?
-  Split dataset into train, validation, test. Run models.
-  Evaluate the model: cross validation..
-  Visualize and Insight.

#### 2. Distribution of test data is sigfinicantly different than training data, what are the issues?
-  Fit training data well but fails the test data. 
-  When there is a change in data distribution, this is called data shift. 
-  Occur when: 
    -  Covariate Shift: P(y|x) are the same but P(x) are different. 
    -  Concept Shift: P(y|x) are different. 
-  Reasons: 
    -  sample selection bias (the training examples have been obtained through a biased method, such as non-uniform selection);
    -  Non-stationary environments: training environment is different from testing one due to a temporal or spatial change, like adversarial classification problems, such as spam filtering and network intrusion detection.
-  Solution: we need importance weighted cv.

#### 3. How to make your model more robust to outliers?
-  Regularization such as L1 and L2 to reduce variance
    -  L1 (lasso): shink some parameters to zero.
    -  L2 (ridge): force the parameters to be relatively small.
    -  L1/L2 (elastic-net): a mix of both L1 and L2 regularizations, a penalty is applied to the sum of the absolute values and to the sum of the squared values.
-  Algotithm changes:
    -  Tree based methods instead of regression method.
    -  For statistical tests, use non parametric tests instead of parametrics one.
    -  Use robust error metrics such as MAE instead of MSE.
-  Data changes:
    -  Winsorize the data.
    -  Transform the data.
    -  Remove if you're certain they're anomalies not worth predicting.

#### 4. Under which situation, use what error metrics?
-  MSE: less robust to outliers, easier to computer the gradient. 
-  MAE: more robust to outliers, harder to fit the model because it cannot be numerically optimized, so when there are less variability in the model and model is computationally easy to fit.
    
#### 5. How to evaluate a binary classifier?
-  Accuracy
-  AUROC
-  Logloss/deviance

#### 6. Regularization
https://www.analyticsvidhya.com/blog/2018/04/fundamentals-deep-learning-regularization-techniques/
   
#### 7. Logistic 
https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
