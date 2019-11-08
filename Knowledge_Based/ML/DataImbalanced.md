Reference: https://www.samueltaylor.org/static/pdf/class_imbalance.pdf

Causes
    
    a. Lack of data
    
    b. Overlapping
    
    c. Noise
    
    d. Biased Estimators

Recognition
    
    a. print(df['class'].value_counts(normalize=True))
    
    b. compare it
    ```
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    dumb_model = DummyClassifier().fit(X_train, y_train)
    y_pred = dumb_model.predict(X_test)
    dumb_accuracy = accuracy_score(y_test, y_pred) # 0.9375
    fancy_model = RandomForestClassifier().fit(X_train, y_train)
    y_pred = fancy_model.predict(X_test)
    fancy_accuracy = accuracy_score(y_test, y_pred) # 0.9675
    ```
    
    c. better metrics
    
    d. be careful with train/test splits

Solution

    a. Gather more data

    b. Preprocessing (Oversampling: random, SMOTE, ADASYN; Undersampling)

    c. Special-purpose learners (randomforestclassier, logisticregression, svc, LGBMclassifier, XGBclassifier) has class weight parameters, but weighting is less effective under high imbalance and more effective with more data.

    d. Post-processing (threshold selection, cost-based classification: use ROC curve or MetaCost to choose a threshold)