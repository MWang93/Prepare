# GLM and GAM

Goal: To solve linear regression model "weak" assumption: gaussian distribution, no interaction, linear relationship ...

Component: link function, weighted sum, a probability distribution

For example: 
-  non-Gaussian outcomes: add a log function if it's a Poisson distribution
-  interactions: add a column to the feature matrix that represents the interaction between the features and fit the model as usual
-  non-linear effects: simple transformation of feature, categorization of the feature, GAM

GAM:

Interpret: Partial dependency plots
Tuning Smoothness and Penalties: n_spline, lam, constrains

1. https://christophm.github.io/interpretable-ml-book/extend-lm.html
2. https://codeburst.io/pygam-getting-started-with-generalized-additive-models-in-python-457df5b4705f
3. https://medium.com/just-another-data-scientist/building-interpretable-models-with-generalized-additive-models-in-python-c4404eaf5515
