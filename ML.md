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


# BERT

InputExample Format(tsv): 
-  guid: Unique ID for the row
-  text_a: The label for the row (should be an int)
-  text_b: A column of the same letter for all rows. BERT wants, but we don’t use
-  labels: The text for row (will be empty for test data)

InputFeature Requires:
-  Convert InputExample to InputFeature
-  Purely numberical data
-  Tokenizing the text, truncate the long sequence and pad the short sequence to the given sequence length (max 512)

InputFeature Format:
- input_ids: list of numberical ids for the tokenised text
- input_mask: will be set to 1 for real tokens and 0 for the padding tokens
- segment_ids: 
- label_ids:one-hot encoded labels for the text

Masked LM

Next Sentence Prediction 

1. https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270
2. https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04
