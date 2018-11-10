# Product
#### 1. Important parameters in RF?
- number of trees
- minimum number of events per tree
- number of features randomly sampled at each split

#### 2. A related to B, we managed to move A in a forced way, but B doesn't change, so A wasn't causing B.
- make the proxy for A.
- select variabes that can relate to that A.
- build a model to predict, select the important ones.
- figure out how to positively effect those variables.

#### 3. Metrics drop in sudden?
- technicsal problem (goes to 0)
  - bug in UI.
  - bug in action to store data into db.
  - bug in query to pull data from db.
  - bug in dashboard to show metrics.
- sudden change in user behavior
  - user segment by various features to dig into which part has problem.
  
#### 4. Metrics is down?
Whenever there's a metric is down, start by breaking it down into its components, make assumption, analyze them independently (build a model to see the reasons).
- numerator
- denominator

#### 5. New feature?
- if feature is successful, would it be a good thing for site, can benefit KPI metrics?
- find a proxy for demand of that feature in current data.
- test until implement.

#### 6. Long term metrics?
- find a short term proxy for the long term metric, this's the point of this as well as other similar questions about long term user engagement, retaining users, or estimating customer lifetime value, 
- eg. estimating subscription retention rate, which is usually the percent of users unsubscribe within 12 months, so we could collecting the customer 12 month ago and see whether they unsubscribe it in 12 month, label them 1 or 0, build model with variables you think might influence the result, and the results shows a short term metric has big influence on that long term metrics, so do testing on that short term metrics instead of the long term metric.

#### 7. A/B test by market?

#### 8. Metrics for any new product launched
Engagement and retention are two sides of the same coin, retained users are the most engaged ones in the high majority of
cases.
- ability to acquire new users
- ability to retain current users

#### 9. Novelty effect
A good proxy to check for novelty effect is: only consider users for which it was the first experience on the site. First time users are obviously not affected by novelty effect.

#### 10. Insrumental variable vs Proxy variable
- An instrumental variable is used to help estimate a causal effect (or to alleviate measurement error). 
  - instrumental variablemust affect the independent variable of interest.
  - only affect the dependent variable through the independent variable of interest, called an exclusion restriction.
- A proxy variable is a variable you use because you think it is correlated with the variable you are really interested in, but have no (or poor) measurement of.

#### 11. We're trying to predict Y and how to find out whether X is a discriminant variable or not? If not, what are the actual discriminant variables？
Use tree to whether X or other variable are the discriminant variable to Y. 
- If the tree doesn't significantly use the X variable, you are sure that it doesn't really matter. In this
case, the tree would split on, for instance, combinations of other variables since that's where most of the
information is. If X is acting as a proxy for those variables, for sure the tree would choose those other
variables first, as they would allow for a better classification. After those splits, there is no information left to
extract from X so the tree won't touch it. It is important that you choose a model that works well with
correlated variables, like trees.
- In other case, if the tree does significantly use X variable, the next step is the find the reason for why it's the discriminant variable.

#### 12. What are the drawbacks of using supervised machine learning to predict frauds?
