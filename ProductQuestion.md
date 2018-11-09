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
- find a short term proxy for the long term metric.

#### 7. A/B test by market?

#### 8. Metrics for any new product launched
engagement and retention are often two sides of the same coin, retained users are the most engaged ones in the high majority of
cases.
- ability to acquire new users
- ability to retain current users
