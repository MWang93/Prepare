# Testing

#### 1. In an A/B test, how can you check if assignment to the various buckets was truly random?
- Plot the distributions of multiple features for both A and B and make sure that they have the same shape. More rigorously, we can conduct a permutation test to see if the distributions are the same.
- MANOVA to compare different means

#### 2. What might be the benefits of running an [A/A test](https://www.optimizely.com/optimization-glossary/aa-testing/), where you have two buckets who are exposed to the exact same product?
- Determine the baseline.
- Verify the sampling algorithm is random.
#### 3. What would be the hazards of letting users sneak a peek at the other bucket in an A/B test?
- The user might not act the same suppose had they not seen the other bucket. You are essentially adding additional variables of whether the user peeked the other bucket, which are not random across groups.
#### 4. What would be some issues if blogs decide to cover one of your experimental groups?
- Same as the previous question. The above problem can happen in larger scale.
#### 5. How would you conduct an A/B test on an opt-in feature? 
- Ask someone for more details.
#### 6. [How would you run an A/B test for many variants, say 20 or more?](https://www.quora.com/How-would-you-run-an-A-B-test-for-many-variants-say-20-or-more)
The more variances  means the higher the chance of a false positive, the higher your chances of finding a winner that is not significant. 
For example: Google 41 shades of blue, which is called 'Multiple Comparison Problem'.  
You can calculate the chance of getting a false positive using the following formula: 

1-(1-a)^m

with m being the total number of variations tested and abeing the significance level. With a significance level of 0.05, the equation would look like this: 1-(1-0.05)^m or 1-0.95^m. 
Solution: Bonferroni Correction
- one control, 20 treatment, if the sample size for each group is big enough.
- ways to attempt to correct for this include changing your confidence level (e.g. Bonferroni Correction) or doing family-wide tests before you dive in to the individual metrics (e.g. Fisher's Protected LSD).
