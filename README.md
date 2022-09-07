# Credit Risk Regression Analysis

## Overview

credit_risk_resampling.ipynb compares two different logistic regression models, which differ only in the training data provided. The purpose is to show the sensitivity of machine learning to data target label bias, and mitigation. The data at hand is loans, categorized as either healthy or high risk. Each loan has seven features, including interest rate, borrower income, and number of accounts. There are way more healthy loans than high risk loans, with a 30:1 sample ratio between healthy and high risk.

We first start with a naive logistic regression model, with training data stratified by the target label. We then follow up with the same logistic regression model, but using a RandomOverSampler to make the minority high-risk loans equal in training size to healthy loans.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Model 1 - Naive logistic regression:
  * Accuracy:  0.9443
  * Precision: 0.9925 (0.9964 for healthy loans, 0.8746 for high risk loans)
  * Recall:    0.9924 (0.9957 for healthy loans, 0.8928 for high risk loans)



* Model 2 - Naive logistic regression with random over sampler:
  * Accuracy:  0.9960
  * Precision: 0.9958 (0.9999 for healthy loans, 0.8725 for high risk loans)
  * Recall:    0.9952 (0.9951 for healthy loans, 0.9968 for high risk loans)

## Summary

Model 2 performs best, with precision/recall for healthy loans nearly identical to model 1, but with a significantly better recall for high risk loans (0.9968 vs 0.8928).

Identifying high risk loans is of higher importance as it directly increases the default risk of the company's holdings. Therefore, high recall on high risk loans (label `1`) is of most importance.
