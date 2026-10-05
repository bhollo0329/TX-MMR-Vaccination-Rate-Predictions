# TX-MMR-Vaccination-Rate-Predictions
Attempting to create a model to predict MMR vaccination rates in Texas at the district level

Primary data sources include ACS and TEA data.

Try 5 models (3 traditional stat regression, 2 ML regression algorithms) to predict MMR vaccination rates.

The 3 traditional methods (Linear, Ridge and Logit) fail to model the low end tail of vaccination rates due to being extreme outliers. The XGBoost performs the best after tuning of hyperparameters using the k-fold method.
