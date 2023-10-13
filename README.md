# SUMMARY
Predict tommorows BTC price and or the percent change in the price. Given 5 days (this can be changed in
the code to add more days) predict the price change/price of Bitcoin. Uses 10 features that are historical 
financial data about bitcoin which i have taken from https://messari.io/report/messari-metrics.
Graph of incomplete results (see Note) can be found in the graphs_results folder.
# NOTE:
Not entirley finished waiting
on getting the results of hyperparamiters so results are not optimal. Also results/predictions
have not been rescaled 

# FILES
[data](/crypto_price_predictions/data.csv)
csv file containing raw data
[data Class](/crypto_price_predictions/dataClass.py)
Class to store and work with data break it into prices and statistics. Also
breaks it down between features and target values as well as scales the data
adds batches adds number of days used to predict etc... It's the class 
that handles all of the data.
[get the Hyperparameter](/crypto_price_predictions/get_best_models.py)
Helper function to determine the best model
[MAIN](/crypto_price_predictions/main.py)
Run file to get results (we don't have optimal hyperparamiters yet).
[Hyperparameter](/crypto_price_predictions/params.json)
A file containing the best hyperparamiters for the models
(Note i did something stupid in my previous validation so we don't
have optimal hyperparams will update when we do but I have a slow 
desktop it is running on so it may take a while)
[percent prediction models](/crypto_price_predictions/percent_pred_models.py)
All 4 models that predict the next days percent change in price.
[price prediction models](/crypto_price_predictions/price_pred_models.py)
All 4 models that predict the next days price.
[Train and test functions](/crypto_price_predictions/train_test_eval_funcs.py)
Functions called by main in order to train and test are finished models.
[Validation Funnctions](/crypto_price_predictions/validation_functions.py)
Functions that are called for validation.
[MAIN file for validation](/crypto_price_predictions/validation_main.py)
Run file to get the Hyperparameter for all 4 models.
[MAIN2](/crypto_price_predictions/main2.py)
For now this file is the same as main but it will become the main function
once I add everything in my todofile. Basically will just impliment more visulization and
scaled results.
[TODO](/crypto_price_predictions/TODO.txt)
A file containing what I have to do not well organized my scrambled thoughts on what I need 
to do.
[GRAPHS/RESULTS](/crypto_price_predictions/graphs_results)

# RESULTS
Unscaled and currently not the best version of each model. Will be updated when 
they are knwon.

# HOW TO RUN
main or main2 can currently be run if you would like to run the file yourself.

# TO-DO:
Rescale data
1. Rescale data
2. Finish getting optimal hyper params
3. Add more visualization
4. Post results 
