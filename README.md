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
## [data](/data.csv)
csv file containing raw data
## [data Class](/dataClass.py)
Class to store and work with data break it into prices and statistics. Also
breaks it down between features and target values as well as scales the data
adds batches adds number of days used to predict, etc... It's the class that handles all of the data.
## [get the Hyperparameter](/get_best_models.py)
Helper function to determine the best model
## [MAIN](/main.py)
Run file to get results (we don't have optimal hyperparameters yet).
## [Hyperparameter](/params.json)
A file containing the best hyperparameters for the models
(Note I did something stupid in my previous validation, so we don't have optimal hyperparams; we will update when we do, but I have a slow desktop it is running on so it may take a while)
## [percent prediction models](/percent_pred_models.py)
All 4 models that predict the next day's percent change in price.
## [price prediction models](/price_pred_models.py)
All 4 models that predict the next day's price.
## [Train and test functions](/train_test_eval_funcs.py)
Functions called by main in order to train and test are finished models.
## [Validation Functions](/validation_functions.py)
Functions that are called for validation.
## [MAIN file for validation](/validation_main.py)
Run file to get the Hyperparameter for all 4 models.
## [MAIN2](/main2.py)
For now, this file is the same as main, but it will become the main function once I add everything in my to-do list. Basically will just implement more visualization and scaled results.
## [TODO](/TODO.txt)
A file containing what I have to do, not well organized, my scrambled thoughts on what I need to do.
## [GRAPHS/RESULTS](/graphs_results)
My current results after training for 1000 epochs with non optimal models (results are unscaled)
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
