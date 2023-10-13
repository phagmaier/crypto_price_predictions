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

| File                     | Description                                                                                          |
|-------------------------- |------------------------------------------------------------------------------------------------------|
| [data.csv](data.csv)      | CSV file containing raw data.                                                                        |
| [dataClass.py](dataClass.py) | Class to store and work with data, breaking it into prices and statistics. Handles data scaling, batching, and prediction day configuration. |
| [get_best_models.py](get_best_models.py) | Helper function to determine the best model. |
| [main.py](main.py)       | Run file to get results (please note that optimal hyperparameters are not available yet).         |
| [params.json](params.json) | JSON file containing the best hyperparameters for the models. |
| [percent_pred_models.py](percent_pred_models.py) | Contains all 4 models that predict the next day's percent change in price. |
| [price_pred_models.py](price_pred_models.py) | Contains all 4 models that predict the next day's price. |
| [train_test_eval_funcs.py](train_test_eval_funcs.py) | Functions called by the main script to train and test the models. |
| [validation_functions.py](validation_functions.py) | Functions used for validation purposes. |
| [validation_main.py](validation_main.py) | Run file for obtaining hyperparameters for all 4 models. |
| [main2.py](main2.py)     | Currently the same as `main.py`, but it will become the main function once more features are added. |
| [TODO.txt](TODO.txt)     | A file containing a to-do list and notes on what needs to be done. |
| [graphs_results](graphs_results) | Directory containing graphs and results (unscaled and currently not the best versions). |

# HOW TO RUN
main or main2 can currently be run if you would like to run the file yourself.

# TO-DO:
1. Rescale data
2. Finish getting optimal hyper params
3. Add more visualization
4. Post results 
