# SUMMARY

Predict tomorrow's BTC price and/or the percent change in the price given 5 days 
(this can be changed in the code to add more or fewer days) of financial features. 
Uses 10 features that are historical financial data about Bitcoin,
which were obtained from [Messari Metrics](https://messari.io/report/messari-metrics). 
Graphs of incomplete results (see Note) can be found in the "graphs_results" folder.

# NOTE

This project is not entirely finished, as we are waiting to obtain the results of hyperparameters, 
so results are not optimal. Current results are based on inaccurate parameters. For prediction models, 
a learning rate of 0.001 is used, and for pure price predictions, a learning rate of 0.01 is employed.
Graphs/results are generated after running for 500 epochs.
The [graphs_results](graphs_results) folder contains current predictions and the average loss per batch for each epoch.

# FILES

| File                                         | Description                                                          |
|---------------------------------------------- |----------------------------------------------------------------------|
| [data.csv](data.csv)                          | CSV file containing raw data.                                        |
| [dataClass.py](dataClass.py)                  | Class to store and work with data, breaking it into prices and statistics. Handles data scaling, batching, and prediction day configuration. |
| [get_best_models.py](get_best_models.py)      | Helper function to determine the best model.                           |
| [main.py](main.py)                            | Run file to get results (please note that optimal hyperparameters are not available yet). |
| [params.json](params.json)                    | JSON file containing the best hyperparameters for the models (NOTE: NOT ACCURATE YET). |
| [Percent Change Prediction Models](percent_pred_models.py) | Contains all 4 models that predict the next day's percent change in price. |
| [Price Prediction Models](price_pred_models.py) | Contains all 4 models that predict the next day's price. |
| [Train Test and Evaluation Functions](train_test_eval_funcs.py) | Functions called by the main script to train and test the models. |
| [validation functions](validation_functions.py) | Functions used for validation purposes. |
| [validation driver](validation_main.py)       | Run file for obtaining hyperparameters for all 4 models. |
| [TODO](TODO.txt)                              | A file containing a to-do list and notes on what needs to be done. |
| [graphs and results](graphs_results)         | Directory containing graphs and results (unscaled and currently not the best versions). |
| [Ensemble Functions](ensembl_funcs.py)        | Functions that help combine a weighted prediction from all models.

# HOW TO RUN/Dependencies

## DEPENDENCIES
- PYTORCH
- SKLEARN
- PANDAS
- NUMPY
- PANDAS

USES PYTHON 3

(Sorry, I'm a 'data scientist,' which means I just use libraries all day.)

Simply run `main.py` to replicate the results.

# TO-DO

1. Finish getting optimal hyperparameters.
2. Add more visualization.
3. Post results.
4. Refactor my DataClass so I don't have to manually rescale.
5. Refactor to allow for easier customization, whether that means adding or reducing features or increasing the number of days used to predict the next day.
6. Record results of predicting tomorrow's price predictions to see how much money I would have gained or lost if I pulled out my investments on predictions of it going down versus putting money back in when I predict it going up.
7. Stop training when the learning rate stops moving or drops below a certain threshold.


# NOTES ON ENSEMBLE

The Ensemble prediction is now working, but technically the first of the testing data ensemble predictions
is not completely accurate since I don't have the raw input for the very first price. 
This can easily be fixed, but as of now, that first ensemble prediction for the very first day in the testing 
isn't as 'accurate' as I'd like it to be. This would not matter in a non-evaluation setting, though,
since you'd have the unscaled price data if you were to do this every day to get the price,
and it's only one data point in the evaluation, so it's not a big deal.
Probably should have just taken this out of the graph, but as it stands, it is present.

# HYPERPARAMETERS

I DON'T HAVE THE OPTIMAL HYPERPARAMETERS YET! The validation process is running on my (incredibly) slow desktop, 
and I'll use those once it's finished, and then repost the results with what should be better hyperparameters.
I'll test on various different epoch sizes. Once those are in, I'll report back in greater detail about the accuracy loss, etc.

# PARAMS.json

See above. Those aren't accurate; I made a mistake in my previous validation, which led to inaccurate params.

## BEST MODEL AT THE MOMENT

Currently, the ensemble model seems to be getting the best results with out current suboptimal params

