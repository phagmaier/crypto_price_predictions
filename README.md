# SUMMARY
Predict tommorows BTC price and or the percent change in the price given 5 days (this can be changed in
the code to add more or less days) of financial features. Uses 10 features that are historical 
financial data about bitcoin which i have taken from https://messari.io/report/messari-metrics.
Graph of incomplete results (see Note) can be found in the graphs_results folder.
# NOTE:
Not entirely finished waiting still
on getting the results of hyperparamiters so results are not optimal. 
Current results based on innacuarte params and for perdiction models used a learning rate of .001 and 
for pure price predictions used a learning rate of .01.
Graphs/results are the from running for 500 epochs.
graphs_results folder contains current predictions and the average loss per batch for each epoch
# FILES

| File                     | Description                                                                                          |
|-------------------------- |------------------------------------------------------------------------------------------------------|
| [data.csv](data.csv)      | CSV file containing raw data.                                                                        |
| [dataClass.py](dataClass.py) | Class to store and work with data, breaking it into prices and statistics. Handles data scaling, batching, and prediction day configuration. |
| [get_best_models.py](get_best_models.py) | Helper function to determine the best model. |
| [main.py](main.py)       | Run file to get results (please note that optimal hyperparameters are not available yet).         |
| [params.json](params.json) | JSON file containing the best hyperparameters for the models. (NOTE NOT ACCURATE YET)|
| [percent_pred_models.py](percent_pred_models.py) | Contains all 4 models that predict the next day's percent change in price. |
| [price_pred_models.py](price_pred_models.py) | Contains all 4 models that predict the next day's price. |
| [train_test_eval_funcs.py](train_test_eval_funcs.py) | Functions called by the main script to train and test the models. |
| [validation_functions.py](validation_functions.py) | Functions used for validation purposes. |
| [validation_main.py](validation_main.py) | Run file for obtaining hyperparameters for all 4 models. |
| [TODO.txt](TODO.txt)     | A file containing a to-do list and notes on what needs to be done. |
| [graphs_results](graphs_results) | Directory containing graphs and results (unscaled and currently not the best versions). |

# HOW TO RUN/Dependencies
## DEPENDENCIES
### PYTORCH
### SKLEARN
### PANDAS
### NUMPY
### PANDAS 

USES PYTHON3
(Sorry i'm a 'data scientist' which means i just use libraries all day.) 
SIMPLY RUN MAIN TO REPLICATE RESULTS

# TO-DO:
1. Finish getting optimal hyper params
2. Add more visualization
3. Post results 
4. Refactor my DataClass so I don't have to manually rescale
5. Refactor to allow for easier customization whether that be to add and or reduce features
or increases the number of days used to predict the next day
6. Record results of predicting tomorows price predictions see how much money I would have 
lost/gained if I pulled out my investments on predictions of it
going down/vs putting money back in when I preict it going up
7. Combine all models and predictions weighing it by the accuracy (possibly see what model is best at predicting 
large/small changes and weight it using that) and essentially combine the predictions to make a more 
informed decision
8. Stop training when learning rate stops moving and or drops below a certain threshold


