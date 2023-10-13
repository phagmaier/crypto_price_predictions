#file 3 validation_main
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from validation_functions import get_combos, getBestModel
from dataClass import DataClass
from price_pred_models import Price_model_1,Price_model_2,Price_model_3,Price_model_4
from percent_pred_models import Percent_model_1, Percent_model_2,Percent_model_3,Percent_model_4
from itertools import product
from get_best_models import get_best
import json

def main():
	results = get_best()

	with open('bitcoin_models_params', "w") as json_file:
		json.dump(results, json_file, indent=4)

	return 




main()