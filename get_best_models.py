#file 2 get_best_models
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


def get_best():

	#epochs = 200

	#want smaller for this example
	epochs = 200
	my_data_class = DataClass()

	percent_data_list = my_data_class.get_percent_val_data()
	price_data_list = my_data_class.get_price_val_data()

	
	m1_m2_params = {'lstm_hidden_size':[i for i in range(5,35,5)],
	'fc_hidden_size':[i for i in range(5,35,5)],
	'num_layers': [i for i in range(1,4)],
	'dropout_prob': [0,.1,.2,.3,.4,.5],
	'learning_rate': [0.1,0.01,0.001,0.0001]}

	m3_m4_params = {'lstm_hidden_size': [i for i in range(5,35,5)],
	'fc_hidden_size': [i for i in range(5,35,5)],
	'fc_hidden_size_2': [i for i in range(5,35,5)],
	'num_layers': [i for i in range(1,4)],
	'dropout_prob': [0,.1,.2,.3,.4,.5],
	'learning_rate': [0.1,0.01,0.001,0.0001]}
	
	models_m1_m2 = {"Price_model_1": Price_model_1, 'Percent_model_1': Percent_model_1,
	"Price_model_2": Price_model_2, "Percent_model_2":Percent_model_2}

	models_m3_m4 = {"Price_model_3": Price_model_3, 'Percent_model_3': Percent_model_3,
	'Price_model_4': Price_model_4, 'Percent_model_4': Percent_model_4}

	results = {}
	
	for i,model in enumerate(models_m1_m2):
		if i % 2 == 0:
			print("Even price model started")
			params, loss, best_lr = getBestModel(price_data_list, models_m1_m2[model], get_combos(m1_m2_params), epochs)
			print("completed even price model")
		else:
			params, loss, best_lr = getBestModel(percent_data_list, models_m1_m2[model], get_combos(m1_m2_params), epochs)
			print("completed odd price model")

		results[model] = {'params': params, 'validation_loss': loss, 'lr':best_lr}

	for i,model in enumerate(models_m3_m4):
		if i % 2 == 0:
			params, loss, best_lr = getBestModel(price_data_list, models_m3_m4[model], get_combos(m3_m4_params), epochs)
			print("completed even percent model")
		else:
			params, loss, best_lr = getBestModel(percent_data_list, models_m3_m4[model], get_combos(m3_m4_params), epochs)
			print("completed odd percent model")

		results[model] = {'params': params, 'validation_loss': loss, 'lr':best_lr}

	return results
