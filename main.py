from get_models import *
from dataClass import *
from train_test_eval_funcs import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import copy
import numpy as np
from ensembl_funcs import *

'''
TO DO: 
BEFORE PASSING UNSCALED TO THE ENSEMBLE FUNC MAKE SURE TO CUT OFF THE FIRST 5 
NEED TO FLATTEN PREDICTIONS
'''

def main():
	epochs = 500
	training_losses = []
	predictions = []
	avg_price_diff = []
	test_losses = []
	params = get_all_model_params()
	models_and_lr = generate_models(params)
	learning_rates = [i[1] for i in models_and_lr]
	models = [i[0] for i in models_and_lr]
	data_class = DataClass()

	price_train_data,price_test_data = data_class.get_price_data()
	unscaled_price,scaler_price = data_class.get_unscaled_data()
	unscaled_price = copy.deepcopy(unscaled_price)
	scaler_price = copy.deepcopy(scaler_price)

	percent_train_data, percent_test_data = data_class.get_percent_data()
	unscaled_percent,scaler_percent = data_class.get_unscaled_data()
	unscaled_percent = copy.deepcopy(unscaled_percent)
	scaler_percent = copy.deepcopy(scaler_percent)
	
	#scaler = data_class.get_scaler()	
	price = "PRICE MODEL "
	percent = "PERCENT MODEL "
	model_labels = []
	for i in range(1,5):
		model_labels.append(price + str(i))
		model_labels.append(percent + str(i))

	
	price_model = 1
	percent_model = 2

	for i in range(len(models)):
		if i%2 == 0: 
			training_losses.append(train(models[i], price_train_data,epochs,.01))
			preds,loss= test(models[i], price_test_data,scaler_price,unscaled_price)
			predictions.append(preds)
			test_losses.append(loss)
			price_model+=1
		else:
			training_losses.append(train(models[i], percent_train_data,epochs,.001))
			preds,loss = test(models[i], percent_test_data,scaler_percent,unscaled_percent)
			predictions.append(preds)
			test_losses.append(loss)
			percent_model +=1
	
	plot_train_loss(model_labels,training_losses,True)
	plot_test_results(model_labels,predictions,[unscaled_price[5:],unscaled_percent[5:]],True)

	#MAY HAVE TO USE NEW NAMES CAUSE I THINK IT FUCKS WITH PLOTTING
	ensemble_predictions = []
	targets = unscaled_price[5:].flatten()
	for i in range(len(predictions)):
		ensemble_predictions.append(predictions[i].flatten())

	preds = []
	errors = []
	for i in range(len(ensemble_predictions)):
		if i%2==0:
			preds.append(ensemble_predictions[i])
			errors.append(price_error(ensemble_predictions[i],targets))
		else:
			temp1,temp2 = percent_to_val(ensemble_predictions[i],targets)
			preds.append(temp1)
			errors.append(temp2)



	ensemble_preds = get_ensemble_pred(preds,errors)

	plot_ensemble(ensemble_preds,targets,True)
	plt.show()


def plot_train_loss(title, data,save_plot=False):
	for x,i in enumerate(data):
		plt.figure()
		plt.plot(i)
		plt.xlabel('EPOCH')
		plt.ylabel("AVERAGE BATCH LOSS PER EPOCH")
		plt.title(f"{title[x]} AVERAGE BATCH LOSS PER EPOCH")
		if save_plot:
			plt.savefig(f'graphs_results/{title[x]} train_losses.png')

def plot_test_results(title, preds,actual,save_plot=False):
	for x,i in enumerate(preds):
		plt.figure()
		x_axis = [point for point in range(len(i))]
		plt.scatter(x_axis,i,color='red',marker='o', label="PREDICTED PRICE")
		temp = actual[0] if x%2 == 0 else actual[1]
		plt.scatter(x_axis,temp,color='blue',marker='x',label="REAL PRICE")
		plt.xlabel('INDEX/DAY')
		plt.ylabel("PRICE")
		plt.title(f"{title[x]} PREDICTIONS VS ACTUAL PRICE")
		plt.legend()
		if save_plot:
			plt.savefig(f'graphs_results/{title[x]} Prediction_results.png')

def plot_ensemble(preds, targets,save_plot=False):
	plt.figure()
	x_axis = [point for point in range(len(targets))]
	plt.scatter(x_axis, preds, color = 'red', marker='o', label="PREDICTED PRICE")
	plt.scatter(x_axis,targets,color='blue',marker='x',label="REAL PRICE")
	plt.xlabel('INDEX/DAY')
	plt.ylabel("PRICE")
	plt.title(f"ENSEMBLE PREDICTIONS VS ACTUAL PRICE")
	plt.legend()
	if save_plot:
		plt.savefig(f'graphs_results/ENSEMBLE_Prediction_results.png')





if __name__ == '__main__':
	main()