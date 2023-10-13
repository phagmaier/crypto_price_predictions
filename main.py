from get_models import *
from dataClass import *
from train_test_eval_funcs import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def main():
	epochs = 100
	training_losses = []
	predictions = []
	avg_price_diff = []
	test_losses = []
	ys = []
	params = get_all_model_params()
	models_and_lr = generate_models(params)
	learning_rates = [i[1] for i in models_and_lr]
	models = [i[0] for i in models_and_lr]
	data_class = DataClass()
	_, price_train_data, price_test_data, _, percent_train_data, percent_test_data =\
	data_class.get_all_data()
	scaler = data_class.get_scaler()	
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
			training_losses.append(train(models[i], price_train_data,epochs))
			preds,Y,loss= test(models[i], price_test_data)
			ys.append(Y)
			predictions.append(preds)
			test_losses.append(loss)
			
			price_model+=1
		else:
			#training_losses.append(train(models[i], percent_train_data,epochs,learning_rates[i]))
			training_losses.append(train(models[i], percent_train_data,epochs))
			#preds,Y,loss, diff = test_for_preds(models[i], price_test_data,scaler)
			preds,Y,loss = test_for_preds(models[i], percent_test_data,scaler)
			ys.append(Y)
			predictions.append(preds)
			test_losses.append(loss)
			percent_model +=1
	plot_train_loss(model_labels,training_losses,True)
	plot_test_results(model_labels, predictions,ys,True)
	plt.show()


def plot_train_loss(title, data,save_plot=False):
	for x,i in enumerate(data):
		plt.figure()
		#x = [i for i in range(len(i))]
		plt.plot(i)
		plt.xlabel('EPOCH')
		plt.ylabel("AVERAGE BATCH LOSS PER EPOCH")
		plt.title(f"{title[x]} AVERAGE BATCH LOSS PER EPOCH")
		if save_plot:
			plt.savefig(f'{title[x]} train_losses.png')

def plot_test_results(title, preds,actual,save_plot=False):
	for x,i in enumerate(preds):
		plt.figure()
		x_axis = [point for point in range(len(i))]
		plt.scatter(x_axis,i,color='red',marker='o', label="PREDICTED PRICE")
		plt.scatter(x_axis,actual[x],color='blue',marker='x',label="REAL PRICE")
		plt.xlabel('INDEX/DAY')
		plt.ylabel("PRICE")
		plt.title(f"{title[x]} PREDICTIONS VS ACTUAL PRICE")
		plt.legend()
		if save_plot:
			plt.savefig(f'{title[x]} Prediction_results.png')



if __name__ == '__main__':
	main()