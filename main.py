from get_models import *
from dataClass import *
from train_test_eval_funcs import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import copy

'''
TO DO: 

TEST TO MAKE SURE THAT WE ELIMINATE THE FIRST 5 IN UNSCALED Y'S AND NOT THE LAST 5
JUST WHATEVER DOES BETTER WHEN YOU TEST BOTH ON 1000 EPOCHS
MAKE SURE YOU CHANGE NOT ONLY THE TRAIN_TST_EVALS FUNCTIONS BUT FOR GRAPGING YOU
NEED TO ADJUST THEM WHEN YOU PASS IT YOU'LL SEE IT BELLOW YOU PASS THEM IN AS LIKE 
[UNSCALED_PRICE[5:], UNSCALED_PERCENT[5:]]

MAIN 2 SHOULD CONVER THE PRICES AND THE PERCENTAGES TO UNSCALED VALUES
BOTH THE PREDICTIONS AND THE ACTUAL VALUES SO WE CAN SEE HOW FAR OFF IT REALLY IS 
AT A PRICE/PERCENTAGE SCALE THAT IS NOT STANDARDIZED
DID THIS WHOLE THING WEIRD COULD PROBABLY BE IMPROVED THINK THERE WAS A REASON
I SCALED EVERYTHING THINK IT SEEMED TO IMPROVE EVERYTHING DURING TESTING BEEN A WHILE DON'T 
ENTIRLEY REMEMBER BUT 
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
			#training_losses.append(train(models[i], percent_train_data,epochs,learning_rates[i]))
			training_losses.append(train(models[i], percent_train_data,epochs,.001))
			#preds,Y,loss, diff = test_for_preds(models[i], price_test_data,scaler)
			preds,loss = test(models[i], percent_test_data,scaler_percent,unscaled_percent)
			predictions.append(preds)
			test_losses.append(loss)
			percent_model +=1
	plot_train_loss(model_labels,training_losses,True)
	'''
	NOTE THAT IF IT'S NOT THE FIRST 5 even though i'm PRETTY SURE IT IS WE WILL HAVE TO CHANGE THIS
	'''
	plot_test_results(model_labels,predictions,[unscaled_price[5:],unscaled_percent[5:]],True)
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
		temp = actual[0] if x%2 == 0 else actual[1]
		plt.scatter(x_axis,temp,color='blue',marker='x',label="REAL PRICE")
		plt.xlabel('INDEX/DAY')
		plt.ylabel("PRICE")
		plt.title(f"{title[x]} PREDICTIONS VS ACTUAL PRICE")
		plt.legend()
		if save_plot:
			plt.savefig(f'{title[x]} Prediction_results.png')



if __name__ == '__main__':
	main()