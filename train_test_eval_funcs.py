import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

'''
NOTES:
BECAUSE YOU NEED 5 DAYS OF DATA AND BECAUSE YOU DID SOMETHING STUPID IN THE DATA CLASS
YOU HAVE TO (COULD NOT HAVE TO COULD JUST USE SCALED Y'S) PASS THE UNSCALED Y'S AND 
THEN CUT OFF THE FIRST 5 BECAUSE YOU TAKE 5 DAYS WORTH OF DATA TO PREDICT THE NEXT DAY
THIS SHOULD PROBABLY BE CHANGED SO IT'S HANDLED IN THE DATA CLASS
'''

def train(model, data, epochs, lr=.001):
	avg_epoch_loss = [] 
	torch.manual_seed(42)
	loss_fn = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(),lr=lr)
	model.train()
	for epoch in range(epochs):
		batch_loss = []
		for X, y in data:
			#y = torch.tensor(y)
			optimizer.zero_grad()
			temp = model(X)
			preds = model(X).reshape(y.shape)
			loss = loss_fn(preds,y)
			batch_loss.append(loss)
			loss.backward()
			optimizer.step()
		avg_epoch_loss.append(sum([abs(i) for i in batch_loss]) / len(batch_loss))
	avg_epoch_loss = [i.item() for i in avg_epoch_loss]
	return avg_epoch_loss


def test(model,test_data,scaler,Y):
	Y = torch.tensor(Y[5:]).reshape(-1,1)
	with torch.inference_mode():
		loss_fn = nn.MSELoss()
		for X,_ in test_data:
			preds = model(X)
			preds = convert(scaler,preds)
			test_loss = loss_fn(preds.reshape(Y.shape),Y)

	return preds.numpy(),test_loss.item()

def convert(scaler, predictions):
	numpied = predictions.numpy()
	unscaled_preds = scaler.inverse_transform(predictions.reshape(-1,1))
	back_to_torch = torch.from_numpy(unscaled_preds)
	return back_to_torch







