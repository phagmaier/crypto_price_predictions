import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def train(model, data, epochs, lr=.001):
	avg_epoch_loss = [] #don't append anything to this currently
	torch.manual_seed(42)
	loss_fn = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(),lr=lr)
	model.train()
	for epoch in range(epochs):
		batch_loss = []
		for X, y in data:
			optimizer.zero_grad()
			temp = model(X)
			#preds = torch.reshape(model(X), y.shape)
			preds = model(X).reshape(y.shape)
			loss = loss_fn(preds,y)
			batch_loss.append(loss)
			loss.backward()
			optimizer.step()
		avg_epoch_loss.append(sum(batch_loss)/len(batch_loss))
	avg_epoch_loss = [i.item() for i in avg_epoch_loss]
	return avg_epoch_loss


def test(model, test_data):
	with torch.inference_mode():
		loss_fn = nn.MSELoss()
		for X,y in test_data:
			preds = model(X).reshape(y.shape)
			test_loss = loss_fn(preds,y)
	#return preds.numpy(), y.numpy(), test_loss.item(), eval_prct_diff(preds,y)
	return preds.numpy(), y.numpy(), test_loss.item()

def test_for_preds(model,test_data,scaler):
	with torch.inference_mode():
		loss_fn = nn.MSELoss()
		for X,y in test_data:
			#preds = torch.reshape(model(X), y.shape)
			preds = model(X).reshape(y.shape)
			#preds = torch.reshape(model(X), y.shape)
			#preds,Y = convert(scaler,preds,y)
			test_loss = loss_fn(preds,y)
	#return preds.numpy(), y.numpy(),test_loss.item(), eval_prct_diff(preds,y)
	return preds.numpy(), y.numpy(),test_loss.item()

'''
def eval_prct_diff(predictions, actual):
	return ((predictions - actual) / actual) * 100
'''

'''
ERROR:
Reshape your data either using array.reshape(-1, 1) 
if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
'''
def convert(scaler, predictions,X):
	numpied = predictions.numpy()
	#may have to transform the data 
	unscaled_preds = scaler.inverse_transform(predictions.reshape(-1,1))
	unscaled_target = scaler.inverse_transform(X.reshape(-1,1))
	back_to_torch = torch.from_numpy(unscaled_preds)
	back_to_torch_X = torch.from_numpy(unscaled_target)
	return back_to_torch,back_to_torch_X





