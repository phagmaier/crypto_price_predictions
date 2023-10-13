#file 1 validate_function
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from itertools import product


def get_combos(params):
    combinations = list(product(*params.values()))
    all_combos = [dict(zip(params.keys(), combo)) for combo in combinations]
    real_combos = [i for i in all_combos if (i['num_layers'] > 1) 
                   or (i['num_layers'] == 1 and i['dropout_prob'] == 0)]

    #print(f"the number of combos is: {len(real_combos)}")
    return real_combos
    #return real_combos



def getBestModel(validation_list, model_class, combinations,epochs=200):
    torch.manual_seed(42)
    loss_fn = nn.MSELoss()
    lowest_loss = float('inf')
    best_params = None
    best_lr = None

    for hyper_params in combinations:
        lr = hyper_params['learning_rate']
        del hyper_params['learning_rate']
        validation_loss = 0
        for train, val in validation_list:
            model = model_class(**hyper_params)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            model.train()
            for epoch in range(epochs):
                for X,y in train:
                    optimizer.zero_grad()
                    preds = torch.reshape(model(X), y.shape)
                    loss = loss_fn(preds,y)
                    loss.backward()
                    optimizer.step()
            with torch.inference_mode():
                for X,y in val:
                    preds = torch.reshape(model(X), y.shape)
                    validation_loss += loss_fn(preds,y).item()
            if validation_loss < lowest_loss:
                lowest_loss = validation_loss
                #hyper_params['lr'] = lr
                best_lr = lr
                best_params = hyper_params
    return best_params, lowest_loss, best_lr