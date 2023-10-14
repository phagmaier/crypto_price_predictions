import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

'''
IDEALLY I WOULD EDIT THE Y's at the end when i put them in the data loader 
And probably scale them there as well so I don't have to do it in my main function but I 
am currently not doing that
'''

class DataClass:
    """
    Takes the following arguments (they have default values):
    file_name: name of the csv file default is data.csv
    train_percent: percentage of data to be used for training default is 0.8
    num_days: number of days to be used to predict the next day default is 5
    folds: k number of folds for validation default is 5
    batch_size:the batch size. Default is 32
        
    To get validation training and testing dataloaders for both types data
    Both percentage change and price use: get_all_data
    Will return a tuple with the following (all data loaders): 
    price_val_dl,price_train_dl,price_test_dl,percent_val_dl,percent_train_dl,percent_test_dl 
    You get all the price data first and then percent data returned in a tuple
    validation is a list of tuples first one is to train second to evaluate
    then you get the training dataloader then testing dataloader
        
    For just next days price:
    To get just validation datasets for model evaluation: get_price_val_data
    For just model training and evaluation: get_price_data
    
    For just next days percentage change:
    Validation dataset for model evaluation: get_percent_val_data
    For model training and evaluation: get_percent_data    
    """
    def __init__(self, file_name = 'data.csv', train_percent=0.8,
                num_days=5, folds=5,batch_size=32):
        self.file_name = file_name
        self.df = pd.read_csv(file_name)
        self.train_percent = train_percent
        self.num_days=num_days
        self.folds = folds
        self.batch_size=32
        self.num_features = len(self.df.columns)
        self.scaler = None
        self.unscaled_ys = None

    def get_scaler(self):
        return self.scaler
        
    def get_num_features(self):
        return self.num_features
        
    #just in case
    def get_original_dataframe(self):
        return self.df
    
    def get_unscaled_data(self):
        return self.unscaled_ys, self.scaler
        
    def get_percent_val_data(self):
        X,y,= self.percent_change_data()
        X_train, X_test, y_train, y_test,unscaled_ys = self.prct_train_test_split_scale(X,y)
        self.unscaled_ys = unscaled_ys
        X_train, y_train = self.put_days_together_percentage(X_train,y_train)
        X_test, y_test = self.put_days_together_percentage(X_test,y_test)
        val_dataloader = self.validation_dataLoaders(X_train,y_train)
        return val_dataloader
        
    
    #calls functions to get percent chnage data
    def get_percent_data(self):
        X,y = self.percent_change_data()
        X_train, X_test, y_train, y_test,unscaled_ys = self.prct_train_test_split_scale(X,y)
        self.unscaled_ys = unscaled_ys
        X_train, y_train = self.put_days_together_percentage(X_train,y_train)
        X_test, y_test = self.put_days_together_percentage(X_test,y_test)
        train_dataloader, test_dataloader =\
        self.get_dataloader(X_train,X_test,y_train,y_test)
        return train_dataloader, test_dataloader
    
    def get_price_val_data(self):
        X,y= self.get_price_change()
        X_train, X_test, y_train, y_test, unscaled_y =\
        self.train_test_split_scale(X,y)
        self.unscaled_ys = unscaled_y
        
        X_train, y_train = self.put_days_together(X_train,y_train)
        X_test, y_test = self.put_days_together(X_test,y_test)
        _, unscaled_y = self.put_days_together([], unscaled_y)
        val_dataloader = self.validation_dataLoaders(X_train,y_train)
        return val_dataloader
    
    def get_price_data(self):
        X,y= self.get_price_change()
        X_train, X_test, y_train, y_test, unscaled_y =\
        self.train_test_split_scale(X,y)
        
        self.unscaled_ys = unscaled_y
        
        X_train, y_train = self.put_days_together(X_train,y_train)
        X_test, y_test = self.put_days_together(X_test,y_test)
        _, unscaled_y = self.put_days_together([], unscaled_y)
        train_dataloader, test_dataloader =\
        self.get_dataloader(X_train, X_test, y_train, y_test)
        return train_dataloader, test_dataloader
        
    def get_all_data(self):
        price_train_dl, price_test_dl = self.get_price_data()
        price_val_dl = self.get_price_val_data()
        percent_train_dl,percent_test_dl  = self.get_percent_data()
        percent_val_dl = self.get_percent_val_data()
        return price_val_dl,price_train_dl,price_test_dl,percent_val_dl,percent_train_dl,percent_test_dl 
    
    def percent_change_data(self):
        df = pd.read_csv(self.file_name)
        
        # Calculate the next day's percentage change
        df['next_day_price'] = df['price'].shift(-1)
        df['target'] = ((df['next_day_price'] - df['price']) / df['price']) * 100
        
        # Drop the last row since there's no next day for it
        df.dropna(inplace=True)
        y = df['target'].values
        df.drop(['next_day_price', 'target'], axis=1, inplace=True)
        
        return df.values, y
    
    
    def prct_train_test_split_scale(self,X,y):    
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        train_size = int(self.train_percent * len(X))
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        feature_scaler.fit(X_train)
        target_scaler.fit(y_train.reshape(-1,1))
        self.scaler = target_scaler
        X_train = feature_scaler.transform(X_train)
        X_test = feature_scaler.transform(X_test)
        y_train = target_scaler.transform(y_train.reshape(-1,1))
        y_test = target_scaler.transform(y_test.reshape(-1,1))
        return X_train, X_test, y_train, y_test,y[train_size:]
    
    def put_days_together_percentage(self, X, y):
        xs = np.array([X[i:i+self.num_days] for i in range(len(X)-self.num_days)])
        ys = np.array([y[i+self.num_days-1] for i in range(len(y)-self.num_days)])
        return xs, ys
    
    
    def validation_dataLoaders(self, X,y):
        kf = KFold(n_splits=self.folds, shuffle=False)
    
        validation_dataloader = []

        for train_index, test_index in kf.split(X):
            x_train = torch.tensor(X[train_index], dtype=torch.float32) 
            x_test = torch.tensor(X[test_index], dtype=torch.float32)
        
            y_train = torch.tensor(y[train_index], dtype=torch.float32)
            y_test = torch.tensor(y[test_index], dtype=torch.float32)
        
            temp = TensorDataset(x_train, y_train)
            train_dataloader = DataLoader(temp, batch_size=self.batch_size, shuffle=False)
        
            temp = TensorDataset(x_test, y_test)
            val_dataloader = DataLoader(temp, batch_size=len(y_test), shuffle=False)
        
            validation_dataloader.append((train_dataloader,val_dataloader))
        return validation_dataloader
    
    
    def get_dataloader(self,X_train, X_test, y_train, y_test):
    
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
    
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=len(y_test), shuffle=False)  # Using entire testing dataset

        return train_dataloader, test_dataloader
    
    
    def get_price_change(self):
        df = pd.read_csv(self.file_name)
        price_column = df['price']
        df.drop('price', axis=1,inplace=True)
        return df.values, price_column.values
    
    def train_test_split_scale(self,X, y):
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        train_size = int(self.train_percent * len(X))
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

        feature_scaler.fit(X_train)
        target_scaler.fit(y_train.reshape(-1,1))
        self.scaler = target_scaler
        X_train = feature_scaler.transform(X_train)
        X_test = feature_scaler.transform(X_test)
        y_train = target_scaler.transform(y_train.reshape(-1,1))
        y_test = target_scaler.transform(y_test.reshape(-1,1))

        X_train = np.hstack((y_train, X_train))
        X_test = np.hstack((y_test.reshape(-1,1), X_test))

        return X_train, X_test, y_train, y_test, y[train_size:]
    
    def put_days_together(self, X, y):
        xs = np.array([X[i:i+self.num_days] for i in range(len(X)-self.num_days)])
        ys = np.array([y[i+self.num_days] for i in range(len(y)-self.num_days)])
        #return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)
        return xs, ys
        

