import torch
import torch.nn as nn


class Price_model_1(nn.Module):
    def __init__(self, num_features=10,lstm_hidden_size=10,
                fc_hidden_size=10,num_layers=2, dropout_prob=0):
        
        super().__init__()
        
        self.lstm = nn.LSTM(num_features, lstm_hidden_size, 
                           batch_first=True, bidirectional=True,
                           num_layers=num_layers, dropout=dropout_prob)
        
        self.fc1 = nn.Linear(lstm_hidden_size*2, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        
    def forward(self, x): 
        _, (hidden,cell) = self.lstm(x)
        out = torch.cat((hidden[-2, :, :],hidden[-1, :, :]), dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class Price_model_2(nn.Module):
    def __init__(self, num_features=10,lstm_hidden_size=10,
                fc_hidden_size=10,num_layers=2, dropout_prob=0):
        
        super().__init__()
        
        self.lstm = nn.LSTM(num_features, lstm_hidden_size, 
                           batch_first=True, bidirectional=True,
                           num_layers=num_layers, dropout=dropout_prob)
        
        self.rnn = nn.RNN(num_features, lstm_hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers,
                         dropout=dropout_prob)
        
        self.fc1 = nn.Linear(lstm_hidden_size*4, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        
    def forward(self, x): 
        _, (hidden,cell) = self.lstm(x)
        out = torch.cat((hidden[-2, :, :],hidden[-1, :, :]), dim=1)
        _, hidden = self.rnn(x)
        rnn_out = torch.cat((hidden[-2, :, :],hidden[-1, :, :]), dim=1)
        out = torch.cat((out,rnn_out), dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class Price_model_3(nn.Module):
    def __init__(self, num_features=10,lstm_hidden_size=10,
                fc_hidden_size=10, fc_hidden_size_2 = 10, num_layers=2, dropout_prob=0):
        
        super().__init__()
        
        self.lstm = nn.LSTM(num_features, lstm_hidden_size, 
                           batch_first=True, bidirectional=True,
                           num_layers=num_layers, dropout=dropout_prob)
        
        self.fc1 = nn.Linear(lstm_hidden_size*2, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, fc_hidden_size_2)
        self.fc3 = nn.Linear(fc_hidden_size_2, 1)
        
    def forward(self, x): 
        _, (hidden,cell) = self.lstm(x)
        out = torch.cat((hidden[-2, :, :],hidden[-1, :, :]), dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out



class Price_model_4(nn.Module):
    def __init__(self, num_features=10,lstm_hidden_size=10,
                fc_hidden_size=10,fc_hidden_size_2=10, num_layers=2, dropout_prob=0):
        
        super().__init__()
        
        self.lstm = nn.LSTM(num_features, lstm_hidden_size, 
                           batch_first=True, bidirectional=True,
                           num_layers=num_layers, dropout=dropout_prob)
        
        self.rnn = nn.RNN(num_features, lstm_hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers,
                         dropout=dropout_prob)
        
        self.fc1 = nn.Linear(lstm_hidden_size*4, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, fc_hidden_size_2)
        self.fc3 = nn.Linear(fc_hidden_size_2, 1)
        
    def forward(self, x): 
        _, (hidden,cell) = self.lstm(x)
        out = torch.cat((hidden[-2, :, :],hidden[-1, :, :]), dim=1)
        _, hidden = self.rnn(x)
        rnn_out = torch.cat((hidden[-2, :, :],hidden[-1, :, :]), dim=1)
        out = torch.cat((out,rnn_out), dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

