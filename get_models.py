'''
FOR SOME REASON IT WON'T LET ME PASS A DIC TO THE MODELS SO I HAVE TO DO IT IN A 
VERY TEDIOUS WAY
SHOULD PROBABLY REFACTOR AND FIND OUT WHY
'''

import json
from price_pred_models import *
from percent_pred_models import *

def get_params(file_name='params.json'):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def parse_params(dic):
    new_dic = dic['params']
    new_dic = {i:new_dic[i] for i in new_dic.keys() if i != 'validation_loss' or i != 'lr'}
    lr = dic['lr']
    return new_dic, lr


def get_all_model_params():
    output = []
    params = get_params()
    for i in params:
        temp = parse_params(params[i])
        temp_dic = {i:temp[0]}
        output.append((temp_dic, temp[1]))
    return output

#also returns the desired learning_rate
def generate_models(model_params):
    num_features = 10
    models = []
    for i in model_params:
        dic = i[0]
        lr = i[1]
        key = list(dic.keys())[0]
        if key == 'Price_model_1':
            temp = [num_features]
            for x in dic[key]:
                temp.append(dic[key][x])
            models.append((Price_model_1(*temp),lr))
        elif key == 'Price_model_2':
            temp = [num_features]
            for x in dic[key]:
                temp.append(dic[key][x])
            models.append((Price_model_2(*temp),lr))
        elif key == 'Price_model_3':
            temp = [num_features]
            for x in dic[key]:
                temp.append(dic[key][x])
            models.append((Price_model_3(*temp),lr))
        elif key == 'Price_model_4':
            temp = [num_features]
            for x in dic[key]:
                temp.append(dic[key][x])
            models.append((Price_model_4(*temp),lr))

        elif key == 'Percent_model_1':
            temp = [num_features]
            for x in dic[key]:
                temp.append(dic[key][x])
            models.append((Percent_model_1(*temp),lr))

        elif key == 'Percent_model_2':
            temp = [num_features]
            for x in dic[key]:
                temp.append(dic[key][x])
            models.append((Percent_model_2(*temp),lr))

        elif key == 'Percent_model_3':
            temp = [num_features]
            for x in dic[key]:
                temp.append(dic[key][x])
            models.append((Percent_model_3(*temp),lr))

        elif key == 'Percent_model_4':
            temp = [num_features]
            for x in dic[key]:
                temp.append(dic[key][x])
            models.append((Percent_model_4(*temp),lr))

    return models




