import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from preprocess import * #Preprocess, ChampDataset
from model import * #CNNModel, MUCNNModel, RNNModel, MURNNModel, OHModel, MUOHModel
from arguments import get_args
import train
from datetime import datetime
now = datetime.now()
import os
import pdb

preprocess = Preprocess()
arg = get_args()
'''
def oneHotEncodding(labels):
    onehot_encoded = list()
    for value in labels:
        target = [0 for _ in range(2)]
        target[value] = 1
        onehot_encoded.append(target)

    return onehot_encoded
'''
def main():

    allChamp, matchComp, blueWin = preprocess.lolDataSet(arg, "train")
    _, test_x, test_y = preprocess.lolDataSet(arg, "test")

    test_x = torch.tensor(test_x).float()
    test_y = torch.tensor(test_y.reshape(-1)).long()

    val_x = torch.tensor(matchComp[0:arg.val_set_size]).float()
    val_y = torch.tensor(blueWin[0:arg.val_set_size].reshape(-1)).long()
    train_x = matchComp[arg.val_set_size:-1]
    train_y = blueWin[arg.val_set_size:-1].reshape(-1)

    champDataset = ChampDataset(train_x,train_y)

    arg.champ_num = len(allChamp)

    if arg.in_game:
        arg.log_path = arg.log_path + 'in-game/'
        arg.model_save_path = arg.model_save_path + 'in-game/'
    else:
        arg.log_path = arg.log_path + 'pregame/'
        arg.model_save_path = arg.model_save_path + 'pregame/'

    arg.log_path = arg.log_path + arg.embed_type + '/' + arg.uncertainty + '/' + now.strftime("%Y%m%d-%H%M%S")
    arg.model_save_path = arg.model_save_path + arg.embed_type + '/' + arg.uncertainty
    if not(os.path.isdir(arg.model_save_path)):
        os.makedirs(os.path.join(arg.model_save_path))
    if arg.saved_model_time == None:
        arg.model_save_path = arg.model_save_path + '/' + now.strftime("%Y%m%d-%H%M%S") + '.pt'

    if arg.embed_type == 'one_hot' and arg.uncertainty == 'None':
        model = OHModel(arg)

        if arg.saved_model_time == None:
            train.NNtrain(model, arg, champDataset, val_x, val_y)
        else:
            arg.model_save_path = arg.model_save_path + '/' +arg.saved_model_time + '.pt'
            model.load_state_dict(torch.load(arg.model_save_path))
            option = 'mat' #vec, mat, tem
            train.opt_matrix(model, val_x,val_y, option)

        train.eval(test_x,test_y,model,arg)


    elif arg.embed_type == 'one_hot' and arg.uncertainty == 'Data':
        model = DUOHModel(arg)

        if arg.saved_model_time == None:
            train.DUNNtrain(model, arg, champDataset, val_x, val_y)
        else:
            arg.model_save_path = arg.model_save_path + '/' +arg.saved_model_time + '.pt'
            model.load_state_dict(torch.load(arg.model_save_path))
        
        train.eval(test_x,test_y,model,arg)


if __name__ == '__main__':
    main()