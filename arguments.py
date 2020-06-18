import os

from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description="LoL outcome predict")
    parser.add_argument('--val_set_size', type=int, default=1000)

    #in-game setting
    parser.add_argument('--champ_num', type=int)
    parser.add_argument('--patch_ver', type=str, default='9.19')

    #train data
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--uncertainty', choices=['None', 'Model', 'Data', 'Both'], default='None')
    parser.add_argument('--in_game', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--mc_samples', type=int, default=20)
    parser.add_argument('--use_cuda', type=bool, default=False)

    parser.add_argument('--embed_type', choices=['one_hot', 'champ2vec'], default='champ2vec')

    #tensorboard
    parser.add_argument('--log_path', type=str, default='./logs/')

    #model_save
    parser.add_argument('--model_save_path', type=str, default='./save_models/')
    parser.add_argument('--saved_model_time', type=str, default=None)
    
    args = parser.parse_args()

    return args
