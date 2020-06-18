import torch
from torch.utils.data import Dataset
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import pickle
import pdb

class Preprocess(object):
    def lolDataSet(self, arg, sep):

        path='./LOLData/'
        if arg.in_game:
            match = pd.read_csv(path+"inGameMatchData"+arg.patch_ver+sep+".csv")
        else:
            match = pd.read_csv(path+"preMatchData"+arg.patch_ver+sep+".csv")
        match.head()

        teams = ["blue", "red"]
        positions = ["_1", "_2", "_3", "_4", "_5"]
        champ_cols = []

        all_champs = set()
        for t in teams:
            for p in positions:
                col = t + p
                all_champs |= set(match[col].unique())
                champ_cols.append(col)
        allChamp = list(all_champs)

        blueChampComp = match.loc[:,['blue_1','blue_2','blue_3','blue_4','blue_5']].values
        redChampComp = match.loc[:,['red_1','red_2','red_3','red_4','red_5']].values
        if arg.in_game:
            inGameData = match.loc[:,['blue_death_diff','red_death_diff','gold_diff','xp_diff', 'timeline']].values.astype(np.int32)
        champComp = np.concatenate([blueChampComp,redChampComp]).tolist()
        matchComp = np.concatenate([blueChampComp,redChampComp], axis = -1)
        blueWin = match.loc[:,['blue_win']].values

        champ2num = {champ:i for i, champ in enumerate(allChamp)}

        for i in range(len(blueChampComp)):
            for j in range(len(blueChampComp[i])):
                blueChampComp[i][j] = champ2num[blueChampComp[i][j]]
        for i in range(len(redChampComp)):
            for j in range(len(redChampComp[i])):
                redChampComp[i][j] = champ2num[redChampComp[i][j]]

        blueChampComp = blueChampComp.astype(np.int32)
        redChampComp = redChampComp.astype(np.int32)
        blueOneHot, redOneHot = np.zeros((len(blueChampComp),len(allChamp))).astype(int), np.zeros((len(redChampComp),len(allChamp))).astype(int)
        for i in range(len(blueChampComp)):
            for j in range(len(blueChampComp[i])):
                blueOneHot[i][blueChampComp[i][j]] = 1
        for i in range(len(redChampComp)):
            for j in range(len(redChampComp[i])):
                redOneHot[i][redChampComp[i][j]] = 1
        
        if arg.in_game:
            matchComp = np.concatenate([blueOneHot,redOneHot,inGameData], axis = -1)
        else:
            matchComp = np.concatenate([blueOneHot,redOneHot], axis = -1)

        return allChamp, matchComp, blueWin

class ChampDataset(Dataset):
    def __init__(self, train_x, train_y):
        self.len = train_x.shape[0]
        self.train_x = torch.from_numpy(train_x)
        self.train_y = torch.from_numpy(train_y)

    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index]

    def __len__(self):
        return self.len
