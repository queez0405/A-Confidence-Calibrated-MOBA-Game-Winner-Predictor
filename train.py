import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import pdb

def NNtrain(model, arg, champDataset, val_x, val_y):
    writer = SummaryWriter(log_dir=arg.log_path)
    train_loader = DataLoader(dataset=champDataset,
                        batch_size=arg.batch_size,
                        shuffle=True,
                        num_workers=2)

    optimizer = optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
    criterion = nn.NLLLoss()
    iter_num = 0

    for epoch in range(1, arg.epochs+1):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            feature, target = data
            feature, target = Variable(feature), Variable(target)
            if arg.use_cuda:
                feature, target = feature, target

            optimizer.zero_grad()
            logit = model(feature.float())
            logit = nn.LogSoftmax(dim=1)(logit)
            loss = criterion(logit, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            iter_num += 1
            if i % 100 == 99 and i != 0:
                print('[%d, %5d] loss: %.4f' %
                    (epoch, i + 1 , running_loss / 100))
                writer.add_scalar('Loss/train', running_loss / 100, iter_num)
                running_loss = 0.0
        val_acc = eval(val_x, val_y, model, arg)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.close()

    torch.save(model.state_dict(), arg.model_save_path)

    print('Finished Training')

def DUNNtrain(model, arg, champDataset, val_x, val_y):
    writer = SummaryWriter(log_dir=arg.log_path)
    train_loader = DataLoader(dataset=champDataset,
                        batch_size=arg.batch_size,
                        shuffle=True,
                        num_workers=2)

    optimizer = optim.Adam(model.parameters(), lr=arg.lr, weight_decay=arg.weight_decay)
    criterion = nn.NLLLoss()
    iter_num = 0

    for epoch in range(1, arg.epochs+1):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            feature, target = data
            feature, target = Variable(feature), Variable(target)

            optimizer.zero_grad()
            if arg.embed_type == 'champ2vec':
                mu, sigma = model(feature.long())
            elif arg.embed_type == 'one_hot':
                mu, sigma = model(feature.float())
            target_hat = get_hat(mu, sigma, arg)
            loss = criterion(target_hat, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            iter_num += 1
            if i % 100 == 99 and i != 0:
                print('[%d, %5d] loss: %.4f' %
                    (epoch, i + 1 , running_loss / 100))
                writer.add_scalar('Loss/train', running_loss / 100, iter_num)
                running_loss = 0.0
        val_acc = eval(val_x, val_y, model, arg)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.close()

    torch.save(model.state_dict(), arg.model_save_path)

    print('Finished Training')

def get_hat(mu, sigma, arg):
    softmax = nn.Softmax()
    normal_dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    y_hat = torch.zeros(mu.shape).float()
    for _ in range(arg.mc_samples):
        normal_samples = normal_dist.sample(sample_shape=mu.shape).squeeze(-1)
        y_sample = mu + sigma.expand(-1,2) * normal_samples
        y_hat += softmax(y_sample)
    y_hat /= arg.mc_samples

    return torch.log(y_hat + torch.Tensor([1e-20]))

def eval(data_x, data_y, model, arg):
    softmax = nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        split_num = 10
        W = torch.tensor([[1.0643,-1.5438],[1.2420,0.7887]])
        b1 = torch.tensor([[0.8721],[1.0325]])
        T = torch.tensor(1.1562)
        pdb.set_trace()
        if arg.uncertainty == 'None':
            predict = model(data_x.float())
        elif arg.uncertainty == 'Data':
            predict, var = model(data_x.float())

        #predict = predict/T #tmp scal
        #predict = (W * (predict.t())).t() #vec scal
        #predict = (torch.mm(W,predict.t())+b1).t() #mat scal

        predict = softmax(predict)
        for i in range(split_num):
            low_prob = 0.5+(0.5/split_num)*i
            high_prob = (0.5+(0.5/split_num)*(i+1))
            index = (predict.max(1).values > low_prob) & (predict.max(1).values < high_prob)
            data_x_unc = data_x[index]
            data_y_unc = data_y[index]
            if arg.uncertainty == 'None':
                pred_unc = model(data_x_unc)
                
                #pred_unc = pred_unc/T #tmp scal
                #pred_unc = (W * (pred_unc.t())).t() #vec scal
                #pred_unc = (torch.mm(W,pred_unc.t())+b1).t() #mat scal
                
                pred_unc = softmax(pred_unc)                
                acc = (pred_unc.max(1).indices == data_y_unc).sum().item()/len(pred_unc)
                print('Now low prob:{}, high prob:{}'.format(low_prob,high_prob))
                print('test data length:{0}, all accuracy:{1:0.5f}, predicted mean:{2:0.5f}'.format(len(pred_unc), acc, pred_unc.max(1).values.mean().item()))

            elif arg.uncertainty == 'Data':
                pred_unc, var_unc = model(data_x_unc)            
                pred_unc = softmax(pred_unc)
                acc = (pred_unc.max(1).indices == data_y_unc).sum().item()/len(var_unc)
                index = sorted(range(data_x_unc.shape[0]), key=var_unc.squeeze(1).tolist().__getitem__)
                data_x_unc = data_x_unc[index]
                data_y_unc = data_y_unc[index]

                print('Now low prob:{}, high prob:{}'.format(low_prob,high_prob))
                print('test data length:{0}, all accuracy:{1:0.5f}, predicted mean:{2:0.5f}'.format(len(var_unc), acc, pred_unc.max(1).values.mean().item()))

    acc = (predict.max(1).indices == data_y).sum().item()/len(data_y)
    nll = nn.NLLLoss()(torch.log(predict),data_y)
    print('unc : {}'.format(arg.uncertainty))
    print('accuracy is {0:0.5f}, nll is {1:0.5f}'.format(acc, nll))
    return acc

'''
def shuffle(data_x,data_y):
    idx = torch.randperm(len(data_x))
    data_x_shuffle = data_x[idx]
    data_y_shuffle = data_y[idx]

    return data_x_shuffle, data_y_shuffle
'''

def opt_matrix(model, val_x,val_y, option):
    
    criterion = nn.NLLLoss()
    logsoftmax = nn.LogSoftmax(dim=1)
    
    if option == 'mat':
        W = Variable(torch.randn(2,2), requires_grad=True)
        b = Variable(torch.randn(2,1), requires_grad=True)
        optimizer = optim.RMSprop([W, b], lr=0.0005)
    elif option == 'vec':
        W = Variable(torch.randn(2,1), requires_grad=True)
        optimizer = optim.RMSprop([W], lr=0.0005)
    elif option == 'tem':
        T = Variable(torch.tensor(1.0), requires_grad=True)
        optimizer = optim.RMSprop([T], lr=0.0005)
    for epoch in range(2000):
        if option == 'mat':
            logit = torch.t(model(val_x)).data
            logit = torch.mm(W,logit) + b
            logit = logsoftmax(logit.t())
        elif option == 'vec':
            logit = torch.t(model(val_x)).data
            logit = W * (logit)
            logit = logsoftmax(logit.t())
        elif option == 'tem':
            logit = model(val_x).data
            logit = (logit) / T
            logit = logsoftmax(logit)
        loss = criterion(logit,val_y)
        loss.backward()
        #print(loss)
        optimizer.step()
    pdb.set_trace()
