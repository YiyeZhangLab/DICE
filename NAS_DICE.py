#!/usr/bin/env python
import numpy as np 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
import torch
from torch.utils.data import *
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler 
import torch.optim as optim 

import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import matplotlib 
from matplotlib import pyplot
import matplotlib.pyplot as plt 
import argparse
import os 
import shutil
import random
from sklearn.cluster import AgglomerativeClustering
import math

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.stats import chi2
import pandas as pd
import statsmodels.api as sm
import numpy as np 
import statsmodels.api as sm
from sklearn.metrics import auc, roc_auc_score, roc_curve

class yf_dataset_withdemo(Dataset):
    def __init__(self, path, file_name, n_z):
        self.path = path
        self.file_name = file_name
        self.n_z = n_z
        
        infile = open(self.path + self.file_name, 'rb')
        new_list = pickle.load(infile)
        
        self.n_samples = len(new_list[0])
        # init categary parameter, the following need to be initial outside here. 
        self.n_cat = None # number of categaries, Tensor.
        self.M = None # [n_hidden, n_clusters] centroid of clusters, the k-th column is the centroid of clusters, Tensor
        self.C = torch.LongTensor(np.array([0 for i in range(self.n_samples)])) # the cluster membership. the i-th emement is corresponding to the original data idx = i.
        self.pred_C = torch.LongTensor(np.array([0 for i in range(self.n_samples)])) # the cluster membership. the i-th 
        self.rep = None # [n_samples, n_hidden] the representations of each sample. the i-th element is also corresponding to idx = i.

        data_x = new_list[0]
        data_v = new_list[1]
        data_y = new_list[2]
        
        self.data_x = data_x
        self.data_y = data_y # list 
        self.data_v = data_v

        samples_list = []
        for i in range(len(data_x)):
            totensor_data_x = torch.FloatTensor(np.array(data_x[i]))
            totensor_data_v = torch.FloatTensor(np.array(data_v[i]))
            totensor_data_y = torch.LongTensor(np.array([data_y[i]]))
            samples_list.append([totensor_data_x,  totensor_data_v, totensor_data_y])
        self.samples = samples_list
        self.mylength = len(data_x)
    
    def __len__(self):
        return self.mylength

    def __getitem__(self, idx):
        return idx, self.samples[idx], self.C[idx]

class EncoderRNN(nn.Module):
    def __init__(self, input_size, nhidden, nlayers, dropout, cuda):
        super(EncoderRNN, self).__init__()
        self.nhidden = nhidden
        self.feasize = input_size
        self.nlayers = nlayers
        self.dropout = dropout
        self.cuda = cuda 
        self.lstm = nn.LSTM(input_size=self.feasize,
                               hidden_size=self.nhidden,
                               num_layers=self.nlayers,
                               dropout=self.dropout,
                               batch_first=True)
        self.init_weights()

    def init_weights(self):
        #nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        for p in self.lstm.parameters():
            p.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        batch_size = x.size()[0]
        output, state = self.lstm(x) #output [batch_size, seq_size, hidden_size]
        hn, cn = state
        #hidden = hidden_state[-1]  # get hidden state of last layer of encoder
        output = torch.flip(output, [1])
        newinput = torch.flip(x,[1])        
        zeros = torch.zeros(batch_size, 1, x.shape[-1]) #zeros = torch.zeros(batch_size, 1, x.shape[-1])
        if self.cuda:
            zeros = zeros.cuda()
        newinput = torch.cat((zeros, newinput),1)
        newinput = newinput[:, :-1, :]
        return output, (hn, cn), newinput

class DecoderRNN(nn.Module):
    def __init__(self, input_size, nhidden, nlayers, dropout):
        super(DecoderRNN, self).__init__()
        self.nhidden = nhidden
        self.feasize = input_size
        self.nlayers = nlayers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=self.feasize,
                               hidden_size=self.nhidden,
                               num_layers=self.nlayers,
                               dropout=self.dropout,
                               batch_first=True)
        self.init_weights()

    def init_weights(self):
        #nn.init.orthogonal_(self.lstm.weight_ih_l0, gain=np.sqrt(2))
        for p in self.lstm.parameters():
            p.data.uniform_(-0.1, 0.1)

    def forward(self, x, h):
        output, state = self.lstm(x, h)
        fin = torch.flip(output, [1])
        return fin

class model_2(nn.Module):
    def __init__(self, input_size, nhidden, nlayers, dropout, n_clusters, n_dummy_demov_fea, para_cuda):
        super(model_2, self).__init__()
        self.nhidden = nhidden
        self.input_size = input_size
        self.nlayers = nlayers
        self.dropout = dropout
        self.n_clusters = n_clusters
        self.n_dummy_demov_fea = n_dummy_demov_fea
        self.para_cuda = para_cuda
        self.encoder = EncoderRNN(self.input_size, self.nhidden, self.nlayers, self.dropout, self.para_cuda)
        self.decoder = DecoderRNN(self.input_size, self.nhidden, self.nlayers, self.dropout)
        self.linear_decoder_output = nn.Linear(self.nhidden, self.input_size)
        self.linear_classifier_c = nn.Linear(self.nhidden, self.n_clusters) 
        self.activateion_classifier = nn.Softmax(dim=1)
        self.linear_regression_c = nn.Linear(self.n_clusters, 1)
        self.linear_regression_demov = nn.Linear(self.n_dummy_demov_fea, 1)
        self.activation_regression = nn.Sigmoid()
        self.init_weights()


    def init_weights(self):
        #nn.init.orthogonal_(self.linear.weight, gain=np.sqrt(2))
        self.linear_decoder_output.bias.data.fill_(0)
        self.linear_decoder_output.weight.data.uniform_(-0.1,0.1)
        
        self.linear_classifier_c.bias.data.fill_(0)
        self.linear_classifier_c.weight.data.uniform_(-0.1,0.1)
        
        self.linear_regression_c.bias.data.fill_(0)
        self.linear_regression_c.weight.data.uniform_(-0.1,0.1)
        
        self.linear_regression_demov.bias.data.fill_(0)
        self.linear_regression_demov.weight.data.uniform_(-0.1,0.1)
    
    def forward(self, x, function, demov = None, mask_BoolTensor = None):
        '''
        mask = 1, mask one cluster. 
        mask = 2, mask two cluster. 
        mask_index: list() of index. 
        '''
        if function =="autoencoder":
            encoded_x, (hn, cn), newinput = self.encoder(x)
            decoded_x = self.decoder(newinput, (hn, cn))
            decoded_x = self.linear_decoder_output(decoded_x)
            return encoded_x, decoded_x
        elif function == "get_representation":
            encoded_x, (hn, cn), newinput = self.encoder(x)
            return encoded_x  
        elif function == "classifier":
            encoded_x, (hn, cn), newinput = self.encoder(x)
            output = self.linear_classifier_c(encoded_x)
            output = self.activateion_classifier(output)
            return encoded_x, output 
        elif function == "outcome_logistic_regression":
            encoded_x, (hn, cn), newinput = self.encoder(x)
            decoded_x = self.decoder(newinput, (hn, cn))
            decoded_x = self.linear_decoder_output(decoded_x)
            
            encoded_x = encoded_x[:,0,:]
            output_c_no_activate = self.linear_classifier_c(encoded_x)
            output_c = self.activateion_classifier(output_c_no_activate)

            # output_c dimension [batch_size, n_clusters]
            if mask_BoolTensor!=None:
                if self.cuda:
                    mask_BoolTensor = mask_BoolTensor.cuda()
                output_c = output_c.masked_fill(mask = mask_BoolTensor, value=torch.tensor(0.0) )
            
            output_from_c = self.linear_regression_c(output_c)
            output_from_v = self.linear_regression_demov(demov)
            output_cpv = output_from_c + output_from_v
            output_outcome = self.activation_regression(output_cpv)
            return encoded_x, decoded_x, output_c_no_activate, output_outcome
        else:
            print(" No corresponding function, check the function you want to for model_2")
            return "Wrong!"    

def parse_args():
    # begin main function
    parser = argparse.ArgumentParser(description='ppd-aware clustering')
    # require para
    parser.add_argument('--init_AE_epoch', type=int, required=False,
                        help='number of epoch for representation initialization')
    parser.add_argument('--n_hidden_fea', type=int, required=False,
                        help='number of hidden size in LSTM')
    parser.add_argument('--input_path', type=str, required=True,
                        help='location of input dataset')

    parser.add_argument('--filename_train', type=str, required=True,
                        help='location of the data corpus')

    parser.add_argument('--filename_valid', type=str, required=True,
                        help='filename_valid')

    parser.add_argument('--filename_test', type=str, required=True,
                        help='file_name_test')
    
    parser.add_argument('--training_output_path', type=str, required=True,
                        help='location of training output')
    
    parser.add_argument('--n_input_fea', type=int, required=True,
                        help='number of original input feature size')

    parser.add_argument('--n_dummy_demov_fea', type=int, required=True,
                        help='number of dummy demo feature size')
    parser.add_argument('--lstm_layer', type=int, default=1,
                        help='number of hidden size in LSTM')
    
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--lstm_dropout', type=float, default=0.0, help='dropout in LSTM')
    
    parser.add_argument('--K_clusters', type=int, required=False,
                        help='number of initial clusters')
    
    parser.add_argument('--iter', type=int, default=20,
                        help='maximum of iterations in iteration merge clusters')
    
    parser.add_argument('--epoch_in_iter', type=int, default=1,
                        help='maximum of iterations in iteration merge clusters')
    
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--input_trained_data_train', type=str, required=False,
                        help='location of the data corpus')
    
    parser.add_argument('--input_trained_model', type=str, required=False,
                        help='location of the data corpus')
    
    parser.add_argument('--cuda', type=int, default=0,
                        help='If use cuda')
    
    parser.add_argument('--lambda_AE', type=float, default=1.0, help='lambda of AE in iteration')
    parser.add_argument('--lambda_classifier', type=float, default=1.0, help='lambda_classifier of classifier in iteration')
    parser.add_argument('--lambda_outcome', type=float, default=10.0, help='lambda of outcome in iteration')
    parser.add_argument('--lambda_p_value', type=float, default=1.0, help='lambda of p value in iteration')
    args = parser.parse_args()
    return args

def func_analysis_test_error_D0420(args, model, data_test, dataloader_test):
    model.eval()
    criterion_MSE = nn.MSELoss()
    criterion_BCE = nn.BCELoss()
    error_AE = []
    error_outcome_likelihood = []
    correct = 0 
    total = 0 
    correct_outcome = 0
    outcome_auc = 0 
    outcome_true_y = []
    outcome_pred_prob = [] 
    for batch_idx, (index, batch_xvy, batch_c) in enumerate(dataloader_test):
        data_x, data_v, target = batch_xvy
        data_x = torch.autograd.Variable(data_x)
        data_v = torch.autograd.Variable(data_v)
        target = torch.autograd.Variable(target)
        batch_c = torch.autograd.Variable(batch_c)

        if args.cuda:
            data_x = data_x.cuda()
            data_v = data_v.cuda()
            target = target.cuda()
            batch_c = batch_c.cuda()
        
        encoded_x, decoded_x, output_c_no_activate, output_outcome = model(x=data_x, function="outcome_logistic_regression", demov=data_v)
        
        loss_AE = criterion_MSE(data_x, decoded_x)
        loss_outcome = criterion_BCE(output_outcome, target.float())
        error_outcome_likelihood.append(loss_outcome.data.cpu().numpy())
        error_AE.append(loss_AE.data.cpu().numpy())

        total += batch_c.size(0)

        outcome_true_y.append(target.data.cpu())
        outcome_pred_prob.append(output_outcome.data.cpu()) 
    
    test_AE_loss = np.mean(error_AE)
    test_outcome_likelihood = np.mean(error_outcome_likelihood)
    
    aucscore = outcome_auc_score = roc_auc_score(np.concatenate(outcome_true_y, 0), np.concatenate(outcome_pred_prob, 0))
    fpr, tpr, thresholds= roc_curve(np.concatenate(outcome_true_y, 0),  np.concatenate(outcome_pred_prob, 0))
    return test_AE_loss, test_outcome_likelihood,  outcome_auc_score, fpr, tpr, thresholds

def ppv_item(true_y, predict_results):
    acc=-1
    ppv=-1
    acc= accuracy_score(true_y, predict_results)
    conf_mat = confusion_matrix(true_y, predict_results)
    conf_mat_tolist = conf_mat.tolist()
    number_TN = conf_mat_tolist[0][0]
    number_FN = conf_mat_tolist[1][0]
    number_FP = conf_mat_tolist[0][1]
    number_TP = conf_mat_tolist[1][1]
    try:
        ppv=number_TP/(number_TP + number_FP) #positive and negative predictive values
    except:
        ppv=-1

    if number_FN == 0:
        FNR = 0
    else:
        FNR = number_FN/(number_TP+number_FN)
    if number_TP == 0:
        TPR = 0
    else:
        TPR = number_TP/(number_TP+number_FN)
    return acc, ppv ,TPR, FNR, conf_mat

def update_curset_pred_C_and_repD0420(args, model, data_cur, dataloader_cur, varname, datatrainM):
    final_embed = torch.randn(len(data_cur), args.n_hidden_fea, dtype=torch.float)
    model.eval()
    for batch_idx, (index, batch_xvy, batch_c) in enumerate(dataloader_cur):
        data_x, data_v, target = batch_xvy
        data_x = torch.autograd.Variable(data_x)
        data_v = torch.autograd.Variable(data_v)
        target = torch.autograd.Variable(target)

        if args.cuda:
            data_x = data_x.cuda()
            data_v = data_v.cuda()
            target = target.cuda()

        encoded_x, decoded_x, output_c_no_activate, output_outcome = model(x=data_x, function="outcome_logistic_regression", demov=data_v)
        embed = encoded_x.data.cpu()
        final_embed[index] = embed

    data_cur.rep = final_embed 
    representations = data_cur.rep
    for i in range(data_cur.rep.size()[0]):
        embed = representations[i,:]
        trans_embed = embed.view(embed.size()+(1,))
        xj = torch.norm(trans_embed - datatrainM.M, dim=0)
        new_cluster = torch.argmin(xj)
        data_cur.C[i] = new_cluster


def analysis_cluster_number_byclustering(data_cur, num_clusters, if_check, varname):
    data_C = data_cur.C
    data_v = data_cur.data_v
    data_y = data_cur.data_y

    list_c = data_C.tolist()
    list_onehot = []
    dict_c_count = {}
    dict_outcome_in_c_count = {}
    for i in range(num_clusters):
        dict_c_count[i] = 0 
        dict_outcome_in_c_count[i] = 0 
    
    for i in range(len(list_c)):
        temp = [0 for i in range(num_clusters)]
        temp[list_c[i]] = 1 
        list_onehot.append(temp)

        dict_c_count[list_c[i]] += 1 
        if data_y[i]==1:
            dict_outcome_in_c_count[list_c[i]] += 1 
    
    if if_check:
        print("--------")
        print("num_clusters=", num_clusters)
        print()
        print("list_c[0]=",list_c[0])
        print("list_onehot[0]=", list_onehot[0])
        print()
        print("list_c[1]=",list_c[1])
        print("list_onehot[1]=", list_onehot[1])
        print("--------")   

    dict_outcome_ratio = {}
    for keyc in dict_c_count:
        if dict_c_count[keyc] == 0:
            dict_outcome_ratio[keyc] = 0
        else:
            dict_outcome_ratio[keyc] = dict_outcome_in_c_count[keyc]/dict_c_count[keyc]
    return dict_outcome_ratio, dict_c_count


def change_label_from_highratio_to_lowratio(K, oldlabel, data_train):
    data_v = data_train.data_v
    data_y = data_train.data_y

    list_c = oldlabel.tolist()
    dict_c_count = {}
    dict_outcome_in_c_count = {}
    for i in range(K):
        dict_c_count[i] = 0 
        dict_outcome_in_c_count[i] = 0 
    
    for i in range(len(list_c)):
        dict_c_count[list_c[i]] += 1 
        if data_y[i]==1:
            dict_outcome_in_c_count[list_c[i]] += 1 

    dict_outcome_ratio = {}
    for keyc in dict_c_count:
        dict_outcome_ratio[keyc] = dict_outcome_in_c_count[keyc]/dict_c_count[keyc]
    
    sorted_dict_outcome_ratio = dict(sorted(dict_outcome_ratio.items(), key=lambda x:x[1], reverse=True))
    order = list(sorted_dict_outcome_ratio.keys())
    order_c_map = {}
    for i in range(len(order)):
        order_c_map[order[i]] = i
    # change c 
    new_list_c = []
    for i in range(len(list_c)):
        new_list_c.append(order_c_map[list_c[i]])
    
    return torch.LongTensor(new_list_c), order_c_map

def analysis_architecture(args, inputmodelpath, inputdatatrainpath, inputnhidden, n_clusters):
    args.input_trained_data_train = inputdatatrainpath
    args.input_trained_model = inputmodelpath
    args.n_hidden_fea = inputnhidden
    args.K_clusters = n_clusters

    # load data
    pkl_file = open(args.input_trained_data_train, 'rb')
    data_train = pickle.load(pkl_file)
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True, drop_last=True)

    data_test = yf_dataset_withdemo(args.input_path, args.filename_test, args.n_hidden_fea)
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, drop_last=True)
    data_valid = yf_dataset_withdemo(args.input_path, args.filename_valid, args.n_hidden_fea)
    dataloader_valid = torch.utils.data.DataLoader(data_valid, batch_size=1, shuffle=False, drop_last=True)

    # Algorithm 2 model
    model = model_2(args.n_input_fea, args.n_hidden_fea, args.lstm_layer, args.lstm_dropout, args.K_clusters, args.n_dummy_demov_fea, args.cuda)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.cuda:
        model = model.cuda()

    device = torch.device("cpu")
    model.load_state_dict(torch.load(args.input_trained_model, map_location=device))
    # model.to(device)
    train_AE_loss, train_outcome_likelihood, train_outcome_auc_score, fpr, tpr, thresholds = func_analysis_test_error_D0420(args, model, data_train, dataloader_train)
    valid_AE_loss, valid_outcome_likelihood, valid_outcome_auc_score, fpr, tpr, thresholds = func_analysis_test_error_D0420(args, model, data_valid, dataloader_valid)
    test_AE_loss, test_outcome_likelihood, test_outcome_auc_score,  fpr, tpr, thresholds = func_analysis_test_error_D0420(args, model, data_test, dataloader_test)


    update_curset_pred_C_and_repD0420(args, model, data_valid, dataloader_valid,"data_valid", data_train)
    update_curset_pred_C_and_repD0420(args, model, data_test, dataloader_test,"data_test", data_train)
    dict_outcome_ratio_train, dict_c_count = analysis_cluster_number_byclustering(data_train, n_clusters, 0, "train")
    dict_outcome_ratio_valid,_ = analysis_cluster_number_byclustering(data_valid, n_clusters, 0, "valid")
    dict_outcome_ratio_test,_ = analysis_cluster_number_byclustering(data_test, n_clusters, 0, "test")

    resx = np.concatenate([data_train.rep.numpy(), data_valid.rep.numpy(), data_test.rep.numpy()], axis=0)
    resy = data_train.data_y + data_valid.data_y + data_test.data_y
    resc = torch.cat((data_train.C, data_valid.C, data_test.C),0)
    
    trainx, trainy, trainc = data_train.rep.numpy(), data_train.data_y, data_train.C
    return data_train, data_valid, data_test, resx, resy, resc, trainx, trainy, trainc, dict_outcome_ratio_train, dict_c_count,test_outcome_likelihood, test_outcome_auc_score, valid_outcome_likelihood, valid_outcome_auc_score


import os
import re 
if __name__ == '__main__':
    args = parse_args()
    path = "./" 
    files= os.listdir(path) 
    s = []
    record = [] 

    print("useful architecture: ")
    for file in sorted(files):
        if file[-4:]!='.log':
            continue
        _,K,hn,_ = re.split('k|hn|.log', file)
        list_line = []
        f = open(path+"/"+file)
        iter_f = iter(f); 
        for line in iter_f: 
            list_line.append(line)
        myline_index= -2
        if '[]' not in list_line[myline_index]:
            lastiter=re.split('\[|,|\]',list_line[myline_index])[-2]
            print("# file={}, K={}, hn={}, last iter={}".format(file, K, hn, lastiter))
            record.append((int(K),int(hn),int(lastiter)))
    print('record=',(record))
    print('len(record)=', len(record))

    list_tuple_k_hn_iter_nll_auc_biggestratio_valid = []
    list_tuple_k_hn_iter_nll_auc_biggestratio_test = []
    list_tuple_k_hn_iter_dict_outcome_ratio_train = []
    for item in record:
        n_clusters, inputnhidden, epoch = item
        taskpath = './'
        inputmodelpath = taskpath + 'hn_'+str(inputnhidden) +'_K_'+str(n_clusters)+'/part2_AE_nhidden_' + str(inputnhidden) + '/model_iter.pt'
        inputdatatrainpath = taskpath + 'hn_'+str(inputnhidden) +'_K_'+str(n_clusters)+'/part2_AE_nhidden_' + str(inputnhidden) +'/data_train_iter.pickle'
        data_train, data_valid, data_test, resx, resy, resc, trainx, trainy, trainc, dict_outcome_ratio_train, dict_c_count,test_outcome_likelihood, test_outcome_auc_score, valid_outcome_likelihood, valid_outcome_auc_score = analysis_architecture(args, inputmodelpath, inputdatatrainpath, inputnhidden,n_clusters)
        if 0 in list(dict_c_count.values()):
            print("degenerated clusters. Some clusters has no pid")
            continue
        from sklearn import manifold, datasets
        list_tuple_k_hn_iter_nll_auc_biggestratio_valid.append((n_clusters, inputnhidden, epoch, valid_outcome_likelihood, valid_outcome_auc_score, dict_outcome_ratio_train[0]))
        list_tuple_k_hn_iter_nll_auc_biggestratio_test.append((n_clusters, inputnhidden, epoch, test_outcome_likelihood, test_outcome_auc_score, dict_outcome_ratio_train[0]))
        list_tuple_k_hn_iter_dict_outcome_ratio_train.append((n_clusters, inputnhidden, epoch, dict_outcome_ratio_train))

    print("--------- valid ---------")
    min_nll= sorted( list_tuple_k_hn_iter_nll_auc_biggestratio_valid, key=lambda x:x[3], reverse=False)
    print("min_nll=",min_nll)
    print()

    max_AUC= sorted( list_tuple_k_hn_iter_nll_auc_biggestratio_valid, key=lambda x:x[4], reverse=True)
    print("max_AUC=",max_AUC)
    print()

    max_biggestratio= sorted( list_tuple_k_hn_iter_nll_auc_biggestratio_valid, key=lambda x:x[5], reverse=True)
    print("max_biggestratio=",max_biggestratio)
    print()
    print("final search result based on the maximum AUC score on validation set, K={}, hn={}".format(max_AUC[0][0], max_AUC[0][1]))

