# end-to-end code 
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

import matplotlib 
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
        #print("output.size()=",output.size()) # output.size()= torch.Size([1, 10, 100])
        #print("hn.size()=",hn.size()) # hn.size()= torch.Size([1, 1, 100])
        #print("hn=",hn)
        #print("output[0]=",output[0])
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
            #print("x.size()=",x.size())
            #print("encoded_x.size()=",encoded_x.size())
            #print("decoded_x.size()=",decoded_x.size())
            #print("output_c_no_activate.size()=",output_c_no_activate.size())
            #print("output_outcome.size()=",output_outcome.size())

            #embed = enc.data.cpu()[:,0,:]
            return encoded_x, decoded_x, output_c_no_activate, output_outcome
        else:
            print(" No corresponding function, check the function you want to for model_2")
            return "Wrong!"          

def parse_args():
    parser = argparse.ArgumentParser(description='ppd-aware clustering')

    parser.add_argument('--init_AE_epoch', type=int, required=True,
                        help='number of epoch for representation initialization')

    parser.add_argument('--n_hidden_fea', type=int, required=True,
                        help='number of hidden size in LSTM')

    parser.add_argument('--output_path', type=str, required=False,
                        help='location of output path')

    parser.add_argument('--input_path', type=str, required=True,
                        help='location of input dataset')

    parser.add_argument('--filename_train', type=str, required=True,
                        help='location of the data corpus')

    parser.add_argument('--filename_test', type=str, required=True,
                        help='location of the data corpus')

    parser.add_argument('--n_input_fea', type=int, required=True,
                        help='number of original input feature size')

    parser.add_argument('--n_dummy_demov_fea', type=int, required=True,
                        help='number of dummy demo feature size')

    parser.add_argument('--lstm_layer', type=int, default=1,
                        help='number of hidden size in LSTM')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--lstm_dropout', type=float, default=0.0, help='dropout in LSTM')

    parser.add_argument('--K_clusters', type=int, required=True,
                        help='number of initial clusters')

    parser.add_argument('--iter', type=int, default=20,
                        help='maximum of iterations in iteration merge clusters')
    
    parser.add_argument('--epoch_in_iter', type=int, default=1,
                        help='maximum of iterations in iteration merge clusters')
          
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', type=int, default=1,
                        help='If use cuda')
    
    parser.add_argument('--lambda_AE', type=float, default=1.0, help='lambda of AE in iteration')
    parser.add_argument('--lambda_classifier', type=float, default=1.0, help='lambda_classifier of classifier in iteration')
    parser.add_argument('--lambda_outcome', type=float, default=1.0, help='lambda of outcome in iteration')
    parser.add_argument('--lambda_p_value', type=float, default=1.0, help='lambda of p value in iteration')


    args = parser.parse_args()
    print("parameters:")
    print(vars(args))
    return args


def test_AE(args, model, dataloader_test):
    criterion_MSE = nn.MSELoss()
    test_error = []
    print("-----------------")
    model.eval()
    for batch_idx, (index, batch_xvy, batch_c) in enumerate(dataloader_test):
        data_x, data_v, target = batch_xvy
        data_x = torch.autograd.Variable(data_x)
        data_v = torch.autograd.Variable(data_v)
        target = torch.autograd.Variable(target)

        if args.cuda:
            data_x = data_x.cuda()
            data_v = data_v.cuda()
            target = target.cuda()

        enc, pred = model(data_x, "autoencoder")
        loss = criterion_MSE(data_x, pred)
        test_error.append(loss.data.cpu().numpy())
    test_AE_error = np.mean(test_error)
    return test_AE_error

def update_M(data_train):
    # assert columns(M) = data_train.n_cat
    if data_train.M.size()[1] != data_train.n_cat:
        raise Exception("Invalid M!")
    
    dict_c_embedding = {}
    representations = data_train.rep
    list_C = data_train.C.tolist()
    
    for i in range(data_train.n_cat):
        dict_c_embedding[i] = []
    
    for i in range(len(list_C)):
        cp = list_C[i]
        dict_c_embedding[cp].append(representations[i,:])

    flag = 0
    for i in range(data_train.n_cat):
        c_key = i
        if len(dict_c_embedding[c_key])==0:
            continue
        embed_list = torch.stack(dict_c_embedding[c_key])
        embed_mean_dim0 = embed_list.mean(dim=0)
        data_train.M[:,c_key] = embed_mean_dim0 
    print("    update M!")

def test_AE(args, model, dataloader_test):
    criterion_MSE = nn.MSELoss()
    test_error = []
    print("-----------------")
    model.eval()
    for batch_idx, (index, batch_xvy, batch_c) in enumerate(dataloader_test):
        data_x, data_v, target = batch_xvy
        data_x = torch.autograd.Variable(data_x)
        data_v = torch.autograd.Variable(data_v)
        target = torch.autograd.Variable(target)

        if args.cuda:
            data_x = data_x.cuda()
            data_v = data_v.cuda()
            target = target.cuda()

        enc, pred = model(data_x, "autoencoder")
        loss = criterion_MSE(data_x, pred)
        test_error.append(loss.data.cpu().numpy())
    test_AE_error = np.mean(test_error)
    return test_AE_error

def update_testset_R_C_M_K(args, model, data_test, dataloader_test, data_train):
    print("-----------------")
    print("    update_testset_R_C_M_K")
    # update date_test.rep
    final_embed = torch.randn(len(data_test), args.n_hidden_fea, dtype=torch.float)
    model.eval()
    for batch_idx, (index, batch_xvy, batch_c) in enumerate(dataloader_test):
        data_x, data_v, target = batch_xvy
        data_x = torch.autograd.Variable(data_x)
        data_v = torch.autograd.Variable(data_v)
        target = torch.autograd.Variable(target)

        if args.cuda:
            data_x = data_x.cuda()
            data_v = data_v.cuda()
            target = target.cuda()

        enc, pred = model(data_x, "autoencoder")
    
        embed = enc.data.cpu()[:,0,:]
        final_embed[index] = embed

    data_test.rep = final_embed 
    print("        update data_test R!")

    data_test.n_cat = data_train.n_cat
    print("        update data_test n_cat")
    data_test.M = data_train.M
    print("        update data_test M")
    # update date_test.C
    representations = data_test.rep
    for i in range(representations.size()[0]):
        embed = representations[i,:]
        trans_embed = embed.view(embed.size()+(1,))
        xj = torch.norm(trans_embed - data_train.M, dim=0)
        new_cluster = torch.argmin(xj)
        data_test.pred_C[i] = new_cluster
    print("        update pred data_test C")


def p_value_calculate(X, y, is_intercept, X_null=None):
    # with null variable
    print("X.shape={}, y.shape={}".format(X.shape, y.shape))

    lr_model = LogisticRegression(C=1e8,solver='lbfgs', max_iter=1000)
    lr_model.fit(X, y)
    alt_prob = lr_model.predict_proba(X)

    alt_log_likelihood = -log_loss(y,
                               alt_prob,
                               normalize=False)  
    if is_intercept:      
        # if we want the p-value of beta_0
        null_prob = sum(y) / float(y.shape[0]) * \
                    np.ones(y.shape)
        null_log_likelihood = -log_loss(y,
                                    null_prob,
                                    normalize=False)
        df = 1
        G = 2 * (alt_log_likelihood - null_log_likelihood)
        p_value = chi2.sf(G, df)
    else: 
        # without null variable
        lr_model.fit(X_null, y)
        null_prob = lr_model.predict_proba(X_null)[:,1]
        null_log_likelihood = -log_loss(y,
                                    null_prob,
                                    normalize=False)
        
        df = X.shape[1] - X_null.shape[1]
        G = 2 * (alt_log_likelihood - null_log_likelihood)
        p_value = chi2.sf(G, df)
    return p_value 

def analysis_p_value_related(data_train, num_clusters, if_check):
    data_C = data_train.C
    data_v = data_train.data_v
    data_y = data_train.data_y

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
    print("dict_c_count=", dict_c_count)     
    print("dict_outcome_in_c_count=", dict_outcome_in_c_count)     

    dict_outcome_ratio = {}
    for keyc in dict_c_count:
        dict_outcome_ratio[keyc] = dict_outcome_in_c_count[keyc]/dict_c_count[keyc]
    print("dict_outcome_ratio=",dict_outcome_ratio)
    
    var_c = np.array(list_onehot)
    var_v = np.array(data_v)
    depend_y = np.array(data_y) 

    var_cpv = np.concatenate((var_c, var_v), axis=1)
    if if_check:
        print("var_c.shape={}, var_v.shape={}, depend_y.shape={}".format(var_c.shape, var_v.shape, depend_y.shape))
        print("var_cpv.shape={}".format(var_cpv.shape))
    
    # calculative p value 
    print("analysis done!")
    dict_p_value = {}
    for k1 in range(num_clusters):
        X_remove_k1 = var_cpv.copy()
        slices_k1 = list(range(var_cpv.shape[1]))
        slices_k1.remove(k1)
        X_remove_k1 = X_remove_k1[:, slices_k1]

        for k2 in range(k1+1, num_clusters):
            X_remove_k1k2 = var_cpv.copy()
            slices_k1k2 = list(range(var_cpv.shape[1]))
            # k1<k2, first remove k2, than remove k1 
            slices_k1k2.remove(k2)
            slices_k1k2.remove(k1)
            print("---------")
            print("k1={}, k2={}".format(k1,k2))
            print("slices_k1={}".format(slices_k1))
            print("slices_k1k2={}".format(slices_k1k2))
            X_remove_k1k2 = X_remove_k1k2[:, slices_k1k2]
            p_value_k1k2 = p_value_calculate(X_remove_k1, depend_y, 0, X_remove_k1k2)
            dict_p_value[(k1,k2)] = p_value_k1k2
    print("dict_p_value=", dict_p_value)
    return dict_outcome_ratio, dict_p_value

def analysis_p_value_related_onlyc(data_train, num_clusters, if_check):
    data_C = data_train.C
    data_v = data_train.data_v
    data_y = data_train.data_y

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
    print("dict_c_count=", dict_c_count)     
    print("dict_outcome_in_c_count=", dict_outcome_in_c_count)     

    dict_outcome_ratio = {}
    for keyc in dict_c_count:
        dict_outcome_ratio[keyc] = dict_outcome_in_c_count[keyc]/dict_c_count[keyc]
    print("dict_outcome_ratio=",dict_outcome_ratio)
    
    var_c = np.array(list_onehot)
    var_v = np.array(data_v)
    depend_y = np.array(data_y) 
    '''
    var_cpv = np.concatenate((var_c, var_v), axis=1)
    if if_check:
        print("var_c.shape={}, var_v.shape={}, depend_y.shape={}".format(var_c.shape, var_v.shape, depend_y.shape))
        print("var_cpv.shape={}".format(var_cpv.shape))
    '''
    # calculative p value 
    print("analysis done!")
    dict_p_value = {}
    for k1 in range(num_clusters):
        X_remove_k1 = var_c.copy()
        slices_k1 = list(range(var_c.shape[1]))
        slices_k1.remove(k1)
        X_remove_k1 = X_remove_k1[:, slices_k1]

        for k2 in range(k1+1, num_clusters):
            X_remove_k1k2 = var_c.copy()
            slices_k1k2 = list(range(var_c.shape[1]))
            # k1<k2, first remove k2, than remove k1 
            slices_k1k2.remove(k2)
            slices_k1k2.remove(k1)
            print("---------")
            print("k1={}, k2={}".format(k1,k2))
            print("slices_k1={}".format(slices_k1))
            print("slices_k1k2={}".format(slices_k1k2))
            X_remove_k1k2 = X_remove_k1k2[:, slices_k1k2]
            p_value_k1k2 = p_value_calculate(X_remove_k1, depend_y, 0, X_remove_k1k2)
            dict_p_value[(k1,k2)] = p_value_k1k2
    print("dict_p_value only c=", dict_p_value)
    print("analysis done!")

def func_analysis_test_error_D0406(args, model, data_test, dataloader_test):
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
    print("-----------------")
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
        
        # x, function, demov, mask_BoolTensor
        encoded_x, decoded_x, output_c_no_activate, output_outcome = model(x=data_x, function="outcome_logistic_regression", demov=data_v)
        
        loss_AE = criterion_MSE(data_x, decoded_x)
        loss_outcome = criterion_BCE(output_outcome, target.float())
        error_outcome_likelihood.append(loss_outcome.data.cpu().numpy())
        error_AE.append(loss_AE.data.cpu().numpy())

        # test datatest.C. 
        # #print("output_c_no_activate =", output_c_no_activate)
        # print()
        # print("output_c_no_activate.data=",output_c_no_activate.data.cpu())
        _, predicted = torch.max(output_c_no_activate.data, 1)
        correct += (predicted == batch_c).sum().item()
        total += batch_c.size(0)

        outcome_true_y.append(target.data.cpu())
        outcome_pred_prob.append(output_outcome.data.cpu()) 

        data_test.pred_C[index] = predicted.cpu()
    
    test_classifier_c_accuracy = correct/total 
    test_AE_loss = np.mean(error_AE)
    test_outcome_likelihood = np.mean(error_outcome_likelihood)

    #false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(outcome_true_y, outcome_pred_prob)
    #auc_score = metrics.auc(false_positive_rate, true_positive_rate)
    outcome_auc_score = roc_auc_score(np.concatenate(outcome_true_y, 0), np.concatenate(outcome_pred_prob, 0))
    return test_AE_loss, test_classifier_c_accuracy, test_outcome_likelihood,  outcome_auc_score

def change_label_from_highratio_to_lowratio(args, oldlabel, data_train):
    # oldlabel = clustering.labels_
    #print(oldlabel)
    #print("type(oldlabel)=", type(oldlabel))

    data_v = data_train.data_v
    data_y = data_train.data_y

    list_c = oldlabel.tolist()
    dict_c_count = {}
    dict_outcome_in_c_count = {}
    for i in range(args.K_clusters):
        dict_c_count[i] = 0 
        dict_outcome_in_c_count[i] = 0 
    
    for i in range(len(list_c)):
        dict_c_count[list_c[i]] += 1 
        if data_y[i]==1:
            dict_outcome_in_c_count[list_c[i]] += 1 

    dict_outcome_ratio = {}
    for keyc in dict_c_count:
        dict_outcome_ratio[keyc] = dict_outcome_in_c_count[keyc]/dict_c_count[keyc]
    print(" before change dict_outcome_ratio=",dict_outcome_ratio)
    
    sorted_dict_outcome_ratio = dict(sorted(dict_outcome_ratio.items(), key=lambda x:x[1], reverse=True))
    order = list(sorted_dict_outcome_ratio.keys())
    order_c_map = {}
    print("sorted_dict_outcome_ratio=",sorted_dict_outcome_ratio)
    for i in range(len(order)):
        order_c_map[order[i]] = i
    print(order_c_map)
    # change c 
    new_list_c = []
    for i in range(len(list_c)):
        new_list_c.append(order_c_map[list_c[i]])
    #print(new_list_c)
    return torch.LongTensor(new_list_c), order_c_map


def main(args):
    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # load data
    data_train = yf_dataset_withdemo(args.input_path, args.filename_train, args.n_hidden_fea)
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True, drop_last=True)
    data_test = yf_dataset_withdemo(args.input_path, args.filename_test, args.n_hidden_fea)
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, drop_last=True)

    # Algorithm 2 model
    model = model_2(args.n_input_fea, args.n_hidden_fea, args.lstm_layer, args.lstm_dropout, args.K_clusters, args.n_dummy_demov_fea, args.cuda)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion_MSE = nn.MSELoss()
    criterion_BCE = nn.BCELoss()
    criterion_CrossEntropy = nn.CrossEntropyLoss()

    print(model)
    if args.cuda:
        model = model.cuda()

    # Autoencoder, initalize the representation 
    print("/////////////////////////////////////////////////////////////////////////////")
    print("part 1: train AE and for representation initialization")

    args.output_path = "./hn_"+str(args.n_hidden_fea)+"_K_"+str(args.K_clusters)
    if os.path.isdir(args.output_path):
        shutil.rmtree(args.output_path)
    os.makedirs(args.output_path)

    part1_foldername = args.output_path + "/part1_AE_nhidden_"+str(args.n_hidden_fea)
    if os.path.isdir(part1_foldername):
        shutil.rmtree(part1_foldername)
    os.makedirs(part1_foldername)

    loss_list = []
    train_AE_loss_list = []
    test_AE_loss_list = []
    number_reassign_list = []
    random_state_list = []

    for epoch in range(args.init_AE_epoch):
        error = []
        print("-----------------")
        model.train()
        for batch_idx, (index, batch_xvy, batch_c) in enumerate(dataloader_train):
            data_x, data_v, target = batch_xvy
            data_x = torch.autograd.Variable(data_x)
            data_v = torch.autograd.Variable(data_v)
            target = torch.autograd.Variable(target)

            if args.cuda:
                data_x = data_x.cuda()
                data_v = data_v.cuda()
                target = target.cuda()

            enc, pred = model(data_x, "autoencoder")

            optimizer.zero_grad()
            loss = criterion_MSE(data_x, pred)
            loss.backward()
            optimizer.step()
            error.append(loss.data.cpu().numpy())
        loss_list.append(np.mean(error))

        train_AE_loss = np.mean(error)
        test_AE_loss = test_AE(args, model, dataloader_test)
        print("Epoch: %s, train AE loss: %s, test AE loss: %s."%(epoch, train_AE_loss, test_AE_loss))   

        train_AE_loss_list.append(train_AE_loss)
        test_AE_loss_list.append(test_AE_loss)

        torch.save(model.state_dict(), part1_foldername+'/AE_model_'+ str(epoch) +'.pt')
        print("    Saving AE models")

    plt.plot(train_AE_loss_list, color='g')
    plt.plot(test_AE_loss_list, color='b')
    plt.legend(["train_AE_loss","test_AE_loss_list"])
    plt.savefig(part1_foldername+"/part1_loss_AE.png")
    plt.close()

    print("part 1, initial done!")
    print("////////////////////////////////////////////////////////////////////////////////////")
    print("part 2, start optimizaiton the main loss")
    part2_foldername = args.output_path + "/part2_AE_nhidden_"+str(args.n_hidden_fea)
    if os.path.isdir(part2_foldername):
        shutil.rmtree(part2_foldername)
    os.makedirs(part2_foldername)

    iter_train_negloglikeli_list = []
    iter_test_negloglikeli_list = []
    iter_train_classifier_acc_list = []
    iter_test_classifier_acc_list = [] 

    iter_train_AE_list = []
    iter_test_AE_list = [] 

    iter_train_auc_list = []
    iter_test_auc_list = [] 

    min_test_negloglikeli_record = 1000000

    saved_iter = -1 
    saved_iter_list = []
    for iter_i in range(args.iter):
        print("****************************************************************************************")
        print("iter_i=", iter_i)
        # part 2, clustering.
        final_embed = torch.randn(len(data_train), args.n_hidden_fea, dtype=torch.float)
        model.eval()
        for batch_idx, (index, batch_xvy, batch_c) in enumerate(dataloader_train):
            data_x, data_v, target = batch_xvy
            data_x = torch.autograd.Variable(data_x)
            data_v = torch.autograd.Variable(data_v)
            target = torch.autograd.Variable(target)

            if args.cuda:
                data_x = data_x.cuda()
                data_v = data_v.cuda()
                target = target.cuda()

            enc, pred = model(data_x, "autoencoder")
            embed = enc.data.cpu()[:,0,:]
            final_embed[index]=embed

        #final_embed = np.vstack(embedding_list)
        final_embed = final_embed.numpy()
        print("    final_embed.shape=",final_embed.shape)

        #clustering = AgglomerativeClustering(n_clusters=args.K_clusters).fit(final_embed)
        random_state = np.random.randint(1234)
        print("random_state=", random_state)
        random_state_list.append(random_state)
        kmeans = KMeans(n_clusters=args.K_clusters, random_state=random_state).fit(final_embed)
        final_embed = torch.from_numpy(final_embed)
        data_train.rep = final_embed

        # always put the high-risk outcome in the begining. 
        # oldlabel = clustering.labels_
        oldlabel = kmeans.labels_
        new_labels, order_c_map = change_label_from_highratio_to_lowratio(args, oldlabel, data_train)
        number_reassign = len(data_train)-(torch.LongTensor(new_labels) == data_train.C).sum().item()
        print("number_reassign=",number_reassign)
        number_reassign_list.append(number_reassign)
        data_train.C = torch.LongTensor(new_labels)

        data_train.n_cat = args.K_clusters 
        data_train.M = torch.FloatTensor(args.n_hidden_fea, args.K_clusters)
        print("***************************************")
        print("data_train.M[0,:]=",data_train.M[0,:])
        update_M(data_train)
        print("data_train.M[0,:]=",data_train.M[0,:])

        print("    data_train.M.shape=", data_train.M.shape)
        print("    data_train.C.shape=", data_train.C.shape)
        print("    data_train.rep.shape=", data_train.rep.shape)
        print("4. init train *.M, *.C, *.rep done!")


        # create pseudo-label.
        # already in *.C.
        # update pseudo-label for testdata
        update_testset_R_C_M_K(args, model, data_test, dataloader_test, data_train)
        # test cluster label from clusters 
        
        # use the kmeans label (I think the result maybe the same.)
        test_final_embed = data_test.rep 
        test_final_embed = test_final_embed.numpy()
        test_cluster_old_labels = kmeans.predict(test_final_embed)
        test_list_c = test_cluster_old_labels.tolist()
        test_new_list_c = []
        for i in range(len(test_list_c)):
            test_new_list_c.append(order_c_map[test_list_c[i]])
        test_new_list_c = torch.LongTensor(test_new_list_c)
        data_test.C = torch.LongTensor(test_new_list_c)
        

        # classification and regression 
        # loss_AE, loss_classifier, loss_regression, loss_p_value (likelihood ratio)
        list_train_AE_loss = []
        list_train_classifier_loss = []
        list_train_outcome_loss = []
        list_train_p_value_loss = []
        list_train_p_value_max = []
        list_train_p_value_min = []
        list_outcome_likelihood = []

        for epoch in range(args.epoch_in_iter):
            print("epoch=",epoch)
            print("---------------------------------------------------------------------------------")
            print("iter_i = {}, epoch={}".format(iter_i, epoch))
            total = 0
            correct = 0 
            error_AE = []
            error_classifier = []
            error_outcome = []
            error_p_value = []
            error_outcome_likelihood = []
            outcome_true_y = []
            outcome_pred_prob = [] 
            print("-----------------")
            model.train()
            for batch_idx, (index, batch_xvy, batch_c) in enumerate(dataloader_train):
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
                
                # x, function, demov, mask_BoolTensor
                encoded_x, decoded_x, output_c_no_activate, output_outcome = model(x=data_x, function="outcome_logistic_regression", demov=data_v)

                #############################
                # For calculate p-value, mask k1, or, mask k1 and k2 together, to calculate G. 
                # G = -2 log((likelihood of mask k1 and k2)/( likelihood of mask k1))
                # G = -2( (log-likelihood of mask k1 and k2)  - (log-likelihood of mask k1))
                # G = 2[ (negative-log-likelihood of mask k1 and k2) - (negative-log-likelihood of mask k1) ] 
                # G = 2[ (cross_entropy_loss of mask k1 and k2) - (cross_entropy_loss of mask k1)]

                list_k_in_c = list(range(args.K_clusters))
                k1, k2 = random.sample(list_k_in_c, 2)
                #print("{},{}".format(k1, k2))

                list_mask_k1 = [0 for i in range(args.K_clusters)]
                list_mask_k1[k1] = 1 
                
                list_mask_k1k2 = [0 for i in range(args.K_clusters)]
                list_mask_k1k2[k1] = 1
                list_mask_k1k2[k2] = 1 

                mask_k1_tensor = torch.BoolTensor(list_mask_k1)
                mask_k1k2_tensor = torch.BoolTensor(list_mask_k1k2)
                encoded_x_mask_k1, decoded_x_mask_k1, output_c_no_activate_mask_k1, output_outcome_mask_k1 = \
                                        model(x=data_x, function="outcome_logistic_regression", demov=data_v, mask_BoolTensor=mask_k1_tensor)
                encoded_x_mask_k1k2, decoded_x_mask_k1k2, output_c_no_activate_mask_k1k2, output_outcome_mask_k1k2 = \
                                        model(x=data_x, function="outcome_logistic_regression", demov=data_v, mask_BoolTensor=mask_k1k2_tensor)
                ###############################

                optimizer.zero_grad()
                loss_classifier = criterion_CrossEntropy(output_c_no_activate, batch_c)
                loss_AE = criterion_MSE(data_x, decoded_x)
                loss_outcome = criterion_BCE(output_outcome, target.float())
                loss_outcome_mask_k1 = criterion_BCE(output_outcome_mask_k1, target.float())
                loss_outcome_mask_k1k2 = criterion_BCE(output_outcome_mask_k1k2, target.float())
                loss_G = 2*(loss_outcome_mask_k1k2 - loss_outcome_mask_k1)
                loss_p_value = 3.841 - loss_G #  we want loss_p_value < 0 as much as possible. 

                loss =  args.lambda_AE*loss_AE \
                        + args.lambda_classifier*loss_classifier \
                        + args.lambda_outcome*loss_outcome \
                        + args.lambda_p_value*loss_p_value 
                loss.backward()
                optimizer.step()

                error_AE.append(loss_AE.data.cpu().numpy())
                error_classifier.append(loss_classifier.data.cpu().numpy())
                error_outcome.append(loss_outcome.data.cpu().numpy())
                error_p_value.append(loss_p_value.data.cpu().numpy())
                error_outcome_likelihood.append(loss_outcome.data.cpu().numpy())
                
                _, predicted = torch.max(output_c_no_activate.data, 1)
                data_train.pred_C[index] = predicted.cpu()
                total += batch_c.size(0)
                correct += (predicted == batch_c).sum().item()

                outcome_true_y.append(target.data.cpu())
                outcome_pred_prob.append(output_outcome.data.cpu())
        
            train_outcome_auc_score = roc_auc_score(np.concatenate(outcome_true_y,0), np.concatenate(outcome_pred_prob,0))
            print("total=", total)
            classifier_c_accuracy = correct/total


            
            train_AE_loss = np.mean(error_AE)
            train_classifier_loss = np.mean(error_classifier)
            train_outcome_loss = np.mean(error_outcome)
            train_p_value_loss = np.mean(error_p_value)
            train_outcome_likeilhood = np.mean(error_outcome_likelihood)

            list_train_AE_loss.append(train_AE_loss)
            list_train_classifier_loss.append(train_classifier_loss)
            list_train_outcome_loss.append(train_outcome_loss)

            list_train_p_value_max.append(max(error_p_value))
            list_train_p_value_min.append(min(error_p_value))
            list_train_p_value_loss.append(train_p_value_loss)
            
            list_outcome_likelihood.append(train_outcome_likeilhood)

            test_AE_loss, test_classifier_c_accuracy, test_outcome_likelihood, test_outcome_auc_score = func_analysis_test_error_D0406(args, model, data_test, dataloader_test)


            iter_train_negloglikeli_list.append(train_outcome_likeilhood)
            iter_test_negloglikeli_list.append(test_outcome_likelihood)

            iter_train_classifier_acc_list.append(classifier_c_accuracy)
            iter_test_classifier_acc_list.append(test_classifier_c_accuracy)
            iter_train_AE_list.append(train_AE_loss)
            iter_test_AE_list.append(test_AE_loss)

            iter_train_auc_list.append(train_outcome_auc_score)
            iter_test_auc_list.append(test_outcome_auc_score)

            print("epoch {:2d}: train AE loss= {:.4e}, c acc= {:.4e}, outcome nll= {:.4e}, outcome_auc_score= {:.4e}, classifier loss= {:.4e}, outcome loss= {:.4e}, p_value loss= {:.4e},".format(epoch, train_AE_loss, classifier_c_accuracy, train_outcome_likeilhood, train_outcome_auc_score, train_classifier_loss, train_outcome_loss, train_p_value_loss))
            print("        : test  AE loss= {:.4e}, c acc= {:.4e}, outcome nll= {:.4e}, outcome_auc_score= {:.4e}".format(test_AE_loss, test_classifier_c_accuracy, test_outcome_likelihood, test_outcome_auc_score))


            # check p-value
            # first use data_train.C to calculate p-value, then use the preict c to calculate. (From classification, we can clearly see the accuracy is very high, so I think equal)
            # prepare the variable for regression. Bring data_x, data_v, data_y, data_train.C. (the order is the same in dataset).
            dict_outcome_ratio, dict_p_value = analysis_p_value_related(data_train, args.K_clusters, 1) 
            dict_p_value_list = list(dict_p_value.values())
            flag_morethan_0p05 = 0 
            for item in dict_p_value_list:
                if item>0.05:
                    flag_morethan_0p05 = 1 
            if (test_outcome_likelihood < min_test_negloglikeli_record) and (flag_morethan_0p05==0):
                print("save model here! iter_i={}, epoch={}".format(iter_i, epoch))
                min_test_negloglikeli_record = test_outcome_likelihood
                torch.save(model.state_dict(), part2_foldername+'/model_iter.pt')
                print("    Saving model")

                with open(part2_foldername+'/data_train_iter.pickle', 'wb') as output:
                        pickle.dump(data_train, output)
                print("    save data_train")

                saved_iter_list.append(iter_i)

                saved_iter = iter_i 

    print("number_reassign_list=",number_reassign_list)

    min_iter_test_negloglikel = min(iter_test_negloglikeli_list)
    index_min_iter_test_negloglikeli = iter_test_negloglikeli_list.index(min_iter_test_negloglikel)
    number_reassign_list_tosee = number_reassign_list[:index_min_iter_test_negloglikeli]
    train_negloglikeli_tosee = iter_train_negloglikeli_list[index_min_iter_test_negloglikeli]
    classifier_acc_tosee = iter_train_classifier_acc_list[index_min_iter_test_negloglikeli]

    print("random_state_list=",random_state_list)
    print("saved_iter_list=",saved_iter_list)
    print("saved_iter = ", saved_iter)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
