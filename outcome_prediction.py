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

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn import metrics


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

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


def update_curset_pred_C_and_repD0420(args, model, data_cur, dataloader_cur, varname, datatrainM):
    print("-----------------")
    print("Deal with:", varname)
    #print("    update pred_C and pred_C")
    # update date_cur.rep
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


def calculate_cluster_metrics(data_train):
    labels_pred = data_train.C.tolist()
    labels_true = data_train.data_y
    X = data_train.rep.numpy()
    print(type(X))
    score = {}
    score['silhouette_score'] = metrics.silhouette_score(X, labels_pred, metric='euclidean')
    score['calinski_harabasz_score'] = metrics.calinski_harabasz_score(X, labels_pred)
    score['davies_bouldin_score'] = metrics.davies_bouldin_score(X, labels_pred)
    return score


def parse_args():
    parser = argparse.ArgumentParser(description='ppd-aware clustering')
    parser.add_argument('--training_output_path', type=str, required=True,
                        help='location of training output')
    parser.add_argument('--n_hidden_fea', type=int, required=True,
                        help='number of hidden size in LSTM')
    parser.add_argument('--input_path', type=str, required=True,
                        help='location of input dataset')
    parser.add_argument('--filename_train', type=str, required=True,
                        help='location of the data corpus')
    parser.add_argument('--filename_valid', type=str, required=True,
                        help='filename_valid')
    parser.add_argument('--filename_test', type=str, required=True,
                        help='file_name_test')
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
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--input_trained_data_train', type=str, required=False,
                        help='location of the data corpus')
    parser.add_argument('--input_trained_model', type=str, required=False,
                        help='location of the data corpus')
    parser.add_argument('--cuda', type=int, default=0,
                        help='If use cuda')
    args = parser.parse_args()
    return args

def calculate_rate_value(true_y, predict_results):
    conf_mat = confusion_matrix(true_y, predict_results)
    conf_mat_tolist = conf_mat.tolist()
    print("conf_mat_tolist=",conf_mat_tolist)
    number_TN = conf_mat_tolist[0][0]
    number_FN = conf_mat_tolist[1][0]
    number_FP = conf_mat_tolist[0][1]
    number_TP = conf_mat_tolist[1][1]
    acc=accuracy_score(true_y, predict_results)
    try:
        fpr=number_FP/(number_FP + number_TN)
        tpr=number_TP/(number_TP + number_FN)
        fnr=number_FN/(number_FN + number_TP)
        tnr=number_TN/(number_TN + number_FP)
        PPV=number_TP/(number_TP + number_FP) #positive and negative predictive values
        NPV=number_TN/(number_FN + number_TN)
        #message = 'acc: {:.4f}, fpr: {:.4f}, tpr: {:.4f}, fnr: {:.4f}, tnr: {:.4f}, PPV: {:.4f}, NPV: {:.4f}'.format(acc,fpr,tpr,fnr,tnr,PPV,NPV)
        #print(message)
        message = {'acc': acc, 'fpr':fpr, 'tpr':tpr, 'fnr':fnr, 'tnr':tnr, 'PPV':PPV, 'NPV':NPV}
    except:
        if number_FP + number_TN != 0:
            fpr=number_FP/(number_FP + number_TN)
        elif number_FP ==0:
            fpr = 0 
        else:
            fpr = 1 
        
        if (number_TP + number_FN) != 0:
            tpr=number_TP/(number_TP + number_FN)
        elif number_TP==0:
            tpr = 0
        else:
            trp = 1
        
        if (number_FN + number_TP) != 0:
            fnr=number_FN/(number_FN + number_TP)
        elif number_FN==0:
            fnr = 0
        else:
            fnr = 1 
        
        if (number_TN + number_FP)!=0:
            tnr=number_TN/(number_TN + number_FP)
        elif number_TN ==0:
            tnr = 0
        else:
            tnr = 1 
        
        if (number_TP + number_FP)!=0:
            PPV=number_TP/(number_TP + number_FP) #positive and negative predictive values
        elif number_TP ==0:
            PPV = 0
        else:
            PPV = 1 
        
        if (number_FN + number_TN)!=0:
            NPV=number_TN/(number_FN + number_TN)
        elif number_TN ==0:
            NPV = 0
        else:
            PPV = 1 
        message = {'acc': acc, 'fpr':fpr, 'tpr':tpr, 'fnr':fnr, 'tnr':tnr, 'PPV':PPV, 'NPV':NPV}
    return message 


def calculate_metrice(true_y, prediction_prob):
    pos = sum(true_y)
    neg = len(true_y) - pos
    true_y = np.array(true_y)
    prediction_prob = np.array(prediction_prob)
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(true_y, prediction_prob)
    #print("AUC=", metrics.auc(fpr, tpr))
    auc_score = metrics.auc(false_positive_rate, true_positive_rate)
    #print("auc_score=",auc_score)
    #print("-----------------")

    youden = []
    for i in range(len(thresholds)):
        thres = thresholds[i]
        fpr = false_positive_rate[i]
        tpr = true_positive_rate[i]
        youden.append(tpr - fpr)
    
    max_youden = max(youden)
    optim_thres = thresholds[youden.index(max_youden)]
    print("use optim_thres from auc=",optim_thres,", type(optim_thres)=",type(optim_thres))
    predict_results = [1 if p>=optim_thres else 0 for p in prediction_prob ]
    message1 = calculate_rate_value(true_y, predict_results) 
    message1["thres"]= optim_thres
    #print("-----------------")
    #print("use threshold 0.5:")
    predict_results = [1 if p>=0.5 else 0 for p in prediction_prob ]
    message2 = calculate_rate_value(true_y, predict_results)
    message2["thres"]= 0.5
    return auc_score, message1, message2  


def plot_roc(labels, predict_prob):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()
    
    youden = []
    for i in range(len(thresholds)):
        thres = false_positive_rate[i]
        tpr = true_positive_rate[i]
        fpr = false_positive_rate[i]
        youden.append(tpr - fpr)
    
    max_youden = max(youden)
    optim_thres = thresholds[youden.index(max_youden)]
    print("optim_thres=",optim_thres)
    
    print("after use optim_thres")
    #print("predict_prob=",predict_prob)
    predict_results = [1 if p>=optim_thres else 0 for p in predict_prob ]
    #print("predict_results=",predict_results)
    conf_mat = confusion_matrix(labels, predict_results)
    print("conf_mat=",conf_mat)
    print("\nclassification_report=\n",classification_report(labels, predict_results))
    print("accuracy_score=",accuracy_score(labels, predict_results))

    return optim_thres


if __name__ == '__main__':
    args = parse_args()
    print("(K,hn)=", args.K_clusters, args.n_hidden_fea)
    n_clusters, inputnhidden = args.K_clusters, args.n_hidden_fea
    taskpath = './'
    args.input_trained_model = taskpath + 'hn_'+str(inputnhidden) +'_K_'+str(n_clusters)+'/part2_AE_nhidden_' + str(inputnhidden) + '/model_iter.pt'
    args.input_trained_data_train = taskpath + 'hn_'+str(inputnhidden) +'_K_'+str(n_clusters)+'/part2_AE_nhidden_' + str(inputnhidden) +'/data_train_iter.pickle'

    pkl_file = open(args.input_trained_data_train, 'rb')

    data_train = pickle.load(pkl_file)
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True, drop_last=True)
    data_test = yf_dataset_withdemo(args.input_path, args.filename_test, args.n_hidden_fea)
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, drop_last=True)
    data_valid = yf_dataset_withdemo(args.input_path, args.filename_valid, args.n_hidden_fea)
    dataloader_valid = torch.utils.data.DataLoader(data_valid, batch_size=1, shuffle=False, drop_last=True)

    model = model_2(args.n_input_fea, args.n_hidden_fea, args.lstm_layer, args.lstm_dropout, args.K_clusters, args.n_dummy_demov_fea, args.cuda)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #print(model)
    if args.cuda:
        model = model.cuda()

    device = torch.device("cpu")
    model.load_state_dict(torch.load(args.input_trained_model, map_location=device))

    # update the rep and c first, then calculate the result. Here we evaluate the clustering metrics on data_test.
    update_curset_pred_C_and_repD0420(args, model, data_test, dataloader_test,"data_test", data_train)
    update_curset_pred_C_and_repD0420(args, model, data_valid, dataloader_valid,"data_valid", data_train)
    
    feature_train = data_train.rep.numpy()
    target_train = np.array(data_train.data_y)
    feature_test = data_valid.rep.numpy()
    target_test = np.array(data_valid.data_y)


    # power_ = 0
    # model =  LogisticRegression(C=10**power_, multi_class='multinomial', solver='lbfgs',max_iter=200)

    model = LogisticRegression(max_iter=200)
    model.fit(feature_train, target_train)

    print("------------------------")
    print("in train set")
    predict_results = model.predict(feature_train)
    predict_prob = model.predict_proba(feature_train)
    predict_prob1 = predict_prob[:,1]

    fpr, tpr, thresholds = metrics.roc_curve(target_train, predict_prob1)
    print("auc=",metrics.auc(fpr, tpr))

    plot_roc(target_train, predict_prob1)
    auc_score, message1, message2   = calculate_metrice(target_train, predict_prob1)
    print("auc_score=", auc_score)
    print("message1=", message1)
    print("message2=", message2)


    print("--------------------------")
    print("in test set")
    predict_results = model.predict(feature_test)
    predict_prob = model.predict_proba(feature_test)
    predict_prob1 = predict_prob[:,1]

    fpr, tpr, thresholds = metrics.roc_curve(target_test, predict_prob1)
    print("auc=",metrics.auc(fpr, tpr))

    plot_roc(target_test, predict_prob1)
    auc_score, message1, message2   = calculate_metrice(target_test, predict_prob1)
    print("auc_score=", auc_score)
    print("message1=", message1)
    print("message2=", message2)