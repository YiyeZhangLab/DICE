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
import os 
from torch.nn.utils.rnn import pad_sequence

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


def pad_collate(batch):
    indices, samples, batch_c = zip(*batch)
    data_x, data_v, target = zip(*samples)
    padded_x = pad_sequence(data_x, batch_first=True)
    stacked_v = torch.stack(data_v)
    stacked_target = torch.stack(target)
    index_tensor = torch.tensor(indices, dtype=torch.long)
    stacked_c = torch.stack(batch_c)
    return index_tensor, (padded_x, stacked_v, stacked_target), stacked_c


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
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training and testing')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print("(K,hn)=", args.K_clusters, args.n_hidden_fea)
    n_clusters, inputnhidden = args.K_clusters, args.n_hidden_fea
    # taskpath = './'
    taskpath = args.training_output_path
    args.input_trained_model = taskpath + 'hn_'+str(inputnhidden) +'_K_'+str(n_clusters)+'/part2_AE_nhidden_' + str(inputnhidden) + '/model_iter.pt'
    args.input_trained_data_train = taskpath + 'hn_'+str(inputnhidden) +'_K_'+str(n_clusters)+'/part2_AE_nhidden_' + str(inputnhidden) +'/data_train_iter.pickle'

    print('args.input_trained_data_train: ', args.input_trained_data_train)
    pkl_file = open(args.input_trained_data_train, 'rb')
    data_train = pickle.load(pkl_file)
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=pad_collate)

    # dict_outcome_ratio_train, dict_c_count = analysis_cluster_number_byclustering(data_train, n_clusters, 0, "train")
    dict_outcome_ratio_train, dict_c_count = analysis_cluster_number_byclustering(data_train, n_clusters, 1, "train")

    X, y, c = data_train.rep.numpy(), data_train.data_y, data_train.C

    default_perplexity = 30
    perplexity = min(default_perplexity, len(X)-1)
    # tsne = manifold.TSNE(n_components=3, random_state=888)
    tsne = manifold.TSNE(n_components=3, random_state=888, perplexity=perplexity)

    X_tsne = tsne.fit_transform(X)
    print("X.shape=", X.shape)
    print("X_tsne.shape=", X_tsne.shape)

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  

    # figure2
    numK = n_clusters
    c_k_dict = {}
    for i in range(numK):
        c_k_dict[i] = [] 

    for i in range(len(y)):
        curk = c[i].item()
        c_k_dict[curk].append(X_tsne[i,:])

    for key in c_k_dict:
        c_k_dict[key] = np.concatenate([x.reshape((1,3)) for x in c_k_dict[key]], axis=0)
        print(c_k_dict[key].shape)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    colorlist=['red','orange','blue','green','cyan','purple']
    for key in c_k_dict:
        ax.scatter(c_k_dict[key][:,0],c_k_dict[key][:,1],c_k_dict[key][:,2],s=30,color=colorlist[key],marker='.',alpha=0.5,label='cluster '+str(key+1)+ ', '+ str(round(dict_outcome_ratio_train[key]*100,2))+'% of outcome 1') 

    plt.legend(fontsize = 14, bbox_to_anchor=(0.8, 0.1), loc="lower right")
    ax.view_init(elev=-73, azim= -0)
    ax.set_xlim(-16, 12)
    plt.savefig(args.training_output_path + "hn_"+str(args.n_hidden_fea)+"_K_"+str(args.K_clusters) + '/figs/tsne_3d.png', bbox_inches='tight')


    """
    print("\n--- Generating Cluster Characteristics Table ---")
    demo_data_path = 'C:/Users/jil4047/Desktop/repos/dice_repo/dataset/mis_cat_250716/demo_data.csv'
    X = pd.read_csv(demo_data_path, index_col=False).iloc[:, 1:]

    X = X.to_numpy()

    print('X shape: ', X.shape)

    # Step 1 & 2: Identify data points for each cluster and extract original features
    cluster_data = {i: [] for i in range(numK)}
    for i in range(len(c)):
        cluster_idx = c[i].item() # Get the cluster assignment for the current data point
        cluster_data[cluster_idx].append(X[i, :]) # Append the original features (X)

    # Convert lists of arrays to numpy arrays for easier manipulation
    for cluster_idx in cluster_data:
        if cluster_data[cluster_idx]: # Check if the list is not empty
            cluster_data[cluster_idx] = np.array(cluster_data[cluster_idx])
        else:
            cluster_data[cluster_idx] = np.empty((0, X.shape[1])) # Handle empty clusters

    # Step 3: Calculate summary statistics for each feature within each cluster
    cluster_characteristics = pd.DataFrame()

    # You might want to name your features, if you have feature names
    # For now, let's use 'Feature_0', 'Feature_1', etc.
    feature_names = [f'Feature_{j}' for j in range(X.shape[1])]

    # summary_methods = {
    #     'Min': lambda arr: np.min(arr, axis=0),
    #     'Max': lambda arr: np.max(arr, axis=0),
    #     'Mean': lambda arr: np.mean(arr, axis=0),
    #     'Median': lambda arr: np.median(arr, axis=0),
    #     'Std Dev': lambda arr: np.std(arr, axis=0),
    #     'Count': lambda arr: len(arr) # To see how many samples in each cluster
    # }
    summary_methods = {
    'Min': lambda arr: np.min(arr, axis=0) if arr.shape[0] > 0 else np.full(arr.shape[1], np.nan),
    'Max': lambda arr: np.max(arr, axis=0) if arr.shape[0] > 0 else np.full(arr.shape[1], np.nan),
    'Mean': lambda arr: np.mean(arr, axis=0) if arr.shape[0] > 0 else np.full(arr.shape[1], np.nan),
    'Median': lambda arr: np.median(arr, axis=0) if arr.shape[0] > 0 else np.full(arr.shape[1], np.nan),
    # 'Std Dev': lambda arr: np.std(arr, axis=0) if arr.shape[0] > 1 else np.full(arr.shape[1], 0.0), # std dev of 1 item is 0 or NaN
    'Count': lambda arr: len(arr)
    }

    for cluster_idx in range(numK):
        cluster_df = pd.DataFrame(index=feature_names)
        current_cluster_data = cluster_data[cluster_idx]

        if current_cluster_data.shape[0] == 0:
            print(f"Warning: Cluster {cluster_idx+1} is empty. Skipping statistics.")
            for method_name in summary_methods:
                cluster_df[f'Cluster {cluster_idx+1}_{method_name}'] = ['N/A'] * len(feature_names) # Fill with N/A
            cluster_characteristics = pd.concat([cluster_characteristics, cluster_df], axis=1)
            continue

        print(f"\nProcessing Cluster {cluster_idx+1} (contains {current_cluster_data.shape[0]} samples)")

        for method_name, method_func in summary_methods.items():
            if method_name == 'Count':
                # Count applies to the cluster as a whole, not per feature
                cluster_df[f'Cluster {cluster_idx+1}_{method_name}'] = [current_cluster_data.shape[0]] * len(feature_names)
                # You might want to put 'Count' as a separate row or just print it
                # For table uniformity, we'll assign it to all features for now.
                # A more refined table might have 'Count' as a header for the cluster column.
            else:
                stats = method_func(current_cluster_data)
                cluster_df[f'Cluster {cluster_idx+1}_{method_name}'] = stats

        # Transpose to have features as rows and statistics as columns for the current cluster
        # This might be counter-intuitive based on initial request, but often more readable
        # Let's stick to the initial thought: a column per cluster, rows for features.
        # So we'll append columns for each cluster.

        # If you want a column for each cluster, with features as rows:
        # We need to construct the dataframe differently.
        # Let's create a temporary DataFrame for each cluster's stats
        temp_cluster_summary = pd.DataFrame()
        temp_cluster_summary['Feature'] = feature_names
        for method_name, method_func in summary_methods.items():
            if method_name == 'Count':
                temp_cluster_summary[method_name] = [current_cluster_data.shape[0]] * len(feature_names)
            else:
                temp_cluster_summary[method_name] = method_func(current_cluster_data)
        
        # Rename columns to reflect the cluster
        temp_cluster_summary = temp_cluster_summary.set_index('Feature')
        temp_cluster_summary.columns = [f'Cluster {cluster_idx+1}_{col}' for col in temp_cluster_summary.columns]
        
        if cluster_characteristics.empty:
            cluster_characteristics = temp_cluster_summary
        else:
            cluster_characteristics = pd.concat([cluster_characteristics, temp_cluster_summary], axis=1)

    print("\n--- Cluster Characteristics Table ---")
    print(cluster_characteristics)

    # Save the table to a CSV file
    output_dir = args.training_output_path + "hn_"+str(args.n_hidden_fea)+"_K_"+str(args.K_clusters)
    os.makedirs(output_dir, exist_ok=True)
    # cluster_table_path = os.path.join(output_dir, 'cluster_characteristics.csv')
    cluster_table_path = os.path.join(output_dir, 'demo_cluster_characteristics.csv')

    cluster_characteristics.to_csv(cluster_table_path)
    print(f"\nCluster characteristics table saved to: {cluster_table_path}")
    """