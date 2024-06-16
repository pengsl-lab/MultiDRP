import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import networkx as nx
import csv
from scipy.stats import pearsonr
import torch.nn.functional as F


path="./data/PRISM/"


def multiomics_data():
    copynumber = np.genfromtxt("{}{}.csv".format(path, "copynumber_23316dim"), delimiter=',', skip_header=3, dtype=np.dtype(str))
    RNAseq = np.genfromtxt("{}{}.csv".format(path, "RNAseq_48392dim"), delimiter=',', skip_header=2, dtype=np.dtype(str))

    for i in range(copynumber.shape[0]):
        for j in range(copynumber.shape[1]):
            copynumber[i,j]=copynumber[i,j].replace('"', '')

    for i in range(RNAseq.shape[0]):
        for j in range(RNAseq.shape[1]):
            RNAseq[i,j]=RNAseq[i,j].replace('"', '')  
            
    cell=list(set(copynumber[:, 0]) & set(RNAseq[:, 0])) 
    copynumber = multiomics_fusion(copynumber, cell)
    RNAseq = multiomics_fusion(RNAseq, cell)

    copynumber = scale(copynumber)
    RNAseq = scale(RNAseq)

    pca = PCA(n_components=128)
    copynumber = pca.fit_transform(np.array(copynumber, dtype=float))
    pca = PCA(n_components=128)
    RNAseq = pca.fit_transform(np.array(RNAseq, dtype=float))

    cell_number = len(cell)
    copynumber_adj = sim_graph(copynumber, cell_number)

    fingerprint = np.genfromtxt("{}{}.csv".format(path, "fingerprint_881dim"), delimiter=',', dtype=np.dtype(str))
    physicochemical = np.genfromtxt("{}{}.csv".format(path, "physicochemical_269dim"), delimiter=',', dtype=np.dtype(str))

    final_fingerprint, final_physicochemical = [], []
    drug = list(fingerprint[:, 0])
                
    drugtemp = []
    lablefile = csv.reader(open("{}{}.csv".format(path, "drug_cell_lable"), 'r'))
    lablefile = list(lablefile)
    drug_cell_lable = []
    for i in range(1, len(lablefile)):
        if lablefile[i][6] != 'NA' and float(lablefile[i][6])>0:
            drug_cell_lable.append(lablefile[i][0:1]+lablefile[i][3:7])
    drug_cell_lable = data_select(cell, np.array(drug_cell_lable))
    drugtemp = set(drug_cell_lable[:,1])
        
    for i in range(len(drug)):
        if drug[i] in drugtemp:
            final_fingerprint.append(fingerprint[i])
            final_physicochemical.append(physicochemical[i])
            
    fingerprint = np.array(final_fingerprint)
    physicochemical = np.array(final_physicochemical)
    
    drug = list(fingerprint[:,0])
    pca = PCA(n_components=128)
    fingerprint = pca.fit_transform(np.array(fingerprint[:, 1:], dtype=float))
    
    physicochemical = scale(np.array(physicochemical[:,1:], dtype=float))
    pca = PCA(n_components=128)
    physicochemical = pca.fit_transform(physicochemical)
    physicochemical_adj = sim_graph(physicochemical, len(drug))
    
    sample_set=data_index(cell, drug, drug_cell_lable)
    print("drug cell lable", sample_set.shape)
    
    RNAseq, copynumber_adj = torch.FloatTensor(RNAseq), torch.FloatTensor(copynumber_adj)
    fingerprint, physicochemical_adj = torch.FloatTensor(fingerprint), torch.FloatTensor(physicochemical_adj)
    return RNAseq, copynumber_adj, fingerprint, physicochemical_adj, sample_set
    

def multiomics_fusion(omics_data, cell_fusion):
    finalomics = []
    cell_index = omics_data[:,0].tolist()
    for cell in cell_fusion:
        index = cell_index.index(cell)
        finalomics.append(np.array(omics_data[index, 1:],dtype=float))
    return np.array(finalomics)


def sim_graph(omics_data, cell_number):
    sim_matrix = np.zeros((cell_number, cell_number), dtype=float)
    adj_matrix = np.zeros((cell_number, cell_number), dtype=float)
    
    for i in range(cell_number):
        for j in range(i+1):
            sim_matrix[i,j] = np.dot(omics_data[i], omics_data[j]) / (np.linalg.norm(omics_data[i]) * np.linalg.norm(omics_data[j]))
            sim_matrix[j,i] = sim_matrix[i,j]
    
    for i in range(cell_number):
        topindex = np.argsort(sim_matrix[i])[-10:]
        for j in topindex:
            adj_matrix[i,j] = 1
    return adj_matrix   


def data_select(cell, lable):
    Reglable = np.array(lable[:,4], dtype = float)
    Reglable = np.log(Reglable)
    Percentile = np.percentile(Reglable,[0,25,50,75,100])
    IQR = Percentile[3] - Percentile[1]
    UpLimit = Percentile[3] + IQR * 1.5
    DownLimit = Percentile[1] - IQR * 1.5
    print("DownLimit and UpLimit:", DownLimit, UpLimit)
    sample_set = []
    for i in range(len(Reglable)):  
        if Reglable[i] > DownLimit and Reglable[i] < UpLimit and lable[i,0] in cell:
            sample_set.append(lable[i])
    return np.array(sample_set)


def data_index(cell, drug, lable):
    Reglable = np.array(lable[:,4], dtype = float)
    Reglable = np.log(Reglable)
    sample_set = []
    
    for i in range(len(lable)):  # len(lable)
        sample_set.append([cell.index(lable[i,0]),drug.index(lable[i,1]), Reglable[i]])
    print("the number of cell and drug:",len(set(np.array(sample_set)[:,0])), len(set(np.array(sample_set)[:,1])))
    sample_set = torch.Tensor(sample_set)
    return sample_set


