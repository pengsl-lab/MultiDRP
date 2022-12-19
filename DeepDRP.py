import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.autograd import Variable
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils import multiomics_data
from model import MultiDeep
import torch.utils.data as Dataset


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--batch', type=int, default=128, help='Number of batch size')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=4, help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=5, help='Patience')

args = parser.parse_args()

random.seed(args.seed)       
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.manual_seed(args.seed)
else:
    device = torch.device("cpu")
    

# Load data
cell_features, cell_adj, drug_features, drug_adj, sample_set=multiomics_data()
cell_features, cell_adj = Variable(cell_features), Variable(cell_adj)
drug_features, drug_adj = Variable(drug_features), Variable(drug_adj)
cell_features, cell_adj = cell_features.to(device), cell_adj.to(device)
drug_features, drug_adj = drug_features.to(device), drug_adj.to(device)


# Model and optimizer 
model = MultiDeep(ncell=cell_features.shape[0],
            ndrug=drug_features.shape[0],
            ncellfeat=cell_features.shape[1],
            ndrugfeat=drug_features.shape[1],
            nhid=args.hidden,
            nheads=args.nb_heads,
            alpha=args.alpha)
optimizer = optim.Adam(model.parameters(), lr=args.lr) 

loss_func = nn.MSELoss()
loss_func.to(device)
best_value = [1000, 1]
MHAresult = []


def train(epoch, index_tra, y_tra, index_val, y_val):
    t = time.time()
    tra_dataset = Dataset.TensorDataset(index_tra, y_tra)
    train_dataset = Dataset.DataLoader(tra_dataset, batch_size=args.batch, shuffle=True)
    model.train()
    for index_trian, y_train in train_dataset:
        y_train = y_train.to(device)
        y_tpred = model(cell_features, cell_adj, drug_features, drug_adj, index_trian.numpy().astype(int), device) 
        loss_train = loss_func(y_tpred, y_train)
        optimizer.zero_grad()
        loss_train.backward() 
        optimizer.step() 
        
    model.eval()
    loss_valid, RMSE_valid, PCC_valid, R2_valid = [], [], [], []
    val_dataset = Dataset.TensorDataset(index_val, y_val)
    valid_dataset = Dataset.DataLoader(val_dataset, batch_size=args.batch, shuffle=True)
    pred_valid, true_valid = [], []
    
    for index_valid, y_valid in valid_dataset:
        y_valid = y_valid.to(device)
        y_vpred = model(cell_features, cell_adj, drug_features, drug_adj, index_valid.numpy().astype(int), device)  
        loss = loss_func(y_vpred, y_valid) 
        pred_valid.extend( y_vpred.cpu().detach().numpy())
        true_valid.extend( y_valid.cpu().detach().numpy())
        
    loss_valid = mean_squared_error(true_valid, pred_valid)
    RMSE_valid = np.sqrt(loss_valid)
    MAE_valid = mean_absolute_error(true_valid, pred_valid)
    MAPE_valid = np.mean(np.abs((np.array(true_valid) - np.array(pred_valid)) / np.array(true_valid))) * 100
    PCC_valid = pearsonr(true_valid, pred_valid)[0]
    R2_valid = r2_score(true_valid, pred_valid)
    
    pred_train = y_tpred.cpu().detach().numpy()
    true_train = y_train.cpu().detach().numpy()
    RMSE_train = np.sqrt(loss_train.item(), out=None)
    MAE_train = mean_absolute_error(true_train, pred_train)
    MAPE_train = np.mean(np.abs((np.array(true_train) - np.array(pred_train)) / np.array(true_train))) * 100
    R2_train = r2_score(true_train, pred_train)
    PCC_train = pearsonr(true_train, pred_train)[0]
                
    print('Epoch: {:04d}'.format(epoch+1),
                'loss_train: {:.4f}'.format(loss_train.item()),
                'RMSE_train: {:.4f}'.format(RMSE_train),
                'MAE_train: {:.4f}'.format(MAE_train),
                'MAPE_train: {:.4f}'.format(MAPE_train),
                'PCC_train: {:.4f}'.format(PCC_train),
                'R2_train: {:.4f}'.format(R2_train),
                '\t loss_valid: {:.4f}'.format(loss_valid),
                'RMSE_valid: {:.4f}'.format(RMSE_valid),
                'MAE_valid: {:.4f}'.format(MAE_valid),
                'MAPE_valid: {:.4f}'.format(MAPE_valid),
                'PCC_valid: {:.4f}'.format(PCC_valid),
                'R2_valid: {:.4f}'.format(R2_valid),
                'time: {:.4f}s'.format(time.time() - t))
        
    if RMSE_valid <= best_value[0]:
        best_value[0] = RMSE_valid
        best_value[1] = epoch+1
        torch.save(model.state_dict(),"./output/models.pkl")
    return best_value[1], RMSE_valid


    
def compute_test(index_test, y_test):
    model.eval()
    loss_test, PCC_test, RMSE_test, R2_test = [], [], [], []
    pred_test, true_test = [], []
    dataset = Dataset.TensorDataset(index_test, y_test)
    test_dataset = Dataset.DataLoader(dataset, batch_size=args.batch, shuffle=True)
    for index_test, y_test in test_dataset:
        y_test = y_test.to(device)
        y_pred = model(cell_features, cell_adj, drug_features, drug_adj, index_test.numpy().astype(int), device)  
        loss_test = loss_func(y_pred, y_test)
        pred_test.extend(y_pred.cpu().detach().numpy())
        true_test.extend(y_test.cpu().detach().numpy())
    
    loss_test = mean_squared_error(true_test, pred_test)
    RMSE_test = np.sqrt(loss_test)
    MAE_test = mean_absolute_error(true_test, pred_test)
    MAPE_test = np.mean(np.abs((np.array(true_test) - np.array(pred_test)) / np.array(true_test))) * 100
    PCC_test = pearsonr(true_test, pred_test)[0]
    R2_test = r2_score(true_test, pred_test)

    with open("MATDRP1", 'a') as f:
        f.write(str(RMSE_test) + " " + str(MAE_test) + " " +str(MAPE_test) + " " +str(PCC_test) + " " + str(R2_test) + "\n")
    
    print("Test set results:",
          "loss= {:.4f}".format(loss_test),
          "RMSE= {:.4f}".format(RMSE_test),
          'MAE= {:.4f}'.format(MAE_test),
          'MAPE= {:.4f}'.format(MAPE_test),
          "PCC_test= {:.4f}".format(PCC_test),
          "R2_test= {:.4f}\n".format(R2_test))

    


# Train model

t_total = time.time()
train_set, test_set = train_test_split(np.arange(sample_set.shape[0]), test_size=0.1, random_state=np.random.randint(0,1000)) 
train_set, valid_set = train_test_split(train_set, test_size=1/9, random_state=np.random.randint(0,1000))
    
index_train, y_train = sample_set[train_set[:], :2], sample_set[train_set[:], 2]
index_valid, y_valid = sample_set[valid_set[:], :2], sample_set[valid_set[:], 2]
index_test, y_test = sample_set[test_set[:], :2], sample_set[test_set[:], 2]
y_train, y_test, y_valid = Variable(y_train, requires_grad=True), Variable(y_test, requires_grad=True), Variable(y_valid, requires_grad=True)
    
model.to(device)
pcc_valid = [0]
bad_counter = 0
for epoch in range(args.epochs):
    best_epoch, avg_pcc_valid = train(epoch, index_train, y_train, index_valid, y_valid)
    pcc_valid.append(avg_pcc_valid)
        
    if abs(pcc_valid[-1]-pcc_valid[-2])<0.005:
        bad_counter += 1
    else:
        bad_counter = 0
            
    if bad_counter >= args.patience:
        break
        
print("Optimization Finished. Total time: {:.4f}s".format(time.time() - t_total))
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('./output/models.pkl'))
# Testing
compute_test(index_test, y_test)
