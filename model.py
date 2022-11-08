import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *


class MultiDeep(nn.Module):
    def __init__(self, ncell, ndrug, ncellfeat, ndrugfeat, nhid, nheads, alpha):
        """Dense version of GAT."""
        super(MultiDeep, self).__init__()
        
        self.cell_attentions1 = [GraphAttentionLayer(ncellfeat, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.cell_attentions1):
            self.add_module('attentioncell1_{}'.format(i), attention)
        self.cell_MultiHead1 = [selfattention(ncell, nhid, ncell) for _ in range(nheads)]
        for i, attention in enumerate(self.cell_MultiHead1):
            self.add_module('selfattentioncell1_{}'.format(i), attention)
        self.cell_prolayer1 = nn.Linear(nhid*nheads, nhid*nheads, bias = False)
        self.cell_LNlayer1 = nn.LayerNorm(nhid*nheads)
        
            
        self.cell_attentions2 = [GraphAttentionLayer(nhid * nheads, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.cell_attentions2):
            self.add_module('attentioncell2_{}'.format(i), attention)
        self.cell_MultiHead2 = [selfattention(ncell, nhid, ncell) for _ in range(nheads)]
        for i, attention in enumerate(self.cell_MultiHead2):
            self.add_module('selfattentioncell2_{}'.format(i), attention)
        self.cell_prolayer2 = nn.Linear(nhid*nheads, nhid*nheads, bias = False)
        self.cell_LNlayer2 = nn.LayerNorm(nhid*nheads)

        self.drug_attentions1 = [GraphAttentionLayer(ndrugfeat, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.drug_attentions1):
            self.add_module('attentiondrug1_{}'.format(i), attention)
        self.drug_MultiHead1 = [selfattention(ndrug, nhid, ndrug) for _ in range(nheads)]
        for i, attention in enumerate(self.drug_MultiHead1):
            self.add_module('selfattentiondrug1_{}'.format(i), attention)
        self.drug_prolayer1 = nn.Linear(nhid*nheads, nhid*nheads, bias = False)
        self.drug_LNlayer1 = nn.LayerNorm(nhid*nheads)
        
        
        self.drug_attentions2 = [GraphAttentionLayer(nhid * nheads, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.drug_attentions2):
            self.add_module('attentiondrug2_{}'.format(i), attention)
        self.drug_MultiHead2 = [selfattention(ndrug, nhid, ndrug) for _ in range(nheads)]
        for i, attention in enumerate(self.drug_MultiHead2):
            self.add_module('selfattentiondrug2_{}'.format(i), attention)
        self.drug_prolayer2 = nn.Linear(nhid*nheads, nhid*nheads, bias = False)
        self.drug_LNlayer2 = nn.LayerNorm(nhid*nheads)
        
        self.FClayer1 = nn.Linear(nhid*nheads*2, nhid*nheads*2)
        self.FClayer2 = nn.Linear(nhid*nheads*2, nhid*nheads*2)
        self.FClayer3 = nn.Linear(nhid*nheads*2, 1)
        self.output = nn.Sigmoid()
    
    def forward(self, cell_features, cell_adj, drug_features, drug_adj, idx_cell_drug, device):
        cellx = torch.cat([att(cell_features, cell_adj) for att in self.cell_attentions1], dim=1)
        cellx = self.cell_prolayer1(cellx)
        cellayer = cellx
        temp = torch.zeros_like(cellx)
        for selfatt in self.cell_MultiHead1:
            temp = temp+selfatt(cellx)
        cellx = temp + cellayer
        cellx = self.cell_LNlayer1(cellx)
        
        cellx = torch.cat([att(cellx, cell_adj) for att in self.cell_attentions2], dim=1)
        cellx = self.cell_prolayer2(cellx)
        cellayer = cellx
        temp = torch.zeros_like(cellx)
        for selfatt in self.cell_MultiHead2:
            temp = temp+selfatt(cellx)
        cellx = temp + cellayer
        cellx = self.cell_LNlayer2(cellx)
        
        drugx = torch.cat([att(drug_features, drug_adj) for att in self.drug_attentions1], dim=1)
        drugx = self.drug_prolayer1(drugx)
        druglayer = drugx
        temp = torch.zeros_like(drugx)
        for selfatt in self.drug_MultiHead1:
            temp = temp+selfatt(drugx)
        drugx = temp + druglayer
        drugx = self.drug_LNlayer1(drugx)
        
        drugx = torch.cat([att(drugx, drug_adj) for att in self.drug_attentions2], dim=1)
        drugx = self.drug_prolayer2(drugx)
        druglayer = drugx
        temp = torch.zeros_like(drugx)
        for selfatt in self.drug_MultiHead2:
            temp = temp+selfatt(drugx)
        drugx = temp + druglayer
        drugx = self.drug_LNlayer2(drugx) 
        
        cell_drug_x = torch.cat((cellx[idx_cell_drug[:, 0]], drugx[idx_cell_drug[:, 1]]), dim=1)
        cell_drug_x = cell_drug_x.to(device)
        cell_drug_x = self.FClayer1(cell_drug_x)
        cell_drug_x = F.relu(cell_drug_x)
        cell_drug_x = self.FClayer2(cell_drug_x)
        cell_drug_x = F.relu(cell_drug_x)
        cell_drug_x = self.FClayer3(cell_drug_x)
        cell_drug_x = cell_drug_x.squeeze(-1)
        return cell_drug_x

