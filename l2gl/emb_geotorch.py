import torch
import torch.nn as nn
import torch.optim as optim
import geotorch
import numpy as np
import matplotlib.pyplot as plt
import local2global as l2g
import local2global.example as ex
import local2global_embedding
import scipy.sparse as ss
import scipy.sparse.linalg as sl
from scipy.stats import ortho_group
from scipy.linalg import sqrtm
from scipy.sparse import diags, csr_matrix, csc_matrix, coo_matrix
from tqdm.notebook import tqdm
from itertools import chain
import itertools
import random
import pandas as pd
import torch_scatter as ts
import torch_geometric as tg
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
import networkx as nx
import glob
import os
import autograd.numpy as anp
from local2global import Patch
from local2global_embedding.network import tgraph
from local2global_embedding.patches import create_patch_data
from local2global_embedding.clustering import louvain_clustering
import Local2Global_embedding.local2global_embedding.embedding.svd as svd
import Local2Global_embedding.local2global_embedding.embedding.gae as gae
import matplotlib.cm as cm






def get_error(patches, result, nodes):
    n=len(patches)
    rot=[result.transformation[i].weight.detach().numpy() for i in range(n)]
    shift=[result.transformation[i].bias.detach().numpy() for i in range(n)]

    emb_problem = l2g.AlignmentProblem(patches)
    embedding = np.empty((emb_problem.n_nodes, emb_problem.dim))
    for node, patch_list in enumerate(emb_problem.patch_index):
        embedding[node] = np.mean([emb_problem.patches[p].get_coordinate(node)@rot[i] + shift[i] for i, p in enumerate(patch_list)], axis=0)

    prob=l2g.AlignmentProblem(patches)
    old_embedding=prob.get_aligned_embedding()
    embedding=embedding[nodes]
    old_embedding=old_embedding[nodes]
    error= l2g.utils.procrustes_error(embedding,old_embedding)

    return embedding, old_embedding, error



def double_intersections_nodes(patches):
    double_intersections=dict()
    for i in range(len(patches)):
        for j in range(i+1, len(patches)):
            double_intersections[(i,j)]=list(set(patches[i].nodes.tolist()).intersection(set(patches[j].nodes.tolist())))
    return double_intersections

def preprocess_patches(list_of_patches, nodes_dict):
    emb_list=[]
    for i in range(len(list_of_patches)-1):
        emb_list.append([torch.tensor(list_of_patches[i].get_coordinates(list(nodes_dict[i,i+1]))),
                         torch.tensor(list_of_patches[i+1].get_coordinates(list(nodes_dict[i,i+1])))])
    emb_list=list(itertools.chain.from_iterable(emb_list))
    return emb_list    



def get_embedding(patches, trained_model):
    n=len(patches)
    rot=[result.transformation[i].weight.detach().numpy() for i in range(n)]
    shift=[result.transformation[i].bias.detach().numpy() for i in range(n)]

    emb_problem = l2g.AlignmentProblem(patches)
    embedding = np.empty((emb_problem.n_nodes, emb_problem.dim))
    for node, patch_list in enumerate(emb_problem.patch_index):
        embedding[node] = np.mean([emb_problem.patches[p].get_coordinate(node)@rot[i] + shift[i] for i, p in enumerate(patch_list)], axis=0)
    
    return embedding



def get_error(new_emb, patches): #, nodes):
    

    prob=l2g.AlignmentProblem(patches)
    old_embedding=prob.get_aligned_embedding()
    
    error= l2g.utils.procrustes_error(new_emb,old_embedding)

    return error
    



def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, dim, n_patches, device):
        super().__init__()
        self.device = device
        self.transformation = nn.ParameterList([nn.Linear(dim, dim).to(device) for _ in range(n_patches)])
        [geotorch.orthogonal(self.transformation[i], 'weight') for i in range(n_patches)]
    
    def forward(self, patch_emb):
        m = len(patch_emb)
        transformations = [self.transformation[0]] + [item for i in range(1, len(self.transformation)-1) for item in (self.transformation[i], self.transformation[i])] + [self.transformation[-1]]
        transformed_emb = [transformations[i](patch_emb[i]) for i in range(m)]
        return transformed_emb

def loss_function(transformed_emb):
    m = len(transformed_emb)
    diff = [transformed_emb[i] - transformed_emb[i+1] for i in range(0, m-1, 2)]
    loss = sum([torch.norm(d) ** 2 for d in diff])
    return loss

def train_model(patch_emb, dim, n_patches, num_epochs=100, learning_rate=0.05):
    device = get_device()
    patch_emb = [p.to(device) for p in patch_emb]
    
    model = Model(dim, n_patches, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_hist = []
    
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        transformed_patch_emb = model(patch_emb)
        loss = loss_function(transformed_patch_emb)
        loss.backward(retain_graph=True)
        optimizer.step()
        loss_hist.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    return model, loss_hist



#res, loss_hist= train_model( preprocess_patches , dim, n_patches , num_epochs=100, learning_rate=0.05)