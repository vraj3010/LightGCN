import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from train_test import train_test_split_per_user
from tqdm import tqdm
# import matplotlib.pyplot as plt
from model import LightGCN
from utils import *
import torch
from collections import defaultdict
import random

ratings = pd.read_csv('ml-latest-small/ratings.csv')

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

# print(ratings['userId'])
ratings['userId']  = user_encoder.fit_transform(ratings['userId'])
ratings['movieId'] = movie_encoder.fit_transform(ratings['movieId'])

neg_samples=get_negative_items(ratings)

num_users  = ratings['userId'].nunique()
num_movies = ratings['movieId'].nunique()
# print(num_users)
# print(num_movies)

interaction_counts = ratings['movieId'].value_counts().to_dict()

# Separate Head and Tail Items
head_items, tail_items = separate_head_tail_items(interaction_counts, head_threshold=15)
head_items = torch.tensor([i + num_users for i in head_items], dtype=torch.long)
tail_items = torch.tensor([i + num_users for i in tail_items], dtype=torch.long)
print(len(head_items))
print(len(tail_items))
int_edges = create_interaction_edges(ratings['userId'], ratings['movieId'], ratings['rating'])
user_ids = int_edges[0].to(dtype=torch.long)
indices = torch.arange(0, int_edges.shape[1], dtype=torch.long)
train_idx, test_idx = train_test_split_per_user(indices,user_ids, test_size=0.2)
train_edges = int_edges[:, train_idx]
test_edges  = int_edges[:, test_idx]

train_adj = create_adj_matrix(train_edges, num_users, num_movies)
test_adj  = create_adj_matrix(test_edges, num_users, num_movies)

# print(train_adj.shape)
train_r = adj_to_r_mat(train_adj, num_users, num_movies)
test_r  = adj_to_r_mat(test_adj, num_users, num_movies)

train_set = set(zip(train_r[0].tolist(), train_r[1].tolist()))
# test_set = set(zip(test_r[0].tolist(), test_r[1].tolist()))
# print(len(train_set & test_set) == 0,"**")

''' ------------ Training Loop ------------ '''

NUM_ITER   = 1000
BATCH_SIZE = 512

model = LightGCN(num_users, num_movies)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print the number of trainable parameters
print(f"Number of trainable parameters: {total_params}")

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

int_edges[1]+=num_users
test_edges[1]+=num_users
test_set=create_test_set(test_edges)
ndcg_calculation_2(model, test_set, neg_samples, num_users,int_edges,head_items,k=10)
ndcg_calculation_head(model, test_set, neg_samples, num_users,int_edges,head_items,k=10)
ndcg_calculation_tail(model, test_set, neg_samples, num_users,int_edges,tail_items,k=2)
ndcg_calculation_2(model, test_set, neg_samples, num_users,int_edges,head_items,k=10,N=80)
ndcg_calculation_head(model, test_set, neg_samples, num_users,int_edges,head_items,k=10,N=80)
ndcg_calculation_tail(model, test_set, neg_samples, num_users,int_edges,tail_items,k=10,N=80)

iterator = tqdm(range(NUM_ITER))

row,col=train_adj[0],train_adj[1]
# print(max(row),min(row))
# print(max(col),min(col))
for i in iterator:
    model.train()
    optimizer.zero_grad()

    user_emb, user_emb_0, item_emb, item_emb_0 = model(train_adj)
    loss = bpr_loss(
        user_emb,
        user_emb_0,
        item_emb,
        item_emb_0,
        train_r,
        BATCH_SIZE
        )
    #Computes gradient via backpropagation
    loss.backward()
    # Updates model parameters
    optimizer.step()

    if i % 100 == 0 and i != 0:
        scheduler.step()
        print(loss.item())
        ndcg_calculation_2(model, test_set, neg_samples, num_users, int_edges, head_items, k=10)
        ndcg_calculation_head(model, test_set, neg_samples, num_users, int_edges, head_items, k=10)
        ndcg_calculation_tail(model, test_set, neg_samples, num_users, int_edges, tail_items, k=2)
        ndcg_calculation_2(model, test_set, neg_samples, num_users, int_edges, head_items, k=10, N=80)
        ndcg_calculation_head(model, test_set, neg_samples, num_users, int_edges, head_items, k=10, N=80)
        ndcg_calculation_tail(model, test_set, neg_samples, num_users, int_edges, tail_items, k=10, N=80)
        catalog_coverage_head_tail(model,test_set,num_users,neg_samples,head_items,tail_items)

ndcg_calculation_2(model, test_set, neg_samples, num_users,int_edges,head_items,k=10)
ndcg_calculation_head(model, test_set, neg_samples, num_users,int_edges,head_items,k=10)
ndcg_calculation_tail(model, test_set, neg_samples, num_users,int_edges,tail_items,k=2)
ndcg_calculation_2(model, test_set, neg_samples, num_users,int_edges,head_items,k=10,N=80)
ndcg_calculation_head(model, test_set, neg_samples, num_users,int_edges,head_items,k=10,N=80)
ndcg_calculation_tail(model, test_set, neg_samples, num_users,int_edges,tail_items,k=10,N=80)