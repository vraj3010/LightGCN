import numpy as np
import random

import torch
# print(torch.__version__)
# from torch_geometric.utils.negative_sampling import structured_negative_sampling
from torch_geometric.utils import structured_negative_sampling


def create_interaction_edges(userids, movieids, ratings_, threshold=3.5):
    ''' Interaction edges in COO format.'''
    mask = ratings_ > threshold
    edges = np.stack([userids[mask], movieids[mask]])
    return torch.LongTensor(edges)


def create_adj_matrix(int_edges, num_users, num_movies):

    n = num_users + num_movies
    adj = torch.zeros(n,n)

    r_mat = torch.sparse_coo_tensor(int_edges, torch.ones(int_edges.shape[1]), size=(num_users, num_movies)).to_dense()
    adj[:num_users, num_users:] = r_mat.clone()
    adj[num_users:, :num_users] = r_mat.T.clone()

    adj_coo = adj.to_sparse_coo()
    adj_coo = adj_coo.indices()

    return adj_coo

def adj_to_r_mat(adj, num_users, num_movies):
    n = num_users + num_movies
    adj_dense = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1]), size=(n, n)).to_dense()
    r_mat = adj_dense[:num_users, num_users:]
    r_coo = r_mat.to_sparse_coo()
    return r_coo.indices()


def r2r_mat(r, num_users, num_movies):
    r_mat = torch.zeros(num_users, num_movies)
    r_mat[r[0], r[1]] = 1
    return r_mat

## edges = (i, j, k) (i, j) positive edge (i, k) negative edge
def sample_mini_batch(edge_index, batch_size):

    '''
    Args:
        edge_index: edge_index of the user-item interaction matrix

    Return:
        structured_negative_sampling return (i,j,k) where
            (i,j) positive edge
            (i,k) negative edge
    '''
    edges = structured_negative_sampling(edge_index.detach())
    edges = torch.stack(edges, dim=0)
    random_idx = random.choices(list(range(edges[0].shape[0])), k=batch_size)
    batch = edges[:, random_idx]
    user_ids, pos_items, neg_items = batch[0], batch[1], batch[2]
    return user_ids, pos_items, neg_items


def bpr_loss(user_emb, user_emb_0, item_emb, item_emb_0, edge_index_r, batch_size = 128, lambda_= 1e-6):

    user_ids, pos_items, neg_items = sample_mini_batch(edge_index_r, batch_size=batch_size)

    user_emb_sub = user_emb[user_ids]
    pos_item_emb = item_emb[pos_items]
    neg_item_emb = item_emb[neg_items]

    pos_scores = torch.diag(user_emb_sub @ pos_item_emb.T)
    neg_scores = torch.diag(user_emb_sub @ neg_item_emb.T)


    reg_loss = lambda_*(
        user_emb_0[user_ids].norm(2).pow(2) +
        item_emb_0[pos_items].norm(2).pow(2) +
        item_emb_0[neg_items].norm(2).pow(2) # L2 loss
    )

    loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss
    return loss


def NDCG_K(model, train_r, test_r, K=20):

    # Calculate the ratings for each user-item pair
    userEmbeddings = model.userEmb.weight
    itemEmbeddings = model.itemEmb.weight
    ratings = userEmbeddings @ itemEmbeddings.T

    # Set the ratings that are inside the training set to very negative number to ignore them
    ratings[train_r[0], train_r[1]] = -1e12

    # For each user get the item ids(indices) that user positively interacted
    interaction_mat_test = r2r_mat(test_r, model.num_users, model.num_items) # shape: (num_users, num_items)
    pos_items_each_user_test = [row.nonzero().squeeze(1) for row in interaction_mat_test]

    # Get top K recommended items (not their ratings but item ids) by the model, ratings are sorted in descending order
    _, topk_items_idxs_pred = torch.topk(ratings, k=K)
    # print(topk_items_idxs_pred.shape)

    # Turn those recommendation ids into binary, by preserving their recommended positions.
    rec_pred_binary = torch.zeros_like(topk_items_idxs_pred)
    # print(rec_pred_binary.shape)

    for i in range(topk_items_idxs_pred.shape[0]):
        for j in range(topk_items_idxs_pred.shape[1]):
            if topk_items_idxs_pred[i,j] in pos_items_each_user_test[i]:
                # if the recommended item is in the list that user is positively interacted
                # meaning the recommendation is good
                rec_pred_binary[i,j] = 1

    # Turn positive items for each user into binary 2D array.
    rec_gt_binary = torch.zeros_like(rec_pred_binary)
    for i in range(rec_gt_binary.shape[0]):
        l = min(len(pos_items_each_user_test[i]), K)
        rec_gt_binary[i, :l] = 1

    # Now calculate the NDGC
    idcg = (rec_gt_binary / torch.log2(torch.arange(K).float() + 2)).sum(dim=1)
    dcg = (rec_pred_binary / torch.log2(torch.arange(K).float() + 2)).sum(dim=1)

    ndcgs = dcg / idcg
    ndcg = ndcgs[~torch.isnan(ndcgs)]

    return ndcg.mean().item()




def ndcg_calculation_2(model, test_set, neg_samples,num_users,int_edges,head_items,k=5):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_embeddings = model.userEmb.weight
    item_embeddings = model.itemEmb.weight
    total_ndcg = 0
    count = 0

    for user_id, pos_items in test_set.items():

        if not pos_items:
            continue

        h=len(pos_items)

        neg_items = random.sample(neg_samples[user_id], h)

        if len(neg_items)*2<k:
            continue
        test_items = [item-num_users for item in pos_items] + neg_items
        test_items = torch.tensor(test_items, dtype=torch.long)
        user_emb = user_embeddings[user_id]
        item_embs = item_embeddings[test_items]
        scores = torch.matmul(item_embs, user_emb)

        sorted_indices = torch.argsort(scores, descending=True)
        sorted_items = [test_items[i] for i in sorted_indices.tolist()]
        dcg = 0
        for i, item in enumerate(sorted_items[:k]):

            if item+num_users in pos_items:
                dcg += 1 / np.log2(i + 2)

        idcg = sum(1/ np.log2(i + 2) for i in range(min(len(pos_items),k)))

        ndcg = dcg / idcg if idcg > 0 else 0

        total_ndcg += ndcg
        count += 1

    avg_ndcg= total_ndcg / count if count > 0 else 0
    print(f"NDCG@10: {avg_ndcg}")


def ndcg_calculation_head(model, test_set, neg_samples, num_users, int_edges, head_items, k=5):

    user_embeddings = model.userEmb.weight
    item_embeddings = model.itemEmb.weight
    total_ndcg = 0
    count = 0

    for user_id, pos_items in test_set.items():
        if not pos_items:
            continue

        # Filter positive items to only include head items
        head_pos_items = [item for item in pos_items if item in head_items]
        if not head_pos_items:
            continue

        h = len(head_pos_items)
        neg_items = random.sample(neg_samples[user_id], h)

        if len(neg_items) * 2 < k:
            continue

        # Convert head items to item indices (subtracting num_users)
        test_items = [item - num_users for item in head_pos_items] + neg_items
        test_items = torch.tensor(test_items, dtype=torch.long)
        user_emb = user_embeddings[user_id]
        item_embs = item_embeddings[test_items]
        scores = torch.matmul(item_embs, user_emb)

        sorted_indices = torch.argsort(scores, descending=True)
        sorted_items = [test_items[i] for i in sorted_indices.tolist()]

        dcg = 0
        for i, item in enumerate(sorted_items[:k]):
            # Check if the item is in the head positive items (convert back to original ID)
            if item + num_users in head_pos_items:
                dcg += 1 / np.log2(i + 2)

        idcg = sum(1 / np.log2(i + 2) for i in range(min(k,len(head_pos_items))))
        ndcg = dcg / idcg if idcg > 0 else 0

        # print(ndcg,user_id)
        total_ndcg += ndcg
        count += 1

    avg_ndcg = total_ndcg / count if count > 0 else 0
    print(f"NDCG@{k} for head items: {avg_ndcg},{count}")
    return avg_ndcg


def ndcg_calculation_tail(model, test_set, neg_samples, num_users, int_edges, tail_items, k=5):

    user_embeddings = model.userEmb.weight
    item_embeddings = model.itemEmb.weight
    total_ndcg = 0
    count = 0

    for user_id, pos_items in test_set.items():
        if not pos_items:
            continue

        # Filter positive items to only include head items
        head_pos_items = [item for item in pos_items if item in tail_items]
        if not head_pos_items:
            continue
        # print(len(head_pos_items),end=" ")
        if len(head_pos_items)*2 < k:
            continue
        h = len(head_pos_items)
        neg_items = random.sample(neg_samples[user_id], h)
        # if user_id==3:
        #     print(neg_items)
        # Convert head items to item indices (subtracting num_users)
        test_items = [item - num_users for item in head_pos_items] + neg_items
        test_items = torch.tensor(test_items, dtype=torch.long)
        user_emb = user_embeddings[user_id]
        item_embs = item_embeddings[test_items]
        scores = torch.matmul(item_embs, user_emb)

        sorted_indices = torch.argsort(scores, descending=True)
        sorted_items = [test_items[i] for i in sorted_indices.tolist()]

        dcg = 0
        for i, item in enumerate(sorted_items[:k]):

            if item + num_users in head_pos_items:
                dcg += 1 / np.log2(i + 2)

        idcg = sum(1 / np.log2(i + 2) for i in range(min(k,len(head_pos_items))))

        ndcg = dcg / idcg if idcg > 0 else 0

        total_ndcg += ndcg
        count += 1

    avg_ndcg = total_ndcg / count if count > 0 else 0

    print(f"NDCG@{k} for tail items: {avg_ndcg}")
    return avg_ndcg

def separate_head_tail_items(interaction_counts, head_threshold=50):
    head_items = [item for item, count in interaction_counts.items() if count >= head_threshold]
    tail_items = [item for item, count in interaction_counts.items() if count < head_threshold]

    return torch.tensor(head_items, dtype=torch.long), \
        torch.tensor(tail_items, dtype=torch.long)

def get_negative_items(ratings):
    all_items = set(ratings['movieId'].unique())  # All available items
    # print(len(all_items))
    user_interactions = ratings.groupby('userId')['movieId'].apply(set)  # Get interacted items per user

    negative_samples = {}  # Store negative samples for each user

    for user, interacted_items in user_interactions.items():
        negative_samples[user] = list(all_items - interacted_items)  # Items user has NOT interacted with

    return negative_samples  # Dictionary {userId: [negative_item1, negative_item2, ...]}

def create_test_set(test_edges):
    test_set = {}

    users = test_edges[0].tolist()  # Convert user tensor to list
    items = test_edges[1].tolist()  # Convert item tensor to list

    for user, item in zip(users, items):
        if user not in test_set:
            test_set[user] = []
        test_set[user].append(item)  # Store as tuple (item, rating)

    min_items = min(len(items) for items in test_set.values()) if test_set else 0
    # print(f"Minimum number of items rated by any user: {min_items}")
    # print(len(test_set))
    return test_set

def bpr_loss(user_emb, user_emb_0, item_emb, item_emb_0, edge_index_r, batch_size = 128, lambda_= 1e-6):

    user_ids, pos_items, neg_items = sample_mini_batch(edge_index_r, batch_size=batch_size)

    user_emb_sub = user_emb[user_ids]
    pos_item_emb = item_emb[pos_items]
    neg_item_emb = item_emb[neg_items]

    pos_scores = torch.diag(user_emb_sub @ pos_item_emb.T)
    neg_scores = torch.diag(user_emb_sub @ neg_item_emb.T)


    reg_loss = lambda_*(
        user_emb_0[user_ids].norm(2).pow(2) +
        item_emb_0[pos_items].norm(2).pow(2) +
        item_emb_0[neg_items].norm(2).pow(2) # L2 loss
    )

    loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss
    return loss


def ndcg_calculation_3(model, test_set, neg_samples,num_users,int_edges,head_items,k=5):

    user_embeddings=model.userEmb.weight
    item_embeddings=model.itemEmb.weight
    total_ndcg = 0
    count = 0

    for user_id, pos_items in test_set.items():

        if not pos_items:
            continue

        neg_items = neg_samples[user_id]
        test_items = [item-num_users for item in pos_items] + neg_items
        test_items=torch.tensor(test_items,dtype=torch.long)
        user_emb = user_embeddings[user_id]
        item_embs = item_embeddings[test_items]
        scores = torch.matmul(item_embs, user_emb)
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_items = [test_items[i] for i in sorted_indices.tolist()]
        dcg = 0
        for i, item in enumerate(sorted_items[:k]):

            if item+num_users in pos_items:
                dcg += 1 / np.log2(i + 2)

        idcg = sum(1/ np.log2(i + 2) for i in range(min(len(pos_items),k)))

        ndcg = dcg / idcg if idcg > 0 else 0
        total_ndcg += ndcg
        count += 1

    avg_ndcg= total_ndcg / count if count > 0 else 0
    print(f"NDCG@10: {avg_ndcg}")


def ndcg_calculation_head_2(model, test_set, neg_samples, num_users, int_edges, head_items, k=5):

    user_embeddings = model.userEmb.weight
    item_embeddings = model.itemEmb.weight
    total_ndcg = 0
    count = 0

    for user_id, pos_items in test_set.items():
        if not pos_items:
            continue

        # Filter positive items to only include head items
        head_pos_items = [item for item in pos_items if item in head_items]
        if not head_pos_items:
            continue


        neg_items = neg_samples[user_id]
        # Convert head items to item indices (subtracting num_users)
        test_items = [item - num_users for item in head_pos_items] + neg_items
        test_items = torch.tensor(test_items, dtype=torch.long)
        user_emb = user_embeddings[user_id]
        item_embs = item_embeddings[test_items]
        scores = torch.matmul(item_embs, user_emb)

        sorted_indices = torch.argsort(scores, descending=True)
        sorted_items = [test_items[i] for i in sorted_indices.tolist()]

        dcg = 0
        for i, item in enumerate(sorted_items[:k]):
            # Check if the item is in the head positive items (convert back to original ID)
            if item + num_users in head_pos_items:
                dcg += 1 / np.log2(i + 2)

        idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(head_pos_items))))
        ndcg = dcg / idcg if idcg > 0 else 0

        # print(ndcg,user_id)
        total_ndcg += ndcg
        count += 1

    avg_ndcg = total_ndcg / count if count > 0 else 0
    print(f"NDCG@{k} for head items: {avg_ndcg},{count}")
    return avg_ndcg


def ndcg_calculation_tail_2(model, test_set, neg_samples, num_users, int_edges, tail_items, k=5):
    user_embeddings = model.userEmb.weight
    item_embeddings = model.itemEmb.weight
    total_ndcg = 0
    count = 0

    for user_id, pos_items in test_set.items():
        if not pos_items:
            continue

        # Filter positive items to only include head items
        head_pos_items = [item for item in pos_items if item in tail_items]
        if not head_pos_items:
            continue

        # h = len(head_pos_items)
        neg_items = neg_samples[user_id]
        # if user_id==3:
        #     print(neg_items)
        # Convert head items to item indices (subtracting num_users)
        test_items = [item - num_users for item in head_pos_items] + neg_items
        test_items = torch.tensor(test_items, dtype=torch.long)
        user_emb = user_embeddings[user_id]
        item_embs = item_embeddings[test_items]
        scores = torch.matmul(item_embs, user_emb)

        sorted_indices = torch.argsort(scores, descending=True)
        sorted_items = [test_items[i] for i in sorted_indices.tolist()]

        dcg = 0
        for i, item in enumerate(sorted_items[:k]):

            if item + num_users in head_pos_items:
                dcg += 1 / np.log2(i + 2)

        idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(head_pos_items))))

        ndcg = dcg / idcg if idcg > 0 else 0

        total_ndcg += ndcg
        count += 1

    avg_ndcg = total_ndcg / count if count > 0 else 0

    print(f"NDCG@{k} for tail items: {avg_ndcg}")
    return avg_ndcg