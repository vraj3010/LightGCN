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


def ndcg_calculation_2(model, test_set, neg_samples,num_users,head_items,k=10,N=None):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_embeddings=model.userEmb.weight
    item_embeddings=model.itemEmb.weight
    total_ndcg = 0
    count = 0

    for user_id, pos_items in test_set.items():

        if not pos_items:
            continue

        h=len(pos_items)
        N2=N
        # print(h,end=" ")
        if N2==None:
            N2=h
        # print(N2,end=" ")
        neg_items = random.sample(neg_samples[user_id], N2)

        # print(len(pos_items),end=" ")
        test_items = [item-num_users for item in pos_items] + neg_items
        # print(len(test_items))
        if len(test_items)<k:
            continue
        test_items = torch.tensor(test_items,dtype=torch.long)
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

def ndcg_calculation_head(model, test_set, neg_samples, num_users, head_items, k=10,N=None):
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
        N2=N
        if N2 is None:
            N2=h
        neg_items = random.sample(neg_samples[user_id], N2)
        # Convert head items to item indices (subtracting num_users)
        test_items = [item - num_users for item in head_pos_items] + neg_items
        if len(test_items)<k:
            continue
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


def ndcg_calculation_tail(model, test_set, neg_samples, num_users, tail_items, k=10,N=None):
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

        h = len(head_pos_items)
        N2=N
        if N2 is None:
            N2=h
        neg_items = random.sample(neg_samples[user_id], N2)
        test_items = [item - num_users for item in head_pos_items] + neg_items
        if len(test_items)<k:
            continue
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

def catalog_coverage_head_tail(model, test_set, num_users,neg_samples, head_items, tail_items, k=10, device="cpu"):
    user_embeddings = model.userEmb.weight
    item_embeddings = model.itemEmb.weight
    num_items = item_embeddings.shape[0]

    recommended_items = set()
    recommended_head = set()
    recommended_tail = set()

    for user_id in range(num_users):
        # get user embedding
        user_emb = user_embeddings[user_id]
        # compute scores for all items
        neg_items=torch.tensor(neg_samples[user_id],dtype=torch.long,device=device)
        item_emb=item_embeddings[neg_items]
        scores = torch.matmul(item_emb, user_emb)
        # get top-k items
        topk_indices = torch.topk(scores, k).indices.tolist()
        # store recommended items
        for idx in topk_indices:
            recommended_items.add(idx)
            if idx in head_items:
                recommended_head.add(idx)
            if idx in tail_items:
                recommended_tail.add(idx)

    overall_coverage = len(recommended_items) / num_items
    head_coverage = len(recommended_head) / len(head_items) if len(head_items) > 0 else 0
    tail_coverage = len(recommended_tail) / len(tail_items) if len(tail_items) > 0 else 0

    print(f"Catalog Coverage@{k} (Overall): {overall_coverage:.4f}")
    print(f"Catalog Coverage@{k} (Head): {head_coverage:.4f}")
    print(f"Catalog Coverage@{k} (Tail): {tail_coverage:.4f}")

    return overall_coverage, head_coverage, tail_coverage