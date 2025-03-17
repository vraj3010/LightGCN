# import random
import torch
from collections import defaultdict
import random


def train_test_split_per_user(indices, user_ids, test_size=0.2, seed=None):

    # if seed is not None:
    #     random.seed(seed)

    user_to_indices = defaultdict(list)
    # print(user_to_indices)

    for idx, user_id in zip(indices, user_ids):
        user_to_indices[user_id.item()].append(idx.item())

    train_indices = []
    test_indices = []

    # Split indices for each user
    for user_id, user_indices in user_to_indices.items():
        random.shuffle(user_indices)
        split_point = int(len(user_indices) * (1 - test_size))
        train_indices.extend(user_indices[:split_point])
        test_indices.extend(user_indices[split_point:])

    # Convert back to torch.Tensor
    train_idx = torch.tensor(train_indices, dtype=torch.long)
    test_idx = torch.tensor(test_indices, dtype=torch.long)

    return train_idx, test_idx
