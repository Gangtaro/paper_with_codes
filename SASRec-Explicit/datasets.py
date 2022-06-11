import random

import torch
from torch.utils.data import Dataset

from utils import neg_sample

class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, rating_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.rating_seq = rating_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]
        ratings = self.rating_seq[index]

        assert self.data_type in {"train", "valid", "test", "submission"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]

        # submission [0, 1, 2, 3, 4, 5, 6]
        # answer None

        if self.data_type == "train":
            input_ids = items[:-3]
            input_ratings = ratings[:-3]
            target_pos = items[1:-2]
            target_ratings = ratings[1:-2]
            answer = [0]  # no use
            ratings_answer = [0]

        elif self.data_type == "valid":
            input_ids = items[:-2]
            input_ratings = ratings[:-2]
            target_pos = items[1:-1]
            target_ratings = ratings[1:-1]
            answer = [items[-2]]
            ratings_answer = [ratings[-2]]

        elif self.data_type == "test":
            input_ids = items[:-1]
            input_ratings = ratings[:-1]
            target_pos = items[1:]
            target_ratings = ratings[1:]
            answer = [items[-1]]
            ratings_answer = [ratings[-1]]
        else:
            input_ids = items[:]
            input_ratings = ratings[:]
            target_pos = items[:]  # will not be used
            target_ratings = ratings[:] 
            answer = []
            ratings_answer = []



        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        input_ratings = [0] * pad_len + input_ratings
        target_pos = [0] * pad_len + target_pos

        target_ratings = [0] * pad_len + target_ratings

        input_ids = input_ids[-self.max_len :]
        input_ratings = input_ratings[-self.max_len :]
        target_pos = target_pos[-self.max_len :]

        target_ratings = target_ratings[-self.max_len :]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_ratings, dtype=torch.float32), ######## new ##########
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor([0], dtype=torch.long), # torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(target_ratings, dtype=torch.float32), ######## new ##########
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(ratings_answer, dtype=torch.float32), ######## new ##########
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(input_ratings, dtype=torch.float32), ######## new ##########
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor([0], dtype=torch.long), # torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(target_ratings, dtype=torch.float32), ######## new ##########
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(ratings_answer, dtype=torch.float32), ######## new ##########
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)
