import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocess import text2ids


class LCQMC(Dataset):
    def __init__(self, dataset_dir,
                 max_seq_len,
                 w2i,
                 charmode = False, sep = '\t', cached_dir = None):
        self.dataset_dir = dataset_dir
        self.train_csv = os.path.join(self.dataset_dir, 'train.txt')
        self.dev_csv = os.path.join(self.dataset_dir, 'dev.txt')
        self.test_csv = os.path.join(self.dataset_dir, 'test.txt')
        self.columns = ['text_1', 'text_2', 'label']
        self.sep = sep
        self.charmode = charmode
        self.w2i = w2i

        if cached_dir is not None:
            self.cathed_dir = cached_dir
        else:
            self.cached_dir = dataset_dir

    def to(self, mode = 'train', with_labels = True):
        self.mode = mode
        self.with_labels = with_labels
        self.cached_ids_dif =  cached_ids_file = os.path.join(self.cached_dir, '{}_ids.pkl'.format_map(self.mode))
        try:
            with open(cached_ids_file, 'rb') as f:
                ids = pickle.load(f)
        except:
            ids = []
            csv_file = getattr(self, '{}_csv'.format(self.mode))
            df = pd.read_csv(csv_file, sep = self.sep, header = None,
                             names = self.columns,
                             na_filter = False)
            for row in df.itertuples():
                text1 = row.text_1
                text2 = row.text_2
                ids1 = text2ids(text1, self.w2i, self.charmode)
                ids2 = text2ids(text2, self.w2i, self.charmode)
                if len(ids1) > self.max_seq_len:
                    ids1 = ids1[:self.max_seq_len]
                    len1 = self.max_seq_len
                else:
                    len1 = len(ids1)
                    ids1 += [0] * (self.max_seq_len - len1)

                if len(ids2) > self.max_seq_len:
                    ids2 = ids2[:self.max_seq_len]
                    len2 = self.max_seq_len
                else:
                    len2 = len(ids2)
                    ids2 += [0] * (self.max_seq_len - len2)

                ids_dict= {'ids1': ids1, 'ids2': ids2, 'len1': len1,
                           'len2':len2}
                if with_labels:
                    ids_dict['label'] = row.label
                ids.append(ids_dict)

            with open(cached_ids_file, 'wb') as f:
                pickle.dump(ids, f)

        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ids_dict = self.ids[idx]
        res = {k: torch.tensor([v], dtype = torch.int32) for k, v in ids_dict.items()}
        if self.with_labels:
            label = ids_dict['label']
            res['label'] = torch.tensor([label], dtype = torch.float32)
        return res
