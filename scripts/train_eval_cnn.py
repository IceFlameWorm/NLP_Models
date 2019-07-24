import os
import sys

FILE_PATH = os.path.dirname(__file__)
PROJECT_PATH = os.path.join(FILE_PATH, '..')
sys.path.append(PROJECT_PATH)

import os
import random
import numpy as np
import pickle
import json
import torch
from copy import deepcopy
from pytorch_nlp_models.text_pair.siamese_cnn import SiameseCNN
from utils.datasets import LCQMCDataset
from utils.model_utils import model_train, model_eval

from torch.utils.data import DataLoader


random.seed(5555)
np.random.seed(6666)
torch.manual_seed(7777)
torch.cuda.manual_seed_all(8888)
torch.backends.cudnn.deterministic = True


DATA_PATH = os.path.join(PROJECT_PATH, 'data')
LCQMC_PATH = os.path.join(DATA_PATH, 'LCQMC')
WORD_VECTORS_PATH = os.path.join(DATA_PATH, 'word_vectors')
BAIDUBAIKE_PKL = os.path.join(WORD_VECTORS_PATH, 'baidubaike.pkl')

MAX_SEQ_LEN = 40

MODEL_PATH = os.path.join(DATA_PATH, 'model_files/siamese_cnn')
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

MODEL_FILE = os.path.join(MODEL_PATH, 'model.pkl')
RESULTS_TXT = os.path.join(MODEL_PATH, 'results.txt')
MODEL_CONFIG_JSON = os.path.join(MODEL_PATH, 'config.json')
TRAIN_CONFIG_JSON = os.path.join(MODEL_PATH, 'train_config.json')


if os.path.exists(MODEL_FILE) or os.path.exists(RESULTS_TXT):
    print('{} is not empty, exit'.format(MODEL_FILE))
    sys.exit(-1)


with open(BAIDUBAIKE_PKL, 'rb') as f:
    wvs = pickle.load(f)


wi = wvs['wi']
iw = wvs['iw']
dim = wvs['dim']
emb = wvs['emb']

dataset = LCQMCDataset(LCQMC_PATH, MAX_SEQ_LEN, wi, charmode = True)

def get_loader(dataset, mode,
               batch_size = 32,
               shuffle = False):
    _dataset = deepcopy(dataset)
    _dataset.to(mode)
    return DataLoader(_dataset, batch_size=batch_size, shuffle=shuffle)


train_loader = get_loader(dataset, 'train', shuffle= True)

#train_loader = get_loader(dataset, 'test', shuffle= True)
dev_loader = get_loader(dataset, 'dev')

MODEL_CONFIG = {'vocab_size': len(iw),
                'emb_dim': dim,
                'hidden_dim': dim,
                'dropout': 0.5,
               }

with open(MODEL_CONFIG_JSON, 'w') as f:
    json.dump(MODEL_CONFIG, f)

TRAIN_CONFIG = {'LR': 1e-4}

with open(TRAIN_CONFIG_JSON, 'w') as f:
    json.dump(TRAIN_CONFIG, f)


model = SiameseCNN( emb_weights=  torch.tensor(emb, dtype = torch.float32),
                   **MODEL_CONFIG,
                  )
his_train_loss, his_test_loss = model_train(model, train_loader, dev_loader, MODEL_FILE, **TRAIN_CONFIG)

checkpoint = torch.load(MODEL_FILE)
model.load_state_dict(checkpoint['model_state_dict'])
data_ks, data_auc, data_probs, data_gts = model_eval(model, dev_loader)

res = {}
res['train_loss'] = his_train_loss[-1]
res['dev_loss'] = his_test_loss[-1]
res['dev_ks'] = data_ks
res['dev_auc'] = data_auc
print(res)
with open(RESULTS_TXT, 'w') as f:
    json.dump(res, f)
