import re
import jieba
import numpy as np
from tqdm import tqdm
from snownlp import SnowNLP

def normalize(text):
    ## 去除特殊符号
    pattern = r"([^\u4e00-\u9fa5^a-z^A-Z^0-9^\s]|\^)" # 非常见中中英文数字字符, 保留空白字符, 防止相邻的英文单词合并到一起
    ptext = re.sub(pattern, '', text)
    # ptext = re.sub(r'\^', '', ptext)

    ## 中文繁体转简体
    if len(ptext) > 0: # 空字符串会报错
        snow = SnowNLP(ptext)
        ptext = snow.han

    ## 英文统一为小写
    ptext = ptext.lower()
    return ptext


def to_tokens(text, charmode = False):
    if charmode:
        return [c for c in text if c.strip() != '']
    else:
        return jieba.lcut(text)

def to_ids(tokens, w2i):
    # pad = 0
    # unk = 1
    ids = [w2i.get(tk, 1) for tk in tokens]
    return ids

def text2ids(text, w2i, charmode = False):
    ptext = normalize(text)
    tokens = to_tokens(ptext, charmode)
    ids = to_ids(tokens, w2i)
    return ids


## 加载百度百科词向量 (文本文件)
def load_word_vec(path):
    # pad:0
    # unk:1
    lines_num, dim = 0, 0
    vectors = []
    iw = ['<pad>', '<unk>']
    wi = {}

    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in tqdm(f):
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                vectors = [np.zeros((2, dim), dtype = np.float32)]
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vector = np.asarray([float(x) for x in tokens[1:]])
            vectors.append(vector)
            iw.append(tokens[0].strip())

    for i, w in enumerate(iw):
        wi[w] = i
    emb = np.vstack(vectors)
    return iw, wi, dim, emb

