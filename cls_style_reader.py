import torch
import numpy as np
import pickle
import ipdb

root = '/workspace/ssd0/byeongcheol/DGSS/FAMix/cls_style.pkl'
with open(root, 'rb') as f:
    a = pickle.load(f)

print(a.keys())


ipdb.set_trace()