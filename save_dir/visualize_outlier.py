import torch
import numpy as np
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import seaborn as sns
import argparse
import ipdb
import os

class_idx = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle', '255'
]

class_idx = [
    'car'
]


style_idx = ['Ethereal Mist style',
'Cyberpunk Cityscape style',
'Rustic Charm style',
'Galactic Fantasy style',
'Pastel Dreams style',
'Dystopian Noir style',
'Whimsical Wonderland style',
'Urban Grit style',
'Enchanted Forest style',
'Retro Futurism style',
'Monochrome Elegance style',
'Vibrant Graffiti style',
'Haunting Shadows style',
'Steampunk Adventures style',
'Watercolor Serenity style',
'Industrial Chic style',
'Cosmic Voyage style',
'Pop Art Popularity style',
'Abstract Symphony style',
'Magical Realism style',
]

if __name__ == "__main__":
    file = "gta5_3_saved_params_score.pkl"
    tsne_dir = "tsne_result_3"
    os.makedirs(tsne_dir, exist_ok=True)
    with open(file, "rb") as f:
        data = pickle.load(f)
    
    for cls_idx, cls in enumerate(class_idx):
        style_emb = []
        style_txt = []
        img_paths = []

        mu = cls + "_mu"
        std = cls + "_std"

        style_mu_list = data[mu]
        style_std_list = data[std]    

        style_len = len(style_mu_list)

        for idx in range(style_len):            
            mu, meta = style_mu_list[idx]
            std, meta = style_std_list[idx]
            # ipdb.set_trace()
            
            feat = torch.cat((mu, std), dim=0).squeeze(-1).squeeze(-1)
            # feat = std.squeeze(-1).squeeze(-1)
            feat = feat.cpu().numpy()

            style_emb.append(feat)
            style_txt.append(meta['style'])
            # ipdb.set_trace()
            img_paths.append(meta['img_path'])

        # ipdb.set_trace()
        style_emb = np.array(style_emb)
        style_txt = np.array(style_txt)
        img_paths = np.array(img_paths)

        # 평균으로부터의 거리 계산
        distances = np.linalg.norm(style_emb - np.mean(style_emb, axis=0), axis=1)

        # outlier 기준 (평균 거리의 2배 이상 떨어진 점들을 outlier로 간주)
        threshold = np.mean(distances) + 3 * np.std(distances)
        outliers = np.where(distances > threshold)[0]

        print("Outliers:", outliers)

        for outlier in outliers:
            print(f"Outlier: {img_paths[outlier]}")

        ipdb.set_trace() 