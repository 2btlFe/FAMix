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
    file = "gta5_4_saved_params_score.pkl"
    tsne_dir = "tsne_result"
    os.makedirs(tsne_dir, exist_ok=True)
    with open(file, "rb") as f:
        data = pickle.load(f)
    
    for cls_idx, cls in enumerate(class_idx):
        style_emb = []
        style_txt = []

        mu = cls + "_mu"
        std = cls + "_std"

        style_mu_list = data[mu]
        style_std_list = data[std]    

        style_len = len(style_mu_list)

        for idx in range(style_len):            
            # mu, meta = style_mu_list[idx]
            std, meta = style_std_list[idx]
            # ipdb.set_trace()
            
            # feat = torch.cat((mu, std), dim=0).squeeze(-1).squeeze(-1)
            feat = std.squeeze(-1).squeeze(-1)
            feat = feat.cpu().numpy()

            style_emb.append(feat)
            style_txt.append(meta['style'])

        # ipdb.set_trace()
        style_emb = np.array(style_emb)
        style_txt = np.array(style_txt)

        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#1a9850', '#d73027', '#4575b4', '#313695', '#91bfdb',
            '#fc8d59', '#fee08b', '#fdae61', '#a6cee3', '#b2df8a'
        ]
        colors = colors[:len(style_idx)]
        cmap = ListedColormap(colors)

        # t-SNE 적용
        tsne_model = TSNE(n_components=2, perplexity=40, n_iter=300, init='random', learning_rate=200.0)
        tsne_results = tsne_model.fit_transform(style_emb)

        plt.figure(figsize=(10, 10))
        for i in range(len(style_idx)):
            scatter = plt.scatter(tsne_results[style_txt == style_idx[i], 0], tsne_results[style_txt == style_idx[i], 1], c=[colors[i]], label=style_idx[i])

        plt.colorbar(scatter, ticks=np.arange(len(style_idx)), label='Style')
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'{style}') for style, color in zip(style_idx, colors)]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.20, 1), loc='upper left', title='Classes')
        cls_path = os.path.join(tsne_dir, f'{cls_idx}_{cls}_std.png')
        plt.savefig(cls_path)