import torch
import numpy as np
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import seaborn as sns

import ipdb

if __name__ == "__main__":
    
    param_root = "/workspace/ssd0/byeongcheol/DGSS/FAMix/save_dir/gta5_6_saved_params2.pkl"
    with open(param_root, "rb") as f:
        params = pickle.load(f)
    
    ipdb.set_trace()

    label_match = {'road' : 0, 'sidewalk' : 1, 'building' : 2, 'wall' : 3, 'fence' : 4, 'pole' : 5, 'traffic light' : 6, 'traffic sign' : 7, 'vegetation' : 8, 'terrain' : 9, 'sky' : 10, 'person' : 11, 'rider' : 12, 'car' : 13, 'truck' : 14, 'bus' : 15, 'train' : 16, 'motorcycle' : 17, 'bicycle' : 18, '255' : 19}

    param = []
    labels = []

    for key in params.keys():
        if "mu" in key:
            mu_params = torch.from_numpy(np.array([t.numpy() for t in params[key]]))
            std_params = torch.from_numpy(np.array([t.numpy() for t in params[key.replace("mu", "std")]]))
            params_ = mu_params.squeeze(-1).squeeze(-1)
            params_ = torch.cat((mu_params, std_params), dim=1).squeeze(-1).squeeze(-1)
            param_ = params_.cpu().numpy()
            param.append(param_)
            labels.append(np.array([label_match[key.split("_")[0]]] * len(param_)))

    param = np.concatenate(param, axis=0)
    labels = np.concatenate(labels, axis=0)

    # ipdb.set_trace()

    # concatenate

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#1a9850', '#d73027', '#4575b4', '#313695', '#91bfdb',
        '#fc8d59', '#fee08b', '#d73027', '#4575b4', '#313695'
    ]
    colors = colors[:20]
    cmap = ListedColormap(colors)


    # t-SNE 적용
    tsne_model = TSNE(n_components=2, perplexity=40, n_iter=300, init='random', learning_rate=200.0)
    tsne_results = tsne_model.fit_transform(param)

    # label array
    label = np.arange(20)

    # 결과 시각화
    plt.figure(figsize=(10, 6))
    for i in range(20):
        indices = labels == i
        scatter = plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=i)

    plt.colorbar(scatter, ticks=np.arange(20), label='Class') 
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'{label}') for label, color in zip(label, colors)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.20, 1), loc='upper left', title='Classes')
    plt.savefig('/workspace/ssd0/byeongcheol/DGSS/FAMix/save_dir/TSNE_result.png')