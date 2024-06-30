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

if __name__ == "__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--className", type=str, default="road")
    args = args.parse_args()

    param_root3 = "/workspace/ssd0/byeongcheol/DGSS/FAMix/save_dir/gta5_3_saved_params2.pkl"
    param_root4 = "/workspace/ssd0/byeongcheol/DGSS/FAMix/save_dir/gta5_4_saved_params2.pkl"
    param_root6 = "/workspace/ssd0/byeongcheol/DGSS/FAMix/save_dir/gta5_6_saved_params2.pkl"

    with open(param_root3, "rb") as f:
        params3 = pickle.load(f)
    with open(param_root4, "rb") as f:
        params4 = pickle.load(f)
    with open(param_root6, "rb") as f:
        params6 = pickle.load(f)
    
    # ipdb.set_trace()

    param = []
    labels = []

    chk_cls = args.className

    mu_params = torch.from_numpy(np.array([t.numpy() for t in params3[f"{chk_cls}_mu"]]))
    std_params = torch.from_numpy(np.array([t.numpy() for t in params3[f"{chk_cls}_std"]]))
    params_ = mu_params.squeeze(-1).squeeze(-1)
    params_3 = torch.cat((mu_params, std_params), dim=1).squeeze(-1).squeeze(-1)
    param_3 = params_3.cpu().numpy()
    param.append(param_3)
    labels.append(np.array([0] * len(param_3)))

    mu_params = torch.from_numpy(np.array([t.numpy() for t in params4[f"{chk_cls}_mu"]]))
    std_params = torch.from_numpy(np.array([t.numpy() for t in params4[f"{chk_cls}_std"]]))
    params_ = mu_params.squeeze(-1).squeeze(-1)
    params_4 = torch.cat((mu_params, std_params), dim=1).squeeze(-1).squeeze(-1)
    param_4 = params_4.cpu().numpy()
    param.append(param_4)
    labels.append(np.array([1] * len(param_4)))

    mu_params = torch.from_numpy(np.array([t.numpy() for t in params6[f"{chk_cls}_mu"]]))
    std_params = torch.from_numpy(np.array([t.numpy() for t in params6[f"{chk_cls}_std"]]))
    params_ = mu_params.squeeze(-1).squeeze(-1)
    params_6 = torch.cat((mu_params, std_params), dim=1).squeeze(-1).squeeze(-1)
    param_6 = params_6.cpu().numpy()
    param.append(param_6)
    labels.append(np.array([2] * len(param_6)))

    param = np.concatenate(param, axis=0)
    labels = np.concatenate(labels, axis=0)

    # concatenate
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#1a9850', '#d73027', '#4575b4', '#313695', '#91bfdb',
        '#fc8d59', '#fee08b', '#d73027', '#4575b4', '#313695'
    ]
    colors = colors[:3]
    cmap = ListedColormap(colors)


    # t-SNE 적용
    tsne_model = TSNE(n_components=2, perplexity=40, n_iter=300, init='random', learning_rate=200.0)
    tsne_results = tsne_model.fit_transform(param)

    # label array
    label = np.arange(3)

    # 결과 시각화
    plt.figure(figsize=(10, 6))
    for i in range(3):
        indices = labels == i
        scatter = plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=i)

    plt.colorbar(scatter, ticks=np.arange(3), label='Class') 
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'{label}') for label, color in zip(label, colors)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.20, 1), loc='upper left', title='Classes')
    plt.savefig(f'/workspace/ssd0/byeongcheol/DGSS/FAMix/save_dir/TSNE_result_{chk_cls}.png')