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
import clip

# def compose_text_with_templates(text: str, templates) -> list:
#     return [template.format(text) for template in templates]

# imagenet_templates = [
#     'a bad photo of a {}.',
#     'a photo of many {}.',
#     'a sculpture of a {}.',
#     'a photo of the hard to see {}.',
#     'a low resolution photo of the {}.',
#     'a rendering of a {}.',
#     'graffiti of a {}.',
#     'a bad photo of the {}.',
#     'a cropped photo of the {}.',
#     'a tattoo of a {}.',
#     'the embroidered {}.',
#     'a photo of a hard to see {}.',
#     'a bright photo of a {}.',
#     'a photo of a clean {}.',
#     'a photo of a dirty {}.',
#     'a dark photo of the {}.',
#     'a drawing of a {}.',
#     'a photo of my {}.',
#     'the plastic {}.',
#     'a photo of the cool {}.',
#     'a close-up photo of a {}.',
#     'a black and white photo of the {}.',
#     'a painting of the {}.',
#     'a painting of a {}.',
#     'a pixelated photo of the {}.',
#     'a sculpture of the {}.',
#     'a bright photo of the {}.',
#     'a cropped photo of a {}.',
#     'a plastic {}.',
#     'a photo of the dirty {}.',
#     'a jpeg corrupted photo of a {}.',
#     'a blurry photo of the {}.',
#     'a photo of the {}.',
#     'a good photo of the {}.',
#     'a rendering of the {}.',
#     'a {} in a video game.',
#     'a photo of one {}.',
#     'a doodle of a {}.',
#     'a close-up photo of the {}.',
#     'a photo of a {}.',
#     'the origami {}.',
#     'the {} in a video game.',
#     'a sketch of a {}.',
#     'a doodle of the {}.',
#     'a origami {}.',
#     'a low resolution photo of a {}.',
#     'the toy {}.',
#     'a rendition of the {}.',
#     'a photo of the clean {}.',
#     'a photo of a large {}.',
#     'a rendition of a {}.',
#     'a photo of a nice {}.',
#     'a photo of a weird {}.',
#     'a blurry photo of a {}.',
#     'a cartoon {}.',
#     'art of a {}.',
#     'a sketch of the {}.',
#     'a embroidered {}.',
#     'a pixelated photo of a {}.',
#     'itap of the {}.',
#     'a jpeg corrupted photo of the {}.',
#     'a good photo of a {}.',
#     'a plushie {}.',
#     'a photo of the nice {}.',
#     'a photo of the small {}.',
#     'a photo of the weird {}.',
#     'the cartoon {}.',
#     'art of the {}.',
#     'a drawing of the {}.',
#     'a photo of the large {}.',
#     'a black and white photo of a {}.',
#     'the plushie {}.',
#     'a dark photo of a {}.',
#     'itap of a {}.',
#     'graffiti of the {}.',
#     'a toy {}.',
#     'itap of my {}.',
#     'a photo of a cool {}.',
#     'a photo of a small {}.',
#     'a tattoo of the {}.',
# ]

# def clip_gen(target):
#     tokens = clip.tokenize(target).to(device)
#     text_target = clip_model.encode_text(tokens).mean(axis=0, keepdim=True).detach()
#     text_target /= text_target.norm(dim=-1, keepdim=True)

#     return text_target

if __name__ == "__main__":
    # parameter
    param_root3 = "/workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/save_dir/Fine_grained_cls_20240704_11-51-07_saved_params.pkl"
    param_root4 = "/workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/save_dir/Coarse_grained_cls_20240704_11-54-40_saved_params.pkl"
    param_root6 = "/workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/save_dir/Wrong_cls_20240704_11-57-02_saved_params.pkl"

    with open(param_root3, "rb") as f:
        params3 = pickle.load(f)
    with open(param_root4, "rb") as f:
        params4 = pickle.load(f)
    with open(param_root6, "rb") as f:
        params6 = pickle.load(f)

    param = []
    labels = []

    # ipdb.set_trace()

    mu_params = torch.from_numpy(np.array([t[0].numpy() for t in params3["mu"]]))
    std_params = torch.from_numpy(np.array([t[0].numpy() for t in params3["std"]]))
    params_3 = mu_params.squeeze(-1).squeeze(-1)
    # params_3 = torch.cat((mu_params, std_params), dim=1).squeeze(-1).squeeze(-1)
    param_3 = params_3.cpu().numpy()
    param.append(param_3)
    labels.append(np.array([0] * len(param_3)))

    mu_params = torch.from_numpy(np.array([t[0].numpy() for t in params4[f"mu"]]))
    std_params = torch.from_numpy(np.array([t[0].numpy() for t in params4[f"std"]]))
    params_4 = mu_params.squeeze(-1).squeeze(-1)
    # params_4 = torch.cat((mu_params, std_params), dim=1).squeeze(-1).squeeze(-1)
    param_4 = params_4.cpu().numpy()
    param.append(param_4)
    labels.append(np.array([1] * len(param_4)))

    mu_params = torch.from_numpy(np.array([t[0].numpy() for t in params6[f"mu"]]))
    std_params = torch.from_numpy(np.array([t[0].numpy() for t in params6[f"std"]]))
    params_6 = mu_params.squeeze(-1).squeeze(-1)
    # params_6 = torch.cat((mu_params, std_params), dim=1).squeeze(-1).squeeze(-1)
    param_6 = params_6.cpu().numpy()
    param.append(param_6)
    labels.append(np.array([2] * len(param_6)))

    # Generate general embedding 
    # target = args.className
    # target = compose_text_with_templates(target, imagenet_templates)
    # embedding = clip_gen(target)

    # ipdb.set_trace()


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
    plt.savefig(f'/workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/save_dir/TSNE_result_png')