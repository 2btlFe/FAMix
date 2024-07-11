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
import os
from PIL import Image

def compose_text_with_templates(text: str, templates) -> list:
    return [template.format(text) for template in templates]

class_idx = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle', '255'
]

def load_img(path):
    img = Image.open(path)
    ipdb.set_trace()
    img = np.array(img.convert('RGB'))
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img    

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
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

def clip_gen(target):
    tokens = clip.tokenize(target).to(device)
    text_target = clip_model.encode_text(tokens).mean(axis=0, keepdim=True).detach()
    text_target /= text_target.norm(dim=-1, keepdim=True)

    return text_target

def clip_feats_gen(target):
    tokens = clip.tokenize(target).to(device)
    text_target = clip_model.encode_text(tokens).detach()
    text_target /= text_target.norm(dim=-1, keepdim=True)

    return text_target    


if __name__ == "__main__":
    # load clip_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model, preprocess = clip.load('RN50', device, jit=False)

    # class 별로 확인해보기
        
    # 80개 모두 확인해보기

    # style 별로 확인해보기
    embedding = []
    labels = []
    cls_num = 3
    
    file_name = ""
    type_idx = []
    for cls in class_idx[:cls_num]:
        file_name += cls + "_"
        for style in style_idx: # 20개 
            new_style = style + "_" + cls

            cls_style = cls + "_" + style
            type_idx.append(cls_style)
            text = compose_text_with_templates(new_style, imagenet_templates)
            txt_emb = clip_feats_gen(text)
            embedding.append(txt_emb)
            labels.append([cls_style] * len(txt_emb))

            
    file_name = file_name[:-1]
    file_name = file_name + ".png"

    embedding = torch.cat(embedding, dim=0)
    labels = np.concatenate(labels, axis=0)
    embedding = embedding.cpu().numpy()

    # ipdb.set_trace()

    #t-SNE concatenate
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#1a9850', '#d73027', '#4575b4', '#313695', '#91bfdb',
        '#fc8d59', '#fee08b', '#bc5090', '#ff6361', '#ffa600',
        '#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600',
        '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3',
        '#fdb462', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd',
        '#ccebc5', '#ffed6f', '#66c2a5', '#fc8d62', '#8da0cb',
        '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99',
        '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a',
        '#ffff99', '#b15928', '#d4b9da', '#c994c7', '#df65b0',
        '#e7298a', '#ce1256', '#980043', '#dd1c77', '#3182bd'
    ]

    
    colors = colors[:len(type_idx)]
    cmap = ListedColormap(colors)

    # t-SNE 적용
    tsne_model = TSNE(n_components=2, perplexity=40, n_iter=300, init='random', learning_rate=200.0)
    tsne_results = tsne_model.fit_transform(embedding)

    plt.figure(figsize=(10, 10))
    for i in range(len(type_idx)):
        scatter = plt.scatter(tsne_results[labels == type_idx[i], 0], tsne_results[labels == type_idx[i], 1], c=[colors[i]], label=type_idx[i])

    plt.colorbar(scatter, ticks=np.arange(len(type_idx)), label='Style')
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'{style}') for style, color in zip(type_idx, colors)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.20, 1), loc='upper left', title='Classes')
    
    tsne_dir = "tsne_clip_result"
    os.makedirs(tsne_dir, exist_ok=True)
    cls_path = os.path.join(tsne_dir, file_name)
    plt.savefig(cls_path)