import pickle
import os
import clip
import torch
import network
import torch.nn as nn
from datasets import cityscapes
import numpy as np
import random
import argparse
from torch.utils import data
from utils.freeze import freeze_all
from torch.nn.functional import unfold
from Experiment_patch.PPIN import PPIN 
import ipdb
from tqdm import tqdm
from PIL import Image
from datasets import Cityscapes
from utils import ext_transforms as et
from Experiment_patch.patch import PatchDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import seaborn as sns
from utils.stats import calc_mean_std
from network.backbone.resnet_clip import AttentionPool2d

from torch.utils.tensorboard import SummaryWriter

def compose_text_with_templates(text: str, templates) -> list:
    return [template.format(text) for template in templates]

class_idx = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle', '255'
]

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

random_styles = ['Ethereal Mist style',
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

class_idx = [
    'road', 'sidewalk', 'truck', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'building', 'bus', 'train', 'motorcycle',
    'bicycle', '255'
]

def compose_text_with_templates(text: str, templates) -> list:
    return [template.format(text) for template in templates]

def clip_gen(target):
    tokens = clip.tokenize(target).to(device)
    text_target = clip_model.encode_text(tokens).detach()
    text_target /= text_target.norm(dim=-1, keepdim=True)
    return text_target

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--save_dir", type=str, default='results', help="path to save results")
    parser.add_argument("--save_path", type=str, help= "path for learnt parameters saving")
    parser.add_argument("--fine_grained_data_root", type=str, default='/datasets_master/gta5',
                        help="path to Dataset")
    parser.add_argument("--coarse_grained_data_root", type=str, default='/datasets_master/gta5',
                        help="path to Dataset")
    parser.add_argument("--wrong_data_root", type=str, default='/datasets_master/gta5',
                        help="path to Dataset")
    parser.add_argument("--lr", type=float, default=1, help='optimization step')
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet_clip', choices=available_models, help='model name')
    parser.add_argument("--BB", type=str, default = 'RN50', help= "backbone name" )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument("--total_it", type = int, default =100, help= "total number of optimization iterations")
    # learn statistics
    parser.add_argument("--resize_feat",action='store_true',default=False, help="resize the features map to the dimension corresponding to CLIP")
    # random seed
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
    #data augmentation
    parser.add_argument("--data_aug", action='store_true',default=False)

    return parser

if __name__ == "__main__":
    opts = get_argparser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    # INIT
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    
    t1 = nn.AdaptiveAvgPool2d((56,56))

    fine_grained_patch_dst = PatchDataset(opts.fine_grained_data_root , transform=transform)
    coarse_grained_patch_dst = PatchDataset(opts.coarse_grained_data_root, transform=transform) 
    wrong_patch_dst = PatchDataset(opts.wrong_data_root, transform=transform)
    
    # patch_loader = data.DataLoader(patch_dst, batch_size=1, shuffle=False, num_workers=4)
    print("Fine Grained Patch dataset: %d" % len(fine_grained_patch_dst))
    print("Coarse Grained Patch dataset: %d" % len(coarse_grained_patch_dst))
    print("Wrong Patch dataset: %d" % len(wrong_patch_dst))
    
    #load CLIP
    model = network.modeling.__dict__['deeplabv3plus_resnet_clip'](num_classes=19,BB= opts.BB,OS=32)
    freeze_all(model)
    

    # ipdb.set_trace()
    # model.backbone.attnpool = AttentionPool2d(8, 2048, )

    clip_model, preprocess = clip.load(opts.BB, device, jit=False)
    torch.autograd.set_detect_anomaly(True)

    patch_dic = {}
    patch_dic['fg'] = fine_grained_patch_dst
    patch_dic['cg'] = coarse_grained_patch_dst
    patch_dic['w'] = wrong_patch_dst

    # ipdb.set_trace()
    # fg_img : [1, C, H, W]
    # cg_img : [1, C, H, W]
    # w_img : [1, C, H, W]

    param = {}
    with open("/workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/save_dir/Fine_grained_cls_20240704_11-51-07_saved_params.pkl", "rb") as f:
        param['fg'] = pickle.load(f)
    with open("/workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/save_dir/Coarse_grained_cls_20240704_11-54-40_saved_params.pkl", "rb") as f:
        param['cg'] = pickle.load(f)
    with open("/workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/save_dir/Wrong_cls_20240704_11-57-02_saved_params.pkl", "rb") as f:
        param['w'] = pickle.load(f)


    # 실험 1. fine_grained 로 해보기
    # ipdb.set_trace()
    # TrgEmb 만들어두기 
    emb_list = []
    label_list = []
    cls_num = 2

    file_name = ""
    type_idx = []
    markers = []
    sizes = []
    
    for origin_style in ['fg', 'cg', 'w']:        
        chosen_mu = param[origin_style]['mu'][2][0].to(device)     #[256, 1, 1]
        chosen_std = param[origin_style]['std'][2][0].to(device)   #[256, 1, 1]
        chosen_meta = param[origin_style]['mu'][2][1]  #dictionary

        for key in patch_dic.keys():
            dataset = patch_dic[key]
            
            label_name = origin_style + '_' + key
            print(label_name)

            tmp_emb_list = []
            tmp_label_list = []
            
            for i in range(len(dataset)): # patch 10장
                img, label, meta = patch_dic[key][i]    
                
                # ipdb.set_trace()
                img = img.unsqueeze(0).cuda()

                label = label.unsqueeze(0)
                meta = meta
                
                print(meta)

                f = model.backbone(img,trunc1=False,trunc2=False,
                trunc3=False,trunc4=False,get1=True,get2=False,get3=False,get4=False)
                
                mean, std = calc_mean_std(f)
                f = (f - mean.expand(f.size())) / std.expand(f.size())
                
                # style inf
                f = f * chosen_std.unsqueeze(0).expand(f.size()) + chosen_mu.unsqueeze(0).expand(f.size())
                # patches pooling
                f = t1(f)

                # ipdb.set_trace()
                # target_features_from_low: [1, 2048, 7, 7]
                target_features_from_low = model.backbone(f.to(device),trunc0=False,trunc1=True,trunc2=False,
                        trunc3=False,trunc4=False,get0=False,get1=False,get2=False,get3=False,get4=False)
                
                target_features_from_low /= target_features_from_low.norm(dim=-1, keepdim=True).clone().detach()
                
                tmp_emb_list.append(target_features_from_low)
                tmp_label_list.append(label_name)
                 
            # ipdb.set_trace()
            emb_list.append(torch.cat(tmp_emb_list, dim=0))
            label_list.append([label_name] * len(tmp_emb_list)) 
            type_idx.append(label_name)
            sizes.append(200)
            markers.append('s')  

    marker_list = ['o', '*', '+', 'v', '^', '<', '>', 'D', 'd', 'p', 'P', 'H', 'h', 'x', '|', '_']

    for i in range(13, 13+cls_num):
        cls = class_idx[i]
        file_name += cls + "_"
        
        marker = marker_list[i-13]
        
        for style in random_styles:
            target = style + ' ' + cls
            cls_style = cls + ' ' + style
            type_idx.append(cls_style)
            
            if cls == 'car' and style == 'Urban Grit style':
                markers.append('s')
            else:
                markers.append(marker)

            sizes.append(36)

            target = compose_text_with_templates(target, imagenet_templates)
            trgemb = clip_gen(target)   #[80, 1024]
            
            emb_list.append(trgemb)
            label_list.append([cls_style] * len(trgemb)) 
    
    file_name = file_name + ".png"
    

    # label_list.append([f'fine_grained_{chosen_meta["style"]}_{chosen_meta["class"]}'])
    # label_list.append([f'coarse_grained_{chosen_meta["style"]}_{chosen_meta["class"]}'])
    # label_list.append([f'wrong_{chosen_meta["style"]}_{chosen_meta["class"]}'])

    # type_idx.append(f'fine_grained_{chosen_meta["style"]}_{chosen_meta["class"]}')
    # type_idx.append(f'coarse_grained_{chosen_meta["style"]}_{chosen_meta["class"]}')
    # type_idx.append(f'wrong_{chosen_meta["style"]}_{chosen_meta["class"]}')

    embedding = torch.cat(emb_list, dim=0)
    labels = np.concatenate(label_list, axis=0)
    embedding = embedding.cpu().numpy()

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
        # 두 번째 세트 색상
        '#17becf', '#1a9850', '#d73027', '#4575b4', '#313695',
        '#91bfdb', '#fc8d59', '#fee08b', '#bc5090', '#ff6361', 
        '#ffa600', '#003f5c', '#58508d', '#bc5090', '#ff6361', 
        '#ffa600', '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', 
        # 세 번째 세트 색상, 두 번째 세트와 동일
        '#17becf', '#1a9850', '#d73027', '#4575b4', '#313695',
        '#91bfdb', '#fc8d59', '#fee08b', '#bc5090', '#ff6361', 
        '#ffa600', '#003f5c', '#58508d', '#bc5090', '#ff6361', 
        '#ffa600', '#8dd3c7', '#ffffb3', '#bebada', '#fb8072',
        # 네 번째 세트 색상, 두 번째 세트와 동일
        '#17becf', '#1a9850', '#d73027', '#4575b4', '#313695',
        '#91bfdb', '#fc8d59', '#fee08b', '#bc5090', '#ff6361', 
        '#ffa600', '#003f5c', '#58508d', '#bc5090', '#ff6361', 
        '#ffa600', '#8dd3c7', '#ffffb3', '#bebada', '#fb8072',
        # 다섯 번째 세트 색상, 두 번째 세트와 동일
        '#17becf', '#1a9850', '#d73027', '#4575b4', '#313695',
        '#91bfdb', '#fc8d59', '#fee08b', '#bc5090', '#ff6361', 
        '#ffa600', '#003f5c', '#58508d', '#bc5090', '#ff6361', 
        '#ffa600', '#8dd3c7', '#ffffb3', '#bebada', '#fb8072',
    ]

    colors = colors[:len(type_idx)]
    cmap = ListedColormap(colors)

    # t-SNE 적용
    tsne_model = TSNE(n_components=2, perplexity=40, n_iter=300, init='random', learning_rate=200.0)
    tsne_results = tsne_model.fit_transform(embedding)

    plt.figure(figsize=(24, 24))
    for i in range(len(type_idx)):
            scatter = plt.scatter(tsne_results[labels == type_idx[i], 0], tsne_results[labels == type_idx[i], 1], c=[colors[i]], marker=markers[i], s=sizes[i], label=type_idx[i])
    
    plt.colorbar(scatter, ticks=np.arange(len(type_idx)), label='Style')
    legend_elements = [Line2D([0], [0], marker=marker, color='w', markerfacecolor=color, markersize=10, label=f'{style}') for marker, style, color in zip(markers, type_idx, colors)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.20, 1), loc='upper left', title='Classes')
    
    tsne_dir = "tsne_clip_result"
    os.makedirs(tsne_dir, exist_ok=True)
    cls_path = os.path.join(tsne_dir, file_name)
    plt.savefig(cls_path)

