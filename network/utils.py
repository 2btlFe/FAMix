import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.stats import *
import random
from torch.nn.functional import unfold
import ipdb


class _Segmentation(nn.Module):
    def __init__(self, backbone,classifier):
        super(_Segmentation, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
      
    def forward(self, x, transfer=False,mix=False,most_list=None,saved_params=None,activation=None,s=0, div=3):
        
        # ipdb.set_trace()
        input_shape = x.shape[-2:]
        features = {}
        features['low_level'] = self.backbone(x,trunc1=False,trunc2=False,
           trunc3=False,trunc4=False,get1=True,get2=False,get3=False,get4=False)
        d2 ,d3 = features['low_level'].shape[2], features['low_level'].shape[3]

        if transfer:

            mean, std = calc_mean_std(features['low_level'])
            self.size = features['low_level'].size()

            mu_t_f1 = torch.zeros([8,256,d2,d3])
            std_t_f1 = torch.zeros([8,256,d2,d3])
            h=0
            w=0

            new_kernel_size = d2//div

            self.patches = unfold(features['low_level'], kernel_size=new_kernel_size, stride=new_kernel_size).permute(-1,0,1)
            self.patches = self.patches.reshape(self.patches.shape[0],self.patches.shape[1],256,d2//div,d3//div)

            means_orig = torch.zeros([8,256,d2,d3])
            stds_orig = torch.zeros([8,256,d2,d3])

            for i in range(div*div):
                mean , std = calc_mean_std(self.patches[i])
            
                means_orig[:,:,h:h+d2//div,w:w+d3//div] = mean.expand((8,256,d2//div,d3//div))
                stds_orig[:,:,h:h+d2//div,w:w+d3//div] =std.expand((8,256,d2//div,d3//div))
                w+=d2//div
                if (i+1)%div == 0 :
                    w=0
                    h+=d3//div

                self.patches[i] = (self.patches[i] - mean.expand(self.patches[i].size()) ) / std.expand(self.patches[i].size())
            features_low_norm = torch.cat([torch.cat([self.patches[div*i+j] for i in range(div)],dim=2) for j in range(div)],dim=3)
            
            h=0
            w=0
            
            for j,most in enumerate(most_list):  #len(most_list)=div*div   
                for k,el in enumerate(most):  #len(most)=B

                    if not saved_params[str(el)+'_mu']:
                        idx = random.choice([idxx for idxx in range (len(saved_params['255_mu']))])
                        mu_t = saved_params['255_mu'][idx]
                        std_t = saved_params['255_std'][idx]
                    else:
                        #orig
                        idx = random.choice([idxx for idxx in range (len(saved_params[str(el)+'_mu']))])
                        mu_t = saved_params[str(el)+'_mu'][idx]
                        std_t = saved_params[str(el)+'_std'][idx]

                    mu_t_f1[k,:,h:h+features['low_level'].shape[2]//div,w:w+features['low_level'].shape[3]//div]  = mu_t.expand((256,d2//div,d3//div))
                    std_t_f1[k,:,h:h+d2//div,w:w+d3//div] = std_t.expand((256,d2//div,d3//div))
                w+=d2//div
                if (j+1)%div==0:
                    w=0
                    h+=d3//div
            if not mix:
                features['low_level'] = (std_t_f1.to('cuda') * features_low_norm + mu_t_f1.to('cuda'))
            else:

                mu_mix = s * means_orig.to('cuda') + (1-s) *  mu_t_f1.to('cuda')
                std_mix = s * stds_orig.to('cuda') + (1-s) *  std_t_f1.to('cuda')
                features['low_level'] = (std_mix.expand(self.size) * features_low_norm + mu_mix.expand(self.size))
            features['low_level'] = activation(features['low_level'])
           
        features['out'] = self.backbone(features['low_level'],trunc1=True,trunc2=False,
            trunc3=False,trunc4=False,get1=False,get2=False,get3=False,get4=True)
    
        x = self.classifier(features)
        output = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return output, features