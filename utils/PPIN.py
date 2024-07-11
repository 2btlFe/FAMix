import torch
import torch.nn as nn
from torch.nn.functional import unfold
from utils.stats import calc_mean_std
import ipdb

class PPIN(nn.Module):
    def __init__(self,content_feat,div=3,ind=[]):
        super(PPIN,self).__init__()
        self.ind = ind
        self.div = div
        self.content_feat = content_feat.clone().detach() #(B,C,H,W)
        self.B = self.content_feat.shape[0]
        self.C = self.content_feat.shape[1]

        side = self.content_feat.size()[2]
        new_side = int(side // self.div)
        
        self.patches = unfold(self.content_feat, kernel_size=new_side, stride=new_side).permute(-1,0,1).reshape(-1,256,new_side,new_side)
        self.patches = self.patches[ind] #(len(ind),C,H/div,W/div)
        
        self.style_mean = torch.zeros(len(ind),self.C,1,1) #(len(ind),C,1,1)
        self.style_std =  torch.zeros(len(ind),self.C,1,1) #(len(ind),C,1,1)
      
        for i in range(len(self.ind)):

            mean , std = calc_mean_std(self.patches[i].unsqueeze(0))
            self.patches[i] = (self.patches[i] - mean.expand(self.patches[i].unsqueeze(0).size()) ) / std.expand(self.patches[i].unsqueeze(0).size())
            self.style_mean[i] = mean
            self.style_std[i] = std

        self.size = self.patches[0].size()   #(C,H/div,W/div)

        self.style_mean = nn.Parameter(self.style_mean, requires_grad = True) # (len(ind),C,1,1)
        self.style_std = nn.Parameter(self.style_std, requires_grad = True) #(len(ind),C,1,1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self):
        
        patches_prime = torch.zeros_like(self.patches.clone().detach()) # (len(ind),C,H/div,W/div)
        
        for i in range(len(self.ind)):
            patches_prime[i] = self.patches[i] * self.style_std[i].expand(self.size)  + self.style_mean[i].expand(self.size)
        
        patches_prime = self.relu(patches_prime)

        return patches_prime
    

# 24/07/10
class PPIN_DC(nn.Module):
    def __init__(self, content_feat, patch_meta):
        super(PPIN_DC, self).__init__()
        self.patch_meta = patch_meta
        self.content_feat = content_feat.clone().detach() #(B,C,H,W)
        self.B = self.content_feat.shape[0]
        self.C = self.content_feat.shape[1]
        self.t1 = nn.AdaptiveAvgPool2d((56, 56))
        
        side = self.content_feat.size()[2]
        
        # ipdb.set_trace()
        self.patches = []   #(len(patch_meta), C, side, side)
        
        scale = 768//side

        for idx in range(len(self.patch_meta)):
            k, h, w, side, dominant_cls = self.patch_meta[idx]
            k, h, w, side = int(k), int(h), int(w), int(side)
            self.patches.append(self.content_feat[k, :, h//scale:(h+side)//scale, w//scale:(w+side)//scale])
        
        self.style_mean = torch.zeros(len(self.patches),self.C,1,1) #(len(patch_meta),C,1,1)
        self.style_std =  torch.zeros(len(self.patches),self.C,1,1) #(len(patch_meta),C,1,1)

        for i in range(len(self.patch_meta)):
            mean , std = calc_mean_std(self.patches[i].unsqueeze(0))
            self.patches[i] = (self.patches[i] - mean.expand(self.patches[i].unsqueeze(0).size()) ) / std.expand(self.patches[i].unsqueeze(0).size())
            self.style_mean[i] = mean
            self.style_std[i] = std

        self.size = self.patches[0].size()   #(C,side,side)

        self.style_mean = nn.Parameter(self.style_mean, requires_grad = True) # (len(self.patches),C,1,1)
        self.style_std = nn.Parameter(self.style_std, requires_grad = True) #(len(self.patch.patches),C,1,1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self):
        
        patches_prime = torch.zeros((len(self.patches), self.C, 56, 56)) # (len(patch),C, 56, 56)
        
        # Patch 마다 크기가 다르기 때문에 다르게 적용해줘야 한다
        for i in range(len(self.patch_meta)):
            tmp = self.patches[i] * self.style_std[i].expand_as(self.patches[i])  + self.style_mean[i].expand_as(self.patches[i])
            tmp = self.relu(tmp)
            patches_prime[i] = self.t1(tmp)

        # patches_prime = self.relu(patches_prime)

        return patches_prime