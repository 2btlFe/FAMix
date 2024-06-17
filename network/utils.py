import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.stats import *
import random
from torch.nn.functional import unfold
import ipdb
import queue


class _Segmentation(nn.Module):
    def __init__(self, backbone,classifier):
        super(_Segmentation, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    # Adjust new div   
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



            # 24/6/12 Patch finding
            # ipdb.set_trace()
            # most_list = [16, 8]
            # B = len(most_list[0])
            # P = len(most_list)
            # direction = [[-1, 0], [1, 0], [0, -1], [0, 1]]

            # cnt = 0

            # for i in range(B):
            #     chk = [[False for _ in range(div)] for _ in range(div)]
                
            #     for j in range(P):
            #         r, c = j // div, j % div
                    
            #         if chk[r][c]:
            #             continue                    

            #         # BFS 시작
            #         el = most_list[j][i]
            #         if not saved_params[str(el) + '_mu']:
            #             idx = random.choice([idxx for idxx in range(len(saved_params['255_mu']))])
            #             mu_t = saved_params['255_mu'][idx]
            #             std_t = saved_params['255_std'][idx]
            #         else: # 원본
            #             idx = random.choice([idxx for idxx in range(len(saved_params[str(el) + '_mu']))])
            #             mu_t = saved_params[str(el) + '_mu'][idx]
            #             std_t = saved_params[str(el) + '_std'][idx]

            #         q = queue.Queue()
            #         pivot = most_list[j][i]
            #         chk[r][c] = True
            #         q.put((r, c))

            #         while not q.empty():
            #             cur_r, cur_c = q.get()  # BFS 루프 내에서 로컬 변수를 사용
            #             # print(i, j, cur_r, cur_c)  # 디버그 출력문

            #             cnt += 1
            #             h = cur_r * d2 // div
            #             w = cur_c * d3 // div

            #             mu_t_f1[i, :, h:h+features['low_level'].shape[2]//div, w:w+features['low_level'].shape[3]//div] = mu_t.expand((256, d2//div, d3//div))
            #             std_t_f1[i, :, h:h+d2//div, w:w+d3//div] = std_t.expand((256, d2//div, d3//div))

            #             for ori in range(4):
            #                 nr, nc = cur_r + direction[ori][0], cur_c + direction[ori][1]

            #                 if nr < 0 or nc < 0 or nr >= div or nc >= div or chk[nr][nc]:
            #                     continue

            #                 nj = nr * div + nc
            #                 if most_list[nj][i] == pivot:
            #                     chk[nr][nc] = True
            #                     q.put((nr, nc))

                #print(cnt, div * div)
                # assert cnt == div * div

            # 24/6/12 Stylization - Patch Wise Stylization
            for j,most in enumerate(most_list):  #len(most_list)=div*div   
                for k,el in enumerate(most):  #len(most)=B

                    if not saved_params[str(el)+'_mu']:
                        idx = random.choice([idxx for idxx in range (len(saved_params['255_mu']))])
                        mu_t = saved_params['255_mu'][idx]
                        std_t = saved_params['255_std'][idx]
                    else: #orig

                        #TODO: Adjust New Sampling Method 
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
                # Mix up
                features['low_level'] = (std_mix.expand(self.size) * features_low_norm + mu_mix.expand(self.size))
            features['low_level'] = activation(features['low_level'])
           
        features['out'] = self.backbone(features['low_level'],trunc1=True,trunc2=False,
            trunc3=False,trunc4=False,get1=False,get2=False,get3=False,get4=True)
    
        x = self.classifier(features)
        output = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return output, features