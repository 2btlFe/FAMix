import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.stats import *
import random
from torch.nn.functional import unfold
import ipdb
import queue

# class LinearFusion(nn.Module):
    
#     def __init__(self, input_size=256, output_size=256):
#         super(LinearFusion, self).__init__()
#         # Linear layer to reduce the number of elements from 6 to 2
#         self.reduce_layer = nn.Linear(6, 2)
        
#     def forward(self, x):
#         # x is expected to be of shape [batch_size, 16, 6, 256, 1, 1]
#         batch_size = x.size(0)
#         num_elements = x.size(1)  # 16
#         # Reshape x to [batch_size, num_elements, 6, 256]
#         x = x.view(batch_size, num_elements, 6, 256).transpose(2, 3)  # Shape: [batch_size, num_elements, 256, 6]
#         x = self.reduce_layer(x)  # Shape: [batch_size, num_elements, 256, 2]
#         x = x.transpose(2, 3).view(batch_size, num_elements, 2, 256, 1, 1)  # Shape: [batch_size, num_elements, 2, 256, 1, 1]

#         return x

# feature disentangle 생각 
class MLPFusion(nn.Module):

    def __init__(self, layer = 0, input_size=256, output_size=256):
        super(MLPFusion, self).__init__()
        # Linear layer to reduce the number of elements from 6 to 2
        
        self.layer = nn.ModuleList()
        for i in range(layer):
            self.layer.append(nn.Linear(6, 6))
        self.layer.append(nn.Linear(6, 2))
        
    def forward(self, x):
        # x is expected to be of shape [batch_size, 16, 6, 256, 1, 1]
        batch_size = x.size(0)
        num_elements = x.size(1)  # 16
        # Reshape x to [batch_size, num_elements, 6, 256]
        x = x.view(batch_size, num_elements, 6, 256).transpose(2, 3)  # Shape: [batch_size, num_elements, 256, 6]
        
        # ipdb.set_trace()
        for layer in self.layer:
            x = layer(x)
           
        x = x.transpose(2, 3).view(batch_size, num_elements, 2, 256, 1, 1)
        
        return x

class _Segmentation(nn.Module):
    def __init__(self, backbone,classifier, blender=None):
        super(_Segmentation, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

        if blender is not None:
            self.mlp_fusion = blender


    # Adjust new div   
    def forward(self, x, transfer=False,mix=False,most_list=None,saved_params=None, saved_params_4=None, saved_params_6=None, activation=None,s=0, div=3, mode="default", single=False):
        
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
            
            # 24/6/24 - patch fusion stylization - by linear and mlp
            if "fusion" in mode:
                mu_bin = torch.zeros([8, div*div, 3, 256, 1, 1])
                std_bin = torch.zeros([8, div*div, 3, 256, 1, 1])
                # ipdb.set_trace()
            
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

            # 24/6/20 - patch fusion stylization - by mlp 
            for j,most in enumerate(most_list):  #len(most_list)=div*div   
                for k,el in enumerate(most):  #len(most)=B
                    

                    if not saved_params[str(el)+'_mu']:
                        if single:
                            idx = 0
                        else:
                            idx = random.choice([idxx for idxx in range (len(saved_params['255_mu']))])
                        mu_t = saved_params['255_mu'][idx]
                        std_t = saved_params['255_std'][idx]
                    else: 
                        #TODO: Adjust New Sampling Method 
                        if single:
                            idx =0
                        else:
                            idx = random.choice([idxx for idxx in range (len(saved_params[str(el)+'_mu']))])
                        
                        if "fusion" in mode:
                            mu_t_3 = saved_params[str(el)+'_mu'][idx]
                            std_t_3 = saved_params[str(el)+'_std'][idx]
                            mu_t_4 = saved_params_4[str(el)+'_mu'][idx]
                            std_t_4 = saved_params_4[str(el)+'_std'][idx]
                            mu_t_6 = saved_params_6[str(el)+'_mu'][idx]
                            std_t_6 = saved_params_6[str(el)+'_std'][idx]
                            
                            # ipdb.set_trace()
                            mu_t = torch.stack([mu_t_3, mu_t_4, mu_t_6], dim=0)  # [3, 256, 1, 1]
                            std_t = torch.stack([std_t_3, std_t_4, std_t_6], dim=0)  #[3, 256, 1, 1]
                            
                            # res = self.mlp_fusion(input)
                            # mu_t, std_t = res[0], res[1]
                            # ipdb.set_trace()
                        else:    
                            mu_t = saved_params[str(el)+'_mu'][idx]
                            std_t = saved_params[str(el)+'_std'][idx]
                    
                    if "fusion" in mode:                    
                        # ipdb.set_trace()
                        mu_bin[k, j, :] = mu_t      #[8, div*div, 3, 256, 1, 1]
                        std_bin[k, j, :] = std_t    #[8, div*div, 3, 256, 1, 1]
                    else:
                        mu_t_f1[k,:,h:h+features['low_level'].shape[2]//div,w:w+features['low_level'].shape[3]//div]  = mu_t.expand((256,d2//div,d3//div))
                        std_t_f1[k,:,h:h+d2//div,w:w+d3//div] = std_t.expand((256,d2//div,d3//div))
                w+=d2//div
                if (j+1)%div==0:
                    w=0
                    h+=d3//div


            if "fusion" in mode:
                # 1. 모두 다 Fusion
                # ipdb.set_trace()
                param = torch.cat([mu_bin, std_bin], dim=2).to('cuda') # [8, div*div, 6, 256, 1, 1]
                res = self.mlp_fusion(param)   # [8, div*div, 2, 256, 1, 1]

                num_patches = div*div
                for j in range(num_patches):
                    # ipdb.set_trace()
                    h=(j//div)*d3//div
                    w=(j%div)*d2//div
                    
                    mu_t_f1[:,:,h:h+d2//div,w:w+d3//div] = res[:, j, 0].expand((8, 256, d2//div, d3//div))
                    std_t_f1[:,:,h:h+d2//div,w:w+d3//div] = res[:, j, 1].expand((8, 256, d2//div, d3//div))

                # 2. mu, std 각각 Fusion
                    
                # 3. MLP 쓰기 
                    
                # 4. Adversarial
                    


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