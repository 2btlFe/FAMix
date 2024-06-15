from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from torch.utils import data
from datasets import Cityscapes
from metrics import StreamSegMetrics
import torch
import torch.nn as nn
import pickle
from copy import deepcopy
from utils.get_dataset import get_dataset
from utils.freeze import * 
from torch.nn.functional import unfold
import ipdb
import logging



from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--dataset", type=str, default='gta5',
                        choices=['cityscapes','ACDC','gta5','synthia','mapillary','bdd100k'], help='Name of dataset')
    parser.add_argument("--data_root", type=str, default='/datasets_master/gta5',
                        help="path to Dataset")
    parser.add_argument("--ACDC_sub", type=str, default="night",
                        help = "specify which subset of ACDC  to use")

    # Backbone Options

    parser.add_argument("--BB", type = str, default = "RN50",
                        help = "backbone of the segmentation network")
    parser.add_argument("--OS", type= int, default=16,
                        help = "output stride")

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=40e3,
                        help="epoch number (default: 40k)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate (default: 0.1)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help='batch size (default: 8)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=768)
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--val_interval", type=int, default=750,
                        help="iteration interval for eval (default: 750)")

    parser.add_argument("--ckpts_path", type = str ,
                        help="path for checkpoints saving")
    parser.add_argument("--data_aug", action='store_true', default=False)
    parser.add_argument("--num_classes", type=int, default=19,
                        help="number of classes to be considered for segmentation")
    parser.add_argument("--path_for_stats",type=str, help="path for the optimized stats")
    
    parser.add_argument("--path_for_3stats", type=str, help="path for the optimized 3 stats")
    parser.add_argument("--path_for_4stats", type=str, help="path for the optimized 4 stats")
    parser.add_argument("--path_for_6stats", type=str, help="path for the optimized 6 stats")

    parser.add_argument("--transfer", action='store_true',default=True)
    parser.add_argument("--div", type=int, default=3, help="number of divisions for the image")
    parser.add_argument("--work_dir", type=str, default='model_ckpt', help="path to save model")
    parser.add_argument("--patch_method", type=str, default="default")
    return parser

def validate(model, loader, device, metrics, dataset):
    """Do validation and return specified samples"""
    metrics.reset()

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs, _ = model(images)
            
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
           
            metrics.update(targets, preds)

        score = metrics.get_results(dataset)
    return score


def main():
    opts = get_argparser().parse_args()
    # 240613_adjust_logger

   # StreamHandler 설정 (콘솔에 로그 출력)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)  # 로그 레벨 설정
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # FileHandler 설정 (파일에 로그 기록)
    os.makedirs(opts.work_dir, exist_ok=True)
    file_handler = logging.FileHandler(f'{opts.work_dir}/train.log')
    file_handler.setLevel(logging.DEBUG)  # 로그 레벨 설정
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # 로거 설정
    logger = logging.getLogger('FAMix')
    logger.setLevel(logging.DEBUG)  # 로거의 기본 로그 레벨 설정
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device: %s" % device)

    # Setup random seed
    # INIT
    torch.manual_seed(opts.random_seed) # set to 1 
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
  
    train_dst,val_dst = get_dataset(opts.dataset,opts.data_root,opts.crop_size,opts.ACDC_sub,
                                    data_aug=opts.data_aug)

    if not opts.test_only:
        train_loader = data.DataLoader(
            train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=4,
        drop_last=True)  # drop_last=True to ignore single-image batches. Drop last batch if it doesn't batch 10

    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=4)
 
    logger.info("Dataset: %s, Train set: %d, Val set: %d" %
    (opts.dataset, len(train_dst), len(val_dst)))


    # Set up model
    model = network.modeling.__dict__['deeplabv3plus_resnet_clip'](num_classes=opts.num_classes, BB= opts.BB, OS=opts.OS)
    model.backbone.attnpool = nn.Identity()

    # freeze layers
    if opts.dataset == "gta5":
        freeze_1_2_3_p4(model)
    elif opts.dataset == "cityscapes":
        freeze_1_2_3(model)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)
    
    # Set up optimizers
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean',label_smoothing=0.1)

    def save_ckpt(path,model,optimizer,scheduler,best_score):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        logger.info("Model saved as %s" % path)
    
    if not opts.test_only:
        utils.mkdir(opts.ckpts_path)
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        if opts.continue_training:
            checkpoint["optimizer_state"] = deepcopy(checkpoint["optimizer_state"])
            checkpoint["scheduler_state"] = deepcopy(checkpoint["scheduler_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            logger.info("Training state restored from %s" % opts.ckpt)
        logger.info("Model restored from %s" %opts.ckpt)
        del checkpoint  # free memory
        
    else:
        logger.info("[!] Retrain")
        model.to(device)
        
    # ==========   Train Loop   ==========#


    # return

    interval_loss = 0

    # load various patch_stats 

    if not opts.test_only: 
        
        if opts.patch_method != "default":
            
            loaded_dict_patches_list = []

            with open(opts.path_for_3stats, 'rb') as f:
                loaded_dict_patches_list.append(pickle.load(f))
            with open(opts.path_for_4stats, 'rb') as f:
                loaded_dict_patches_list.append(pickle.load(f))
            with open(opts.path_for_6stats, 'rb') as f:
                loaded_dict_patches_list.append(pickle.load(f))
        else:
            with open(opts.path_for_stats, 'rb') as f:
                loaded_dict_patches = pickle.load(f)

        relu = nn.ReLU(inplace=True)


    if opts.test_only:
       
        model.eval()

        val_score = validate(model=model, loader=val_loader, device=device, metrics=metrics, dataset=opts.dataset)

        logger.info(metrics.to_str(val_score))
        logger.info(val_score["Mean IoU"])
        logger.info(val_score["Class IoU"])
        
        save_txt = f"{opts.work_dir}_logs_PODA_val_gta5_{opts.dataset}.txt"

        save_txt = f"logs_PODA_val_gta5_{opts.dataset}.txt"
        with open(save_txt, 'a') as f:
            f.write(f'{val_score["Mean IoU"]}\n')
            f.write(f'{val_score["Class IoU"]}\n') 
        return

    # initial div
    div = opts.div
    div_list = [3, 4, 6]

    # previous performance
    # if opts.patch_method == "ranking":
    #     previous_cls_iou = [0] * opts.num_classes

    while True:  # cur_itrs < opts.total_itrs:
    # =====  Train  =====

        if opts.dataset == "gta5":
            model.backbone.layer4[2].train()
        elif opts.dataset == "cityscapes":
            model.backbone.layer4.train()
        
        model.classifier.train()

        cur_epochs += 1

        # Division
        if opts.patch_method == "division":
            if cur_epochs < 7:
                div = 3
                loaded_dict_patches = loaded_dict_patches_list[0]
            elif cur_epochs < 14:
                div = 4
                loaded_dict_patches = loaded_dict_patches_list[1]
            else:
                div = 6
                loaded_dict_patches = loaded_dict_patches_list[2]
        
        # Rever Division
        elif opts.patch_method == "reverse_division":
            if cur_epochs < 7:
                div = 6
                loaded_dict_patches = loaded_dict_patches_list[2]
            elif cur_epochs < 14:
                div = 4
                loaded_dict_patches = loaded_dict_patches_list[1]
            else:
                div = 3
                loaded_dict_patches = loaded_dict_patches_list[0]

        # Random
        elif opts.patch_method == "random":
            idx = random.randint(0,2)
            loaded_dict_patches = loaded_dict_patches_list[idx]
            div = div_list[idx]

        # Alternate
        elif opts.patch_method == "alternate":
            if cur_epochs % 3 == 0:
                div = 3
                loaded_dict_patches = loaded_dict_patches_list[0]
            elif cur_epochs % 3 == 1:
                div = 4
                loaded_dict_patches = loaded_dict_patches_list[1]
            else:
                div = 6
                loaded_dict_patches = loaded_dict_patches_list[2]

        # # ranking
        # elif opts.patch_method == "ranking":
            
        for i, (images, labels) in tqdm(enumerate(train_loader), desc="Training", unit="batch", unit_scale=True):
            
            cur_itrs += 1
            images = images.to(device, dtype=torch.float32)
            # labels = labels.to(device, dtype=torch.long)
           
            optimizer.zero_grad()
            
            labels_ = labels.unsqueeze(1)  # (B,1,768,768)
            # lbl_patches = divide_in_patches(labels_,3)

            side = labels.size()[2]
            new_side = int(side // div)

            lbl_patches = unfold(labels_, kernel_size=new_side, stride=new_side).permute(-1,0,1)
            lbl_patches = lbl_patches.reshape(lbl_patches.shape[0],lbl_patches.shape[1],1,new_side,new_side) #### (div*div, B, 1, H/div, W/div)
            lbl_patches = lbl_patches.to(torch.long)

            most_list = []
            for j in range(len(lbl_patches)): ### iterate on dim 0 (div*div)
                most = [Cityscapes.name(torch.mode(torch.flatten(lbl_patches[j][k])).values) if torch.mode(torch.flatten(lbl_patches[j][k])).values != 255 else 255 for k in range(lbl_patches[0].shape[0])]
                most_list.append(most) #len=div*div , each element list of B elements
            
            # Mix proportion for the mixup
            beta_dist = torch.distributions.beta.Beta(0.1, 0.1)
            s = beta_dist.sample((opts.batch_size, 256, 1, 1)).to('cuda')
            
            outputs,features = model(images, transfer=opts.transfer,mix=True,most_list=most_list,saved_params=loaded_dict_patches,activation=relu,s=s, div=div)
        
            ##############################################################################################################################################
            labels = labels.to(device, dtype=torch.long)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
          
            writer.add_scalar("loss",loss,cur_itrs)

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                logger.info("Epoch %d, Itrs %d/%d, Loss_clip=%f" %
                    (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            # Performance of GTAV can be used to tradeoff 
            if (cur_itrs) % opts.val_interval == 0 or cur_itrs == opts.total_itrs:
             
                save_ckpt(opts.ckpts_path+'/latest_%s_%s_'%
                        ('deeplabv3plus_resnet_clip', opts.dataset)+str(cur_itrs)+'.pth' ,model,optimizer,scheduler,best_score)
               
                logger.info("validation...")
                
                model.eval()

                val_score = validate(model=model, loader=val_loader,device=device, metrics=metrics, dataset=opts.dataset)
                
                logger.info(metrics.to_str(val_score))
                
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt(opts.ckpts_path+'/best_%s_%s.pth' %
                            ('deeplabv3plus_resnet_clip', opts.dataset),model,optimizer,scheduler,best_score)

                writer.add_scalar("mIoU", val_score['Mean IoU'] ,cur_itrs)

                # 아니 이거는 도대체가 무슨 심보지 - transfer learning을 최대한 활용하겠다는 건가
                if opts.dataset == "gta5":
                    model.backbone.layer4[2].train()
                elif opts.dataset == "cityscapes":
                    model.backbone.layer4.train()
        
                model.classifier.train()

            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return
            

if __name__ == '__main__':
    main()
