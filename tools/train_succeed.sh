DATASET=gta5
DATAROOT=/workspace/ssd0/byeongcheol/DGSS/Data/GTA5
PATCH_SIZE=$1
DATE=$(date +"%Y%m%d")
TIME=$(date +"%H-%M-%S")
# PATCH_METHOD=('default' 'random' 'alternate' 'division' 'reverse_division')




python3 main.py \
--dataset ${DATASET} \
--data_root ${DATAROOT} \
--total_itrs  40000 \
--batch_size 8 \
--val_interval 750 \
--transfer \
--data_aug \
--continue_training \
--ckpt /workspace/ssd0/byeongcheol/DGSS/FAMix/model_ckpt_4_20240615_11-26-06/latest_deeplabv3plus_resnet_clip_gta5_33750.pth \
--ckpts_path /workspace/ssd0/byeongcheol/DGSS/FAMix/model_ckpt_4_20240615_11-26-06 \
--path_for_stats save_dir/${DATASET}_${PATCH_SIZE}_saved_params2.pkl  \
--path_for_3stats save_dir/${DATASET}_3_saved_params2.pkl \
--path_for_4stats save_dir/${DATASET}_4_saved_params2.pkl \
--path_for_6stats save_dir/${DATASET}_6_saved_params2.pkl \
--div ${PATCH_SIZE}