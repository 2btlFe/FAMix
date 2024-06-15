DATASET=gta5
DATAROOT=/workspace/ssd0/byeongcheol/DGSS/Data/GTA5
PATCH_SIZE=$1
DATE=$(date +"%Y%m%d")
TIME=$(date +"%H-%M-%S")



python3 main.py \
--dataset ${DATASET} \
--data_root ${DATAROOT} \
--total_itrs  40000 \
--batch_size 8 \
--val_interval 750 \
--transfer \
--data_aug \
--ckpts_path model_ckpt_${PATCH_SIZE}_${DATE}_${TIME} \
--path_for_stats save_dir/${DATASET}_${PATCH_SIZE}_saved_params2.pkl  \
--div ${PATCH_SIZE}