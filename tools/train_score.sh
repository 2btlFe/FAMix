DATASET=gta5
DATAROOT=/workspace/ssd0/byeongcheol/DGSS/Data/GTA5
PATCH_SIZE=$1
DATE=$(date +"%Y%m%d")
TIME=$(date +"%H-%M-%S")
# PATCH_METHOD=('default' 'random' 'alternate' 'division' 'reverse_division')

# Fusion
# PATCH_METHOD='fusion' 'fusion_mlp' 'fusion_adv' 

PATCH_METHOD=$2

python3 main.py \
--dataset ${DATASET} \
--data_root ${DATAROOT} \
--total_itrs  40000 \
--batch_size 8 \
--val_interval 750 \
--transfer \
--data_aug \
--ckpts_path model_ckpt_${PATCH_SIZE}_${PATCH_METHOD}_${DATE}_${TIME} \
--path_for_stats save_dir/${DATASET}_${PATCH_SIZE}_saved_params_score.pkl  \
--patch_method ${PATCH_METHOD} \
--div ${PATCH_SIZE} \