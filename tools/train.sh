DATASET=gta5
DATAROOT=/workspace/ssd0/byeongcheol/DGSS/Data/GTA5
PATCH_SIZE=$1
DATE=$(date +"%Y%m%d")
TIME=$(date +"%H-%M-%S")
# PATCH_METHOD=('default' 'random' 'alternate' 'division' 'reverse_division')

# Fusion
# PATCH_METHOD='fusion' 'fusion_mlp' 'fusion_adv' 

PATCH_METHOD=$2
NUM_LAYER=$3

python3 main.py \
--dataset ${DATASET} \
--data_root ${DATAROOT} \
--total_itrs  40000 \
--batch_size 8 \
--val_interval 750 \
--transfer \
--data_aug \
--ckpts_path model_ckpt_${PATCH_SIZE}_${PATCH_METHOD}_${DATE}_${TIME} \
--path_for_stats save_dir/${DATASET}_${PATCH_SIZE}_saved_params2.pkl  \
--path_for_3stats save_dir/${DATASET}_3_saved_params2.pkl \
--path_for_4stats save_dir/${DATASET}_4_saved_params2.pkl \
--path_for_6stats save_dir/${DATASET}_6_saved_params2.pkl \
--patch_method ${PATCH_METHOD} \
--num_layer ${NUM_LAYER} \
--div ${PATCH_SIZE}