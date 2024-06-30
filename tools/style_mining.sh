PATCH_SIZE=$1
DATASET=gta5
DATAROOT=/workspace/ssd0/byeongcheol/DGSS/Data/GTA5
DATE=$(date +"%Y%m%d")
TIME=$(date +"%H-%M-%S")

python3 patch_PIN.py  \
    --dataset ${DATASET}  \
    --data_root ${DATAROOT}  \
    --resize_feat  \
    --save_dir /workspace/ssd0/byeongcheol/DGSS/FAMix/save_dir \
    --save_path /workspace/ssd0/byeongcheol/DGSS/FAMix/save_dir/${DATASET}_${PATCH_SIZE}_${DATE}_${TIME}_saved_params.pkl \
    --mining_time ${DATE}_${TIME} \
    --div ${PATCH_SIZE}
