PATCH_SIZE=6
DATASET=gta5
DATAROOT=/workspace/ssd0/byeongcheol/DGSS/Data/GTA5

python3 patch_PIN.py  \
    --dataset ${DATASET}  \
    --data_root ${DATAROOT}  \
    --resize_feat  \
    --save_dir /workspace/ssd0/byeongcheol/DGSS/FAMix/save_dir \
    --save_path /workspace/ssd0/byeongcheol/DGSS/FAMix/save_dir/${DATASET}_${PATCH_SIZE}_saved_params.pkl \
    --patch_size ${PATCH_SIZE}  
