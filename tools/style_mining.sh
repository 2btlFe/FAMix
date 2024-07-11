#!/bin/bash

PATCH_SIZE=$1
USE_DIVIDE_CONQUER=$2  # 두 번째 인자 추가
DATASET=gta5
DATAROOT=/workspace/ssd0/byeongcheol/DGSS/Data/GTA5
DATE=$(date +"%Y%m%d")
TIME=$(date +"%H-%M-%S")

# 기본 명령어 구성
CMD="python3 patch_PIN.py \
    --dataset ${DATASET} \
    --data_root ${DATAROOT} \
    --resize_feat \
    --save_dir /workspace/ssd0/byeongcheol/DGSS/FAMix/save_dir \
    --save_path /workspace/ssd0/byeongcheol/DGSS/FAMix/save_dir/${DATASET}_${PATCH_SIZE}_${DATE}_${TIME}_saved_params.pkl \
    --mining_time ${DATE}_${TIME} \
    --div ${PATCH_SIZE}"

# 두 번째 인자가 존재하고 "true"인 경우에만 --divide_conquer 옵션 추가
if [ ! -z "$USE_DIVIDE_CONQUER" ] && [ "$USE_DIVIDE_CONQUER" = "true" ]; then
    CMD="$CMD --divide_conquer"
fi

# 최종 명령어 실행
eval $CMD