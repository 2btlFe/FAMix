DATASET=("bdd100k" "mapillary" "synthia", "cityscapes")
DATASET_ROOT=(  "/workspace/ssd0/byeongcheol/DGSS/Data/BDD100k/images/val" \
                "/workspace/ssd0/byeongcheol/DGSS/Data/Mapillary/validation/images" \
                "/workspace/ssd0/byeongcheol/DGSS/Data/Synthia/RGB/val", \
                "/workspace/ssd0/byeongcheol/DGSS/Data/cityscapes/leftImg8bit/val")


CKPT="/workspace/ssd0/byeongcheol/DGSS/FAMix/model_ckpt_6_0612/best_deeplabv3plus_resnet_clip_gta5.pth"
SAVE_DIR="/workspace/ssd0/byeongcheol/DGSS/FAMix/results"

for i in ${!DATASET[@]}; do
    python3 predict.py \
        --ckpt $CKPT \
        --save_val_results_to ${SAVE_DIR}/${DATASET[$i]} \
        --dataset_root ${DATASET_ROOT[$i]}
done