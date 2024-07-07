FILENAME=TSNE_result
DATE=$(date +"%Y%m%d")
TIME=$(date +"%H-%M-%S")

python3 experiment_CLIP.py \
    --fine_grained_data_root /workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/Fine_grained_cls \
    --coarse_grained_data_root /workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/Coarse_grained_cls \
    --wrong_data_root /workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/Wrong_cls \
    --save_dir /workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/save_dir \
    --save_path /workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/save_dir/${FILENAME}_${DATE}_${TIME}_saved_params.pkl \