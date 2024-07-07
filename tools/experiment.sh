DATE=$(date +"%Y%m%d")
TIME=$(date +"%H-%M-%S")

TYPE=('fg' 'cg' 'w')

for i in ${TYPE[@]}
do
    python3 experiment.py \
    --data_root /workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/Building_Patch/Train/${i} \
    --resize_feat \
    --save_dir /workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/Building_Patch/save_dir \
    --save_path /workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/Building_Patch/save_dir/${i}_saved_params.pkl
done
