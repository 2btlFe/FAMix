
PATCH_STYLE=('fg' 'cg')


for patch_style in ${PATCH_STYLE[@]}
do    
    python experiment_cossim.py \
        --patch_root /workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/Building_Patch/Test/${patch_style} \
        --fine_grained_param_root /workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/Building_Patch/save_dir/fg_saved_params.pkl \
        --coarse_grained_param_root /workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/Building_Patch/save_dir/cg_saved_params.pkl \
        --wrong_param_root /workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/Building_Patch/save_dir/w_saved_params.pkl \
        --result_root /workspace/ssd0/byeongcheol/DGSS/FAMix/Experiment_patch/Building_Patch/Test/${patch_style} \
        --style "Urban Grit style" \
        --class_name building
done
