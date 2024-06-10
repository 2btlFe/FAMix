python3 main.py \
--dataset cityscapes \
--data_root /workspace/ssd0/byeongcheol/DGSS/Data/cityscapes \
--ckpt /workspace/ssd0/byeongcheol/DGSS/FAMix/model_ckpt/best_deeplabv3plus_resnet_clip_gta5.pth \
--test_only \
--ACDC_sub ACDC_subset_if_tested_on_ACDC

python3 main.py \
--dataset bdd100k \
--data_root /workspace/ssd0/byeongcheol/DGSS/Data/BDD100k \
--ckpt /workspace/ssd0/byeongcheol/DGSS/FAMix/model_ckpt/best_deeplabv3plus_resnet_clip_gta5.pth \
--test_only \
--ACDC_sub ACDC_subset_if_tested_on_ACDC

python3 main.py \
--dataset mapillary \
--data_root /workspace/ssd0/byeongcheol/DGSS/Data/Mapillary \
--ckpt /workspace/ssd0/byeongcheol/DGSS/FAMix/model_ckpt/best_deeplabv3plus_resnet_clip_gta5.pth \
--test_only \
--ACDC_sub ACDC_subset_if_tested_on_ACDC

python3 main.py \
--dataset synthia \
--data_root /workspace/ssd0/byeongcheol/DGSS/Data/Synthia \
--ckpt /workspace/ssd0/byeongcheol/DGSS/FAMix/model_ckpt/best_deeplabv3plus_resnet_clip_gta5.pth \
--test_only \
--ACDC_sub ACDC_subset_if_tested_on_ACDC

python3 main.py \
--dataset gtav\
--data_root /workspace/ssd0/byeongcheol/DGSS/Data/GTA5\
--ckpt /workspace/ssd0/byeongcheol/DGSS/FAMix/model_ckpt/best_deeplabv3plus_resnet_clip_gta5.pth \
--test_only \
--ACDC_sub ACDC_subset_if_tested_on_ACDC



