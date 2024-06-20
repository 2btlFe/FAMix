WORKDIR=$1
CKPT=${WORKDIR}/best_deeplabv3plus_resnet_clip_gta5.pth

python3 main.py \
--dataset cityscapes \
--data_root /workspace/ssd0/byeongcheol/DGSS/Data/cityscapes \
--ckpt ${CKPT} \
--test_only \
--ACDC_sub ACDC_subset_if_tested_on_ACDC \
--ckpts_path ${WORKDIR}

python3 main.py \
--dataset bdd100k \
--data_root /workspace/ssd0/byeongcheol/DGSS/Data/BDD100k \
--ckpt ${CKPT} \
--test_only \
--ACDC_sub ACDC_subset_if_tested_on_ACDC \
--ckpts_path ${WORKDIR}

python3 main.py \
--dataset mapillary \
--data_root /workspace/ssd0/byeongcheol/DGSS/Data/Mapillary \
--ckpt ${CKPT} \
--test_only \
--ACDC_sub ACDC_subset_if_tested_on_ACDC \
--ckpts_path ${WORKDIR}

python3 main.py \
--dataset synthia \
--data_root /workspace/ssd0/byeongcheol/DGSS/Data/Synthia \
--ckpt ${CKPT} \
--test_only \
--ACDC_sub ACDC_subset_if_tested_on_ACDC \
--ckpts_path ${WORKDIR}

python3 main.py \
--dataset gta5 \
--data_root /workspace/ssd0/byeongcheol/DGSS/Data/GTA5 \
--ckpt ${CKPT}  \
--test_only \
--ACDC_sub ACDC_subset_if_tested_on_ACDC \
--ckpts_path ${WORKDIR}



