python3 main.py \
--dataset gta5 \
--data_root /workspace/ssd0/byeongcheol/DGSS/Data/GTA5 \
--total_itrs  40000 \
--batch_size 8 \
--val_interval 750 \
--transfer \
--data_aug \
--ckpts_path model_ckpt \
--path_for_stats save_dir/