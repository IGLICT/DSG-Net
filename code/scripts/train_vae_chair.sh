CUDA_VISIBLE_DEVICES=1 python ./train.py \
  --exp_name 'box_vae_chair' \
  --category 'Chair' \
  --data_path '../data/partnetdata/Chair_dgeo' \
  --pg_dir_train '../data/part_trees/Chair_all_no_other_less_than_10_parts-train' \
  --pg_dir_test '../data/part_trees/Chair_all_no_other_less_than_10_parts-test' \
  --epochs 3000 \
  --model_version 'model_dsmnet_full' \
  --loss_weight_box 100.0 \
  --checkpoint_interval 20000 \
  --lr_decay_every 50000 \
  --loss_weight_kldiv 0.001



