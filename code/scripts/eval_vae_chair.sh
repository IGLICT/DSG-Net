
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --exp_name 'box_vae_chair' \
  --test_dataset 'test_no_other_less_than_10_parts.txt' \
  --model_epoch 0


