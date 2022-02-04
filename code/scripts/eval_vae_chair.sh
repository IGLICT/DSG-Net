
CUDA_VISIBLE_DEVICES=0 python eval_recon_sn.py \
  --exp_name 'exp_ae_Chair' \
  --test_dataset 'test_no_other_less_than_10_parts.txt' \
  --model_epoch 1999
