export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="yanher/pickapic_subset" # alyssa's huggingface subset

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~24 hours / 2000 steps

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=16 \
  --max_train_steps=32 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=4 \
  --learning_rate=1e-8 --scale_lr \
  --cache_dir="/n/home00/caagnes/cache" \
  --checkpointing_steps 8 \
  --beta_dpo 5000 \
   --output_dir="tmp-sd15"