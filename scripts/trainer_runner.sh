#!/bin/bash

watermarks=("Unigram")
export CUDA_VISIBLE_DEVICES=2,5,6,7
# Loop over each value in the list
for wm in "${watermarks[@]}"
do
  echo "Running with Watermark: $wm"
  for model in "meta-llama/Llama-3.2-3B-Instruct" "Qwen/Qwen2.5-3B-Instruct"
    do
    # Run the accelerate launch command with the current wm value
    accelerate launch scripts/dpo_train.py \
      --dataset_name="data/${wm}_new/$model" \
      --model_name_or_path="$model" \
      --per_device_train_batch_size 8 \
      --learning_rate 5e-4 \
      --gradient_accumulation_steps 1 \
      --logging_steps 1 \
      --eval_steps 50 \
      --output_dir="models/${wm}_new/$model" \
      --warmup_ratio 0.4 \
      --report_to wandb \
      --run_name "$wm/$model" \
      --num_train_epochs 1 \
      --seed 42 \
      --overwrite_output_dir \
      --logging_first_step \
      --no_remove_unused_columns \
      --use_peft \
      --lora_r=32 \
      --lora_alpha=16 \
      --gradient_checkpointing=true \
      --max_length=1024 \
      --max_prompt_length=768 \
      --bf16 \
      --eval_strategy=steps \
      --save_strategy=steps \
      --save_steps=50
    done

  echo "Finished running with Watermark: $wm"
done
