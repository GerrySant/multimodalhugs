#!/bin/bash

# Activate the virtual environment
source /path/to/your/environment/bin/activate

export RUN_NAME="<run_name>"
export CONFIG_PATH='/multimodalhugs/examples/multimodal_translation/signwriting2text_translation/configs/example_config.yaml'
export OUTPUT_PATH="/path/to/your/output_directory"
export CUDA_VISIBLE_DEVICES=0

# Ejecuta el script Python y captura sus salidas
output=$(python /examples/multimodal_translation/signwriting2text_translation/example_scripts/signbankplus_training_setup.py \
    --config_path $CONFIG_PATH)

# Extrae las variables de entorno del output del script Python
export MODEL_PATH=$(echo "$output" | grep 'MODEL_PATH' | cut -d '=' -f 2)
export PROCESSOR_PATH=$(echo "$output" | grep 'PROCESSOR_PATH' | cut -d '=' -f 2)
export DATA_PATH=$(echo "$output" | grep 'DATA_PATH' | cut -d '=' -f 2)

# Visualizar las variables declaradas
echo "CONFIG_PATH: $CONFIG_PATH"
echo "OUTPUT_PATH: $OUTPUT_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "MODEL_PATH: $MODEL_PATH"
echo "PROCESSOR_PATH: $PROCESSOR_PATH"
echo "DATA_PATH: $DATA_PATH"

python /multimodalhugs/examples/multimodal_translation/run_translation.py \
    --model_name_or_path $MODEL_PATH \
    --processor_name_or_path $PROCESSOR_PATH \
    --run_name $RUN_NAME \
    --dataset_dir $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --do_train True \
    --do_eval True \
    --fp16 \
    --label_smoothing_factor 0.1 \
    --remove_unused_columns False \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --evaluation_strategy "steps" \
    --eval_steps 10 \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 3 \
    --overwrite_output_dir \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 0.001 \
    --warmup_steps 10000 \
    --max_steps 100000 \
    --predict_with_generate True \
    --lr_scheduler_type "polynomial"