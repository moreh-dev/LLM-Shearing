MPATH=outputs/llama2_7b_pruning_scaling_doremi_to1.3b_sl2048/hf-latest_rank0
OUTPUT_PATH=eval_outputs

lm_eval \
    --model hf \
    --model_args pretrained=${MPATH},max_memory_per_gpu=40GB,max_cpu_memory=1200GB,dtype=float16 \
    --tasks hellaswag \
    --batch_size 8 \
    --device cuda \
    --num_fewshot 10 \
    --write_out \
    --output_path ${OUTPUT_PATH}/hela.json 

lm_eval \
    --model hf \
    --model_args pretrained=${MPATH},max_memory_per_gpu=40GB,max_cpu_memory=1200GB,dtype=float16 \
    --tasks arc_easy \
    --batch_size 8 \
    --device cuda \
    --write_out \
    --output_path ${OUTPUT_PATH}/arc.json 

lm_eval \
    --model hf \
    --model_args pretrained=${MPATH},max_memory_per_gpu=40GB,max_cpu_memory=1200GB,dtype=float16 \
    --tasks piqa \
    --batch_size 8 \
    --device cuda \
    --write_out \
    --output_path ${OUTPUT_PATH}/qa.json

lm_eval \
    --model hf \
    --model_args pretrained=${MPATH},max_memory_per_gpu=40GB,max_cpu_memory=1200GB,dtype=float16 \
    --tasks winogrande \
    --batch_size 8 \
    --device cuda \
    --write_out \
    --output_path ${OUTPUT_PATH}/wino.json

lm_eval \
    --model hf \
    --model_args pretrained=${MPATH},max_memory_per_gpu=40GB,max_cpu_memory=1200GB,dtype=float16 \
    --tasks arc_challenge \
    --batch_size 8 \
    --device cuda \
    --num_fewshot 25 \
    --write_out \
    --output_path ${OUTPUT_PATH}/arc.json 
