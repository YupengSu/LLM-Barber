# Description: Script to unstructured prune llama-7b model

# llama-7b dense
python main.py \
    --model huggyllama/llama-7b \
    --prune_method magnitude \
    --sparsity_ratio 0 --sparsity_type unstructured  \
    --save_model /path/to/save/model --save_ppl /path/to/save/ppl --save_zeroshot /path/to/save/zeroshot \
    --delete

# llama-7b magnitude unstructured 50%
python main.py \
    --model huggyllama/llama-7b \
    --prune_method magnitude \
    --sparsity_ratio 0.5 --sparsity_type unstructured  \
    --save_model /path/to/save/model --save_ppl /path/to/save/ppl --save_zeroshot /path/to/save/zeroshot \
    --delete

# llama-7b magnitude unstructured 50% barber
python main.py \
    --model huggyllama/llama-7b \
    --prune_method magnitude \
    --sparsity_ratio 0.5 --sparsity_type unstructured  \
    --prune_barber --prune_granularity block --threshold 0.1 \
    --save_model /path/to/save/model --save_ppl /path/to/save/ppl --save_zeroshot /path/to/save/zeroshot \
    --delete

# llama-7b sparsegpt unstructured 50% with weight reconstruction (default)
python main.py \
    --model huggyllama/llama-7b \
    --prune_method sparsegpt \
    --sparsity_ratio 0.5 --sparsity_type unstructured  \
    --save_model /path/to/save/model --save_ppl /path/to/save/ppl --save_zeroshot /path/to/save/zeroshot \
    --delete

# llama-7b sparsegpt unstructured 50% without weight reconstruction
python main.py \
    --model huggyllama/llama-7b \
    --prune_method sparsegpt \
    --sparsity_ratio 0.5 --sparsity_type unstructured  \
    --prune_barber --prune_granularity block --threshold 0 \
    --save_model /path/to/save/model --save_ppl /path/to/save/ppl --save_zeroshot /path/to/save/zeroshot \
    --delete

# llama-7b sparsegpt unstructured 50% without weight reconstruction barber
python main.py \
    --model huggyllama/llama-7b \
    --prune_method sparsegpt \
    --sparsity_ratio 0.5 --sparsity_type unstructured  \
    --prune_barber --prune_granularity layer --threshold 0.1 \
    --save_model /path/to/save/model --save_ppl /path/to/save/ppl --save_zeroshot /path/to/save/zeroshot \
    --delete

# llama-7b wanda unstructured 50%
python main.py \
    --model huggyllama/llama-7b \
    --prune_method wanda \
    --sparsity_ratio 0.5 --sparsity_type unstructured  \
    --save_model /path/to/save/model --save_ppl /path/to/save/ppl --save_zeroshot /path/to/save/zeroshot \
    --delete

# llama-7b wanda unstructured 50% barber
python main.py \
    --model huggyllama/llama-7b \
    --prune_method wanda \
    --sparsity_ratio 0.5 --sparsity_type unstructured  \
    --prune_barber --prune_granularity output1 --threshold 0.01 \
    --save_model /path/to/save/model --save_ppl /path/to/save/ppl --save_zeroshot /path/to/save/zeroshot \
    --delete
