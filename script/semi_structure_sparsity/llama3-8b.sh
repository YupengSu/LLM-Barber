# Description: Script to semi-structure prune llama3-8b model.

# llama3-8b magnitude 2:4/4:8
python main.py \
    --model meta-llama/Meta-Llama-3-8B \
    --prune_method magnitude \
    --sparsity_ratio 0.5 --sparsity_type 2:4  \
    --save_model /path/to/save/model --save_ppl /path/to/save/ppl  \
    --delete

# llama3-8b magnitude 2:4/4:8 barber
python main.py \
    --model meta-llama/Meta-Llama-3-8B \
    --prune_method magnitude \
    --sparsity_ratio 0.5 --sparsity_type 2:4  \
    --prune_barber --prune_granularity block --threshold 0.1 \
    --save_model /path/to/save/model --save_ppl /path/to/save/ppl  \
    --delete

# llama3-8b wanda 2:4/4:8
python main.py \
    --model meta-llama/Meta-Llama-3-8B \
    --prune_method wanda \
    --sparsity_ratio 0.5 --sparsity_type 2:4  \
    --save_model /path/to/save/model --save_ppl /path/to/save/ppl  \
    --delete

# llama3-8b wanda 2:4/4:8 barber
python main.py \
    --model meta-llama/Meta-Llama-3-8B \
    --prune_method wanda \
    --sparsity_ratio 0.5 --sparsity_type 2:4  \
    --prune_barber --prune_granularity output1 --threshold 0.01 \
    --save_model /path/to/save/model --save_ppl /path/to/save/ppl  \
    --delete
