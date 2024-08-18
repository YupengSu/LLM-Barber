import argparse
import os 
import numpy as np
import pandas as pd
import torch
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights", seqlen=2048):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = seqlen 
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "wandd", "sparsegpt"])
    parser.add_argument("--prune_barber", action="store_true", help="whether to update the model after pruning")
    parser.add_argument("--prune_metric", type=str, default="WD", choices=["WD", "W", "D"])
    parser.add_argument("--prune_granularity", type=str, default="output1", choices=["layer", "input1", "output1", "block"])
    parser.add_argument("--threshold", type=float, default=0.01, help="threshold for reconstruction")
    parser.add_argument("--retain_reconstruction", action="store_true", help="retain the reconstruction weights of initialization method")
    parser.add_argument("--datasets_cache_dir", default=None, type=str )
    parser.add_argument("--cache_dir", default="llm_weights", type=str )

    parser.add_argument('--save_data', type=str, default=None, help='Path to save data.')

    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--save_ppl', type=str, default=None, help='Path to save perplexity evaluation results.')
    parser.add_argument("--save_zeroshot", type=str, default=None, help="Path to save zero-shot evaluation results")
    parser.add_argument("--delete", action="store_true", help="delete the model after evaluation")
    
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Set cache directory for Huggingface datasets
    os.environ['HF_DATASETS_CACHE'] = os.path.join(os.getcwd(), 'datasets_cache')

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        assert args.prune_granularity == "output1", "structured sparsity only supported for output1 granularity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {model_name}")
    sparse_model = get_llm(args.model, args.cache_dir, args.seqlen)
    dense_model = copy.deepcopy(sparse_model)
    sparse_model.eval()
    dense_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = sparse_model.hf_device_map["lm_head"]
    print("use device ", device)

    if "llama" in args.model:
        from lib.llama import prune_wanda, prune_wandd, prune_magnitude, prune_sparsegpt, check_sparsity, prune_barber
    elif "opt" in args.model:
        from lib.opt import prune_wanda, prune_magnitude, prune_sparsegpt, check_sparsity, prune_barber

    if args.sparsity_ratio != 0:
        print("pruning starts, prune method: ", args.prune_method)
        if args.prune_method == "wanda":
            prune_wanda(args, sparse_model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "wandd":
            prune_wandd(args, sparse_model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, sparse_model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, sparse_model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        print("pruning ends")
        if args.prune_barber:
            print("pruning update starts, granularity: ", args.prune_granularity, ", metric: ", args.prune_metric, ", threshold: ", args.threshold)
            prune_barber(args, dense_model, sparse_model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            del dense_model
            torch.cuda.empty_cache()
            print("pruning update ends")  

    print("*"*30)
    sparsity_ratio = 0
    sparsity_ratio = check_sparsity(sparse_model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)

    if args.save_model:
        sparse_model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

    if args.save_ppl:
        from lib.eval import eval_ppl
        ppl_test = eval_ppl(args, sparse_model, tokenizer, device)

        if not os.path.exists(args.save_ppl):
            os.makedirs(args.save_ppl)
        save_filepath = os.path.join(args.save_ppl, f"log_{args.prune_method}_{args.threshold}.txt")
        with open(save_filepath, "w") as f:
            print(f"{'method':<15}{'actual_sparsity':<15}{'wikitest2':<15}{'ptb':<15}{'c4':<15}", file=f, flush=True)
            print(f"{args.prune_method:<15}{sparsity_ratio:<15.4f}{ppl_test[0]:<15.4f}{ppl_test[1]:<15.4f}{ppl_test[2]:<15.4f}", file=f, flush=True)
            
    if args.save_zeroshot:
        from lib.eval import eval_zero_shot
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, args.save_model, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

        df = pd.DataFrame(results).T
        df_str = df.to_string()

        if not os.path.exists(args.save_zeroshot):
            os.makedirs(args.save_zeroshot)
        save_filepath = os.path.join(args.save_zeroshot, f"log_{args.prune_method}_{args.threshold}.txt")
        with open(save_filepath, "w") as f:
            print(df_str, file=f, flush=True)

    if args.delete:
        os.system(f"rm -rf {args.save_model}")

if __name__ == '__main__':
    main()