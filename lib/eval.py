# Code adapted from https://github.com/locuslab/wanda/blob/main/lib/eval.py

# Import necessary modules
import time
import torch
import torch.nn as nn

# Import get_loaders function from data module within the same directory
from .data import get_loaders 

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    
    ppl_test = []
    for dataset in ["wikitext2", "ptb", "c4"]:
        _, testloader = get_loaders(
            dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer
        )
        # Print status
        print(f"evaluating on {dataset}")
        # Evaluate ppl in no grad context to avoid updating the model
        with torch.no_grad():
            ppl_test.append(eval_ppl_wikitext(model, testloader, 1, device))
    print("wikitext2:" + str(ppl_test[0]))
    print("ptb:" + str(ppl_test[1]))
    print("c4:" + str(ppl_test[2]))
    return ppl_test 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    nlls = []
    print(f"nsamples {nsamples}")

    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        j = min(i+bs, nsamples)

        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        lm_logits = model(inputs).logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    torch.cuda.empty_cache()

    return ppl.item()


def eval_zero_shot(model_name, save_model, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    from lm_eval import tasks, evaluator 
    task_manager = tasks.TaskManager()
    task_names = task_manager.match_tasks(task_list)
    print(task_names)
    model_args = f"pretrained={save_model},cache_dir=./llm_weights"
    limit = None 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={save_model},cache_dir=./llm_weights,use_accelerate=True"
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        limit=limit
    )
    
    return results['results']