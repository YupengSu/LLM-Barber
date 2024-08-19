# LLM-Barber
Code for the paper "LLM-Barber: Block-Aware Rebuilder for Sparsity Mask in One-Shot for Large Language Models".

**LLM-Barber: Block-Aware Rebuilder for Sparsity Mask in One-Shot for Large Language Models** 

Yupeng Su*, Ziyi Guan*, Xiaoqun Liu, Tianlai Jin, Dongkuan Wu, Graziano Chesi, Ngai Wong, Hao Yu(* indicates equal contribution)

![Figure 1a](img/figure1a.png)
LLM-Barber integrates pruning across both Self-Attention and MLP block, mitigates error accumulation, as evidenced by the
lighter orange arrows, facilitating global optimization and
improved model performance.
![Figure 1b](img/figure1b.png)
LLM-Barber identifies weights that, although initially non-salient without a
sparsity mask, gain significance in post-pruning. 

## Setup
To install, follow the instructions in the [INSTALL.md](INSTALL.md) file.

## Usage
The [scripts](./scripts/) directory houses all Bash commands necessary to reproduce the primary findings presented in our paper.

The following command demonstrates pruning LLaMA-7B using LLM-Barber to achieve 50% unstructured sparsity.

```bash
python main.py \
    --model huggyllama/llama-7b \
    --prune_method magnitude \
    --sparsity_ratio 0.5 --sparsity_type unstructured  \
    --save_model /path/to/save/model --save_ppl /path/to/save/ppl --save_zeroshot /path/to/save/zeroshot \
    --delete
```

Here's an overview of the arguments used in the command:

* **`--model huggyllama/llama-7b`**: Specifies the LLaMA model to use from the Hugging Face model hub.
* **`--prune_method magnitude`**: Selects the pruning method, here it's "magnitude" pruning.
* **`--sparsity_ratio 0.5`**: Sets the sparsity ratio to 0.5, meaning 50% of the weights will be pruned.
* **`--sparsity_type unstructured`**: Specifies the type of sparsity as "unstructured".
* **`--save_model /path/to/save/model`**: Defines the directory where the pruned model will be saved.
* **`--save_ppl /path/to/save/ppl`**: Defines the directory where the perplexity results will be saved.
* **`--save_zeroshot /path/to/save/zeroshot`**: Defines the directory where the zero-shot results will be saved.
* **`--delete`**: This flag indicates that the pruned model should be deleted after the experiment. 

This command will run the `main.py` script with the specified arguments, pruning the "huggyllama/llama-7b" model using magnitude pruning with a sparsity ratio of 0.5 and unstructured sparsity. The results will be saved to the specified directories, and the pruned model will be deleted after the experiment.

To implement structured N:M sparsity, set the --sparsity_type argument to either "2:4" or "4:8". An example command is provided below.
```bash
python main.py \
    --model huggyllama/llama-7b \
    --prune_method magnitude \
    --sparsity_ratio 0.5 --sparsity_type 2:4  \
    --save_model /path/to/save/model --save_ppl /path/to/save/ppl  \
    --delete
```



## Acknowledgement
This repository is build upon the [Wanda](https://github.com/locuslab/wanda) and [SparseGPT](https://github.com/IST-DASLab/sparsegpt) repository.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.


