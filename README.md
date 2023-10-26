# CodeChain: Towards Modular Code Generation Through Chain of Self-revisions with Representative Sub-modules

This is the official code for the paper "CodeChain: Towards Modular Code Generation Through Chain of Self-revisions with Representative Sub-modules". Check out our [blog](https://blog.salesforceairesearch.com/codechain/) and [paper](https://arxiv.org/abs/2310.08992) for more details. 

Authors: [Hung Le](https://scholar.google.com/citations?user=jnYI1UgAAAAJ&hl=en), [Hailin Chen](https://aclanthology.org/people/h/hailin-chen/), [Amrita Saha](https://scholar.google.co.uk/citations?user=3Zb5Y2YAAAAJ&hl=en), [Akash Gokul](https://scholar.google.com/citations?user=MYRUJkUAAAAJ&hl=en), [Doyen Sahoo](https://scholar.google.com.sg/citations?user=A61jJD4AAAAJ&hl=en), [Shafiq Joty](https://scholar.google.com/citations?user=hR249csAAAAJ&hl=en) 

### Contents:
* [x] [CodeChain Overview](#codechain-overview)
* [x] [Setup Datasets and Models](#setup-datasets-and-models)
* [x] [Generate and evaluate code](#generate-and-evaluate-code)
* [x] [Citation](#citation)
* [x] [License](#license)

## CodeChain Overview  
<p align="center">
<img src="images/code llm agent - clustering_v2.svg" width="100%" />
</p>
A pretrained LLM is first instructed with chain-of-thought prompting to generate a set of modularised solutions. Generated sub-modules are then extracted from potentially correct solutions and grouped into different semantic clusters. The cluster centroids are selected as representative sub-modules to condition the next self-revision round. The model is instructed to reuse or adapt these modules into its revised solutions.

## Setup Datasets and Models

If you have conda ready, just run to install dependencies:

```
./install.sh
```

### Datasets 

We use the main benchmark [APPS](https://huggingface.co/datasets/codeparrot/apps) in our experiments. Please follow the instructions to donwload the datasets.

To run the code, place the dataset into the `data` folder. For APPS, parition the data into sub-folder by the difficulty level e.g. competition, interview, introductory. We uploaded the `data/mini_val` for the APPS validation split we used in the paper. 

We also uploaded the example test cases we extracted from the problem description on APPS. See `*_example_tests.pkl` files in the `data` folder. 

### Models 

Put a file `openaiapikey.txt` for querying GPT models. The file should just contain your OpenAI api key. By default, our code is compatible to run with GPT3.5 or GPT4. To run on open source models, we recommend to set up following the [VLLM framework](https://github.com/vllm-project/vllm) which can speed the generation significantly. The inference code on this framework is similar to how we run on a GPT model in this repo. 


## Generate and Evaluate Code
### Direct Generation

```
./scripts/run_direct_gen.sh
```
 
Run this script to direct generate code in a normal setting (without CoT prompting and self-revision). 
This process uses the prompt `prompts/direct_gen.txt` to instruct the model to generate code. 
Refer to `src/configs/config_generate.py` for a list of parameters to be configured. 
The output code is saved into a sub-folder under `outputs` folder. 

### Code Chain 

```
./scripts/run_codechain.sh
```

Run this script to generate and revise code iteratively in with CodeChain framework. 
This process uses the prompt `prompts/codechain_gen.txt` to instruct the model to generate code with CoT prompting and `prompts/codechain_revise.txt` to instruct the model to self-revise the code with selected represenatative sub-modules. 
This process contains 4 major sub-processes: generating code, evaluating code with example test cases, post-processing, and clustering. 
Refer to the corresponding `src/configs/config_<sub-process>.py` for a list of parameters to be configured. 
All outputs (including code, processed code, extracted modules and clustered samples) are saved with a template of the `outputs/<exp_name>*` (e.g. `outputs/<exp_name>_data.csv`, `outputs/<exp_name>_xxclusters.csv`).


### Evaluation 
```
./scripts/evaluate_codechain.sh
```

Finally, to evaluate the code on hidden test cases by pass@1, run the evaluation script. This will evaluate the generated code on all generation and revision rounds. 


## Citation   
If you find the paper or the source code useful to your projects, please cite the following bibtex: 

<pre>
@misc{le2023codechain,
      title={CodeChain: Towards Modular Code Generation Through Chain of Self-revisions with Representative Sub-modules}, 
      author={Hung Le and Hailin Chen and Amrita Saha and Akash Gokul and Doyen Sahoo and Shafiq Joty},
      year={2023},
      eprint={2310.08992},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
</pre>

## License 

The code is released under Apache License Version 2.0 - see `LICENSE.txt` for details.

The code is developed from other open source projects: including [APPS](https://github.com/hendrycks/apps/), [BigCode Encoder](https://github.com/bigcode-project/bigcode-encoder), and [BigCode Evaluation Harness](https://github.com/bigcode-project/bigcode-evaluation-harness). We thank the original contributors of these works for open-sourcing their valuable source codes. 




