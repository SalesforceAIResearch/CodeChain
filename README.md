# CodeChain: Towards Modular Code Generation Through Chain of Self-revisions with Representative Sub-modules

This is the official code for the paper "CodeChain: Towards Modular Code Generation Through Chain of Self-revisions with Representative Sub-modules". Check out our [blog](https://blog.salesforceairesearch.com/codechain/) and [paper](https://arxiv.org/abs/2310.08992) for more details. 

Authors: [Hung Le](https://scholar.google.com/citations?user=jnYI1UgAAAAJ&hl=en), [Hailin Chen](https://aclanthology.org/people/h/hailin-chen/), [Amrita Saha](https://scholar.google.co.uk/citations?user=3Zb5Y2YAAAAJ&hl=en), [Akash Gokul](https://scholar.google.com/citations?user=MYRUJkUAAAAJ&hl=en), [Doyen Sahoo](https://scholar.google.com.sg/citations?user=A61jJD4AAAAJ&hl=en), [Shafiq Joty](https://scholar.google.com/citations?user=hR249csAAAAJ&hl=en) 

### Contents:
* [x] [CodeChain Overview](#codechain-overview)
* [x] [Installation](#installation)
* [x] [Datasets](#datasets)
* [x] [Generate code with CoT prompting](#generate-code-with-cot-prompting)
* [x] [Self-Revise generated code](#self-revise-generated-code)
* [x] [Evaluating Generated Code](#evaluating-generated-code)
* [x] [Citation](#citation)

## CodeChain Overview  
<p align="center">
<img src="images/code llm agent - clustering_v2.svg" width="100%" />
</p>
A pretrained LLM is first instructed with chain-of-thought prompting to generate a set of modularised solutions. Generated sub-modules are then extracted from potentially correct solutions and grouped into different semantic clusters. The cluster centroids are selected as representative sub-modules to condition the next self-revision round. The model is instructed to reuse or adapt these modules into its revised solutions.

## Installation

## Datasets

## Generate code with CoT prompting 

## Self-Revise generated code 

## Evaluating generated code 

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


