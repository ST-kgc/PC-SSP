# **PC-SSP**

This is the code for implementation of **Path-based Self-Supervised Pretraining for\
Knowledge Graph Completion.**

## **Library Overview**

This implementation includes the following parts:

#### Datasets:

*   WN18RR

*   FB15k-237

#### Generate paths:

    python generate_path/path_produce.py

*   2-hop paths:

Generated path corpus:

> path\_store\_train\_2hops. pt
>
> path\_store\_dev\_2hops. pt
>
> path\_store\_test\_2hops. pt

*   5-hop paths:

Generated path corpus:

> train\_paths\_5hops. pt
>
> dev\_paths\_5hops. pt
>
> test\_paths\_5hops. pt



#### **Three kinds of model learning style:**

*   P2E

*   P2P

*   Joint

## **Installation**

The starting point is to install PC-SSP framework. To this end, first, create a python 3.9 environment and install dependencies:

    virtualenv -p python3.9.1 pcssp_env
    source pcssp_env/bin/activate
    pip install -r requirements.txt

Then activate your environment:

    conda activate pcssp_env

## **Usage**

To train and evaluate a KG embedding model for the link prediction task, use the main.py:

    python main.py 

## **Citation**

If you use the codes, please cite the following paper \[1] \[2]:

## **Acknowledgement**

We refer to the code of **CPC **\[1] and **kg-reeval **\[2]. Thanks for their contributions:


\[1] Oord, Aaron van den, Yazhe Li, and Oriol Vinyals. "Representation learning with contrastive predictive coding." 2018.

\[2] Sun, Zhiqing, et al. "A re-evaluation of knowledge graph completion methods." 2019.


