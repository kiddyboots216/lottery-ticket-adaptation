# Lottery Ticket Adaptation

## What is this repo?

This repo is the official implementation of Lottery Ticket Adaptation, described in the paper [Lottery Ticket Adaptation](https://arxiv.org/abs/2406.16797). This README, the repo, and the paper are all currently in WIP/preprint status. 

## Navigating this repo

Check out `rlaif` for the implementation of alignment, and `mergekit` for the implementation of model merging.

## A complete example

Please check out the script in "rlaif/scripts/continual_learning.sh" for a complete example with continual learning. We train models with FFT and LoRA on two tasks sequentially, showing that they forget completely. We then update disjoint sets of parameters for Task A and B, showing that this can mitigate forgetting with LoRA, but that this negatively impacts performance because the model's capacity is reduced. We finally show that LoTA, which intelligently chooses disjoint sets of parameters to update, can further mitigate forgetting without compromising performance.

Again, check out line 66 of the above script https://github.com/kiddyboots216/lottery-ticket-adaptation/blob/main/rlaif/scripts/continual_learning.sh#L66 for a complete example with LoTA, but here is a high-level overview.

First, train and save a model on the desired task by following the instructions in `rlaif`. Then, create the task vector with mergekit;

`mergekit-yaml task_vector.yaml "merge_path/" --cuda`

Finally, extract and save the mask. 

`python save_mask.py --merge_path "merge_path/"`

Now, pass this mask to the training code in `rlaif` to train LoTA models. 

# Citation
If our paper or this repository is useful for your research, you can use the following BibTeX entry:

    @article{
        panda2024lottery,
        title={Lottery Ticket Adaptation: Mitigating Destructive Interference in LLMs},
        author={Ashwinee Panda and Berivan Isik and Xiangyu Qi and Sanmi Koyejo and Tsachy Weissman and Prateek Mittal},
        year={2024},
        eprint={2406.16797},
        archivePrefix={arXiv},
        url={https://arxiv.org/abs/2406.16797}
    }
