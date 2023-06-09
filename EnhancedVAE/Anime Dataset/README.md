# Enhancing VAEs for Collaborative Filtering: Flexible Priors & Gating Mechanisms
This is the source code used for experiments for the paper published in RecSys '19:  
"Enhancing VAEs for Collaborative Filtering: Flexible Priors & Gating Mechanisms"    
(arxiv preprint: https://arxiv.org/abs/1911.00936, ACM DL: https://dl.acm.org/citation.cfm?id=3347015)

An example of training a hierarchical VampPrior VAE for Collaborative Filtering on the Netflix dataset is as follows:
`python experiment.py  --dataset_name="ml20m" --max_beta=0.13 --model_name="hvamp1" --gated --input_type="binary" --z1_size=200 --z2_size=100 --hidden_size=300 --num_layers=1 --test_batch_size=100`

### Requirements
Requirements are listed in `requirements.txt`

### Datasets
Datasets should be downloaded and preprocessed according to instructions in `./datasets/`

### Acknowledgements
Many of our code is reformulated based on https://github.com/dawenl/vae_cf and https://github.com/jmtomczak/vae_vampprior
