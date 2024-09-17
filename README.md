# ONE-STAGE PROMPT-BASED CONTINUAL LEARNING
Pytorch implementation code for ONE-STAGE PROMPT-BASED CONTINUAL LEARNING, ECCV 2024.

Paper link: https://arxiv.org/abs/2402.16189


## Abstract 
Prompt-based Continual Learning (PCL) has gained considerable attention as a promising continual learning solution as it achieves state-of-the-art performance while preventing privacy violation and memory overhead issues. Nonetheless, existing PCL approaches face significant computational burdens because of two Vision Transformer (ViT) feed-forward stages; one is for the query
ViT that generates a prompt query to select prompts inside a prompt pool; the other one is a backbone ViT that mixes information between selected prompts and image tokens.  
To address this, we introduce a one-stage PCL framework by directly using the intermediate layer's token embedding as a prompt query. This design removes the need for an additional feed-forward stage for query ViT, resulting in $\sim 50\%$ computational cost reduction for both training and inference with marginal accuracy drop ($\le 1\%$). We further introduce a Query-Pool Regularization (QR) loss that regulates the relationship between the prompt query and the prompt pool to improve representation power. The QR loss is only applied during training time, so there is no computational overhead at inference from the QR loss. With the QR loss, our approach maintains $\sim 50\%$ computational cost reduction during inference as well as outperforms the prior two-stage PCL methods by $\sim 1.4\%$ on public class-incremental continual learning benchmarks including CIFAR-100 and ImageNet-R.

## Prerequisites
* set up conda
```
conda create --name osprompt python=3.8
conda activate osprompt
```
* Install packages
```
sh install_requirements.sh
``` 

## Basic Experiments


* Run CIFAR-100 Training

```
sh cifar-100.sh
```

* Run ImageNet-R Training (short:5-task / 10-task / long: 20-task)

```
sh imagenet-r_short.sh
sh imagenet-r.sh
sh imagenet-r_long.sh
```

## Change Ref ViT model (OS-prompt++)

* poolformer

```
python -u run.py --config $CONFIG --gpuid $1 --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name OSPrompt \
    --prompt_param 100 8 1e-4 --query poolformer \
    --log_dir ${OUTDIR}/os-p
``` 

* swin-v2

```
python -u run.py --config $CONFIG --gpuid $1 --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name OSPrompt \
    --prompt_param 100 8 1e-4 --query swin \
    --log_dir ${OUTDIR}/os-p
``` 

* caformer

```
python -u run.py --config $CONFIG --gpuid $1 --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name OSPrompt \
    --prompt_param 100 8 1e-4 --query caformer \
    --log_dir ${OUTDIR}/os-p
``` 


## Acknowledgement 
This code is based on: https://github.com/GT-RIPL/CODA-Prompt
 

