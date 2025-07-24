#%% the NAN happened in the training process of transcoder of blocks of llada


# Command

```bash
 ...
 torchrun --nproc_per_node 4 -m sparsify ${lm_path} ${dataset_path} \
 ...
```

# Code
```python
# remote/sparsify/sparsify/sparse_coder.py
 def forward(
     self, x: Tensor, y: Union[Tensor] = None, *, dead_mask: Union[Tensor] = None
 ) -> ForwardOutput:
     top_acts, top_indices, pre_acts = self.encode(x)
     # If we aren't given a distinct target, we're autoencoding
     if y is None:
         y = x
     # Decode
     sae_out = self.decode(top_acts, top_indices)
     # Compute the residual
     e = y - sae_out
     # Used as a denominator for putting everything on a reasonable scale
     total_variance = (y - y.mean(0)).pow(2).sum()
     l2_loss = e.pow(2).sum()
     fvu = l2_loss / total_variance
```

# Probelm details
```python
# fvu inf value happped in the start of the training process
{'step': 9,  'fvu/transformer.blocks.0': 0.090}
{'step': 9,  'fvu/transformer.blocks.0': 0.090}
{'step': 19, 'fvu/transformer.blocks.0': 0.059}
{'step': 19, 'fvu/transformer.blocks.0': 0.059}
{'step': 29, 'fvu/transformer.blocks.0': 0.052}
{'step': 29, 'fvu/transformer.blocks.0': 0.052}
{'step': 39, 'fvu/transformer.blocks.0': inf, }
{'step': 39, 'fvu/transformer.blocks.0': inf, }
{'step': 49, 'fvu/transformer.blocks.0': inf, }
{'step': 49, 'fvu/transformer.blocks.0': inf, }
{'step': 59, 'fvu/transformer.blocks.0': inf, }
{'step': 59, 'fvu/transformer.blocks.0': inf, }
{'step': 69, 'fvu/transformer.blocks.0': 0.054}
{'step': 69, 'fvu/transformer.blocks.0': 0.054}
{'step': 79, 'fvu/transformer.blocks.0': 0.041}
{'step': 79, 'fvu/transformer.blocks.0': 0.041}
{'step': 89, 'fvu/transformer.blocks.0': inf, }
{'step': 89, 'fvu/transformer.blocks.0': inf, }
{'step': 99, 'fvu/transformer.blocks.0': 0.041}
{'step': 99, 'fvu/transformer.blocks.0': 0.041}
{'step': 109, 'fvu/transformer.blocks.0': 0.04}
{'step': 109, 'fvu/transformer.blocks.0': 0.04}
{'step': 119, 'fvu/transformer.blocks.0': 0.05}
{'step': 119, 'fvu/transformer.blocks.0': 0.05}
{'step': 129, 'fvu/transformer.blocks.0': inf,}
{'step': 129, 'fvu/transformer.blocks.0': inf,}
{'step': 139, 'fvu/transformer.blocks.0': inf,}
{'step': 139, 'fvu/transformer.blocks.0': inf,}
{'step': 149, 'fvu/transformer.blocks.0': inf,}
{'step': 149, 'fvu/transformer.blocks.0': inf,}
{'step': 159, 'fvu/transformer.blocks.0': 0.04}
{'step': 159, 'fvu/transformer.blocks.0': 0.04}
{'step': 169, 'fvu/transformer.blocks.0': 0.06}
{'step': 169, 'fvu/transformer.blocks.0': 0.06}
{'step': 179, 'fvu/transformer.blocks.0': 0.04}
{'step': 179, 'fvu/transformer.blocks.0': 0.04}
```


# How to find where and why the inf happened more detail?

## suppositions
1. forward process
suppose 1: total variance is the problem
suppose 2: some weight is the problem
suppose 3 : denominator is the problem   


2. backward process
suppose 1: gradient is the problem


## actions
add inf hook function to see where the inf happened
add inf environment to see when the inf happened
add eps to the denominator to avoid inf


## command 1
```bash
export WANDB_ENTITY="kevin__"
export WANDB_PROJECT="sparsify"
export WANDB_RUN_NAME="debug_sparsify_llama38b_latents32768_bs64_wandb_bxy"
export WANDB_UPLOAD="offline"
export WANDB_TAGS="sparsify,llama3-8b,smollm2,expansion_factor-32,bxy"

export SPARSIFY_DISABLE_TRITON=1

port=$((29500 + RANDOM % 1000))
lm_path="/home/bingxing2/home/scx8aiv/.cache/huggingface/hub/models--GSAI-ML--LLaDA-1.5/snapshots/43716255dd5ce01192bcfc073a2557e969a2f271"
dataset_path="/home/bingxing2/home/scx8aiv/.cache/huggingface/hub/datasets--EleutherAI--SmolLM2-135M-10B/snapshots/4c87b0da77c1937f3a08b763cf899c46f39b8b3b"

CUDA_LAUNCH_BLOCKING=1
TORCH_NAN_INF_CHECK=1
cd remote/sparsify
torchrun --nproc_per_node 4 -m sparsify ${lm_path} ${dataset_path} \
--k 192 \
--expansion_factor 32 \
--batch_size 1 \
--grad_acc_steps 16 \
--micro_acc_steps 1 \
--load_in_8bit \
--hookpoints 'transformer.blocks.0' \
--transcode True \
--loss_fn fvu_mdm \
--ctx_len 64 \
--run_name ${WANDB_RUN_NAME} 
```

## result 1
```bash
ValueError: total_variance degenerated to 0.0
# when: step == 35
# where: sparse_coder.forward()
#        line 226: total_variance = (y - y.mean(0)).pow(2).sum()
```


# why total_variance degenerated?

## supposition
suppose: the data in the dim 0 is same, when the data is all mask tokens

## action
add unique_row check code

## result
```bash
unique_tokens: 1
# fequency: 2/100 steps
```

## solve
```python
if total_variance < eps:
    # 视作0方差批，跳过TV归一化 
    fvu = torch.zeros_like(l2_loss)  
else:
    fvu = l2_loss / total_variance
```



