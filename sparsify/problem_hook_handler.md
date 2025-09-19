# Why the GPU util is so low? 30%


## Supposition
suppose 1: the hook function is repeated register in each step of training update process

```python
        for batch in dl:
            x = batch["input_ids"].to(device)
            # Forward pass on the model to get the next batch of activations
            handles = [
                mod.register_forward_hook(hook) for mod in name_to_module.values()
            ]

            try:
                if self.cfg.loss_fn == "fvu_mdm":
                    # before: x = batch["input_ids"].to(device)
                    # before in SMDM: input_ids = train_data[:, 0 : model.config.block_size].contiguous()
                    noisy_input, mask_indices, p_mask = forward_process(x)
                    self.model(noisy_input)
                    avg_losses = dict(avg_fvu)
                else:
                    raise ValueError(f"Unknown loss function '{self.cfg.loss_fn}'")
            finally:
                for handle in handles:
                    handle.remove()
```


## action
move out of the loop, dangerous!
(comment from lantian: training of llama3, the GPU util has no problem)

## Supposition
suppose 2: the data bottleneck: from dataloader to GPU

## Action
add batchsize, num_workers, pin_memory

```python
torch.cuda.synchronize()
t0 = time.time()
...  # forward + backward
torch.cuda.synchronize()
gpu_time = time.time() - t0
```

## Result
work

