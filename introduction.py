#%%
import os
os.environ["HF_ENDPOINT"]       = "https://hf-mirror.com"
from sparsify import Sae
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#%%
# repo_name  = "EleutherAI/sae-pythia-160m-32k"
sae_dir  = "/home/xuekaiwen/code/radd_physics/remote/sparsify/checkpoints/unnamed"
sae_name = "layers.10"

# sae = Sae.load_from_hub(repo_name, hookpoint=sae_name)
sae = Sae.load_from_disk(os.path.join(sae_dir, sae_name))

     
lm_dir = '/home/xuekaiwen/.cache/huggingface/hub/models--EleutherAI--pythia-160m/snapshots/50f5173d932e8e61f858120bcb800b97af589f46'
tokenizer = AutoTokenizer.from_pretrained(lm_dir)
inputs = tokenizer("Hello, world!", return_tensors="pt")


#%%
with torch.inference_mode():
    model = AutoModelForCausalLM.from_pretrained(lm_dir)
    outputs = model(**inputs, output_hidden_states=True)

    # latent_acts = []
    # for sae, hidden_state in zip(saes.values(), outputs.hidden_states):
    #     # (N, D) input shape expected
    #     hidden_state = hidden_state.flatten(0, 1)
    #     latent_acts.append(sae.encode(hidden_state))
    h_id = int(sae_name.rsplit('.')[-1]) + 1
    hidden_state = outputs.hidden_states[h_id].flatten(0, 1)
    latent_act = sae.encode(hidden_state)


# Do stuff with the latent activations

#%%
