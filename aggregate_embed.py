import numpy as np

def pi0_fast_embed(prelogits):
    # load prelogits with client.infer(element)["pre_logits"]
    return np.mean(prelogits, axis=0)

def pi05_embed():
    pass