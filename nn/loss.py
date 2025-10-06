import numpy as np
from core.tensor import Tensor

def cross_entropy(logits, targets):
    batch_size = logits.data.shape[0]
    
    max_logits = logits.data.max(axis=-1, keepdims=True)
    shifted = logits - Tensor(max_logits)
    log_sum_exp = (shifted.exp().sum(axis=-1, keepdims=True)).log()
    log_probs = shifted - log_sum_exp
    
    # Cross entropy: -sum(targets * log_probs) / batch_size
    loss = -(Tensor(targets) * log_probs).sum() / Tensor(np.array((batch_size, ), dtype=np.float64))
    return loss