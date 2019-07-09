import torch
import torch.nn.functional as F
from data import IGNORE_ID

def cross_entropy(logits, ground_truth):
    return F.cross_entropy(logits, ground_truth, 
        ignore_index=IGNORE_ID, reduction='mean')
        
def kd_regulerizer(logits, priors, masks):
    log_probs = F.log_softmax(logits)
    ce = priors * log_probs * masks[:, None]
    n = torch.sum(masks)
    ce = -torch.sum(ce, dim=-1) / n
    return ce
    
def uniform_label_smooth_regulerizer(logits, ground_truth):
    dim = logits.shape[-1]
    masks = (ground_truth != IGNORE_ID).float()    
    priors = torch.ones_like(logits) / dim
    return kd_regulerizer(logits, priors, masks)
    
        
    