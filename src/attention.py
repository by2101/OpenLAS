import torch
import torch.nn as nn
import torch.nn.functional as F
from six.moves import range


class DotProductAttention(torch.nn.Module):
    def __init__(self, q_dim, k_dim, inner_dim, temperature):
        super(DotProductAttention, self).__init__()
        self.q_affine = nn.Linear(q_dim, inner_dim)
        self.k_affine = nn.Linear(k_dim, inner_dim)
        self.t = temperature
        
    def forward(self, query, key, value, mask=None):
        '''
        query: b x T_q x q_dim
        key: b x T_k x k_dim
        value: b x T_q x v_dim
        
        softmax(key x query^T) x value
        '''
        
        q = self.q_affine(query)
        k = self.k_affine(key)
        score = torch.bmm(k, q.permute(0, 2, 1))
        if mask is not None:
            score.masked_fill_(mask.to(score.device), -float('inf'))
        attn = F.softmax(score / self.t, dim=-1)
        return attn, torch.bmm(attn, value)
        
if __name__ == "__main__":
    from utils import get_seq_mask_by_shape, get_seq_mask
    
    enc = torch.randn(5, 10, 128)
    enc_lengths = torch.tensor([5,6,7,8,9]).long()
    dec = torch.randn(5, 6, 64)
    dec_lengths = torch.tensor([3,4,4,5,6]).long()
    attention = DotProductAttention(128, 64, 256, 1.0)
    mask = (1-get_seq_mask_by_shape(10, 6, enc_lengths).transpose(1,2)).byte()
    
    dec_masks = get_seq_mask_by_shape(6, 128, dec_lengths)
    a = attention(enc, dec, enc, mask)
    a = a * dec_masks