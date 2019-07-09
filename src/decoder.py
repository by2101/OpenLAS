import torch
import torch.nn as nn
import torch.nn.functional as F
from six.moves import range
import attention
import math
from loss import cross_entropy, uniform_label_smooth_regulerizer
import utils
from utils import get_seq_mask_by_shape, get_seq_mask

import pdb



class RNNDecoder(torch.nn.Module):
    def __init__(self, config):
        super(RNNDecoder, self).__init__()
        self.config = config

        self.embed_dim = config["embed_dim"]
        self.dropout_rate = config["dropout_rate"]
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.enc_dim = config["enc_dim"]
        self.att_inner_dim = config["att_inner_dim"]

        self.emb = nn.Embedding(self.vocab_size, self.embed_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        rnns = [torch.nn.LSTM(self.embed_dim, self.hidden_size, 1, batch_first=True)]
        
        for _ in range(self.num_layers-1):
            rnns += [torch.nn.LSTM(self.hidden_size+self.enc_dim, self.hidden_size, 1, batch_first=True)]
        
        self.rnns = torch.nn.ModuleList(rnns)
        
        self.attentions = torch.nn.ModuleList(
            [attention.DotProductAttention(self.enc_dim, self.hidden_size, self.att_inner_dim,
                math.sqrt(self.att_inner_dim)) for _ in range(self.num_layers-1)])
        
        self.output_affine = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, enc_outputs, enc_lengths, src_ids, tgt_ids, label_smooth=0):
        bz = enc_outputs.shape[0]
        if bz != src_ids.shape[0]:
            raise ValueError("enc_outputs does not match src_ids.")
            
        encout_max_length = enc_outputs.shape[1]
        dec_max_length = src_ids.shape[1]
        att_masks = (1-get_seq_mask_by_shape(encout_max_length, dec_max_length, enc_lengths).transpose(1,2)).byte() 
        
        rnn_in = self.emb(src_ids)
        rnn_in = self.dropout(rnn_in)
        
        rnn = self.rnns[0]
        rnn_output, _ = rnn(rnn_in)
        
        for l in range(1, self.num_layers):
            att_scores, att = self.attentions[l-1](enc_outputs, rnn_output, enc_outputs, mask=att_masks)               
            rnn_in = torch.cat([rnn_output, att], dim=-1)
            rnn_in = self.dropout(rnn_in)
            rnn_output, _ = self.rnns[l](rnn_in)
        
        rnn_output = self.dropout(rnn_output)
        logits = self.output_affine(rnn_output)

        ce = cross_entropy(logits.view(-1, logits.size(-1)), tgt_ids.view(-1))
        if label_smooth > 0:
            ls = uniform_label_smooth_regulerizer(logits.view(-1, logits.size(-1)), tgt_ids.view(-1))
            loss = (1-label_smooth) * ce + label_smooth * ls
        else:
            loss = ce
        return loss

    def get_attention_scores(self, enc_outputs, enc_lengths, src_ids):
        bz = enc_outputs.shape[0]
        if bz != src_ids.shape[0]:
            raise ValueError("enc_outputs does not match src_ids.")
            
        encout_max_length = enc_outputs.shape[1]
        dec_max_length = src_ids.shape[1]
        att_masks = (1-get_seq_mask_by_shape(encout_max_length, dec_max_length, enc_lengths).transpose(1,2)).byte() 
        
        rnn_in = self.emb(src_ids)
        rnn_in = self.dropout(rnn_in)
        
        rnn = self.rnns[0]
        rnn_output, _ = rnn(rnn_in)
        
        att_score_list = []
        
        for l in range(1, self.num_layers):
            att_scores, att = self.attentions[l-1](enc_outputs, rnn_output, enc_outputs, mask=att_masks)                
            att_score_list.append(att_scores)
            rnn_in = torch.cat([rnn_output, att], dim=-1)
            rnn_in = self.dropout(rnn_in)
            rnn_output, _ = self.rnns[l](rnn_in)
        return att_score_list

    def zero_states(self, batch_size):
        states = []
        for _ in range(len(self.rnns)):
            states.append(None)                
        return states
        
        
    def forward_step(self, enc_outputs, enc_lengths, decoder_states, src_ids):
        '''
        decoder_states
        src_ids: batch_size x 1        
        '''
        bz = enc_outputs.shape[0]
        if bz != src_ids.shape[0]:
            raise ValueError("enc_outputs does not match src_ids.")        
        encout_max_length = enc_outputs.shape[1]
        if src_ids.shape[1] != 1:
            raise ValueError('The src_ids is not for one step.')
        att_masks = (1-get_seq_mask_by_shape(encout_max_length, 1, enc_lengths).transpose(1,2)).byte()         
        
        src_ids = src_ids.to(enc_outputs.device)
        
        next_states = []
        rnn_in = self.emb(src_ids)
        rnn_in = self.dropout(rnn_in)
        
        rnn = self.rnns[0]
        
        rnn_output, states = rnn(rnn_in, decoder_states[0])
        next_states.append(states)
        
        for l in range(1, self.num_layers):
            att_scores, att = self.attentions[l-1](enc_outputs, rnn_output, enc_outputs, mask=att_masks) 

            rnn_in = torch.cat([rnn_output, att], dim=-1)           
            rnn_in = self.dropout(rnn_in)
            rnn_output, states = self.rnns[l](rnn_in, decoder_states[l])        
            next_states.append(states)
            
        rnn_output = self.dropout(rnn_output)
        logits = self.output_affine(rnn_output)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, next_states
        

        
        
        
if __name__ == "__main__":
    # For debugging
    config = {
        "embed_dim": 8,
        "vocab_size": 128,
        "hidden_size": 64,
        "num_layers": 2,
        "enc_dim": 32,
        "att_inner_dim": 32,
        "dropout_rate": 0.5
        }

    decoder = RNNDecoder(config)
    
    enc_outputs = torch.randn(2, 20, 32)
    enc_lengths = torch.tensor([15, 16]).long()
    src_ids = torch.tensor([[1,2,3,4,5],
        [6,7,8,9,10]])
    tgt_ids = torch.tensor([[2, 3, 4, 5, 6],
        [7,8,9,10,-1]])

    log_probs, loss = decoder(enc_outputs, enc_lengths, src_ids, tgt_ids)
        
    states = decoder.zero_states(2)
        
    log_probs2 = []    
    states2 = []    
    for i in range(1):
        res, states = decoder.forward_step(enc_outputs, enc_lengths, states, src_ids[:, i][:, None])
        log_probs2.append(res)
        states2.append(states)
    log_probs2 = torch.cat(log_probs2, dim=1)

    






