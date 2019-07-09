import logging
import numpy as np
import torch
import copy
from six.moves import range
from data import BOS_ID, EOS_ID
from utils import Timer


import pdb

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
# logging.basicConfig(format='train.py [line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger("model")
logger.setLevel(logging.INFO)


def extend_tok(score, state, id, old_tok=None):
    tok = {
            "cum_score": 0,
            "norm_score": 0,
            "state": None,
            "ids": []
        }
    
    if old_tok is not None:
        tok["cum_score"] = old_tok["cum_score"]
        tok["ids"] = copy.deepcopy(old_tok["ids"])
    tok["cum_score"] += score
    tok["state"] = copy.copy(state)
    tok["ids"].append(id)
    tok["norm_score"] = tok["cum_score"] / len(tok["ids"])   
    return tok

    
def prune_hyps(hyps, k=5, key="cum_score"):
    sorted_hyps = sorted(hyps, key=lambda x: x[key], reverse=True)
    return sorted_hyps[:k]
    
def get_end_hyps(hyps):
    ended_hyps = []
    maintained_hyps = []
    for tok in hyps:
        if tok['ids'][-1] == EOS_ID:
            ended_hyps.append(tok)
        else:
            maintained_hyps.append(tok)
    return maintained_hyps, ended_hyps
    
def get_best_hyp(hyps, length_penalty_alpha=0.6):
    best_score = -float('inf')
    best_idx = 0
    for i in range(len(hyps)):
        cum_score = copy.copy(hyps[i]['cum_score'])
        lp = ((5.0+len(hyps[i]['ids']))/(5.0+1.0))**length_penalty_alpha
        final_score = cum_score / lp
        hyps[i]['final_score'] = final_score
        if final_score > best_score:
            best_score = final_score 
            best_idx = i
    return hyps[best_idx] 
        
def show_toks(toks, id2token, key="final_score"):
    msg = ""
    for i, tok in enumerate(toks):
        hyp = " ".join([id2token[id] for id in tok['ids']])
        score_info = tok[key]
        msg += ("top-{} {} {:.4f} hyp: {}\n".format(i+1, key, score_info, hyp))
    return msg
    
class LAS(torch.nn.Module):
    def __init__(self, encoder, decoder, config):
        super(LAS, self).__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, feats, feat_lengths, src_ids, tgt_ids, label_smooth=0):
        enc_outputs, enc_lengths = self.encoder(feats, feat_lengths)
        loss = self.decoder(enc_outputs, enc_lengths, src_ids, tgt_ids, label_smooth)
        return loss
        
    def get_atten_scores(self, feats, feat_lengths, src_ids):
        enc_outputs, enc_lengths = self.encoder(feats, feat_lengths)
        return self.decoder.get_attention_scores(enc_outputs, enc_lengths, src_ids)
        
        
    def beam_search_sentence(self, feats, feat_lengths, id2token, beam_size=5, max_steps=60):
        decode_timer = Timer()
        decode_timer.tic()
        
        assert feats.shape[0] == 1
        
        enc_outputs, enc_lengths = self.encoder(feats, feat_lengths)
        
        # The initial step
        initial_state = self.decoder.zero_states(1)
        log_probs, next_states = self.decoder.forward_step(enc_outputs, enc_lengths, 
            initial_state, torch.tensor([[BOS_ID]]).long())
        # log_probs: [1, 1, vocab_size]
        topk_score, topk_indices = torch.topk(log_probs.view(-1), beam_size)
        
        # pdb.set_trace()
        
        ended_hyps = []
        
        this_hyps = []
        for i in range(topk_score.size(0)):
            this_hyps.append(extend_tok(topk_score[i].item(), next_states, topk_indices[i].item()))
        
        for step in range(max_steps):           
            new_hyps = []
            for tok in this_hyps:
                state = tok["state"]
                id = tok["ids"][-1]
                log_probs, next_states = self.decoder.forward_step(enc_outputs, enc_lengths, 
                    state, torch.tensor([[id]]).long())
                topk_score, topk_indices = torch.topk(log_probs.view(-1), beam_size)
                for i in range(topk_score.size(0)):
                    new_hyps.append(
                        extend_tok(topk_score[i].item(), next_states, topk_indices[i].item(), tok))                        
            new_hyps, new_ended_hyps = get_end_hyps(new_hyps)
            this_hyps = prune_hyps(new_hyps, k=beam_size)
            ended_hyps.extend(new_ended_hyps)
                
        for tok in this_hyps:
            tok['ids'].append(EOS_ID)
                
        ended_hyps.extend(this_hyps)

        best_hyp = get_best_hyp(ended_hyps)
        decode_time_cost = decode_timer.toc()
        num_frames = feats.shape[1]
        rtf = 100.0 * decode_time_cost / num_frames
        
        ended_hyps = prune_hyps(ended_hyps, key="final_score")
        
        
        msg = show_toks(ended_hyps, id2token)
        msg += "Decoding Real-Factor: {:.3f}\n".format(rtf)
        logger.info("ended_hyps:\n"+msg)

        return best_hyp, ended_hyps
        
    # def beam_search(self, feats, feat_lengths, beam_size=5):
        # batch_size = feat_lengths.shape[0]
        # enc_outputs, enc_lengths = self.encoder(feats, feat_lengths)


    
        # first_token = {
            # "cum_score": 0,
            # "states": self.decoder.zero_states,
            
            # }
        
        