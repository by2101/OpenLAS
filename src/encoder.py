import torch
import torch.nn as nn
from six.moves import range

def get_mask(inputs, lengths):
    with torch.no_grad():
        masks = torch.zeros_like(inputs)
        for i in range(lengths.shape[0]):
            masks[i, :lengths[i].long(), i] = 1
    return masks
    
class BiRNN_Torch(torch.nn.Module):
    def __init__(self, config):
        super(BiRNN_Torch, self).__init__()
        self.config = config

        self.input_dim = config["input_dim"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        
        
        self.rnn = nn.LSTM(self.input_dim, self.hidden_size, 
            self.num_layers,bidirectional=True, batch_first=True)
        
        
    def forward(self, feats, feat_lengths):
        _, idx_sort = torch.sort(feat_lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        feats_in = feats.index_select(0, idx_sort)
        lengths_in = feat_lengths.index_select(0, idx_sort)
        
        x_packed = nn.utils.rnn.pack_padded_sequence(input=feats_in, lengths=lengths_in, batch_first=True)
        output = x_packed
        h = None
        output, h = self.rnn(output, h) 
                
        output_padded = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output, output_lengths = output_padded
        output = output.index_select(0, idx_unsort.to(output.device))
        output_lengths = output_lengths.index_select(0, idx_unsort.to(output_lengths.device))
        return output, output_lengths    
    
        
class BiRNN(torch.nn.Module):
    def __init__(self, config):
        super(BiRNN, self).__init__()
        self.config = config

        self.input_dim = config["input_dim"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        
        
        self.rnns = []
        self.rnns.append(nn.LSTM(self.input_dim, 
            self.hidden_size, bidirectional=True, batch_first=True))  
        self.norms = [nn.LayerNorm(self.hidden_size*2, elementwise_affine=False)]
        
        for _ in range(self.num_layers-1):
            self.rnns.append(nn.LSTM(self.hidden_size * 2, 
                self.hidden_size, bidirectional=True, batch_first=True))
            self.norms.append(nn.LayerNorm(self.hidden_size*2, elementwise_affine=False))
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.norms = torch.nn.ModuleList(self.norms)
        
        
    def forward(self, feats, feat_lengths):
        _, idx_sort = torch.sort(feat_lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        feats_in = feats.index_select(0, idx_sort)
        lengths_in = feat_lengths.index_select(0, idx_sort)
        
        x_packed = nn.utils.rnn.pack_padded_sequence(input=feats_in, lengths=lengths_in, batch_first=True)
        output = x_packed
        h = None
        for i in range(len(self.rnns)):
            output, h = self.rnns[i](output, h)
            output_padded = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            output, output_lengths = output_padded
            output = self.norms[i](output)
            output = nn.utils.rnn.pack_padded_sequence(input=output, lengths=output_lengths, batch_first=True)
                
        output_padded = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output, output_lengths = output_padded
        output = output.index_select(0, idx_unsort.to(output.device))
        output_lengths = output_lengths.index_select(0, idx_unsort.to(output_lengths.device))
        return output, output_lengths
       
        
        
        
        
        
if __name__ == "__main__":
    from data import FrameBasedSampler, Collate, SpeechDataset
    fn = "/home/baiye/Speech/las/egs/timit/data/test.json"
    dataset = SpeechDataset(fn)
    sampler = FrameBasedSampler(dataset)
    collate = Collate(left=0, right=0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, shuffle=False)
    dataiter = iter(dataloader)
    feats, feat_lengths, targets, target_lengths = next(dataiter)
    config = {
        "input_dim": 40,
        "hidden_size": 256,
        "num_layers": 3,
    
        }
    rnn = PyramidBiRNN(config)
    output, output_lengths = rnn(feats, feat_lengths)    
