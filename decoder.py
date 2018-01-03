from CharLvl_NMT import *

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_symbols, n_layers=1, dropout_p=0.5,):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.attn = nn.Linear(3*hidden_size, output_size) # due to bi-directional gru
        self.attn_combine = nn.Linear(3*hidden_size, hidden_size)
        
        self.gru = nn.GRU(hidden_size, 2*hidden_size, num_layers = 1) 
        self.out = nn.Linear(2*hidden_size, num_symbols)

    def forward(self, embedded, hidden, encoder_outputs):
        # attention implementation modified from official pytorch tutorial 
        hidden = hidden.view(64, -1)
        
        # compute attention weights 
        all_info = torch.cat((embedded, hidden), 1)
        attn_weights = F.softmax( self.attn(all_info) ) 
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        # apply encoder outputs then pass into gru
        output = torch.cat((embedded, attn_applied), 1)
        output = (self.attn_combine(output) ) 
        output = F.relu(output)
        output, hidden = self.gru(
            output.unsqueeze(0), 
            hidden.unsqueeze(0))
            
        # last linear layer, remove a dim 
        output = (self.out(output)).squeeze(0).float()
        return output, hidden, attn_weights