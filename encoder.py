from CharLvl_NMT import *

class rnn_encoder(nn.Module):
    def __init__(self, hidden_size, n_layers=1):
        super(rnn_encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True)

    def forward(self, input, hidden):
        # input: (SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE) hidden: (1, BATCH_SIZE, HIDDEN_SIZE)
        input = input.contiguous().view(1, 64, self.hidden_size)
        output = input
        for i in range(self.n_layers):
            output, hidden = self.gru(output, 
            hidden)
        return output, hidden

    def initHidden(self):
        return Variable(torch.zeros(2, 64, self.hidden_size))

class cnn_encoder(nn.Module):
    def __init__(self, filter_widths, num_filters_per_width, char_embedding_dim, use_cuda):
        super(cnn_encoder, self).__init__()
        self.filter_size_range = filter_widths
        self.char_embedding_dim = char_embedding_dim
        self.num_filters_per_width = num_filters_per_width  #possible sizes * num_filters = decoder hidden size

        # 1 conv layer with dynamic padding 
        # this enable kernels with varying widths a la Cho's NMT (2017) 
        self.filter_banks = []
        
        for k, num_filters in zip(self.filter_size_range, self.num_filters_per_width):
            padding = k //2 
            
            self.k_filters = nn.Conv2d(
                    1, num_filters, 
                    # for kernel and padding the dimensions are (H, W)
                    (self.char_embedding_dim, k), padding=(0, padding), 
                    stride=1)  
            self.filter_banks.append( self.k_filters )

        self.filter_banks = nn.ModuleList(self.filter_banks) 
        
        self.pool = nn.MaxPool1d( 5 , return_indices=True)  

        self.use_cuda = use_cuda

    def forward(self, x, seq_len):
        all_activations = []
        all_unpool_indices = []
        for k, k_sized_filters in zip(
            self.filter_size_range, self.filter_banks): 
            activations = F.relu(k_sized_filters(x))    
            
            if k % 2 == 0: # even kernel widths: skip last position
                input_indices = torch.LongTensor(range(seq_len))
                if self.use_cuda: 
                    input_indices = input_indices.cuda()

                activations = activations.index_select( 3, Variable(input_indices)) 
    
            #print('convolved width %s kernels for activations in shape %s' %(k, activations.data.shape) )            
            activations = activations.squeeze(2)
            activations, unpool_indices = self.pool(activations)
            activations = activations.unsqueeze(2)
            all_activations.append(activations)

        activation_tensor = torch.cat(all_activations, 1)
        return activation_tensor