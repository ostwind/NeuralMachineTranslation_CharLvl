from CharLvl_NMT import *
from encoder import cnn_encoder, rnn_encoder
from decoder import AttnDecoderRNN


class CharLevel_autoencoder(nn.Module):
      def __init__(self, criterion, num_symbols, use_cuda):
            ''' overview of autoencoder forward:
            1. Input batch is embedded 
            2. CNN+Pool encoder is called on input
            3. BiGRU encoder is called on activations of previous encoder
            4. Attention GRU decoder takes an embedded symbol at current t 
                  - Decoder embedding embeds symbol at current t 
            6. Batch cross entropy is calculated and returned  
            '''
            super(CharLevel_autoencoder, self).__init__()
            self.char_embedding_dim = 128
            self.pooling_stride = 5
            self.seq_len = 300
            self.num_symbols = num_symbols
            self.use_cuda = use_cuda

            self.filter_widths = list(range(1, 9)) 
            # due to cuda limitations, every filter width has 50 less filters
            self.num_filters_per_width = [150, 150, 200, 200, 250, 250, 250, 250] 

            self.encoder_embedding = nn.Embedding(num_symbols, self.char_embedding_dim)
            self.cnn_encoder = cnn_encoder(
            filter_widths = self.filter_widths,
            num_filters_per_width = self.num_filters_per_width,
            char_embedding_dim = self.char_embedding_dim,
            use_cuda = use_cuda)
            
            self.decoder_hidden_size = int(np.sum(np.array(self.num_filters_per_width)) )
            self.rnn_encoder = rnn_encoder(  
            hidden_size = self.decoder_hidden_size )

            # decoder embedding dim dictated by output dim of encoder
            self.decoder_embedding = nn.Embedding(num_symbols, self.decoder_hidden_size)
            self.attention_decoder = AttnDecoderRNN(
                  num_symbols = num_symbols,
                  hidden_size = self.decoder_hidden_size, 
                  output_size = self.seq_len//self.pooling_stride)

            self.criterion = criterion

      def encode(self, data, seq_len):
            encoder_embedded = self.encoder_embedding(data).unsqueeze(1).transpose(2,3) 
            encoded = self.cnn_encoder.forward(encoder_embedded, self.seq_len)
            encoded = encoded.squeeze(2)
      
            encoder_hidden = self.rnn_encoder.initHidden()
            encoder_outputs = Variable(torch.zeros(64, seq_len//self.pooling_stride, 2*self.decoder_hidden_size))
            if self.use_cuda:
                  encoder_outputs = encoder_outputs.cuda()
                  encoder_hidden = encoder_hidden.cuda()

            for symbol_ind in range(self.seq_len//self.pooling_stride):#self.rnn_emits_len): 
                  output, encoder_hidden = self.rnn_encoder.forward(
                        encoded[:,:,symbol_ind], encoder_hidden)
                  encoder_outputs[:, symbol_ind,:] = output[0]
            return encoder_outputs, encoder_hidden

      def decode(self, target_data, decoder_hidden, encoder_outputs, i):   
            use_teacher_forcing = True if random.random() < 0.7 else False
            if type(i) != bool: # given batch  index, then eval mode, no teacher forcing
                  use_teacher_forcing = False
            
            output = []
            # SOS token = 32 after encoding it
            input_embedded = Variable(torch.LongTensor([32]).repeat(64), requires_grad = False)
            if self.use_cuda:
                  input_embedded = input_embedded.cuda()
            input_embedded = self.decoder_embedding( input_embedded )
                                    
            for symbol_index in range(self.seq_len): 
                  # # current symbol, current hidden state, outputs from encoder 
                  decoder_output, decoder_hidden, attn_weights = self.attention_decoder.forward(
                  input_embedded, decoder_hidden, encoder_outputs)  
                  output.append(decoder_output)

                  if use_teacher_forcing:
                        input_symbol = Variable(target_data[:, symbol_index], requires_grad = False)
                        if self.use_cuda:
                              input_symbol = input_symbol.cuda()

                  else:
                        values, input_symbol = decoder_output.max(1)
                  input_embedded = self.decoder_embedding( input_symbol )
            
            # at current batch: conglomerate all true and predicted symbols
            # into one vector then return the batch cross entropy
            # first mask out padding at the end of every sentence    
            actual_sentence_mask = torch.ne(target_data, 31).byte()
            threeD_mask = actual_sentence_mask.unsqueeze(2).repeat(1, 1, 125)#.transpose()
            predicted = torch.stack(output, dim=1)
            
            # if validation loader is called, dump predictions
            if type(i) != bool: 
                  values, indices = predicted.max(2) 
                  print( indices.data.shape)
                  pickle.dump(indices.data.numpy(), open( "./data/%s_predicted.p" %(i), "wb" ), protocol=4 )

            if self.use_cuda:
                  target_data, actual_sentence_mask, threeD_mask = target_data.cuda(), actual_sentence_mask.cuda(), threeD_mask.cuda()

            # calculate cross entropy on non-padding symbols
            masked_target = torch.masked_select(target_data, actual_sentence_mask)
            predicted = predicted.masked_select(Variable(threeD_mask), )
            predicted = predicted.view(-1,125)
            loss = self.criterion(
                  predicted,
                  Variable(masked_target, ) ) 

            return loss 