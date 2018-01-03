from CharLvl_NMT import *  
from loader import loader, ValidLoader
from autoencoder import CharLevel_autoencoder

''' 
An implementation of Fully Character Level Neural Machine Translation proposed in Lee et al (2016)
This particular implementation has the task of German to English translation.

Paper: https://arxiv.org/abs/1610.03017
Project Repo: https://github.com/cer446/NLP_Char_Translation (other models and model evaluation are found here)
Dataset: WMT16 GE-EN 
'''

num_epochs = 100
batch_size = 64
learning_rate = 1e-3
max_batch_len = 300
num_symbols = 125
use_cuda = torch.cuda.is_available()

criterion = nn.CrossEntropyLoss()

model = CharLevel_autoencoder(criterion, num_symbols, use_cuda)
if use_cuda:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

def train(model, optimizer, num_epochs, batch_size, learning_rate):
    model.load_state_dict(torch.load('./300_60_teachers6.pth', map_location=lambda storage, loc: storage))
    train_loader, _ = loader()
    valid_loader = ValidLoader()

    for epoch in range(num_epochs):
        model.train()
        for index, (data, label) in enumerate(train_loader):
            data = Variable(data)
            if use_cuda: 
                data = data.cuda()
            
            # ===================forward=====================
            encoder_outputs, encoder_hidden = model.encode(data, max_batch_len)
            #print(encoder_outputs.data.shape, encoder_hidden.data.shape) 
                
            decoder_hidden = encoder_hidden
            #print('deocder input', decoder_input.shape, 'decoder hidden', decoder_hidden.data.shape)
            
            loss = model.decode(
                 label, decoder_hidden, encoder_outputs, i =  False)
            
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # ===================log========================
        torch.save(model.state_dict(), './300_60_teachers5.pth')
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, num_epochs, loss.data[0]))

def inference_mode(model, optimizer, num_epochs, batch_size, learning_rate):
    state_path = './300_60_teachers6.pth'
    model.load_state_dict(torch.load(state_path, map_location=lambda storage, loc: storage))
    
    valid_loader = ValidLoader()
    model.eval()
    for index, (data, label) in enumerate(valid_loader):
        pickle.dump(label.numpy(), open( "./data/%s_target.p" %(index), "wb" ), protocol=4 )

        data = Variable(data, volatile = True)
        encoder_outputs, encoder_hidden = model.encode(data, max_batch_len)
        loss = model.decode(label, encoder_hidden, encoder_outputs, index)
        print(loss.data[0])
    
if __name__ == '__main__':
    #sys.stdout=open('progress_update_big_data.txt','w')
    train(model, optimizer, num_epochs, batch_size, learning_rate)
    #inference_mode(model, optimizer, num_epochs, batch_size, learning_rate)

