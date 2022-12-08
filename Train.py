from Model import *
from Dataset import *
from Config import *

dataset_init = Dataset(dir_name)
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, dataset_init.vocab_size, num_layers)
criterion = nn.MSELoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters())
optimizer = torch.optim.Adam(params, lr = 0.001)

for e in range(num_epochs):
    for i_batch in range(len(dataset_init.X)):
        hidden = decoder.init_hidden()
        images = Variable(torch.FloatTensor([dataset_init.images[i_batch]]))
        input_seqs = Variable(torch.LongTensor(dataset_init.X[i_batch])).view(1,-1)
        target_seq = Variable(torch.FloatTensor(dataset_init.y[i_batch]))
        encoder.zero_grad()
        decoder.zero_grad()
        features = encoder(images)
        outputs, hidden = decoder(features, input_seqs, hidden)
        loss = 0
        for di in range(target_seq.shape[0]):
            loss += criterion(outputs.squeeze(0)[di], target_seq[di])

        loss.backward()
        optimizer.step()
    
    print('Epoch [%d/%d], Loss: %.4f'%(e+1, num_epochs, loss.data[0])) 
    torch.save(encoder,'EncoderWeights'+str(loss.data[0])+'.pt')
    torch.save(decoder,'DecoderWeights'+str(loss.data[0])+'.pt')
