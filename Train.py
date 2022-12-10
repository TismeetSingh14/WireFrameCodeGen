from Model import *
from Dataset import *
from Config import *
from nltk.translate.bleu_score import sentence_bleu

dataset_init = Dataset(dir_name)
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, dataset_init.vocab_size, num_layers)
criterion = nn.MSELoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
print(params)
optimizer = torch.optim.Adam(params, lr = 0.001)
print("Training Started")
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
    
    print('Epoch [%d/%d], Loss: %.4f'%(e+1, num_epochs, loss.data)) 
    torch.save(encoder,'EncoderWeights'+str(loss.data)+'.pt')
    torch.save(decoder,'DecoderWeights'+str(loss.data)+'.pt')

decoded_words = []
start_text = ' '
hidden = decoder.init_hidden()
image = load_val_images('val/')[0]
image = Variable(torch.FloatTensor([image]))
predicted = ' '

for di in range(9999):
    sequence = dataset_init.tokenizer.texts_to_sequences([start_text])[0]
    decoder_input = Variable(torch.LongTensor(sequence)).view(1,-1)
    features = encoder(image)

    outputs,hidden = decoder(features,decoder_input,hidden)
    topv, topi = outputs.data.topk(1)
    ni = topi[0][0][0]
    word = word2idx(ni,dataset_init.tokenizer)
    if word is None:
        continue
    predicted += word + ' '
    start_text = word
    print(predicted)
    if word == '':
        break

original_gui = load_doc('val/2BC033FD-F097-463B-98A8-C1C9CE50B478.gui')
original_gui = ' '.join(original_gui.split())
original_gui = original_gui.replace(',', ' ,')
original_gui = original_gui.split()

btns_to_replace = ['btn-green','btn-red']
normalized_original_gui = ['btn-orange' if token in btns_to_replace else token for token in original_gui]
normalized_original_gui = ['btn-active' if token == 'btn-inactive' else token for token in normalized_original_gui]

generated_gui = predicted.split()

normalized_generated_gui = ['btn-orange' if token in btns_to_replace else token for token in generated_gui]
normalized_generated_gui = ['btn-active' if token == 'btn-inactive' else token for token in normalized_generated_gui]

print(sentence_bleu([normalized_original_gui],normalized_generated_gui))

# compiler = Compiler('default')
# compiled_website = compiler.compile(predicted.split())

# print(compiled_website)