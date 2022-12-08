import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN,self).__init__()
        resnet = models.resnet34(pretrained = True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum = 0.01)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        if images.shape[0] < 2:
            features = self.linear(features)
            return features
        features = self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.n_layers = num_layers
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.GRU(embed_size*2, hidden_size, num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, hidden):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1).repeat(1, embeddings.shape[1], 1), embeddings), 2)
        output, hidden = self.lstm(embeddings, hidden)
        outputs = self.linear(output)
        return outputs, hidden
    
    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))