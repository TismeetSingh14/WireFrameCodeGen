import os
from keras.preprocessing.text import Tokenizer
from Utils import *
from keras.utils import to_categorical

class Dataset():
    def __init__(self, data_dir, input_transform = None, target_transform = None):
        self.data_dir = data_dir
        self.image_filenames = []
        self.texts = []
        all_filenames = os.listdir(data_dir)
        all_filenames.sort()
        for filename in all_filenames:
            if filename[-3:] == 'png':
                self.image_filenames.append(filename)
            else:
                text = ' ' + load_doc(self.data_dir + filename) + ' '
                text = ' '.join(text.split())
                text = text.replace(',', ' ,')
                self.texts.append(text)
        self.input_transform = input_transform
        self.target_transforms = target_transform

        tokenizer = Tokenizer(filters = '', split = " ", lower = False)
        tokenizer.fit_on_texts([load_doc('vocabulary.vocab')])
        self.tokenizer = tokenizer

        self.vocab_size = len(tokenizer.word_index) + 1
        self.train_sequences = tokenizer.texts_to_sequences(self.texts)
        self.max_sequences = max(len(s) for s in self.train_sequences)
        self.max_length = 48

        X, y, image_data_filenames = [], [], []
        for img_no, seq in enumerate(self.train_sequences):
            in_seq, out_seq = seq[:-1], seq[1:]
            out_seq = to_categorical(out_seq, num_classes = self.vocab_size)
            image_data_filenames.append(self.image_filenames[img_no])
            X.append(in_seq)
            y.append(out_seq)

        self.X = X
        self.y = y
        self.image_data_filenames = image_data_filenames
        self.images = []
        for image_name in self.image_data_filenames:
            image = resize_img(self.data_dir + image_name)
            self.images.append(image)