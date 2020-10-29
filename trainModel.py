from showAndTell import GRUAttention
from tensorflow.keras.layers import Input, Dropout, Dense, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import random
import math
import json
import pickle
from functools import lru_cache
from tensorflow.keras.losses import sparse_categorical_crossentropy


@lru_cache(4000)
def load_img(path):
    return np.load(path)


class ImgCaption(Sequence):

    def __init__(self, annotations_path, preprocess_base, tokenizer=None, maxlen=None,
                 batch_size=100, shuffle=False):
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        captions = []
        self.img_ids = []
        for val in annotations['annotations']:
            captions.append(f"<start> {val['caption']} <end>")
            self.img_ids.append(preprocess_base + '%012d.jpg.npy' % val['image_id'])
        if tokenizer is None or isinstance(tokenizer, int):
            self.tokenizer = Tokenizer(num_words=tokenizer,
                                       oov_token="<unk>",
                                       filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
            self.tokenizer.fit_on_texts(captions)
            self.tokenizer.word_index['<pad>'] = 0
            self.tokenizer.index_word[0] = '<pad>'
        else:
            self.tokenizer = tokenizer
        self.captions = pad_sequences(self.tokenizer.texts_to_sequences(captions), maxlen=maxlen, padding='post')
        self.indexes = list(range(len(self.img_ids)))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.img_shape = load_img(self.img_ids[0]).shape
        pass

    def __getitem__(self, index):
        idx = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        text = self.captions[idx, ...]
        y = np.zeros_like(text)
        y[:, :-1] = text[:, 1:]
        imgs = np.zeros((len(idx),) + self.img_shape)
        for i, v in enumerate(idx):
            imgs[i, ...] = load_img(self.img_ids[v])
        return (imgs, text), y

    def __len__(self):
        return math.ceil(len(self.indexes) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.indexes)
        pass


def loss(y_true, y_pred):
    print(y_true)
    print(y_pred)
    return sparse_categorical_crossentropy(y_true, y_pred, True)


if __name__ == '__main__':
    train = ImgCaption('annotations/captions_train2014.json',
                       'train_processed2014/COCO_train2014_', batch_size=50, shuffle=True)
    '''print(len(train))
    (x, text), y = train[0]
    print(x.shape)
    print(text)
    print(y)
    print(len(train.tokenizer.word_index))
    map = train.tokenizer.index_word
    print(' '.join([map[w] for w in text[0, :]]))'''
    with open('tokenizer.pickle', 'wb') as f:
        pickle.dump(train.tokenizer, f)
    img = Input((64, 2048))
    d_img = Dense(300)(img)

    txt = Input((None, ))
    emb = Embedding(len(train.tokenizer.word_index), 300)(txt)

    d = GRUAttention(300, mask_zeros=True, return_sequences=True)([d_img, emb])
    d = Dense(len(train.tokenizer.word_index), activation='softmax')(d)

    model = Model([img, txt], d)

    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['sparse_categorical_accuracy'])
    #model.compile(loss=loss, optimizer='nadam', metrics=['sparse_categorical_accuracy'])


    #print(model.predict(train[0][0]))
    model.fit(train, epochs=10,
              callbacks=[ModelCheckpoint(filepath='model.{epoch:02d}-{loss:.4f}-{sparse_categorical_accuracy:.4f}.h5')])
