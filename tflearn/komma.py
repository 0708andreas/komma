import ipdb

import tflearn as tf
from tflearn.data_utils import pad_sequences, to_categorical
import nltk
import re
import math

to_batch = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

vocab = []
vocab_dict = {}

with open("words.txt", 'r') as f:
    vocab = f.read().strip().split(" ")

vocab_dict = {x: i for i, x in enumerate(vocab)}

buff = ""
text = ""
f = open("input.txt", "r")
text = f.read().lower()
f.close()

words = nltk.word_tokenize(text)
del text

trainX = []
trainY = []

for i, word in enumerate(words):
    if word == ",":
        continue
    elif words[min(i + 1, len(words) - 1)] == ",":
        trainX.append(vocab_dict.get(word, len(vocab)))
        trainY.append(1)
    else:
        trainX.append(vocab_dict.get(word, len(vocab)))
        trainY.append(0)

# ipdb.set_trace()

trainY = to_categorical(trainY, nb_classes=2)

trainX = to_batch(trainX, 200)
trainY = to_batch(trainY, 200)

net = tf.input_data(shape=[None, 200])
net = tf.embedding(net, input_dim=22343, output_dim=128)
net = tf.lstm(net, 128)
net = tf.dropout(net, 0.5)
net = tf.fully_connected(net, 2, activation="softmax")
net = tf.regression(net, optimizer='adam', loss='categorical_crossentropy')

print("Network created. Starting training\n=================================")
ipdb.set_trace()
model = tf.DNN(net, clip_gradients=0., tensorboard_verbose=2)
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=1)
ipdb.set_trace()