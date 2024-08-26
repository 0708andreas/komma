import ipdb

import tflearn as tf
from tflearn.data_utils import pad_sequences, to_categorical
import nltk
import re
import math

vocab = []
vocab_dict = {}

puncts = "[.?!:]"

with open("words.txt", 'r') as f:
    vocab = f.read().strip().split(" ")

vocab_dict = {x: i for i, x in enumerate(vocab)}
sentences = []

print("Vocaulary read")

buff = ""
text = ""
f = open("input.txt", "r")
text = f.read().lower()
f.close()
sentences = nltk.sent_tokenize(text)
del text

trainX = []
trainY = []

print("Input text read")

# ipdb.set_trace()
for sent in sentences:
    tokens = nltk.word_tokenize(sent)
    x = []
    y = []
    for i, word in enumerate(tokens):
        if word == ",":
            continue
        elif tokens[min(i + 1, len(tokens) - 1)] == ",":
            x.append(vocab_dict.get(word, len(vocab)))
            y.append(1)
        else:
            x.append(vocab_dict.get(word, len(vocab)))
            y.append(0)
    trainX.append(x)
    trainY.append(y)

# ipdb.set_trace()

print("Input text processed")

trainX = pad_sequences(trainX, maxlen=200, value=0.)
trainY = pad_sequences(trainY, maxlen=200, value=0,)

# testX = trainX[-200:]
# trainX = trainX[:-200]

# testY = trainY[-200:]
# trainY = trainY[:-200]

# trainY = to_categorical(trainY, nb_classes=2)
# testY = to_categorical(testY, nb_classes=2)

seq_len = 20
vec_size = 100
dropout = 0.5

net = tf.input_data(shape=[None, 200])
net = tf.embedding(net, input_dim=22343, output_dim=128)
net = tf.lstm(net, 128)
net = tf.dropout(net, dropout)
net = tf.fully_connected(net, 200, activation="softmax")
net = tf.regression(net, optimizer='adam', loss='categorical_crossentropy')

print("Network created. Starting training\n=================================")

model = tf.DNN(net, clip_gradients=0., tensorboard_verbose=2)
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=3)
ipdb.set_trace()