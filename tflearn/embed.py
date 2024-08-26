import tflearn as tf
import ipdb
import nltk

vocab = []
vocab_dict = {}

with open("words.txt", 'r') as f:
    vocab = f.read().strip().split(" ")

vocab_dict = {x: i for i, x in enumerate(vocab)}

text = ""
with open('input.txt', 'r') as f:
    text = f.read().lower()

words = [vocab_dict.get(x, len(vocab)) for x in nltk.word_tokenize(text)]
del text

to_batch = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

words = to_batch(words, 25)

net = tf.input_data(shape=[None, 25])
net = tf.embedding(net, input_dim=22343, output_dim=128)

model = tf.DNN(net, clip_gradients=0., tensorboard_verbose=2)
ipdb.set_trace()