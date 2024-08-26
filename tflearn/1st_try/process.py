import re
import nltk
vocab = []

with open("input.txt", 'r') as f:
    for line in f:
        # line = re.sub("[^a-zA-Z\ ]+", "", line)
        words = [x.lower() for x in nltk.word_tokenize(line)]
        words = list(set(words))
        vocab += words
 
vocab = list(set(vocab))

with open("words.txt", 'w') as f:
    f.write(" ".join(vocab))