# coding: utf-8
import nltk
txt = open("input.txt", 'r').read()
nltk.sent_tokenize(txt)
from gensim.models import Word2Vec
sents = nltk.sent_tokenize(txt)
vec = Word2Vec(sents)
vec
vec['hi']
vec['man']
vec['Man']
vec['lived']
vec = Word2Vec(sents)
sents[0]
vec.vocab
vec['g']
vec = Word2Vec(txt)
vec['lived']
vec.vocab
