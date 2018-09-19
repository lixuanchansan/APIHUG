#!/usr/bin/env python
"""A simple API to do almost nothing"""
import hug
import gensim.downloader as api
import os

# To read files
import codecs

# To read filepaths
import glob

# Logging
import logging

# regular expression operators
import re

import nltk
import pprint

import gensim.models.word2vec as w2v

# non-linear dimensional reduction
import sklearn.manifold

# set logging config
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

nltk.download("punkt")
nltk.download("stopwords")
nltk.download('wordnet')

# Importing GloVe

from gensim.scripts.glove2word2vec import glove2word2vec

#glove_input_file = 'glove_50d.txt'
#word2vec_output_file = 'word2vec.txt.word2vec'
#glove2word2vec(glove_input_file, word2vec_output_file)

from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec

# Loading the word2vec model
#glove_w2v = KeyedVectors.load_word2vec_format(word2vec_output_file, binary = False)

#i added this

#corpus = api.load('text8')  # download the corpus and return it opened as an iterable
#glove_w2v  = Word2Vec(corpus)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

#file_names = glob.glob("data/insurance.txt")


# Input: text
# Output: Tree JSON
@hug.post() 
@hug.cli()
def getIntents(body):
    """Returns the result of parsing through nlp engine"""
    return body

@hug.post() 
@hug.cli()
def Treeify(body):
    """Returns the result of magic"""
    return body
