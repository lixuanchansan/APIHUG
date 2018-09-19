#!/usr/bin/env python
"""A simple API to do almost nothing"""
import hug
import gensim.downloader as api
import os

<<<<<<< HEAD
users = [
    {
        'uid': u'jsmith',
        'name': u'John Smith',
        'email': u'jsmith@example.com',
        'phone': u'123-456-7890'
    },
    {
        'uid': u'jdoe',
        'name': u'Jane Doe',
        'email': u'jdoe@example.com',
        'phone': u'234-567-8901'
    },
    {
        'uid': u'ssample',
        'name': u'Sally Sample',
        'email': u'ssample@example.com',
        'phone': u'345-678-9012'
    }
]

@hug.get('/users', versions=1)
def user(user_id):
    return 'I do nothing useful.'
    #return user_id

@hug.local()
@hug.get('/users', versions=2)
def user():
    return {'users': users}

if __name__ == '__main__':
    user.interface.local()

@hug.get() 
@hug.cli()
def add(text: hug.types.text, categories:hug.types.text):
    """Returns the result of adding number_1 to number_2"""
    return "text +"" Cat: " +categories
=======
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
def getIntents(body):
    """Returns the result of parsing through nlp engine"""
    return body

@hug.post() 
def Treeify(body):
    """Returns the result of magic"""
    return body
>>>>>>> b66d7ed81fc9b83dd092c49df5dda4190d4db2dc
