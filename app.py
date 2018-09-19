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

<<<<<<< HEAD
@hug.post() 
@hug.cli()
def Treeify(body):
    """Returns the result of magic"""
    return body
=======
@hug.get() 
@hug.cli()
def query_array(input: hug.types.text):
    """Assuming input to be an array of strings"""

    # Loading the word2vec model
    glove_w2v = KeyedVectors.load_word2vec_format(word2vec_output_file, binary = False)

    tokenizer = nltk.data.load('tokenizer/punkt/english.pickle')
    stop_words.add("is")

    unprocessed_raw_sentences = tokenizer.tokenize(input)

    raw_sentences = []

    for raw_sentence in unprocessed_raw_sentences:
        raw_sentences.extend(raw_sentence.split("\n"))

    lemmatized_words = []

    for raw_sentence in raw_sentences:
        words = re.sub("[^a-zA-Z]"," ", raw_sentence)
        words = words.split()

        for word in words:
            lemmatized_word = lemmatizer.lemmatize(word.lower())
            if lemmatized_word not in stop_words:
                lemmatized_words.append(lemmatized_word)

    lemmatized_vocab = {}

    for lemmatized_word in lemmatized_words:
        if lemmatized_word not in lemmatized_vocab:
            lemmatized_vocab[lemmatized_word] = 1
        else:
            lemmatized_vocab[lemmatized_word] += 1

    categories = []

    for word, count in lemmatized_vocab.items():
        is_added = False
        
        if len(categories) == 0:
            categories.append(Category(word, count, 0.8))
            
        for category in categories:
            if category.is_similar(word):
                category.add_word(word, count)
                is_added = True
                break
                
        if not is_added:
            categories.append(Category(word, count, 0.8))
    
    tagged_sentences = []

    for raw_sentence in raw_sentences:
        tagged_sentences.append(Sentence(raw_sentence, categories, lemmatized_words))

    categories.sort(key = lambda category: -category.count)
    num_of_sentences = len(raw_sentences)
    filtered_categories = list(filter(lambda category: category.count > num_of_sentences/5,categories))

    tree_structure = TreeStructure(tagged_sentences, filtered_categories)
    tree_structure = tree_structure.treefy()

    return tree_structure

>>>>>>> 11ac3f3cad360fb7438b0b005d5e1416eeb29837
