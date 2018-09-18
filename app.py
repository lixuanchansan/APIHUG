#!/usr/bin/env python
"""A simple API to do almost nothing"""
import hug
import gensim.downloader as api

info = api.info()  # show info about available models/datasets
model = api.load("glove-twitter-25") # download the model and return as object ready for use
print(api.load("glove-twitter-25", return_path=True)) 
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
        'email': u'ssampele@example.com',
        'phone': u'345-678-9012'
    }
]


# Helper methods

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

# Input: text
# Output: Tree JSON
@hug.get() 
@hug.cli()
def add(text: hug.types.text, categories:hug.types.text):
    """Returns the result of adding number_1 to number_2"""
    return text +" Cat: " +categories + "Willie McBride"

@hug.get() 
@hug.cli()
def query_array(input: hug.types.text):
    """Assuming input to be an array of strings"""

    # General Inports
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download('wordnet')

    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords

    # Importing GloVe

    glove_input_file = 'glove_50d.txt'
    word2vec_output_file = 'word2vec.txt.word2vec'
    glove2word2vec(glove_input_file, word2vec_output_file)

    from gensim.models import KeyedVectors

    # Loading the word2vec model
    glove_w2v = KeyedVectors.load_word2vec_format(word2vec_output_file, binary = False)

    tokenizer = nltk.data.load('tokenizer/punkt/english.pickle')
    stop_words.add("is")

    unprocessed_raw_sentences = tokenizer.tokenize(corpus_raw)

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

