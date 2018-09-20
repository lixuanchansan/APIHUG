#!/usr/bin/env python
import hug

@hug.get('/users', versions=1)
def user(user_id):
    return 'I do nothing useful.'
    #return user_id

@hug.get()
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