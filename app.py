#!/usr/bin/env python
"""A simple API to do almost nothing"""
import hug
import nltk
from gensim.scripts.glove2word2vec import glove2word2vec

def generate_word_list(raw_sentence, lemmatized_words):
    word_list = []
    
    words = re.sub("[^a-zA-Z]"," ", raw_sentence)
    words = words.split()
    
    for word in words:
        lemmatized_word = lemmatizer.lemmatize(word.lower())
        if lemmatized_word in lemmatized_words:
            word_list.append(lemmatized_word)
        
    return word_list

def generate_categories_in_sentence(word_list, categories):
    categories_in_sentence = set()
    
    for word in word_list:
        for category in categories:
            if word in category.words:
                categories_in_sentence.add(category)
    
    return list(categories_in_sentence)
    
class Sentence:
    def __init__(self, raw_sentence, categories, lemmatized_words):
        self.raw_sentence = raw_sentence
        self.word_list = generate_word_list(raw_sentence, lemmatized_words)
        self.child_cat = generate_categories_in_sentence(self.word_list, categories)
        self.parent_cat = []
        
    def __repr__(self):
        return (
            'Raw_sentence: ' + self.raw_sentence
        )
    
# greedy Category implementation 

class Category:
    
    def __init__(self, word, count, threshold = 0.9):
        self.threshold = threshold
        
        self.words = set()
        self.words.add(word)
        
        self.num_of_vocab = 1 
        self.count = count
        self.average_word = word
        self.average_vector = glove_w2v.get_vector(word)
        self.most_representative_word = word
        
    def is_similar(self, word):
        return glove_w2v.similarity(self.average_word, word) > self.threshold
        
    def add_word(self, word, count):
        self.count += count
        
        if word not in self.words:
            self.words.add(word)
            
            self.average_vector = (
                glove_w2v.get_vector(word)
                + self.average_vector * self.num_of_vocab
            )/(self.num_of_vocab + 1)
            
            self.num_of_vocab += 1
            
            self.average_word = glove_w2v.similar_by_vector(self.average_vector, topn = 1)[0][0]
            self.most_representative_word = glove_w2v.most_similar_to_given(self.average_word, list(self.words))
            
    def __repr__(self):
        return 'Category: ' + self.most_representative_word

        # return (
        #     "\nA Category most represented by: " + self.most_representative_word +
        #     "\nwith num_of_vocab: " + str(self.num_of_vocab) +
        #     "\nand count of: " + str(self.count) +
        #     "\nwords:" + str(self.words) +
        #     "\n"
        # )

def select_category(filtered_categories):
    if len(filtered_categories) > 0:
        return filtered_categories[0]
    else:
        return None

class TreeStructure:
    def __init__(self, tagged_sentences, filtered_categories, root = "root"):

        self.tagged_sentences = tagged_sentences

        for sentence in self.tagged_sentences:
            new_categories = []

            for category in filtered_categories:
                for word in category.words:
                    for child_category in sentence.child_cat:
                        if word in child_category.words:
                            new_categories.append(category)
            
            set_of_cats = set(new_categories)
            sentence.child_cat = list(set_of_cats)
    
        all_categories = set()
        
        for sentence in self.tagged_sentences:
            all_categories.update(sentence.child_cat)

        self.filtered_categories = list(all_categories)
        self.root = root
        self.children = dict()
        
    def to_dictionary(self):
        return {
            'root': self.root,
            'tagged_sentences': self.tagged_sentences,
            'categories': self.filtered_categories,
            'children': self.children
        }

    def treefy(self):
        while len(self.filtered_categories) > 0:
            
            # Category Object
            selected_category = select_category(self.filtered_categories)
            
            selected_sentences = []
            
            for sentence in self.tagged_sentences:
                
                if selected_category in sentence.child_cat:
                    
                    print("The above sentence has the category")
                    print(selected_category)
                    unknown_variable = {selected_category}.pop()
                    sentence.child_cat.remove(unknown_variable)
                    sentence.parent_cat.append(selected_category)
                    selected_sentences.append(sentence)
                    
            
            print("\n\n\nSelected Sentences for category", selected_category)
            
            for sentence in selected_sentences:
                print(sentence)
                
            print()
            
            # obtain the categories from the remaining sentences in this tagged_sentences
            # TODO
            self.filtered_categories.remove(selected_category)
            
            for sentence in selected_sentences:
                self.tagged_sentences.remove(sentence)
            
            print("Updated Filtered_categories")
            print(self.filtered_categories)
            print()
            
            print("Updated Tagged Sentences: ")
            print(self.tagged_sentences)
            
            print("GOING INTO RECURSION")
            # obtain the categories from selected_sentences
            # TODO
            internal_tree_structure = TreeStructure(selected_sentences, filtered_categories, selected_category.most_representative_word)
            self.children[selected_category.most_representative_word] = internal_tree_structure.treefy()
            print("UPDATING FILTERED_CAT")
            
            updated_categories = set()
            for sentence in self.tagged_sentences:
                updated_categories.update(sentence.child_cat)
                
            # filtered_categories attribute
            self.filtered_categories = list(updated_categories)
            
        print("END OF END OF END OF RECURSION!!!!!!!!!!!!!!!!!!!")
        
        return self
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

