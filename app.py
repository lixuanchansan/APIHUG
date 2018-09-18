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

