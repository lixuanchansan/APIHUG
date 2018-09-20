#!/usr/bin/env python
"""A simple API to do almost nothing"""
import hug

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

@hug.post()
def getIntents(body):
    """Returns the result of parsing through nlp engine"""
    return body

@hug.local()
@hug.post()
def treeify():
    return {'users': users}

if __name__ == '__main__':
    user.interface.local()
