#!/usr/bin/env python
"""A simple API to do almost nothing"""
import hug
@hug.default_input_format("application/json")

@hug.post()
def getIntents(body):
  """  GETS TEXT IN JSON"""
    parsed = hug.input_format.json(body) 
    return parsed

@hug.local()
@hug.post()
def treeify(body):
      """  GETS FLAT ARRAY IN JSON"""
    parsed = hug.input_format.json(body) 
    return parsed

if __name__ == '__main__':
    user.interface.local()
