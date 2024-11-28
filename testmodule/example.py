"""
Example module for documentation. 

Here I am writing some random text, which will be included in the documentation. 
Most of it is AI-generated, so it doesn't make much sense.
I'm mostly just rambling on about nothing in particular, trying to fill up space.

Example usage:


.. highlight:: python

::

    from testmodule import ExampleClass
    e = ExampleClass('test')
"""


class ExampleClass:
    '''Docstring for example class'''
    def __init__(self, name):

        '''Name of the object'''
        self.name = name

    def method(self, prop):
        '''Docstring for example method'''
        return f"{self.name} is {prop}"