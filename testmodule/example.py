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
    '''
    Docstring for example class. This class can be used for nothing much
    in particular, its just an example. 


    Attributes:
        name (str)
            The name of the object. 

    '''
    
    def __init__(self, name):
        self.name = name

    def method(self, prop):
        '''
        Docstring for example method.

        Args:
            prop (str)
                An arbitrary property to assign to ExampleClass. Not used for anything.
        '''
        return f"{self.name} is {prop}"