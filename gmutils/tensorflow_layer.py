""" tensorflow_layer.py

    Template for building TensorFlow Layers and combinations thereof

"""
import os, sys, re
import numpy as np
import tensorflow as tf

from gmutils.utils import err, argparser
from gmutils.objects import Object

################################################################################
# CONFIG

default = {
    'dtype'              : tf.float16,
    'activation'         : tf.nn.relu,
}

################################################################################
# OBJECTS

class TensorflowLayer(Object):
    """
    IMPORTANT: below the word "graph" may be used in the documentation below in different senses -- either referring to a tensorflow Graph (tf.Graph)
    or the abstract mathematical concept from Graph Theory.  I will try to be clear.  Where I remember, I will likely use lowercase to refer to the
    abstract mathematical concept.

    This object is to assist the construction and interconnection of layers in TF (tf.Layer objects).  The reason this object exists is because the
    functional interface (e.g. tf.layers.dense()) is lacking.  Perhaps this is what motivated the Keras folks...  For now it will be similar to a 
    keras Sequential model.

    Attributes
    ----------
    layers : array of tf.Layer

    """
    def __init__(self, graph, options={}):
        """
        Instantiate the object and set options

        graph : array of dict (NOT a tf.Graph)
            A representation of any layers comprising this object -- what they are like and how they are connected

        """
        self.set_options(options, default)
        self.layers = []
        for desc in graph:
            self.layers.append( self.generate_layer(desc) )


    def generate(self, input):
        """
        Generate the layers and connect them according to the graph

        Assumes some 'input' tensor as a starting point.  The output shape of which will define the input shape of the
        first layer of this object.

        Returns
        -------
        tf.Layer

        """
        output = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                output = layer(input)
            elif i > 0:
                output = layer(output)

        return output
    

    def generate_layer(self, desc):
        """
        Generate a tf.Layer object based on a description

        Parameters
        ----------
        desc : dict
            { 'type' : str,
              'units' : int,
              'activation' : tf.nn.activation,
              'name' : str }

        Returns
        -------
        tf.Layer

        """
        if desc['type'] == 'dense':
            return tf.layers.Dense(units=desc.get('units'), activation=desc.get('activation'), name=desc.get('name'))

        return None


    
################################################################################
# FUNCTIONS


################################################################################
##   MAIN   ##

if __name__ == '__main__':
    parser = argparser_ml({'desc': "Tools to train and use TensorFlow models: tensorflow_model.py"})
    args = parser.parse_args()   # Get inputs and options

        
################################################################################
################################################################################
