""" tensorflow_model.py

    Tools for building ML models with TensorFlow

"""
import os, sys, re
import numpy as np
import tensorflow as tf

from gmutils.utils import err, argparser
from gmutils.model import Model

################################################################################
# CONFIG

default = {
    'batch_size'         : 100,
    'epochs'             : 50,
    'vec_dtype'          : tf.float16,
    'learning_rate'      : 0.01,
}

################################################################################
# OBJECTS

class TensorflowModel(Model):
    """
    An object to manage the training, storage, and utilizating of TensorFlow models

    Attributes  (depends on subclass)
    ----------
    label : str
        The supervised label

    model : a TensorFlow Model

    model_dir : str
        Path to where the model is stored

    model_file : str
        File path where the model is stored

    estimators : array of str
        Names of the underlying estimator(s) to be used

    """
    def __init__(self, options=None):
        """
        Instantiate the object and set options
        """
        self.set_options(options, default)
        super().__init__(options)


    def print(self, tensors):
        """
        Make it easier to generate a print statement.  Prints multiple tensors, but shape info only about the first one.

        Parameters
        ----------
        tensors : array of Tensor
            Tensors to be printed

        """
        null_a = tf.Print(tensors[0], tensors, "\ninput %s  "% str(tensors[0].get_shape()), summarize=100 )
        return tf.identity(null_a, name='null')
            
        
    def local_string(self, name, value):
        # return tf.get_variable(name, shape=(), dtype=tf.string, initializer=init, collections=[tf.GraphKeys.LOCAL_VARIABLES])
        return tf.Variable(value, name=name, dtype=tf.string, collections=[tf.GraphKeys.LOCAL_VARIABLES])

    
    def local_vec(self, name, shape):
        """
        Generate a local variable for use by a TF graph
        """
        if isinstance(shape, int):
            shape = (shape)
        return tf.get_variable(name, shape=shape, dtype=tf.float16, collections=[tf.GraphKeys.LOCAL_VARIABLES])

    
    def placeholder_float(self):
        """
        Generate a placeholder float for use by a TF graph
        """
        return tf.placeholder(self.get('vec_dtype'), shape=(1))
                                              
        
    def placeholder_vec(self, shape):
        """
        Generate a placeholder vec for use by a TF graph
        """
        if isinstance(shape, int):
            shape = (shape)
        return tf.placeholder(self.get('vec_dtype'), shape=shape)
                                              

    def node_tensor(self):
        """
        Return a basic placeholder for a Node
        """
        return tf.placeholder(self.get('vec_dtype'), shape=(self.get('dim')))


    def node_layer(self):
        """
        Return a basic dense layer for processing a node tensor
        """
        x = tf.placeholder(self.get('vec_dtype'), shape=[None, 3])
        linear_model = tf.layers.Dense(units=self.get('dim'))
        y = linear_model(x)


        
    
################################################################################
# FUNCTIONS


################################################################################
##   MAIN   ##

if __name__ == '__main__':
    parser = argparser_ml({'desc': "Tools to train and use TensorFlow models: tensorflow_model.py"})
    args = parser.parse_args()   # Get inputs and options

        
################################################################################
################################################################################
