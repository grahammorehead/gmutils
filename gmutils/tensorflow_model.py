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
    'activation'         : tf.nn.relu,
}

################################################################################
# OBJECTS

class TensorflowModel(Model):
    """
    An object to manage the training, storage, and utilizating of TensorFlow models

    Attributes  (depends on subclass)
    ----------
    sess : a TensorFlow session

    label : str
        The supervised label

    model : a TensorFlow Model

    model_dir : str
        Path to where the model is stored

    model_file : str
        File path where the model is stored


    global_initializer : TF variable initializer

    graph : initial default tf.Graph

    finals : list of final output Tensors

    feed_dict : dict
        dict of vars to be fed into graph just before running.  Best case scenario run once

    to_print : dict
        array of tensors to print when the session is run

    loss : array of tensor
        Where loss tensors are stored

    label : array of label - typically a float on (0,1)
        One-to-one correspondence with self.loss

    """
    def __init__(self, options=None):
        """
        Instantiate the object and set options
        """
        self.set_options(options, default)
        super().__init__(options)
        self.graph = tf.get_default_graph()
        self.feed_dict = {}
        self.finals = []
        self.to_print = []


    def initialize(self, sess):
        """
        Initialize all global vars in the graph
        """
        global_vars          = tf.global_variables()
        is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

        # print [str(i.name) for i in not_initialized_vars] # only for testing
        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))


    def add_to_feed(self, placeholder, val):
        """
        Add this placeholder and val to self.feed_dict
        """
        self.feed_dict[placeholder] = [val]   # listified to add a dimension
            

    def add_node(self, val):
        """
        Add a new node to the graph and add it's val to the feed_dict
        """
        placeholder = self.node_placeholder()
        self.add_to_feed(placeholder, val)
        return placeholder


    def add_final(self, T):
        """
        Add a tf Tensor T to the list of finals
        """
        self.finals.append(T)
        
        
    def empty_node(self):
        """
        Get the empty node and use it again.  It's a constant.  Assumes all node vectors same dim
        """
        if not self.done():
            self.set('empty_node', tf.constant([0.0]*self.get('dim'), dtype=self.get('dtype')) )
        return self.get('empty_node')

        
    def run(self):
        """
        Run the session
        """
        verbose = False
        with tf.Session() as sess:
            self.initialize(sess)
            targets = self.to_print + self.finals

            if verbose:
                err(self.to_print)
                err(self.finals)
                names = []
                for key in self.feed_dict.keys():
                    names.append(key.name)
                err(names)
            
            output = sess.run(targets, feed_dict=self.feed_dict)
            # print(output)
            
            
    def print(self, tensor, text):
        """
        Make it easier to generate a print statement.  Prints a tensor with shape info and some specified text

        Parameters
        ----------
        tensor : tf Tensor

        text : str

        """
        try:
            shape = str(tensor.get_shape())
        except:
            shape = str(len(tensor))
        try:
            tensor_type = str(type(tensor))
            tensor_type += ' ' + tensor.dtype
        except:
            pass
        
        text = "\n\nTENSOR (%s) shape=%s : |%s|\n"% (tensor_type, shape, text)
        nullT = tf.Print(tensor, [tensor], text, summarize=1000 )
        self.to_print.append(nullT)
        
        
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

    
    def float_placeholder(self, name=None):
        """
        Return a basic placeholder for a float
        """
        try:
            self.placeholder_i += 1
        except:
            self.placeholder_i  = 1
        if name is not None:
            name += '_' + str(self.placeholder_i)
            
        return tf.placeholder(self.get('vec_dtype'), shape=[None, 1], name=name)


    def node_placeholder(self, name=None):
        """
        Return a basic placeholder for a Node
        """
        try:
            self.placeholder_i += 1
        except:
            self.placeholder_i  = 1
        if name is not None:
            name += '_' + str(self.placeholder_i)
            
        return tf.placeholder(self.get('vec_dtype'), shape=[None, self.get('dim')], name=name)


    def node_layer(self):
        """
        Return a basic dense layer for processing a node tensor
        """
        return tf.layers.Dense(units=self.get('dim'), activation=self.get('activation'))


    def average(self, tensors):
        """
        Output a tensor which is the average of the inputs

        Parameters
        ----------
        array of TF Tensors all having identical shape
        """
        return tf.reduce_mean(tensors, 0)

        
    
################################################################################
# FUNCTIONS


################################################################################
##   MAIN   ##

if __name__ == '__main__':
    parser = argparser_ml({'desc': "Tools to train and use TensorFlow models: tensorflow_model.py"})
    args = parser.parse_args()   # Get inputs and options

        
################################################################################
################################################################################
