""" tensorflow_graph.py

    Template for building TensorFlow graphs

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

class TensorflowGraph(Object):
    """
    IMPORTANT: below the word "graph" (lowercase) refers to a set of connected tf Tensors.  It does NOT correspond exactly to a tf.Graph.  The same is
    true about the word "Graph" in the name of this object.

    An object to assist the generation of TensorFlow graphs.  It is not a "model", but rather a generalization of the
    concept of a layer.  Whereas a typicaly TensorFlow layer might be multidimensional, it is almost always some type of hyperrectangle.  This object
    facilitates construction of graphs which are more complex.

    Attributes  (depends on subclass)
    ----------
    finals : list of final output Tensors

    feed_dict : dict
        dict of vars to be fed into graph just before running.  Best case scenario run once

    to_print : list of tf Tensor
        array of tensors to print when the session is run

    saver : a tf.train.Saver
        For the purpose of saving the current state

    graph : tf.Graph

    """
    def __init__(self, options=None):
        """
        Instantiate the object and set options
        """
        self.set_options(options, default)
        self.feed_dict = {}
        self.finals = []
        self.to_print = []


    def generate(self):
        """
        Configure or set up.  Overridden in the subclass
        
        Parameters
        ----------
        sess : tf Session

        """
        self.configure_saver()


    def save(self, sess, path):
        """
        Use a tf Saver to save the state of the model.
        """
        self.saver.save(sess, path)
            
        
    def get_targets(self):
        """
        Return a list of final tensor targets in the graph
        """
        return self.to_print + self.finals
        

    def add_to_feed(self, placeholder, val):
        """
        Add this placeholder and val to self.feed_dict
        """
        self.feed_dict[placeholder] = [val]   # listified to add a dimension
            

    def add_node(self, val, name=None):
        """
        Add a new node to the graph and add it's val to the feed_dict
        """
        placeholder = self.node_placeholder(name)
        self.add_to_feed(placeholder, val)
        return placeholder


    def get_learning_rate(self):
        """
        Add a new node to the graph and add it's val to the feed_dict
        """
        # placeholder = self.float_placeholder('learning_rate')
        placeholder = tf.placeholder(tf.float16, shape=[])
        return placeholder


    def add_final(self, T):
        """
        Add a tf Tensor T to the list of finals
        """
        self.finals.append(T)
        
        
    def empty_float(self):
        """
        Get the empty float and use it again.  It's a constant and it's empty
        """
        if not self.done():
            enode = tf.constant(0.0, dtype=self.get('dtype'), shape=(1, 1))
            self.set('empty_float', enode)
        return self.get('empty_float')

        
    def empty_node(self):
        """
        Get the empty node and use it again.  It's a constant.  Assumes all node vectors same dim
        """
        if not self.done():
            enode = tf.constant( [0.0]*self.get('dim'), dtype=self.get('dtype'), shape=(1, self.get('dim')) )
            self.set('empty_node', enode)
        return self.get('empty_node')
            
            
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
            try:
                shape = str(len(tensor))
            except:
                shape = '(unknown)'
        try:
            tensor_type = str(type(tensor))
            tensor_type += ' ' + tensor.dtype
        except:
            pass
        
        text = "\n\nTENSOR (%s) shape=%s : |%s|\n"% (tensor_type, shape, text)
        nullT = tf.Print(tensor, [tensor], text, summarize=10000 )
        self.to_print.append(nullT)
        
            
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
            
        pl = tf.placeholder(self.get('dtype'), shape=[None, 1], name=name)
        return pl


    def string_placeholder(self, name=None):
        """
        Return a basic placeholder for a float
        """
        try:
            self.placeholder_i += 1
        except:
            self.placeholder_i  = 1
        if name is not None:
            name += '_' + str(self.placeholder_i)
            
        pl = tf.placeholder(tf.string, name=name)
        return pl


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
            
        pl = tf.placeholder(self.get('dtype'), shape=[None, self.get('dim')], name=name)
        return pl
    

    def node_layer(self, name):
        """
        Return a basic dense layer for processing a node tensor
        """
        layer = tf.layers.Dense(units=self.get('dim'), activation=self.get('activation'), name=name)
        return layer
    

    def average(self, tensors):
        """
        Output a tensor which is the average of the inputs

        Parameters
        ----------
        array of TF Tensors all having identical shape
        """
        avg = tf.reduce_mean(tensors, 0)
        return avg


    def print_feed_dict(self):
        """
        Show some info about the feed dict
        """
        print("\nFEED DICT:")
        for k,v in self.feed_dict.items():
            print(k, ':', type(v))
        print()


    def reduce(self, layer, arr):
        """
        Use 'layer' to connect elements in the array 'arr' of Tensors, generating a binary tree (consumes two at a time).

        if there are 3 input tensors in 'arr', layer gets created having input shape of twice that of each element in arr.  In the graph, there is now an
        output_1 connected through 'layer' to arr[1], arr[2].  After the next iteration, there is now and output_2 connected through 'layer' to arr[0], output_1

        arr[0]   arr[1]    arr[2]
          |        |         |
          |        ---layer---
          |             |
          -----layer-----
                 |

        In that way, it proceeds from end to start
        """
        if len(arr) == 1:   # if only one tensor, just send it back unchanged
            err(['Unchanged Tensor:', arr])
            return arr
        
        output = arr[-1]    # Contains last tensor in arr, for example: arr[2]
        i = len(arr)
        j = i - 1           # Contains the penultimate index, for example: 1
        while j > 0:
            j -= 1
            stacked = tf.concat([arr[j], output], axis=1)
            output = layer.generate(stacked)
            
        return output


################################################################################
##   MAIN   ##

if __name__ == '__main__':
    parser = argparser_ml({'desc': "Tools to train and use TensorFlow models: tensorflow_model.py"})
    args = parser.parse_args()   # Get inputs and options

        
################################################################################
################################################################################
