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
    # 'dtype'              : tf.float32,
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

    graph : tf.Graph

    dim : int (optional)

    """
    def __init__(self, graph, design, options={}):
        """
        Instantiate the object and set options

        graph : tf.Graph

        design : array of dict
            A representation of any layers comprising this object -- what they are like and how they are connected

        """
        self.set_options(options, default)
        self.graph = graph
        self.layers = []
        for desc in design:
            self.layers.append( self.generate_layer(desc) )


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
            output = tf.layers.Dense(units=desc.get('units'), activation=desc.get('activation'), name=desc.get('name'))
            return output
            """
            output = self.dense(units=desc.get('units'), activation=desc.get('activation'), name=desc.get('name'))
            return output
            """

        elif desc['type'] == 'lstm':
            output = self.LSTM(units=desc.get('units'), activation=desc.get('activation'), name=desc.get('name'))
            return output
            
        else:
            err([], {'ex':"No Layer type selected!"})


    def generate(self, input):
        """
        Generate the layers and connect them according to the design.  To be used during building of the tf.Graph.

        Assumes some 'input' tensor. The output shape of which will define the input shape of the first layer of this object.

        Parameters
        ----------
        input : Tensor

        Returns
        -------
        Tensor

        """
        for i, layer in enumerate(self.layers):
            input = layer(input)
        return input
    

    def dense(self, units, activation, name):
        """
        Generate a dense layer to be incorporated
        """
        def basic_dense(inputs, units, activation, name):
            return tf.layers.dense(inputs, units, activation=activation, name=name)
            
        return lambda x: basic_dense(x, units, activation, name)

    
    def LSTMCell(self, units):
        """
        Generate another LSTMCell to be incorporated
        """
        # return tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=8, num_units=self.get('dim'), dtype=self.get('dtype')
        return tf.contrib.rnn.BasicLSTMCell(units, forget_bias=1.0)
    
            
    def LSTM(self, units, activation, name):
        """
        Generate a bidirectional LSTM layer
        """
        # activation ignored for now

        def dynamic_brnn(name, inputs):
            num_layers = 2   # EXPERIMENT with this
            lstm_fw_cells = [self.LSTMCell(units) for _ in range(num_layers)]
            lstm_bw_cells = [self.LSTMCell(units) for _ in range(num_layers)]

            # bookend = tf.expand_dims(tf.zeros_like(inputs[0]), 0)
            bookend = tf.zeros_like(inputs[0])
            final = [bookend]
            for x in inputs:
                # final.append( tf.expand_dims(x, 0) )
                final.append( x )
            final.append(bookend)
            final = tf.stack(final)
            outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(lstm_fw_cells, lstm_bw_cells, final, dtype=self.get('dtype'), scope=name)
            outputs = tf.reshape(outputs, [-1, units])
            return outputs
        
        return lambda x: dynamic_brnn(name, x)

            
################################################################################
##   MAIN   ##

if __name__ == '__main__':
    parser = argparser_ml({'desc': "Tools to train and use TensorFlow models: tensorflow_model.py"})
    args = parser.parse_args()   # Get inputs and options

        
################################################################################
################################################################################
