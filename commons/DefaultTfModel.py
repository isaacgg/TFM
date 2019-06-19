# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:37:31 2018

@author: isaac
"""

import tensorflow as tf
import os

class DefaultTfModel():
    _graph = None
    _session = None
    _frozen_session = None
    _frozen_graph = None

    #Open and close session
    def _open_session(self, config = None):
        if self._session is None:
            print("open session")
            self._session = tf.InteractiveSession(graph = self._graph, config = config)

    def _open_frozen_session(self):
        if self._frozen_session is None:
            print("open session")
            self._frozen_session = tf.InteractiveSession(graph = self._frozen_graph)  
    
    def close_session(self):
        if self._session is not None:
            self._session.close()
            self._session = None
        elif self._frozen_session is not None:
            self._frozen_session.close()
            self._frozen_session = None
        else:
            print("There's no open session")
    
        #Imports and exports
    def load_model(self,name):
        self._open_session()
        self.saver.restore(self._session, self.default_folder +name+"/" +  name, )
        
    def freeze_model(self, name, output_node = "Restricted_Boltzmann_Machine/correlation_positive/clamp_phase/h_out"):
        output_graph =  name + "/model.pb"
        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self._session, # The session is used to retrieve the weights
            self._session.graph.as_graph_def(),
#            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node.split(",") # The output node names are used to select the usefull nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        return output_graph_def
    
    def restore_checkpoint(self, logdir):
        self._open_session()
        self.saver.restore(self._session, os.path.join(logdir, "model.ckpt"))
    
    def load_frozen_model(self, name, input_node, output_node):
        self._frozen_graph = tf.Graph()
        frozen_graph = name + "model.pb"
        
        with tf.gfile.GFile(frozen_graph, "rb") as f:
            restored_graph_def = tf.GraphDef()
            restored_graph_def.ParseFromString(f.read())
        
        with self._frozen_graph.as_default() as graph:
            tf.import_graph_def(restored_graph_def,
                                input_map=None,
                                return_elements=None,
                                name=""
            )
            
        ## NOW the complete graph with values has been restored
        self.frozen_output = graph.get_tensor_by_name(output_node+":0")
        ## Let's feed the images to the input placeholders
        if input_node is not None:
            self.frozen_input= graph.get_tensor_by_name(input_node+":0")
        
    def get_features(self, input_data):
        self._open_frozen_session()
        out = self._frozen_session.run(self.frozen_output, feed_dict = {self.frozen_input: input_data})
        return out
    
    def get_graph(self):
        return self._graph
    
    def __init__(self):
        self._graph = tf.Graph()