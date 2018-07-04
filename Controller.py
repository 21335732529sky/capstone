import tensorflow as tf
import numpy as np
import random

class Controller:
    def __init__(self, layers, nodes):
        self.graph = tf.Graph()
        
    def create_graph(self):
        with self.graph.as_default():
            