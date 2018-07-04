import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import cudnn_rnn
from pprint import pprint
from tensorflow.python import debug

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

class LSTMModel:
    def __init__(self, nodes, layers, num_unrolling,
                 input_dim=1, output_dim=1, alpha=0.001, mode='C'):
        self.alpha = alpha
        self.dimI = input_dim
        self.dimO = output_dim
        self.graph = tf.Graph()

        self.create_model(nodes, layers, num_unrolling, mode)
        self.sess = tf.InteractiveSession(graph=self.graph)
        #self.sess = debug.LocalCLIDebugWrapperSession(self.sess)
        self.saver = tf.train.Saver()

        self.initialize()

        self.mode = mode

    def create_model(self, nodes, layers, num_unrolling, mode):
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, num_unrolling, self.dimI])
            self.q = tf.unstack(self.x, axis=1)

            self.keep_prob = tf.placeholder(tf.float32)

            def lstm_cell():
                cell = rnn.BasicLSTMCell(nodes, forget_bias=1.0)
                #cell = rnn.GridLSTMCell(nodes, num_frequency_blocks=3)
                return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            self.stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                [lstm_cell() for _ in range(layers)])
            self.outputs, self.states = rnn.static_rnn(self.stacked_lstm,
                                                       self.q, dtype=tf.float32)

            self.w = weight_variable([nodes, self.dimO])
            self.b = bias_variable([self.dimO])

            self.out = tf.matmul(self.outputs[-1], self.w) + self.b

            self.y_ = tf.placeholder(tf.float32, [None, self.dimO])

            if mode == 'C':
                self.prediction = tf.nn.softmax(self.out)
                self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.out))
                self.label_p = tf.argmax(self.out, axis=1)
                self.label_y = tf.argmax(self.y_, axis=1)
                self.matrix = tf.contrib.metrics.confusion_matrix(self.label_y, self.label_p)
                self.cor = tf.equal(self.label_p, self.label_y)
                self.accuracy = tf.reduce_mean(tf.cast(self.cor, tf.float32))
            elif mode == 'R':
                self.error = tf.reduce_mean(tf.square(self.out - self.y_))
                self.prediction = tf.identity(self.out)

            self.adam = tf.train.AdamOptimizer(self.alpha)
            self.optimizer = self.adam.minimize(self.error)
            self.grad = self.adam.compute_gradients(self.error)

            #self.accuracy = tf.metrics.accuracy(labels=self.y_, predictions=self.prediction)

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def train(self, x, y, keep_prob=1.0):
        feed_dict = {self.x: x, self.y_: y, self.keep_prob: keep_prob}
        self.sess.run(self.optimizer, feed_dict=feed_dict)


    def predict(self, x):
        feed_dict = {self.x: x, self.keep_prob: 1.0}
        return self.sess.run(self.prediction, feed_dict=feed_dict)


    def performance(self, x, y):
        error = 0; acc = 0
        for key in x.keys():
            feed_dict = {self.x: x[key], self.y_: y[key], self.keep_prob: 1.0}
            if self.mode == 'C':
                tmp1, tmp2 = self.sess.run([self.error, self.accuracy], feed_dict=feed_dict)
                error += tmp1; acc += tmp2
            else:
                error += self.sess.run(self.error, feed_dict=feed_dict)

        return (error/len(x.keys()), acc/len(x.keys())) if self.mode == 'C' else error/len(x.keys())

    def confusionMatrix(self, x, y):
        feed_dict = {self.x : x, self.y_ : y, self.keep_prob : 1.0}
        return self.sess.run(self.matrix, feed_dict=feed_dict)

    def save(self, filedir):
        self.saver.save(self.sess, filedir)
        self.saver.export_meta_graph(filedir+'.meta')

    def restore(self, filedir):
        new_saver = tf.train.import_meta_graph(filedir+'.meta')
        new_saver.restore(self.sess, filedir)
