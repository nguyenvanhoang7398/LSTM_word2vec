import tensorflow as tf
import pickle
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell

class LSTM:
    def __init__(self, n_classes=2, vocab_size=30000, n_doc=50000, max_doc_size=500, word_embedding_size = 256,
                 batch_size=128, rnn_size=128,
                 word_embeddings_file='word_embeddings.pickle',
                 doc_dataset_pickle='docs_dataset.pickle',
                 data_file='shuffled-labeled-data.csv',
                 test_size=0.1, log='LSTM_log.pickle', path='/LSTM_model/'):
        print('Creating new Long-Short Term Memory (LSTM) model')

        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.n_doc = n_doc
        self.max_doc_size = max_doc_size
        self.word_embedding_size = word_embedding_size
        self.batch_size = batch_size
        self.rnn_size = rnn_size
        self.word_embeddings_file = word_embeddings_file
        self.doc_dataset_pickle = doc_dataset_pickle
        self.data_file = data_file
        self.test_size = test_size

        with open(word_embeddings_file, 'rb') as f:
            self.word_embeddings = pickle.load(f)
        with open(doc_dataset_pickle, 'rb') as f:
            self.doc_idx, self.word_idx, self.count, self.dictionary, self.reverse_dictionary = pickle.load(f)

        self.log = log
        self.path = path

        x = tf.placeholder('float', [None, self.max_doc_size, word_embedding_size])
        y = tf.placeholder('float')

        self.saver = tf.train.Saver()

    def create_data_vector(self, doc_word2vec_outfile):

        count=0
        doc_word2vec = []
        for i in range(self.n_doc):
            doc = []
            for j in range(self.max_doc_size):
                if self.doc_idx[count] != i or count >= len(self.doc_idx):
                    empty_vector = np.ndarray(shape=(self.word_embedding_size,))
                    empty_vector.fill(0)
                    doc.append(empty_vector)
                else:
                    doc.append(self.word_embeddings[self.word_idx[count]])
                    count += 1
            doc_word2vec.append(doc)

        with open(doc_word2vec_outfile, mode='a') as f:
            pickle.dump(doc_word2vec, f)

    def create_model(self, data):

        layer = {'weights': tf.Variable(tf.random_normal([self.rnn_size, self.n_classes])),
                 'biases': tf.Variable(tf.random_normal([self.n_classes]))}

        data = tf.transpose(data, [1, 0, 2])
        data = tf.reshape(data, [-1, self.word_embedding_size])
        data = tf.split(0, self.max_doc_size, data)

        lstm_cell = rnn_cell.BasicLSTMCell(self.rnn_size)
        outputs, states = rnn.rnn(lstm_cell, data, dtype=tf.float32)

        output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

        return output

    def generate_batch_lstm(self):
        pass

