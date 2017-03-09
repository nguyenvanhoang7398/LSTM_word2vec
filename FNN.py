import tensorflow as tf
import pickle
import numpy as np
from nltk import word_tokenize, WordNetLemmatizer

class FNN:
    def __init__(self, n_node_hl1=500, n_node_hl2=500, n_class=2, vocab_size=30000, n_doc=50000,
                 batch_size=128, word_embedding_size = 256, doc_embedding_size=256,
                 word_embeddings_file='word_embeddings.pickle', doc_embedding_file='doc_embeddings.pickle',
                 doc_dataset_pickle='docs_dataset.pickle',
                 data_file='shuffled-labeled-data.csv',
                 test_size=0.1,
                 log='FNN_log.pickle', path='/FNN_model/'):
        print('Creating new Feed-forward Neural Network (FNN) model')

        self.n_node_hl1 = n_node_hl1
        self.n_node_hl2 = n_node_hl2
        self.n_class = n_class
        self.n_doc = n_doc
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.word_embedding_size = word_embedding_size
        self.doc_embedding_size = doc_embedding_size
        self.data_file = data_file
        self.test_size = test_size

        with open(word_embeddings_file, 'rb') as f:
            self.word_embeddings = pickle.load(f)

        with open(doc_embedding_file, 'rb') as f:
            self.doc_embeddings = pickle.load(f)

        with open(doc_dataset_pickle, 'rb') as f:
            self.doc_idx, self.word_idx, self.count, self.dictionary, self.reverse_dictionary = pickle.load(f)

        self.log = log
        self.path = path

        self.x = tf.placeholder('float')
        self.y = tf.placeholder('float')

        self.hidden_1_layer = {'f_fum': self.n_node_hl1,
                          'weights': tf.Variable(tf.random_normal([self.doc_embedding_size, self.n_node_hl1])),
                          'biases': tf.Variable(tf.random_normal([self.n_node_hl1]))}

        self.hidden_2_layer = {'f_fum': self.n_node_hl2,
                          'weights': tf.Variable(tf.random_normal([self.n_node_hl1, self.n_node_hl2])),
                          'biases': tf.Variable(tf.random_normal([self.n_node_hl2]))}

        self.output_layer = {'f_fum': None,
                        'weights': tf.Variable(tf.random_normal([self.n_node_hl2, self.n_class])),
                        'biases': tf.Variable(tf.random_normal([self.n_class])), }

        self.saver = tf.train.Saver()

    def create_model(self, data):
        l1 = tf.add(tf.matmul(data, self.hidden_1_layer['weights']), self.hidden_1_layer['biases'])
        l1 = tf.nn.relu(l1)
        l2 = tf.add(tf.matmul(l1, self.hidden_2_layer['weights']), self.hidden_2_layer['biases'])
        l2 = tf.nn.relu(l2)
        output = tf.matmul(l2, self.output_layer['weights']) + self.output_layer['biases']
        return output

    def train_neural_network(self, train_inputs, train_labels, n_epoch=10):

        prediction = self.create_model(self.x)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            try:
                with open(self.log, 'rb') as f:
                    epoch = pickle.load(f) + 1
                    print('Starting with epoch', epoch)

            except:
                epoch = 1

            while epoch <= n_epoch:

                if epoch != 1:
                    self.saver.restore(sess, self.path + 'model.ckpt')

                epoch_loss = 0

                batch_x = []
                batch_y = []
                batches_run = 0

                for i, doc_vector in enumerate(train_inputs):
                    line_x = list(doc_vector)
                    if train_labels[i] == '1':
                        line_y = [1, 0]
                    else:
                        line_y = [0, 1]
                    batch_x.append(line_x)
                    batch_y.append(line_y)

                    if len(batch_x) >= self.batch_size:
                        _, c = sess.run([optimizer, cost], feed_dict={self.x: batch_x, self.y: batch_y})

                        epoch_loss += c

                        batch_x = []
                        batch_y = []
                        batches_run += 1

                self.saver.save(sess, save_path=self.path + 'model.ckpt')
                print('Epoch', epoch, 'completed out of', n_epoch, 'loss:', epoch_loss)
                with open(self.log, 'wb') as f:
                    pickle.dump(epoch, f)
                epoch += 1

    def test_neural_network(self, test_inputs, test_labels, n_epoch=10):
        prediction = self.create_model(self.x)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for epoch in range(n_epoch):
                try:
                    self.saver.restore(sess, self.path+ "model.ckpt")
                except Exception as e:
                    print(str(e))

                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

                feature_sets = []
                labels = []
                counter = 0

                for i, doc_vector in enumerate(test_inputs):
                    try:
                        if test_labels[i] == '1':
                            label = [1, 0]
                        else:
                            label = [0, 1]
                        feature_sets.append(doc_vector)
                        labels.append(label)
                        counter += 1
                    except Exception as e:
                        print(str(e))

                print('Tested', counter, 'samples.')
                test_x = np.array(feature_sets)
                test_y = np.array(labels)
                print('Accuracy', accuracy.eval({self.x: test_x, self.y: test_y}))