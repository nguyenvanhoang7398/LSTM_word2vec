from FNN import FNN
from preprocessing_data import create_train_test_data

train_inputs, train_labels, test_inputs, test_labels = create_train_test_data('doc_embeddings.pickle', 'shuffled-labeled-data.csv', test_size=0.1)
fnn = FNN()
fnn.train_neural_network(train_inputs=train_inputs, train_labels=train_labels, n_epoch=10)
fnn.test_neural_network(test_inputs=test_inputs, test_labels=test_labels)