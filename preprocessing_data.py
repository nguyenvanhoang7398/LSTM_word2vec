import pickle
import numpy as np

def create_train_test_data(docs_embeddings_pickle, data_file, test_size=0.1):
    print('Run create_docs_pickle(', docs_embeddings_pickle, ',', data_file, ')')
    train_inputs = []
    train_labels = []
    test_inputs = []
    test_labels = []

    with open(docs_embeddings_pickle, 'rb') as f:
        docs_embeddings = pickle.load(f)

    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('train_pos') or line.startswith('test_pos'):
                if np.random.rand() < test_size:
                    test_labels.append('1')
                    test_inputs.append(docs_embeddings[i])
                else:
                    train_labels.append('1')
                    train_inputs.append(docs_embeddings[i])
            if line.startswith('train_neg') or line.startswith('test_neg'):
                if np.random.rand() < test_size:
                    test_labels.append('0')
                    test_inputs.append(docs_embeddings[i])
                else:
                    train_labels.append('0')
                    train_inputs.append(docs_embeddings[i])


    return train_inputs, train_labels, test_inputs, test_labels