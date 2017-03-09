import numpy as np
import pickle

n_doc = 3
max_doc_size = 7
doc_idx = [0] * 5 + [1] * 6 + [2] * 3
word_embeddings = []
word_embedding_size = 5
for i in range(7):
    word_embeddings.append([(j+i*10) for j in range(word_embedding_size)])
word_idx = np.random.randint(low=0, size=14, high=7)


def create_data_vector():
    count = 0
    doc_word2vec = []
    for i in range(n_doc):
        doc = []
        for j in range(max_doc_size):
            print(count)
            if count >= len(doc_idx) or doc_idx[count] != i:
                empty_vector = np.ndarray(shape=(word_embedding_size,))
                empty_vector.fill(0)
                doc.append(empty_vector)
            else:
                doc.append(word_embeddings[word_idx[count]])
                count += 1
        doc_word2vec.append(doc)
    print(word_idx)
    print(doc_word2vec)

create_data_vector()

with open('word_embeddings.pickle', 'rb') as f:
    we = pickle.load(f)

print(type(we[0]), we[0].shape)