from numpy import asarray
from MasterProject.data_Preprocessing.Datasets import preprocessing

#load the whole embedding into memory


def get_embedding_dict(filename):
    GLove_embeddings = dict()
    file = open(filename)

    for line in file:
        value = line.split()
        word = value[0]
        weight = asarray(value[1:],dtype="float32")
        GLove_embeddings[word] = weight
    file.close()

    return GLove_embeddings




glove_name = '/Users/xiaoyiwen/Desktop/glove.6B.100d.txt'

glove_embedding = get_embedding_dict(glove_name)
print(len(glove_embedding))