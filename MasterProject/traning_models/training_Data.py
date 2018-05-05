#encode the training document as a integers
#1:load the vocabulary and filter out the words (not interested in)
from os import listdir
import os
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import Counter
import re

#####filter those minioccurence "2"

#load a doc into memory
def load_document(filename):
    file = open(filename,'r')
    text = file.read()
    file.close()
    return text

#integrate all the files into a string
#split by whitespace, remove the punctuation , fiter out thoses are not in the voca
def filter_doc(text, vocab):
    #print(text)
    # 1:remove tags from the text
    text = re.sub("<.*?>", "\n", text)
    # 2:split the sentences into a list
    tokens = text.split("\n")
    #print("2:split the sentences into a list")
    # 3:lowercase
    tokens = [token.lower() for token in tokens]
    #4:remove the punctuations
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens if w != '']
    #5:split the tokens -> token
    tokens1 = split_tokens_to_token(tokens)
    #6:filter out tokens not in the vocab
    tokens = [w for w in tokens1 if w in vocab]
    tokens = ' '.join(tokens)

    return tokens

# 5:split the tokens
def split_tokens_to_token(tokens):
    # 5:tokenize and remove those are not alphabate
    tokens1 = []
    for token in tokens:
        #print(token)
        #print("\n")
        token2 = token.split()
        tokens1 += token2

    return tokens1

#load all the document in the direct
def load_directory(dir, vocab):
    documents = list()
    for filename in listdir(dir):

        if not filename.endswith('.txt'):
            continue
        path = dir + '/' + filename
        text = load_document(path)
        tokens = filter_doc(text, vocab)
        #add strings of all the document into a list -> string
        documents.append(tokens)

    return documents

#save to a file
def save_to_file(doc,filename):
    file = open(filename,'w')
    file.write(doc)
    file.close()


#-------------------------main---------------------------------------------------

#min_occurrence is 2

vocab_name = 'review_vocab.txt'
vocab = load_document(vocab_name)
vocab = vocab.split()
vocab = set(vocab)

positive = load_directory('/Users/xiaoyiwen/Desktop/datasets/train/pos',vocab)
print(len(positive))
negative = load_directory('/Users/xiaoyiwen/Desktop/datasets/train/neg',vocab)
print(len(negative))
train_docs = positive + negative
print(len(train_docs))










