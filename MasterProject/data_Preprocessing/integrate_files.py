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

print(vocab)
print(len(vocab))


# #min_occurrence is 5
#
# vocab_name = 'vocab_min_5.txt'
# vocab_5 = load_document(vocab_name)
# vocab_5 = vocab_5.split()
# vocab_5 = set(vocab_5)
#
# print(vocab_5)
# print(len(vocab_5))
#
#
# # min_occuren 8
#
# vocab_name = 'vocab_min_8.txt'
# vocab_8 = load_document(vocab_name)
# vocab_8 = vocab_8.split()
# print("vocab-----size")
# print(len(vocab_8))
# vocab_8 = set(vocab_8)
# print(len(vocab_8))
#
# print(vocab_8)
# print(len(vocab_8))
#
#
# #_----------laod the training reviews------
#
#
# positive = load_directory('/Users/xiaoyiwen/Desktop/datasets/train/pos1',vocab)
#
# negative = load_directory('/Users/xiaoyiwen/Desktop/datasets/train/neg1',vocab)
#
# train_docs = positive + negative
#
# print("train_docs222222")
# print(train_docs)
# print(len(train_docs))
#
#
# #-------------5--------
# positive1 = load_directory('/Users/xiaoyiwen/Desktop/datasets/train/pos1',vocab_5)
#
# negative1 = load_directory('/Users/xiaoyiwen/Desktop/datasets/train/neg1',vocab_5)
#
# train_docs = positive1 + negative1
#
# print("train_docs55555555")
# print(train_docs)
# print(len(train_docs))
#
#
# #---------8-----------
# positive2 = load_directory('/Users/xiaoyiwen/Desktop/datasets/train/pos1',vocab_8)
# print(positive2)
# print(len(positive2))
#
# negative2 = load_directory('/Users/xiaoyiwen/Desktop/datasets/train/neg1',vocab_8)
# print(negative2)
# print(len(negative2))
# train_docs = positive2 + negative2
#
# print("train_docs888888")
# i = 0
# sum = 0
# for doc in train_docs:
#     i = i+1
#     print(doc)
#     print(len(doc))
#     sum += len(doc)
#
#
#
# print(train_docs)
# print(i)
# print(sum)