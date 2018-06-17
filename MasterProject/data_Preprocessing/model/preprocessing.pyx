from os import listdir
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from gensim.models import word2vec
import nltk.data
import logging
from MasterProject.data_Preprocessing.Datasets import get_data

data = get_data.Datasets()

spliter = nltk.data.load('tokenizers/punkt/english.pickle')

#clean the sentence
def clean_text(sentence):
       #remove the tags
       review = BeautifulSoup(sentence).get_text()
       #remove the the',",\
       review = re.sub(r"\\", "", review)
       review = re.sub(r"\'", "", review)
       review = re.sub(r"\"", "", review)
       #return the lower case
       tokens = review.lower().split()
       #remove non-alpha
       tokens = [token for token in tokens if token.isalpha()]

       return tokens
#get each review
def transfer_reviews_to_sentences(review,spliter):

    sentences_lists = []

    sentences = spliter.tokenize(review.strip())


    for s in sentences:
        if(len(s)>0):
            cleaned = clean_text(s)
            sentences_lists.append(cleaned)


    return sentences_lists


#------------------------IO------------------------------------

#1:load all docs in a directory
def gather_all_sentences(dataset):

    #initialize sentences lists
    all_sentences = []

    for review in dataset:

        all_sentences +=transfer_reviews_to_sentences(review,spliter)


    return all_sentences

#2:load the file and return the text
def load_file(filename):
    file = open(filename,'r')
    text = file.read()

    file.close()
    return text


#save a list to a file
def save_to_file(token,filename):
    data = '\n'.join(token)
    file = open(filename,'w')
    file.write(data)
    file.close()



#-------------------------main---------------------------------------------------

train_data = data.get_train_data()
unsup_data = data.get_unlabeled_data()
print(len(train_data)+ len(unsup_data))

train_sentences = gather_all_sentences(train_data["review"])
unsup_sentences = gather_all_sentences(unsup_data["review"])
sentences = []
sentences = train_sentences + unsup_sentences
print(sentences)
print(len(sentences))

#set the format
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

#create a model
num_features = 300 #dimensionality
min_word_count = 30 #filter
num_workers = 8 #
context = 10 #contect window size
downsampling = 1e-3

print("Training models!")

model = word2vec.Word2Vec(sentences,workers=num_workers,size=num_features,min_count=min_word_count,window=context,sample=downsampling)

#init_sims wiil make the model memory efficient
model.init_sims(replace=True)

#save the modle
filename = "300features_30minwords_10window"
model.save(filename)

#save the modle in text form
filename1 = '300dim_30min_10windows.txt'
model.wv.save_word2vec_format(filename1,binary=False)
