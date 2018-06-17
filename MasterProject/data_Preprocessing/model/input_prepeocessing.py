from os import listdir
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from gensim.models import word2vec
import nltk.data
import logging


#------------------------------------------------------------

spliter = nltk.data.load('tokenizers/punkt/english.pickle')

#clean the sentence
def clean_text(sentence,sub_stopwords=False):
    #remove tags
    sentence = BeautifulSoup(sentence,"html5lib").get_text()
    #remove non alphabate
    sentence = re.sub("[^a-zA-Z]", " ", sentence)
    #lowercase
    words = sentence.lower().split()

    if(sub_stopwords):
        stops = set(stopwords.words("English"))
        words = [w for w in words if not w in stops]

    return words


#get each review
def transfer_reviews_to_sentences(review,spliter,sub_stopwords=False):
    sentences_lists = []

    sentences = spliter.tokenize(review.strip())
    for s in sentences:
        if(len(s)>0):
            cleaned = clean_text(s,sub_stopwords=False)
            sentences_lists.append(cleaned)


    return sentences_lists

#------------------------IO------------------------------------

#1:load all docs in a directory
def gather_all_sentences(directory):

    #initialize sentences lists
    all_sentences = []

    for filename in listdir(directory):
       if not filename.endswith(".txt"):
           continue

       path = directory + '/' + filename
       review = load_file(path)

       #print(transfer_reviews_to_sentences(review,spliter,sub_stopwords=False))
       all_sentences +=transfer_reviews_to_sentences(review,spliter,sub_stopwords=False)


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


negative_sentences = gather_all_sentences('/Users/xiaoyiwen/Desktop/datasets/train/neg1')
positive_sentences = gather_all_sentences('/Users/xiaoyiwen/Desktop/datasets/train/pos1')
unlabeled_sentences = gather_all_sentences('/Users/xiaoyiwen/Desktop/datasets/train/unsup')

sentences = negative_sentences + positive_sentences + unlabeled_sentences
print(len(sentences))

#set the format
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

#create a model
num_features = 300 #dimensionality
min_word_count = 40 #filter frequence>40
num_workers = 4 # 4threads to run in parrell
context = 10 #contect window size
downsampling = 1e-3

print("Training models!")

model = word2vec.Word2Vec(sentences,workers=num_workers,size=num_features,min_count=min_word_count,window=context,sample=downsampling)

#init_sims wiil make the model memory efficient
model.init_sims(replace=True)

#save the modle
filename = "300features_40minwords_10window"
model.save(filename)