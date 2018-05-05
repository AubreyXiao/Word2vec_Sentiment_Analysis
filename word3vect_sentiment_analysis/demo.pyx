#import pandas
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from gensim.models import word2vec
import nltk.data
import logging

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

#clean the text
def clean_text(raw_review, remove_stopwords=False):
    #1：remove the URL
    review = BeautifulSoup(raw_review,"html5lib").get_text()
    #2：remove the non alphbate
    letters_only = re.sub("[^a-zA-Z]", " ",review)
    #3：convert to lowercase and split
    words = letters_only.lower().split()
    #4: optional remove the stopwords
    if(remove_stopwords):

      stops = set(stopwords.words("English"))
      words = [w for w in words if not w in stops]

    #return the cleaned text
    return(words)

#transfer a review to sentences ( word2vec will take lists of sentense as input)
def reviews_to_sentences(review, sent_detector,remove_stopwords=False):

    #split the reviews into sentences

    sentences = sent_detector.tokenize(review.strip())

    #loop over each sentence and clean it -> append to a list
    sentences_lists = []

    for sentence in sentences:

        if(len(sentence)>0):

         sentences_lists.append(clean_text(sentence,remove_stopwords))

    return(sentences_lists)




#load the data
train_data = pd.read_csv("Datasets/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test_data = pd.read_csv("Datasets/testData.tsv",header=0, delimiter="\t",quoting= 3)
unlabeled_data = pd.read_csv("Datasets/unlabeledTrainData.tsv",header=0, delimiter="\t",quoting=3)


#gather a list of all the sentences

sentences = []

for review in unlabeled_data["review"]:
    sentences += reviews_to_sentences(review,sent_detector)


for review in train_data["review"] :
    sentences += reviews_to_sentences(review,sent_detector)


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
filename = "300features_40minwords_10window.txt"
model.wv.save_word2vec_format(filename,binary=False)