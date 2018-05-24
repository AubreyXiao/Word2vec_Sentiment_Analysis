#import pandas
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import nltk.data
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# load the data
train_data = pd.read_csv("Datasets/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test_data = pd.read_csv("Datasets/testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_data = pd.read_csv("Datasets/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)


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


#get each datastes  average vector
def get_averageVector(words, model, num_features):

    #initialize a array (300,)
    featurevec = np.zeros((num_features,),dtype="float32")

    #count the words in a paragraph
    nwords = 0

    #get the vocab
    vocab = set(model.wv.index2word)

    #loop all the words in the praragrap

    for w in words:
        if(w in vocab):
            nwords +=1
            featurevec = np.add(featurevec, model[w])

    featurevec = np.divide(featurevec,nwords)


    return featurevec




#get the average of the test or train
def get_total_vector(reviews, model, num_features):


    #initial a counter
    Counter = 0
    #initial a 2 numpy array
    review_arr = np.zeros((len(reviews),num_features),dtype="float32")

    for words in reviews:

        if(Counter%1000==0):
            print("Review %d of %d" % (Counter,len(reviews)))

        review_arr[Counter] = get_averageVector(words,model,num_features)

        Counter = Counter + 1

    return review_arr




#get the datasets
def get_clean_review_to_words_lists(reviews):
    clean_datsets = []
    for review in reviews:
        clean_datsets.append(clean_text(review,remove_stopwords=True))

    return clean_datsets



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



#gather a list of all the sentences

sentences = []

for review in unlabeled_data["review"]:
    sentences += reviews_to_sentences(review,sent_detector)


for review in train_data["review"] :
    sentences += reviews_to_sentences(review,sent_detector)



print("loading the model")

#load the model
modelname = "300features_40minwords_10window"
model = Word2Vec.load(modelname)

#set num_features

num_features = 300

print("cleaning the training datasets")

#get clean train
clean_train_lists = get_clean_review_to_words_lists(train_data["review"])
train_data_vector = get_total_vector(clean_train_lists,model,num_features)

print("cleaning the test datasets")

#get clean test
clean_test_lists = get_clean_review_to_words_lists(test_data["review"])
test_data_vector = get_total_vector(clean_test_lists,model,num_features)



#build up a neural network classifier

#
# #create the RandomForest classifier
# #100 trees
# forest = RandomForestClassifier(n_estimators=100)
# #
# forest = forest.fit(train_data_vector,train_data["sentiment"])
#
# #test the result
# results = forest.predict(test_data_vector)
#
# #write the output to the test results
# output = pd.DataFrame(data={"id":test_data["id"],"sentiment":results})
# output.to_csv("model_prediction.csv",index=False,quoting=3)