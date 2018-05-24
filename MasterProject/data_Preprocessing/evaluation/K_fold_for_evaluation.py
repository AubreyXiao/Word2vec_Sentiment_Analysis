import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import nltk.data
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold


#--------------------load the pickel-------------------------------------------------------

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

#-------------------function---------------------------------------

def return_averaged_vector_review(dimension,review,model):

    review_vector = np.zeros((dimension,),dtype="float32")

    vocab = set(model.wv.index2word)

    num_words = 0

    for word in review:
        if(word in vocab):
            num_words += 1
            review_vector = np.add(review_vector,model[word])


    #divide the result
    review_vector = np.divide(review_vector,num_words)

    return review_vector


#gather all the reviews in the traning datasets
def return_total_vector(reviews, model, dimension):

    #should be 25000(train reviews) * 300(dimension)
    matrix = np.zeros((len(reviews),dimension),dtype="float32")

    count = 0

    for review in reviews:

        if (count % 1000 == 0):
            print("Review %d of %d" % (count, len(reviews)))

        matrix[count] = return_averaged_vector_review(dimension,review,model)
        count = count +1

    return matrix


#get all the review
def get_clean_review_lists(review_datasets):
    clean_review_datasets = []
    for review in review_datasets:
        clean_review_datasets.append(clean_text(review))
    return clean_review_datasets


#preprocess the text and split each review to a list of tokens
def clean_text(review):
    #1: remove the tags
    review = BeautifulSoup(review).get_text()
    #2.remove the non alpha
    text = re.sub("[^a-zA-Z]", " ",review)
    #3.remove split the tokens
    lowercase =  text.lower().split()
    #remove the stopwords
    stops = set(stopwords.words("English"))
    words = [w for w in lowercase if not w in stops]

    return words


#split each review into sentences and clean the sentences -> to a list
def transfer_review_to_sentences(review, sent_detector):

    #split the review to a list
    sentences = sent_detector.tokenize(review.strip())

    sentencs_list = []
    for sentence in sentences:
        if(len(sentence)>0):
         words = clean_text(sentence)
         sentencs_list.append(words)

    return sentencs_list

#-------------------------main---------------------------------------------------



#-----load the training datasets----
dir = "/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/sentiment_classifier/kaggle_dataset/"

train_data = pd.read_csv(dir +"labeledTrainData.tsv",header = 0, delimiter='\t', quoting=3)
unlabeled_data = pd.read_csv(dir + "unlabeledTrainData.tsv",header = 0, delimiter='\t',quoting=3)
test_data = pd.read_csv(dir + "testData.tsv",header=0, delimiter='\t',quoting=3)

#-----split the dataset-----cross validation-----

seed = 2000

x_train = train_data["review"]
y_train = train_data["sentiment"]

x_test = test_data["review"]



#-----------load the model-----------------------------------------------------

modelname = "/Users/xiaoyiwen/Desktop/Sentiment_Analysis_word2vec/MasterProject/data_Preprocessing/300features_40minwords_10window"
model = Word2Vec.load(modelname)
dimension = 300


#--------------------------getcleandatastes-------------------------------------

train_cleaned_reviews = get_clean_review_lists(x_train)
test_cleaned_reviews = get_clean_review_lists(x_test)

#dimensinon=300
x_train = return_total_vector(train_cleaned_reviews, model, dimension)
x_test = return_total_vector(test_cleaned_reviews,model,dimension)


#print(ytrain = np.array(train_data["sentiment"]))

y_train = np.array(y_train)


#---------using K-fold to evaluate the model----------------------------

seed = 7

#define 10 fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
acc_scores = []

for train, validation in kfold.split(x_train,y_train):
    #interate test model
    #define model
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=300))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train[train], y_train[train], epochs=20, batch_size=20, verbose=0)

    #evaluate the model
    scores = model.evaluate(x_train[validation],y_train[validation],verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1],scores[1]*100))
    acc_scores.append(scores[1]*100)

#average
print("%.2f%% (+/- %.2f%%)" % (np.mean(acc_scores), np.std(acc_scores)))




























